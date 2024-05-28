import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from basicts.utils import load_pkl


def sample_gumbel(shape, eps=1e-20, device=None):
    uniform = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(uniform + eps) + eps))

def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps, device=logits.device)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
    # Apply Gumbel-Softmax to logits with optional hard sampling
    y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape).to(logits.device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

class GraphGeneration(nn.Module):

    def __init__(self, k):
        super().__init__()

        self.k = k 
        self.num_nodes = 14
        self.train_length = 3500
        self.global_k = 6
        self.local_k = 14
        # Load node features from preprocessed data
        self.node_feats = torch.from_numpy(load_pkl("datasets/DYG_doz/data_in12_out12.pkl")["processed_data"]).float()[:self.train_length, :, 0]
        self.scores = torch.from_numpy(np.load('result_DYG_doz/F_DYG.npy')).requires_grad_(False)
        # Define dimensions for fully connected layers
        self.dim_fc = 55712
        self.embedding_dim = 100
        
        # Define convolutional layers
        self.conv1 = torch.nn.Conv1d(1, 8, 10, stride=1)  
        self.conv2 = torch.nn.Conv1d(8, 16, 10, stride=1)  
        self.fc = torch.nn.Linear(self.dim_fc, self.embedding_dim)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.bn3 = torch.nn.BatchNorm1d(self.embedding_dim)

        # Define dimensions for mean fully connected layer
        self.dim_fc_mean = 16128
        self.fc_mean = nn.Linear(self.dim_fc_mean, 100)

        # Define additional fully connected layers
        self.fc_cat = nn.Linear(self.embedding_dim, 2)
        self.fc_out = nn.Linear((self.embedding_dim) * 2, self.embedding_dim)
        self.dropout = nn.Dropout(0.5)

        def encode_one_hot(labels):
            # Encode labels into one-hot vectors
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
            labels_one_hot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
            return labels_one_hot

        # Create relational matrices for nodes
        self.rel_rec = torch.FloatTensor(np.array(encode_one_hot(np.where(np.ones((self.num_nodes, self.num_nodes)))[0]), dtype=np.float32))
        self.rel_send = torch.FloatTensor(np.array(encode_one_hot(np.where(np.ones((self.num_nodes, self.num_nodes)))[1]), dtype=np.float32))

    def build_knn_graph(self, data, k=6*14):

        def batch_cosine_similarity(x, y):
            # Compute the L2 norms
            l2_x = torch.norm(x, dim=2, p=2) + 1e-7  
            l2_y = torch.norm(y, dim=2, p=2) + 1e-7  
            
            # Compute the outer product of the L2 norms
            l2_m = torch.matmul(l2_x.unsqueeze(dim=2), l2_y.unsqueeze(dim=2).transpose(1, 2))
            
            # Compute the dot product of the input tensors
            l2_z = torch.matmul(x, y.transpose(1, 2))
            
            # Calculate the cosine similarity
            cos_affnity = l2_z / l2_m
            
            adj = cos_affnity
            return adj
                
        # Compute k-nearest neighbors based on the specified metric
        batch_sim = batch_cosine_similarity(data, data)
        
        batch_size, num_nodes, _ = batch_sim.shape
        adj = batch_sim.view(batch_size, num_nodes*num_nodes)
        res = torch.zeros_like(adj)
        
        # Select top-k similarities and create adjacency matrix
        top_k, indices = torch.topk(adj, k, dim=-1)
        res.scatter_(-1, indices, top_k)
        adj = torch.where(res != 0, 1.0, 0.0).detach().clone()
        adj = adj.view(batch_size, num_nodes, num_nodes)
        adj.requires_grad = False
        return adj

    def forward(self, long_history, encoder):

        device = long_history.device
        batch_size, _, _, _ = long_history.shape
        num_nodes = self.num_nodes
        
        # Load precomputed feature tensor
        scores = self.scores.to(device)

        # Prepare global features and fuse them
        global_feat = self.node_feats.to(device).transpose(1, 0).view(num_nodes+6, 1, -1) 
        global_feat_fusion = torch.einsum('ik, kab -> iab', scores, global_feat[self.local_k:(self.local_k+self.global_k), :, :]) + global_feat[0:self.local_k, :, :]

        # Apply convolutional and batch normalization layers
        global_feat_fusion = self.bn2(F.relu(self.conv2(self.bn1(F.relu(self.conv1(global_feat_fusion))))))
        global_feat_fusion = global_feat_fusion.view(num_nodes, -1) 
        global_feat_fusion = F.relu(self.fc(global_feat_fusion)) 
        global_feat_fusion = self.bn3(global_feat_fusion)
        global_feat_fusion = global_feat_fusion.unsqueeze(0).expand(batch_size, num_nodes, -1) 
        
        # Encode long-term history using the encoder
        embeddings = encoder(long_history[..., [0]]) 
        node_feat = global_feat_fusion

        # Compute sender and receiver features for edges
        receivers = torch.matmul(self.rel_rec.to(node_feat.device), node_feat) 
        senders = torch.matmul(self.rel_send.to(node_feat.device), node_feat)
        edge_feat = torch.cat([senders, receivers], dim=-1) 
        edge_feat = torch.relu(self.fc_out(edge_feat)) 

        # Compute unnormalized Bernoulli logits for edges
        unnorm_probability = self.fc_cat(edge_feat) 

        # Sample adjacency matrix using Gumbel-Softmax
        dynamic_static_graph = gumbel_softmax(unnorm_probability, temperature=0.5, hard=True)
        dynamic_static_graph = dynamic_static_graph[..., 0].clone().reshape(batch_size, num_nodes, -1)
        
        # Mask the diagonal to prevent self-loops
        mask = torch.eye(num_nodes, num_nodes).unsqueeze(0).bool().to(dynamic_static_graph.device)
        dynamic_static_graph.masked_fill_(mask, 0)

        # Compute k-nearest neighbor adjacency matrix
        knn_graph = self.build_knn_graph(embeddings.reshape(batch_size, num_nodes, -1), k=self.k*self.num_nodes)
        mask = torch.eye(num_nodes, num_nodes).unsqueeze(0).bool().to(knn_graph.device)
        knn_graph.masked_fill_(mask, 0)

        return unnorm_probability, embeddings, knn_graph, dynamic_static_graph