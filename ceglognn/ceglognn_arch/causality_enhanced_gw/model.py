import torch
from torch import nn
import torch.nn.functional as F
from basicts.utils import load_adj
"""
    Refference Code: https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py
"""
class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # Ensure adjacency matrix is on the same device as input
        A = A.to(x.device)
        
        # Perform graph convolution using Einstein summation notation
        if len(A.shape) == 3:
            x = torch.einsum('ncvl,nvw->ncwl', (x, A))
        else:
            x = torch.einsum('ncvl,vw->ncwl', (x, A))
        
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        # Define a 2D convolution layer as MLP
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        
    def forward(self, x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        
        # Adjust input channels for multiple supports and orders
        c_in = (order * support_len + 1) * c_in

        # Define a linear layer
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x] 
        for a in support: 
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        # Concatenate along the channel dimension
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class CEGW(nn.Module):
    
    def __init__(self, num_nodes, support_len, dropout=0.3, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2, out_dim=12,
                 residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2, **kwargs):
        super(CEGW, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        # Initialize various convolutional and batch norm layers
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.fc_his = nn.Sequential(nn.Linear(96, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))

        receptive_field = 1
        self.supports_len = support_len

        # Adaptive adjacency matrix initialization
        if gcn_bool and addaptadj:
            if aptinit is None:
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)
                self.supports_len += 1
            else:
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
                self.supports_len += 1

        # Define layers in blocks
        for b in range(blocks):
            additional_scope = kernel_size - 1 
            new_dilation = 1
            for i in range(layers):
                # Define filter convolution layer
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))

                # Define gate convolution layer
                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))

                # Define residual convolution layer
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels, out_channels=residual_channels, kernel_size=(1, 1)))

                # Define skip convolution layer
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels, out_channels=skip_channels, kernel_size=(1, 1)))
                
                # Define batch normalization layer
                self.bn.append(nn.BatchNorm2d(residual_channels))

                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        # Define the final convolution layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1, 1), bias=True)

        self.receptive_field = receptive_field

    def _calculate_random_walk_matrix(self, adj_mx):
        B, N, N = adj_mx.shape

        # Add self-loops to the adjacency matrix
        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[1])).unsqueeze(0).expand(B, N, N).to(adj_mx.device)
        
        # Calculate degree matrix
        d = torch.sum(adj_mx, 2)
        d_inv = 1. / d
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(adj_mx.device), d_inv)
        d_mat_inv = torch.diag_embed(d_inv)
        
        # Calculate random walk matrix
        random_walk_mx = torch.bmm(d_mat_inv, adj_mx)
        return random_walk_mx

    def forward(self, input, embeddings, dynamic_static_graph):
        # Transpose the input tensor for appropriate dimension alignment
        input = input.transpose(1, 3)
        
        # Pad the input tensor to adjust its dimensions
        input = nn.functional.pad(input, (1, 0, 0, 0))

        # Select the first two channels of the input tensor
        input = input[:, :2, :, :]
        in_len = input.size(3)
        
        # Pad the input tensor if its length is less than the receptive field
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        
        # Apply the initial convolution
        x = self.start_conv(x)
        skip = 0

        # Load adjacency matrix and calculate supports
        adj_dyg, _ = load_adj("datasets/DYG_doz/causality_graph.pkl", "doubletransition")
        self.supports = [self._calculate_random_walk_matrix(dynamic_static_graph)] + [torch.tensor(i) for i in adj_dyg]

        new_supports = None
        # If graph convolution and adaptive adjacency matrix are enabled, calculate new supports
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # Iterate through the layers and blocks
        for i in range(self.blocks * self.layers):
            residual = x
            
            # Apply the filter and gate convolutions with non-linear activations
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # Apply the skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            # Apply graph convolution or residual convolution based on the configuration
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            # Add residual connection and apply batch normalization
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        # Process hidden states and add to the skip connection
        embeddings = self.fc_his(embeddings)        
        embeddings = embeddings.transpose(1, 2).unsqueeze(-1) 
        skip = skip + embeddings
        
        # Apply final layers with ReLU activation
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        # Squeeze and transpose the final output tensor
        x = x.squeeze(-1).transpose(1, 2)
        return x
