import torch
from torch import nn
from .encoder import Encoder
from .causality_enhanced_gw import CEGW
from .graph_generation import GraphGeneration

class CEGLo(nn.Module):
    """
        Refference Code: https://github.com/zezhishao/STEP/blob/github/step/step_arch/step.py
    """
    def __init__(self, pretrained_encoder_path, encoder_args, gw_args, gragen_args):
        super().__init__()

        self.pretrained_encoder_path = pretrained_encoder_path
        self.encoder = Encoder(**encoder_args) 
        self.prediction = CEGW(**gw_args)
        # Load the pretrained encoder weights
        self.load_pretrained_encoder()
        # Initialize graph generation module with provided arguments      
        self.graph_generation = GraphGeneration(**gragen_args) 

    def load_pretrained_encoder(self):
        # Load the checkpoint dictionary from the pre-trained model path
        checkpoint_dict = torch.load(self.pretrained_encoder_path)
        self.encoder.load_state_dict(checkpoint_dict["model_state_dict"])
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, history, long_history, future, batch_seen, epoch, **kwargs) -> torch.Tensor:
        # long_history: input into encoder to generate embeddings
        # history: input into CE-GW
        batch_size, _, num_nodes, _ = history.shape
        
        unnorm_probability, embeddings, knn_graph, dynamic_static_graph = self.graph_generation(long_history, self.encoder)
        normalized_probability = unnorm_probability.softmax(-1)[..., 0].clone().reshape(batch_size, num_nodes, num_nodes)
        
        # Use the last hidden state for each node
        embeddings = embeddings[:, :, -1, :]
        
        y_hat = self.prediction(history, embeddings=embeddings, dynamic_static_graph=dynamic_static_graph).transpose(1, 2)
        predicted_data = y_hat.unsqueeze(-1)
        
        lambda_graph = 0.1
        
        return predicted_data, normalized_probability, knn_graph, lambda_graph