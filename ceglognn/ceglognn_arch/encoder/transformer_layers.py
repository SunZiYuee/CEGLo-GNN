import math
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerLayers(nn.Module):
    def __init__(self, hidden_dim, nlayers, mlp_ratio, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.d_model = hidden_dim
        # Single Transformer encoder layer
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * mlp_ratio, dropout)
        # Stack the encoder layers to form a Transformer encoder
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src):

        B, N, L, D = src.shape
        # Scale the input by the square root of the model dimension
        src = src * math.sqrt(self.d_model)
        # Reshape
        src = src.view(B * N, L, D)
        # Required shape: [sequence length, batch size, feature dimension] and pass the input through the Transformer encoder
        src = src.transpose(0, 1)
        output = self.transformer_encoder(src, mask=None)
        
        return output.transpose(0, 1).view(B, N, L, D)
