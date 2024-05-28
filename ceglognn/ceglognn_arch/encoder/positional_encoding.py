import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, max_len: int = 1000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = nn.Parameter(torch.empty(max_len, hidden_dim), requires_grad=True)

    def forward(self, input_data, index=None, abs_idx=None):
        
        batch_size, num_nodes, num_patches, num_feat = input_data.shape
        input_data = input_data.view(batch_size * num_nodes, num_patches, num_feat) 
        
        # Select positional embeddings based on index or use the first part of the positional embedding
        if index is None:
            pe = self.position_embedding[:input_data.size(1), :].unsqueeze(0)
        else:
            pe = self.position_embedding[index].unsqueeze(0)
        
        # Add positional embeddings to input data
        input_data = input_data + pe
        input_data = self.dropout(input_data)
        
        # Reshape input data back to original dimensions
        input_data = input_data.view(batch_size, num_nodes, num_patches, num_feat)
        
        return input_data