from torch import nn

class PatchEmbedding(nn.Module):

    def __init__(self, patch_size, in_channel, embed_dim, norm_layer):
        super().__init__()
        # Initialize parameters
        self.output_channel = embed_dim
        self.len_patch = patch_size
        self.input_channel = in_channel
        self.output_channel = embed_dim
        
        # Define convolutional layer for patch embedding
        self.input_embedding = nn.Conv2d(
            in_channel,
            embed_dim,
            kernel_size=(self.len_patch, 1),
            stride=(self.len_patch, 1)
        )
        
        # Define normalization layer
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()

    def forward(self, long_history):

        batch_size, num_nodes, num_feat, len_time_series = long_history.shape
        long_history = long_history.reshape(batch_size * num_nodes, num_feat, len_time_series, 1)
        # Apply convolutional layer, normalization layer to get patch embeddings
        output = self.norm_layer(self.input_embedding(long_history))
        # Remove the last dimension and reshape to separate batch size and number of nodes
        output = output.squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1)
        # Ensure the output shape is correct
        assert output.shape[-1] == len_time_series / self.len_patch
        
        return output
