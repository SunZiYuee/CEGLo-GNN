import torch
from torch import nn
from timm.models.vision_transformer import trunc_normal_
import numpy as np
from .patch import PatchEmbedding
from .mask import MaskGenerator
from .positional_encoding import PositionalEncoding
from .transformer_layers import TransformerLayers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):

    def __init__(self, patch_size, in_channel, embed_dim, num_heads, mlp_ratio, dropout, num_token, mask_ratio, encoder_depth, decoder_depth, mode="pre-train"):
        super().__init__()
        assert mode in ["pre-train", "forecasting"], "Error mode."
        
        # Initialize parameters
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_token = num_token
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.mode = mode
        self.mlp_ratio = mlp_ratio
        self.selected_feature = 0
        self.global_k = 6
        self.local_k = 14
        self.scores = torch.from_numpy(np.load('result_DYG_doz/F_DYG.npy')).requires_grad_(False)
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, norm_layer=None)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout)
        # Define mask generator
        self.mask = MaskGenerator(num_token, mask_ratio)
        self.encoder = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)
        self.enc_2_dec_emb = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.decoder = TransformerLayers(embed_dim, decoder_depth, mlp_ratio, num_heads, dropout)
        self.output_layer = nn.Linear(embed_dim, patch_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize positional encoding weights
        nn.init.uniform_(self.positional_encoding.position_embedding, -.02, .02)
        trunc_normal_(self.mask_token, std=.02)

    def encoding_local(self, long_term_history, mask=True):
        batch_size, num_nodes, _, _ = long_term_history.shape
        patches = self.patch_embedding(long_term_history)     
        patches = patches.transpose(-1, -2)         
        patches = self.positional_encoding(patches)
        # Apply masking if required
        if mask:
            unmasked_token_index, masked_token_index = self.mask()
            encoder_input = patches[:, :, unmasked_token_index, :]
        else:
            unmasked_token_index, masked_token_index = None, None
            encoder_input = patches 
        # Pass through encoder layers
        embeddings_unmasked = self.encoder(encoder_input) 
        embeddings_unmasked = self.encoder_norm(embeddings_unmasked).view(batch_size, num_nodes, -1, self.embed_dim) 

        return embeddings_unmasked, unmasked_token_index, masked_token_index

    def encoding_global(self, long_term_history, mask=True):
        batch_size, num_nodes, _, _ = long_term_history.shape
        patches = self.patch_embedding(long_term_history)     
        patches = patches.transpose(-1, -2)         
        patches = self.positional_encoding(patches)
        if mask:
            unmasked_token_index, masked_token_index = self.mask()
            encoder_input = patches[:, :, unmasked_token_index, :]
        else:
            unmasked_token_index, masked_token_index = None, None
            encoder_input = patches 

        embeddings_unmasked = self.encoder(encoder_input) 
        embeddings_unmasked = self.encoder_norm(embeddings_unmasked).view(batch_size, num_nodes, -1, self.embed_dim) 

        return embeddings_unmasked, unmasked_token_index, masked_token_index

    def decoding(self, embeddings_unmasked, masked_token_index):

        batch_size, num_nodes, _, _ = embeddings_unmasked.shape
        embeddings_unmasked = self.enc_2_dec_emb(embeddings_unmasked)
        embeddings_masked = self.positional_encoding(
            self.mask_token.expand(batch_size, num_nodes, len(masked_token_index), embeddings_unmasked.shape[-1]),
            index=masked_token_index 
        )
        # Concatenate unmasked and masked tokens
        embeddings_full = torch.cat([embeddings_unmasked, embeddings_masked], dim=-2) 
        # Pass through decoder layers
        embeddings_full = self.decoder(embeddings_full)
        embeddings_full = self.decoder_norm(embeddings_full) 

        reconstruction_full = self.output_layer(embeddings_full.view(batch_size, num_nodes, -1, self.embed_dim)) 

        return reconstruction_full

    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full, unmasked_token_index, masked_token_index):

        batch_size, num_nodes, _, _ = reconstruction_full.shape 
        
        # Extract masked tokens
        reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_index):, :]     
        reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)     
        
        label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)[:, :, :, self.selected_feature, :].transpose(1, 2)  
        label_masked_tokens = label_full[:, :, masked_token_index, :].contiguous() 
        label_masked_tokens = label_masked_tokens.view(batch_size, num_nodes + 6, -1).transpose(1, 2)  

        return reconstruction_masked_tokens, label_masked_tokens

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, epoch: int = None, **kwargs) -> torch.Tensor:
        history_data = history_data.permute(0, 2, 3, 1) 

        # Split history_data into local and global parts
        history_data_local = history_data[:, 0:self.local_k, :, :]
        history_data_global = history_data[:, self.local_k:(self.local_k+self.global_k), :, :]
        scores = self.scores.to(device)
        if self.mode == "pre-train":
            
            # Encode history data
            embeddings_unmasked_local, unmasked_token_index_local, masked_token_index_local = self.encoding_local(history_data_local)
            embeddings_unmasked_global, unmasked_token_index_global, masked_token_index_global = self.encoding_global(history_data_global)
            # Ensure the unmasked token indices match
            assert unmasked_token_index_local == unmasked_token_index_global, "DIFFERENT LIST"
            # Fuse local and global hidden states
            embeddings_unmasked_fusion = torch.einsum('ik, akbc -> aibc', scores, embeddings_unmasked_global) + embeddings_unmasked_local
            # Decode fused hidden states
            reconstruction_full = self.decoding(embeddings_unmasked_fusion, masked_token_index_global)
            # Get reconstructed masked tokens and corresponding labels
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(reconstruction_full, history_data, unmasked_token_index_global, masked_token_index_global)
            label_masked_fusion = torch.einsum('ik, abk -> abi', scores, label_masked_tokens[:, :, self.local_k:(self.local_k+self.global_k)]) + label_masked_tokens[:, :, 0:self.local_k]
            return reconstruction_masked_tokens, label_masked_fusion
        else:
            # Encode history data without masking
            embeddings_full_local, _, _ = self.encoding_local(history_data_local, mask=False)
            embeddings_full_global, _, _ = self.encoding_global(history_data_global, mask=False)

            embeddings_full_fusion = torch.einsum('ik, akbc -> aibc', scores, embeddings_full_global) + embeddings_full_local
            return embeddings_full_fusion