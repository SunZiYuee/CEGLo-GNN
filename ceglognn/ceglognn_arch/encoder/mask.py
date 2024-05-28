import random
from torch import nn

class MaskGenerator(nn.Module):

    def __init__(self, num_tokens, mask_ratio):
        super().__init__()
        # Initialize number of tokens and mask ratio
        self.num_tokens = num_tokens
        self.mask_ratio = mask_ratio
        self.sort = True

    def uniform_rand(self):
        # Create a list of token indices
        mask = list(range(int(self.num_tokens)))

        # Set a random seed
        random.seed(10)
        random.shuffle(mask)
        
        # Calculate the number of tokens to mask
        mask_len = int(self.num_tokens * self.mask_ratio)
        
        # Split the list into masked and unmasked tokens
        self.masked_tokens = mask[:mask_len]
        self.unmasked_tokens = mask[mask_len:]
        
        # Sort the token indices if required
        if self.sort:
            self.masked_tokens = sorted(self.masked_tokens)
            self.unmasked_tokens = sorted(self.unmasked_tokens)
        
        return self.unmasked_tokens, self.masked_tokens

    def forward(self):
        # Generate unmasked and masked tokens using uniform_rand method
        self.unmasked_tokens, self.masked_tokens = self.uniform_rand()
        return self.unmasked_tokens, self.masked_tokens