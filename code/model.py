import torch
import torch.nn as nn
import torch.nn.functional as F

class SplitAutoencoder(nn.Module):
    def __init__(self, gene_dim, embedding_dim): # gene_dim = d, embedding_dim = 2b
        super(SplitAutoencoder, self).__init__()
        self.encoder = nn.Linear(gene_dim, embedding_dim)
        self.activation = nn.ReLU()
        self.decoder = nn.Linear(embedding_dim // 2, gene_dim) # input is H_i + H_j, which is of size b

    def forward(self, x):
        x_encoded = self.encoder(x)                 # shape: (num_persons, num_bacteria, embedding_dim)
        x_encoded = self.activation(x_encoded)
        H_i, H_j = self.split_embeddings(x_encoded) # each: (num_persons, num_bacteria, b)
        x_decoded = self.decoder(H_i+H_j)           # shape: (num_persons, num_bacteria, gene_dim)
        return x_encoded, x_decoded
    
    def split_embeddings(self, H_ij):
        """
        H_ij: Tensor of shape (num_persons, num_bacteria, 2b)
        returns:
            H_i: Tensor of shape (num_persons, num_bacteria, b)
            H_j: Tensor of shape (num_persons, num_bacteria, b)
        """
        b = H_ij.size(-1) // 2
        H_i = H_ij[..., :b]
        H_j = H_ij[..., b:]
        return H_i, H_j