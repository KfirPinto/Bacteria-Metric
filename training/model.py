import torch
import torch.nn as nn
import torch.nn.functional as F

class SplitAutoencoder(nn.Module):
    def __init__(self, gene_dim, embedding_dim=128):  # gene_dim = 30,000, embedding_dim = 2b
        super(SplitAutoencoder, self).__init__()

        # Encoder: compress from gene_dim down to 2b
        self.encoder = nn.Sequential(
            nn.Linear(gene_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim)  # where embedding_dim = 2b
        )

        self.activation = nn.ReLU()

        # Decoder: input is (H_i + H_j) ∈ ℝ^b
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim // 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, gene_dim)
        )

    def forward(self, x):
        """
        x: input tensor of shape (n, d, gene_dim)
        Returns:
            - encoded H_ij ∈ ℝ^{n, d, 2b}
            - decoded x_hat ∈ ℝ^{n, d, gene_dim}
        """
        batch_shape = x.shape[:2]  # (n, d)
        x_flat = x.view(-1, x.shape[-1])  # flatten to (n*d, gene_dim)

        # Encoding
        H_ij = self.encoder(x_flat)  # (n*d, 2b)
        H_ij = self.activation(H_ij)

        # Split into H_i and H_j
        b = H_ij.shape[-1] // 2
        H_i = H_ij[:, :b]  # (n*d, b)
        H_j = H_ij[:, b:]  # (n*d, b)

        # Sum H_i and H_j
        H_sum = H_i + H_j  # (n*d, b)

        # Decoding
        x_reconstructed = self.decoder(H_sum)  # (n*d, gene_dim)

        # Reshape back to (n, d, gene_dim) and (n, d, 2b)
        x_reconstructed = x_reconstructed.view(*batch_shape, -1)
        H_ij = H_ij.view(*batch_shape, -1)

        return H_ij, x_reconstructed
    
    def batch_log_zscore(x):
        """
        Applies log10(x + ε), then z-score normalization per bacteria across samples in the batch.
        Assumes x: (batch, bacteria, genes)
        """
        eps = torch.min(x[x > 0]) if torch.any(x > 0) else 1e-8
        x_log = torch.log10(x + eps)
        mean = x_log.mean(dim=0, keepdim=True)  # mean over batch
        std = x_log.std(dim=0, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std) * eps, std)
        z = (x_log - mean) / std
        return z

    