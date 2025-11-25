import torch
import torch.nn as nn

class PathwayReg(nn.Module):
    def __init__(self, gene_dim, embedding_dim, pathway_dim):  
        super(PathwayReg, self).__init__()

        # Part A: compress from gene_dim down to embedding_dim
        self.encoder = nn.Sequential(
            nn.Linear(gene_dim, 1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(512, embedding_dim) # where embedding_dim = 2b
        )

        self.activation = nn.LeakyReLU(negative_slope=0.1)

        # Part B: input is (H_i + H_j) ∈ ℝ^b, predict pathway activity
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim // 2, 256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(256, pathway_dim),
            nn.LeakyReLU(negative_slope=0.1),
        )

    def forward(self, x):
        """
        x: gene families tensor of shape (n, d, gene_dim)
        Returns:
            - embedding H_ij ∈ ℝ^{n, d, 2b}
            - predictions ∈ ℝ^{n, d, pathway_dim}
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
        predictions = self.decoder(H_sum)  # (n*d, pathway_dim) 

        # Reshape back to (n, d, pathway_dim) and (n, d, embedding_dim)

        predictions = predictions.view(*batch_shape, -1)
        H_ij = H_ij.view(*batch_shape, -1)

        return H_ij, predictions
