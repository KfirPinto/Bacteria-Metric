import torch
import torch.nn as nn
import torch.nn.functional as F

class SplitVAE(nn.Module):
    def __init__(self, gene_dim, embedding_dim=128):
        super(SplitVAE, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(gene_dim, 4096),
            nn.LeakyReLU(0.01),
            nn.Linear(4096, 1024),
            nn.LeakyReLU(0.01)
        )

        # Separate layers for mean and log-variance
        self.fc_mu = nn.Linear(1024, embedding_dim)
        self.fc_logvar = nn.Linear(1024, embedding_dim)

        # Decoder: input is (H_i + H_j) ∈ ℝ^b
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim // 2, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 4096),
            nn.LeakyReLU(0.01),
            nn.Linear(4096, gene_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        # Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1
        eps = torch.randn_like(std) 
        return mu + eps * std

    def forward(self, x):
        """
        x: input tensor of shape (n, d, gene_dim)
        Returns:
            - H_ij: latent representation (n, d, 2b)
            - x_hat: reconstruction (n, d, gene_dim)
            - mu, logvar: for computing KL divergence
        """
        batch_shape = x.shape[:2]  # (n, d)
        x_flat = x.view(-1, x.shape[-1])  # (n*d, gene_dim)

        # Encoder to shared hidden representation
        hidden = self.encoder(x_flat)  # (n*d, 1024)

        # Mean and log-variance for latent space
        mu = self.fc_mu(hidden)       # (n*d, 2b)
        logvar = self.fc_logvar(hidden)

        # Sample from latent space using reparameterization
        z = self.reparameterize(mu, logvar)  # (n*d, 2b)

        # Split into H_i and H_j
        b = z.shape[-1] // 2
        H_i = z[:, :b]
        H_j = z[:, b:]
        H_sum = H_i + H_j  # (n*d, b)

        # Decode
        x_reconstructed = self.decoder(H_sum)  # (n*d, gene_dim)

        # Reshape outputs
        x_reconstructed = x_reconstructed.view(*batch_shape, -1)
        z = z.view(*batch_shape, -1)
        mu = mu.view(*batch_shape, -1)
        logvar = logvar.view(*batch_shape, -1)

        return z, x_reconstructed, mu, logvar