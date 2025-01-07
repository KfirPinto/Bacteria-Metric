import torch
import torch.nn as nn
import torch.nn.functional as F

class SplitAutoencoder(nn.Module):
    def __init__(self, bacteria_dim, gene_dim, embedding_dim):
        super(SplitAutoencoder, self).__init__()
        self.bacteria_dim = bacteria_dim
        self.gene_dim = gene_dim
        self.embedding_dim = embedding_dim

        # Person encoder: Flatten person data (bacteria_dim * gene_dim) and project it to embedding_dim
        self.person_encoder = nn.Sequential(
            nn.Linear(bacteria_dim * gene_dim, embedding_dim),  # Flattened input (x * y) to embedding_dim
            nn.ReLU()
        )

        # Bacteria encoder: Flatten bacteria data (gene_dim) and project it to embedding_dim
        self.bacteria_encoder = nn.Sequential(
            nn.Linear(gene_dim, embedding_dim),  # Flattened input (z) to embedding_dim
            nn.ReLU()
        )

        # Projection matrix: Adjust the combined embedding size to match the size of the gene expression
        self.projection_matrix = nn.Linear(embedding_dim, gene_dim)  # Mapping to gene_dim (matching bacteria row)

    def forward(self, person_data, bacteria_data):
        # Flatten person data (bacteria_dim * gene_dim) into a vector
        person_input_flat = person_data.reshape(-1, self.bacteria_dim * self.gene_dim)  # Flatten to 1D

        # Encode person data to get person embedding
        person_embedding = self.person_encoder(person_input_flat)  # Output: (embedding_dim)

        # Encode bacteria data to get bacteria embedding
        bacteria_embedding = self.bacteria_encoder(bacteria_data)  # Output: (embedding_dim)

        # Combine embeddings (sum of both embeddings)
        combined_embedding = person_embedding + bacteria_embedding  # (embedding_dim)

        # Project the combined embedding to match the original gene expression size (gene_dim)
        projected_embedding = self.projection_matrix(combined_embedding)  # Output: (gene_dim)

        return projected_embedding, person_embedding, bacteria_embedding, combined_embedding
