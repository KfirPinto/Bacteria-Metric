import torch
from torch.utils.data import Dataset

class TensorDataset(Dataset):
    def __init__(self, gene_families, pathways, batch_size=64):
        """
        gene_families: Tensor of shape (N, D, gene_dim)
        pathways: Tensor of shape (N, D, pathway_dim)
        """
        assert gene_families.shape[:2] == pathways.shape[:2], "Input and target dimensions must match in (N, D)"
        self.gene_families = gene_families
        self.pathways = pathways
        self.batch_size = batch_size

    def __len__(self):
        return self.gene_families.size(0)

    def __getitem__(self, idx):
        return self.gene_families[idx], self.pathways[idx]
