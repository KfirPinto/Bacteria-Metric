import torch
from torch.utils.data import TensorDataset, DataLoader

def create_dataloaders(train_gene_families, train_pathways, val_gene_families, val_pathways, batch_size=64):
    train_dataset = TensorDataset(train_gene_families, train_pathways)   
    val_dataset = TensorDataset(val_gene_families, val_pathways)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
    )

    return train_loader, val_loader