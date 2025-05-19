import torch
from torch.utils.data import TensorDataset, DataLoader

def create_dataloaders(train_tensor, val_tensor, batch_size=64):
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False, persistent_workers=False
    )

    return train_loader, val_loader