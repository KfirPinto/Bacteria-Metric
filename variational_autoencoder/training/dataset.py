import torch
from torch.utils.data import TensorDataset, DataLoader

def create_dataloaders(train_tensor, batch_size=64):
    train_dataset = TensorDataset(train_tensor)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, persistent_workers=False
    )
  
    return train_loader