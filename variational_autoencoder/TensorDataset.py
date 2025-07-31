import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TensorDataset(Dataset):
    def __init__(self, tensor, batch_size=128):    
        self.tensor = tensor
        self.batch_size = batch_size

    def __len__(self):
        return self.tensor.size(0)
    
    def __getitem__(self, idx):
        return self.tensor[idx]