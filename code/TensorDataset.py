import torch
from torch.utils.data import Dataset

class TensorDataset(Dataset):
    def __init__(self, tensor):    
        self.tensor = tensor

    def __len__(self):
        # Return the number of batches in the dataset
        return self.tensor.size(0)

    def __getitem__(self, idx):
        # Return a single sample (idx-th element) from the dataset
        return self.tensor[idx]  