import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TensorDataset(Dataset):
    def __init__(self, tensor, batch_size):    
        self.tensor = tensor
        self.batch_size = batch_size

    def __len__(self):
        # Return number of batches in the dataset.
        return (self.tensor.size(0) + self.batch_size - 1) // self.batch_size
    
    def __getitem__(self, idx):
        # Retrieve a whole batch of data.
        start_idx = idx * self.batch_size # calculate the start index for the batch
        end_idx = min(start_idx + self.batch_size, self.tensor.size(0)) 
        batch = self.tensor[start_idx:end_idx]  
        return batch
    
    @staticmethod
    def custom_collate_fn(batch):
        # Batch should already be a list of tensors, we can concatenate them
        return torch.cat(batch, dim=0)

"""
if __name__ == "__main__":
    tensor_path = "/home/bcrlab/barsapi1/Bacteria-Metric/data/data_files/gene_families/Intersection/tensor.npy"
    data_tensor = np.load(tensor_path) 
    data_tensor = torch.tensor(data_tensor, dtype=torch.float32)

    # Initialize dataset and dataloader
    train_dataset = TensorDataset(data_tensor, batch_size=20)
    train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=custom_collate_fn)

    # Verify by printing batch sizes
    for batch in train_loader:
        print(batch.shape)  # Should print torch.Size([64, 10]) for each batch
       # Stop after printing the first batch for debugging
"""