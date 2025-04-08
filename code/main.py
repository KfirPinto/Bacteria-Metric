from model import SplitAutoencoder
from train import train_model
import torch
from torch.utils.data import Dataset, DataLoader
from TensorDataset import TensorDataset
import numpy as np

def main():
    # load data
    tensor_path = "/home/bcrlab/barsapi1/Bacteria-Metric/data/data_files/gene_families/Intersection/tensor.npy"
    data_tensor = np.load(tensor_path) 
    print(f"Data tensor shape: {data_tensor.shape}")

    # Convert to torch tensor and shuffle
    data_tensor = torch.tensor(data_tensor, dtype=torch.float32)
    indices = torch.randperm(data_tensor.size(0)) # Shuffle indices of the *first* dimension
    data_tensor = data_tensor[indices]

    # Split the tensor along the bacteria dimension (dimension 1)
    num_bacteria = data_tensor.shape[1]
    split_index_train = int(0.7 * num_bacteria)  # 70% for training
    split_index_val = int(0.85 * num_bacteria)  # 15% for validation, the remaining 15% for testing

    # Split the tensor into training, validation and test sets (70-15-15)
    # 'x:' from index x to the end
    # ':x' from the start to index x
    train_tensor = data_tensor[:, :split_index_train, :] 
    val_tensor = data_tensor[:, split_index_train:split_index_val, :]   
    test_tensor = data_tensor[:, split_index_val:, :]
    np.save("test_tensor.npy", test_tensor.numpy())  # Save the test tensor for performance evaluation

    # Create Dataset instances for training and testing
    batch_size = 64  
    train_dataset = TensorDataset(train_tensor, batch_size)
    val_dataset = TensorDataset(val_tensor, batch_size)
    test_dataset = TensorDataset(test_tensor, batch_size)

    # Use DataLoader to load batches of data
    train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=TensorDataset.custom_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, shuffle=False, collate_fn=TensorDataset.custom_collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, shuffle=False, collate_fn=TensorDataset.custom_collate_fn, num_workers=4)

    # Initialize model
    gene_dim = data_tensor.shape[-1]  # gene dim
    embedding_dim = 8  # Example embedding size 

    model = SplitAutoencoder(gene_dim=gene_dim, embedding_dim=embedding_dim)

    # Train the model
    print("Training model...")
    trained_model = train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001)

    # Save trained model
    trained_model_path = "split_autoencoder.pt"
    torch.save(trained_model, trained_model_path)

if __name__ == "__main__":
    main()
