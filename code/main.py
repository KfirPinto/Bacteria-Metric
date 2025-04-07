from model import SplitAutoencoder
from train import train_model
import torch
from torch.utils.data import Dataset, DataLoader
from TensorDataset import TensorDataset
import numpy as np

def main():
    # load data
    tensor_path = "data/data_files/gene_families/Intersection/tensor.npy"
    data_tensor = np.load(tensor_path) 
    print(f"Data tensor shape: {data_tensor.shape}")

    # Convert to torch tensor
    data_tensor = torch.tensor(data_tensor, dtype=torch.float32)

    # Split the tensor along the bacteria dimension (dimension 1)
    num_bacteria = data_tensor.shape[1]
    split_index = int(0.7 * num_bacteria)  # 70% for training

    # Split the tensor into training and testing sets (70-30)
    train_tensor = data_tensor[:, :split_index, :]  # First 70% of bacteria
    test_tensor = data_tensor[:, split_index:, :]   # Remaining 30% of bacteria

     # Create Dataset instances for training and testing
    train_dataset = TensorDataset(train_tensor)
    test_dataset = TensorDataset(test_tensor)

    # Use DataLoader to load batches of data
    batch_size = 64  # Choose an appropriate batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    gene_dim = data_tensor.shape[-1]  # gene dim
    embedding_dim = 8  # Example embedding size (2b, where b=4) so the decoding would be from dimension 4 to gene_dim
    model = SplitAutoencoder(gene_dim=gene_dim, embedding_dim=embedding_dim)

    # Train the model
    print("Training model...")
    trained_model = train_model(model, train_tensor, num_epochs=5, learning_rate=0.001)

    # Save trained model
    trained_model_path = "split_autoencoder.pt"
    torch.save(trained_model, trained_model_path)

if __name__ == "__main__":
    main()
