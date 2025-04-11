from model import SplitAutoencoder
from train import train_model
import torch
from torch.utils.data import Dataset, DataLoader
from TensorDataset import TensorDataset
import numpy as np

def main():

    # Load data
    data_tensor = np.load("data/data_files/gene_families/AsnicarF_march_2017/tensor.npy") 
    print(f"Data tensor shape: {data_tensor.shape}")

    # Load labels
    samples = np.load("data/data_files/gene_families/Intersection/AsnicarF_march_2017.npy", allow_pickle=True)
    bacteria = np.load("data/data_files/gene_families/Intersection/AsnicarF_march_2017.npy", allow_pickle=True)
    gene_families = np.load("data/data_files/gene_families/Intersection/AsnicarF_march_2017.npy", allow_pickle=True)

    # Convert the data tensor to a PyTorch tensor
    data_tensor = torch.tensor(data_tensor, dtype=torch.float32)

    # Shuffle the tensor along the first dimension (samples). same order for tensor and labels. 
    indices = torch.randperm(data_tensor.size(0)) 
    data_tensor = data_tensor[indices]
    samples = samples[indices]

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

    # Split the labels corrispondigly
    train_samples = samples[:, :split_index_train, :]
    val_samples = samples[:, split_index_train:split_index_val, :]
    test_samples = samples[:, split_index_val:, :]

    train_bacteria = bacteria[:, :split_index_train, :]
    val_bacteria = bacteria[:, split_index_train:split_index_val, :]
    test_bacteria = bacteria[:, split_index_val:, :]

    train_gene_families = gene_families[:, :split_index_train, :]
    val_gene_families = gene_families[:, split_index_train:split_index_val, :]
    test_gene_families = gene_families[:, split_index_val:, :]

    # Create Dataset instances for training and testing
    batch_size = 64  
    train_dataset = TensorDataset(train_tensor, train_samples, train_bacteria, train_gene_families, batch_size)
    val_dataset = TensorDataset(val_tensor, val_samples, val_bacteria, val_gene_families, batch_size)
    test_dataset = TensorDataset(test_tensor, test_samples, test_bacteria, test_gene_families, batch_size)

    # Use DataLoader to load batches of data
    train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=TensorDataset.custom_collate_fn)
    val_loader = DataLoader(val_dataset, shuffle=False, collate_fn=TensorDataset.custom_collate_fn)
    test_loader = DataLoader(test_dataset, shuffle=False, collate_fn=TensorDataset.custom_collate_fn)

    # Initialize model
    gene_dim = data_tensor.shape[-1]  # gene dim
    embedding_dim = 8  # Example embedding size 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SplitAutoencoder(gene_dim=gene_dim, embedding_dim=embedding_dim)
    model = model.to(device)

    # Train the model
    print("Training model...")
    trained_model = train_model(model, train_loader, val_loader, device, num_epochs=10, learning_rate=0.001)

    # Save trained model
    trained_model_path = "split_autoencoder.pt"
    torch.save(trained_model, trained_model_path)

    # Save validation and test sets with labels
    np.save("data/data_files/val/val_tensor.npy", val_tensor.numpy())
    np.save("data/data_files/test/test_tensor.npy", test_tensor.numpy())

    np.save("data/data_files/val/val_samples.npy", val_samples)
    np.save("data/data_files/test/test_samples.npy", test_samples)

    np.save("data/data_files/val/val_bacteria.npy", val_bacteria)
    np.save("data/data_files/test/test_bacteria.npy", test_bacteria)

    np.save("data/data_files/val/val_genes.npy", val_gene_families)
    np.save("data/data_files/test/test_genes.npy", test_gene_families)

if __name__ == "__main__":
    main()
