from model import SplitAutoencoder
from train import train_model
import torch
from torch.utils.data import Dataset, DataLoader
from TensorDataset import TensorDataset
import numpy as np

def main():

    # Load data
    data_tensor = np.load("data/data_files/gene_families/Intersection/tensor.npy") 
    print(f"Data tensor shape: {data_tensor.shape}")

    # Load labels
    samples = np.load("data/data_files/gene_families/Intersection/sample_list.npy", allow_pickle=True)
    bacteria = np.load("data/data_files/gene_families/Intersection/bacteria_list.npy", allow_pickle=True)
    gene_families = np.load("data/data_files/gene_families/Intersection/gene_families_list.npy", allow_pickle=True)


    # Convert the data tensor to a PyTorch tensor
    data_tensor = torch.tensor(data_tensor, dtype=torch.float32)

    # Shuffle the tensor along the bacteria dimension. same order for tensor and labels. 
    perm = torch.randperm(data_tensor.size(1)) 
    data_tensor = data_tensor[:, perm, :]
    bacteria = bacteria[perm]

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
    train_bacteria = bacteria[:split_index_train]
    val_bacteria = bacteria[split_index_train:split_index_val]
    test_bacteria = bacteria[split_index_val:]

    # save the test labels and test data (tensor) as separate .npy files
    np.save("test_tensor.npy", test_tensor.numpy())
    np.save("test_bacteria.npy", test_bacteria)
    np.save("val_tensor.npy", val_tensor.numpy())
    np.save("val_bacteria.npy", val_bacteria)
    np.save("samples.npy", samples)
    np.save("genes.npy", gene_families)

    # Create Dataset instances for training and testing
    batch_size = 128 
    train_dataset = TensorDataset(train_tensor, batch_size=batch_size)
    val_dataset = TensorDataset(val_tensor, batch_size=batch_size)

    # Use DataLoader to load batches of data
    # noga - added num_workers=4 for parallel data loading
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    # Initialize model
    gene_dim = data_tensor.shape[-1]  # gene dim
    embedding_dim = 8  # Example embedding size 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SplitAutoencoder(gene_dim=gene_dim, embedding_dim=embedding_dim)
    model = model.to(device)
    #model = torch.compile(model)
    print(f"Running on device: {device}") # Check if GPU is available

    # Train the model
    print("Training model...")
    trained_model = train_model(model, train_loader, val_loader, device, num_epochs=100, learning_rate=0.001)

    # Save trained model
    trained_model_path = "split_autoencoder.pt"
    torch.save(trained_model, trained_model_path)
    print(f"Model saved to {trained_model_path}")
    

if __name__ == "__main__":
    main()
