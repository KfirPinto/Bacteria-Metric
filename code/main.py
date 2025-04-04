from model import SplitAutoencoder
from train import train_model
import torch
import numpy as np

def main():
    # load data
    tensor_path = "data/data_files/gene_families/AsnicarF_2017_march/tensor.npy"
    data_tensor = np.load(tensor_path) 

    # Convert to torch tensor
    data_tensor = torch.tensor(data_tensor, dtype=torch.float32)

    # Split the tensor along the bacteria dimension (dimension 1)
    num_bacteria = data_tensor.shape[1]
    split_index = int(0.7 * num_bacteria)  # 70% for training

    # Split the tensor into training and testing sets (70-30)
    train_tensor = data_tensor[:, :split_index, :]  # First 70% of bacteria
    test_tensor = data_tensor[:, split_index:, :]   # Remaining 30% of bacteria

    # Initialize model
    gene_dim = data_tensor.shape[-1]  # gene dim
    embedding_dim = 8  # Example embedding size (2b, where b=4) so the decoding would be from dimension 4 to gene_dim
    model = SplitAutoencoder(gene_dim=gene_dim, embedding_dim=embedding_dim)

    # Train the model
    print("Training model...")
    trained_model = train_model(model, train_tensor, num_epochs=100, learning_rate=0.001)

    # Save trained model
    model_save_path = "split_autoencoder.pt"
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model trained and saved to {model_save_path}!")

if __name__ == "__main__":
    main()
