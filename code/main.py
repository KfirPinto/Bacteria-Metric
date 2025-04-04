from model import SplitAutoencoder
from train import train_model
import torch
import pandas as pd
import os

def main():
    # File path to the new dataset
    file_path = os.path.join("..", "data", "2021-03-31.AsnicarF_2017_gene_families.csv")  # Relative path from 'code' to 'data'
    ########################## LOAD DATA

    # Initialize model
    model = SplitAutoencoder(gene_dim=gene_dim, embedding_dim=embedding_dim)

    # Train the model
    print("Training model...")
    trained_model = train_model(model, data_tensor, num_epochs=100, learning_rate=0.001)

    # Save trained model
    model_save_path = "split_autoencoder.pt"
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model trained and saved to {model_save_path}!")

if __name__ == "__main__":
    main()
