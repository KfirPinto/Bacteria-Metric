from data_processing import load_and_preprocess
from model import SplitAutoencoder
from train import train_model
import torch
import pandas as pd
import os

def main():
    # File path to the new dataset
    file_path = os.path.join("..", "data", "2021-03-31.AsnicarF_2017_gene_families.csv")  # Relative path from 'code' to 'data'
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data_3d, genes, bacteria, samples = load_and_preprocess(file_path)
    print(f"Data shape: {data_3d.shape}")  # Expected: (bacteria_dim × gene_dim × people_dim)
    print(f"Number of genes: {len(genes)}")
    print(f"Number of bacteria: {len(bacteria)}")
    print(f"Number of samples: {len(samples)}")

    # Prepare inputs for training
    person_data = data_3d.permute(2, 0, 1)  # (people × bacteria × gene_dim)
    bacteria_data = data_3d.permute(0, 2, 1)  # (bacteria × people × gene_dim)

    print(f"Person data shape: {person_data.shape}")
    print(f"Bacteria data shape: {bacteria_data.shape}")

    # Initialize model
    gene_dim = data_3d.size(1)  # Gene dimension
    embedding_dim = 64
    model = SplitAutoencoder(bacteria_dim=person_data.size(1), gene_dim=gene_dim, embedding_dim=embedding_dim)

    # Train the model
    print("Training model...")
    trained_model = train_model(model, data_3d, person_data, bacteria_data, num_epochs=100, learning_rate=0.001)

    # Save trained model
    model_save_path = "split_autoencoder.pt"
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model trained and saved to {model_save_path}!")

    # Extract embeddings
    print("Extracting embeddings...")
    with torch.no_grad():
        person_embeddings = []
        bacteria_embeddings = []
        combined_embeddings = []

        for person_idx in range(person_data.size(0)):
            person_input = person_data[person_idx]  # (bacteria_dim × gene_dim)
            bacteria_input = bacteria_data[:, person_idx, :]  # (gene_dim)

            # Forward pass to get embeddings
            _, person_embedding, bacteria_embedding, combined_embedding = trained_model(person_input, bacteria_input)

            person_embeddings.append(person_embedding)
            bacteria_embeddings.append(bacteria_embedding)
            combined_embeddings.append(combined_embedding)

        # Convert lists to tensors
        person_embeddings = torch.stack(person_embeddings)  # (people × embedding_dim)
        bacteria_embeddings = torch.stack(bacteria_embeddings)  # (bacteria × embedding_dim)
        combined_embeddings = torch.stack(combined_embeddings)  # (people × embedding_dim)

    # Save embeddings to files
    embeddings_save_path = "embeddings.pt"
    torch.save({
        "person_embeddings": person_embeddings,
        "bacteria_embeddings": bacteria_embeddings,
        "combined_embeddings": combined_embeddings
    }, embeddings_save_path)
    print(f"Embeddings saved to {embeddings_save_path}!")

    # Optional: Print embedding shapes
    print(f"Person Embeddings shape: {person_embeddings.shape}")
    print(f"Bacteria Embeddings shape: {bacteria_embeddings.shape}")
    print(f"Combined Embeddings shape: {combined_embeddings.shape}")

if __name__ == "__main__":
    main()
