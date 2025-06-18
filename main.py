import torch
from data_utils import load_data_tensor, load_metadata, normalize_tensor
from preprocess import shuffle_bacteria, split_tensor, save_eval_data
from training.model import SplitAutoencoder
from training.dataset import create_dataloaders
from training.train import train_model
import config

def main():

    # Load data and metadata
    data = load_data_tensor(config.TENSOR_PATH)
    samples, bacteria, gene_families = load_metadata(
        config.SAMPLE_LIST_PATH,
        config.BACTERIA_LIST_PATH,
        config.UNANNOTATED_BACTERIA_LIST_PATH,
        config.GENE_LIST_PATH
    )

    # Load additional bacteria, whose don't have pathways, and use them to train the model
    additional_bacteria = load_metadata

    # Normalize
    data = normalize_tensor(data)
    data = torch.tensor(data, dtype=torch.float32)

    # Shuffle and split
    data, bacteria = shuffle_bacteria(data, bacteria)
    split = split_tensor(data, bacteria, config.TRAIN_RATIO, config.VAL_RATIO)

    # Save evaluation sets
    save_eval_data(split, samples, gene_families, out_dir=config.EVAL_OUTPUT_DIR)

    # Create DataLoaders
    train_loader, val_loader = create_dataloaders(
        split["train_tensor"], split["val_tensor"], batch_size=64
    )

    # Initialize model
    gene_dim = data.shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SplitAutoencoder(gene_dim=gene_dim, embedding_dim=config.EMBEDDING_DIM).to(device)
    print(f"Running on device: {device}")

    # Train model
    print("Training model...")
    trained_model = train_model(model, train_loader, val_loader, device, num_epochs=config.NUM_EPOCHS,
                                learning_rate=config.LEARNING_RATE, name=config.NAME, lambda_weight=config.LAMBDA_WEIGHT)
    
    # Save model
    trained_model_path = f"{config.EVAL_OUTPUT_DIR}/split_autoencoder.pt"
    torch.save(trained_model, trained_model_path)
    print(f"Model saved to {trained_model_path}")

if __name__ == "__main__":
    main()