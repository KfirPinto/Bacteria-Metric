import torch
from data_utils import load_data_tensor, load_metadata, normalize_tensor
from preprocess import shuffle_bacteria, split_tensor, save_eval_data
from training.model import SplitVAE
from training.dataset import create_dataloaders
from training.train import train_model
import config as config
import random
import os

def main():

    # ----- Load data and metadata -----

    # Load gene families abundance tensor
    data = load_data_tensor(config.GENE_FAMILIES_TENSOR_PATH)

    # Load gene gamilies abuncdance tensor contains unannotated bacteria (missing pathways)
    unannotated_data = load_data_tensor(config.GENE_FAMILIES_COMPLEMENTRARY_TENSOR_PATH)

    # Load metadata
    samples, bacteria, unannotated_bacteria, gene_families = load_metadata(
        config.SAMPLE_LIST_PATH,
        config.BACTERIA_LIST_PATH,
        config.UNANNOTATED_BACTERIA_LIST_PATH,
        config.GENE_LIST_PATH
    )

    # Normalize
    data_norm = normalize_tensor(data)
    unannotated_data_norm = normalize_tensor(unannotated_data)

    # Convert to pytorch tensors
    data_tensor = torch.tensor(data_norm, dtype=torch.float32)
    unannotated_data_tensor = torch.tensor(unannotated_data_norm, dtype=torch.float32)

    for i in range (11):

        out_dir = f"{config.EVAL_OUTPUT_DIR}/Run_{i}/"
        os.makedirs(out_dir, exist_ok=True)

        # Shuffle and split
        data_tendor_shuffled, bacteria_shuffled = shuffle_bacteria(data_tensor, bacteria)
        split = split_tensor(data_tendor_shuffled, bacteria_shuffled, unannotated_data_tensor, unannotated_bacteria, config.TRAIN_RATIO)

        # Save evaluation sets
        save_eval_data(split, samples, gene_families, out_dir=out_dir)

        # Create DataLoaders
        train_loader = create_dataloaders(
            split["train_tensor"], batch_size=config.BATCH_SIZE
        )

        # Initialize model
        gene_dim = data.shape[-1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_seed = random.randint(0, 99999) # random seed for model weight initialization
        torch.manual_seed(model_seed)

        model = SplitVAE(gene_dim=gene_dim, embedding_dim=config.EMBEDDING_DIM).to(device)
        print(f"Running on device: {device}")

        # Train model
        print("Training model...")
        trained_model = train_model(model, train_loader, device, num_epochs=config.NUM_EPOCHS,
                                    learning_rate=config.LEARNING_RATE, name=config.NAME, lambda_weight=config.LAMBDA_WEIGHT, weight_decay=config.WEIGHT_DACAY)
        
        # Save model
        trained_model_path = os.path.join(out_dir, f"split_autoencoder.pt")
        torch.save(trained_model.state_dict(), trained_model_path)
        print(f"Model saved to {trained_model_path}")

if __name__ == "__main__":
    main()