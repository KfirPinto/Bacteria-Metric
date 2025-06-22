import torch
from data_utils import load_data_tensor, load_metadata, normalize_tensor
from preprocess import shuffle_bacteria, split_tensor, save_eval_data
from training.model import PathwayReg
from training.dataset import create_dataloaders
from training.train import train_model
import config

def main():

    # Load data and metadata
    gene_families = load_data_tensor(config.GENE_FAMILIES_TENSOR_PATH)
    pathways = load_data_tensor(config.PATHWAYS_TENSOR_PATH)
    sample_mt, bacteria_mt, gene_mt, pathway_mt = load_metadata(
        config.SAMPLE_LIST_PATH,
        config.BACTERIA_LIST_PATH,
        config.GENE_LIST_PATH,
        config.PATHWAYS_LIST_PATH
    )

    # Normalize
    gene_families_norm = normalize_tensor(gene_families)
    gene_families_norm = torch.tensor(gene_families, dtype=torch.float32)

    pathways_norm = normalize_tensor(pathways) # Normalized same way as gene_families
    pathways_norm = torch.tensor(pathways, dtype=torch.float32)

    # Shuffle and split
    gene_families_perm, pathways_perm, bacteria_mt_perm = shuffle_bacteria(gene_families_norm, pathways_norm, bacteria_mt)
    split = split_tensor(gene_families_perm, pathways_perm, bacteria_mt_perm, config.TRAIN_RATIO, config.VAL_RATIO)

    # Save evaluation sets
    save_eval_data(split, sample_mt, gene_mt, pathway_mt, out_dir=config.EVAL_OUTPUT_DIR)

    # Create DataLoaders
    train_loader, val_loader = create_dataloaders(
        split["train_gene_families"], split["train_pathways"],
        split["val_gene_families"], split["val_pathways"],
        batch_size=64
    )

    # Initialize model
    gene_dim = gene_families.shape[-1]
    pathway_dim = pathways.shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PathwayReg(gene_dim, config.EMBEDDING_DIM, pathway_dim).to(device)
    print(f"Running on device: {device}")

    # Train model
    print("Training model...")
    trained_model = train_model(model, train_loader, val_loader, device, num_epochs=config.NUM_EPOCHS,
                                learning_rate=config.LEARNING_RATE, name=config.NAME, lambda_weight=None)
    
    # Save model
    trained_model_path = f"{config.EVAL_OUTPUT_DIR}/model_weights.pt"
    torch.save(model.state_dict(), trained_model_path)
    print(f"Model saved to {trained_model_path}")

if __name__ == "__main__":
    main()