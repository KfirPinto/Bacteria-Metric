import torch
import numpy as np
from data_utils import load_data_tensor, load_metadata, normalize_tensor, cal_embedding
from preprocess import shuffle_bacteria, split_tensor, save_eval_data
from training.model import SplitVAE
from training.dataset import create_dataloaders
from variational_autoencoder.training.train_first import train_model
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import config
import random
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

    out_dir = f"{config.EVAL_OUTPUT_DIR}"
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

    # Save tensor of the embddings of the test set
    test_tensor = split["test_tensor"]
    trained_model.eval()
    with torch.no_grad():  # Disable gradients for inference
        test_tensor_embeddings, _, _, _ = trained_model.forward(test_tensor.to(device))   
        test_tensor_embeddings = test_tensor_embeddings.cpu().detach().numpy()

        # Average over the sample dimension (axis=0)
        # Shape: (samples, bacteria, embedding_dim) -> (bacteria, embedding_dim)
        bacteria_embeddings = np.mean(test_tensor_embeddings, axis=0)
        
        #rel_abundance_path = config.REL_ABUNDANCE_PATH
        #test_set_path = config.BACTERIA_LIST_EXPANDED_PATH
        #bacteria_embeddings = cal_embedding(test_tensor_embeddings, rel_abundance_path, test_set_path)
        # Extract the first half embedding (bacteria representation)
        bacteria_embeddings = bacteria_embeddings[:, :config.EMBEDDING_DIM//2]

        test_tensor_embeddings_path = os.path.join(out_dir, "test_tensor_embeddings.npy")
        np.save(test_tensor_embeddings_path, bacteria_embeddings)

if __name__ == "__main__":
    main()