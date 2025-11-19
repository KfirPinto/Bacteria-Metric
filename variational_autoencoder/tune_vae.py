import os
import sys
import optuna
import torch
import wandb
import numpy as np

# Add parent dir
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import config
from variational_autoencoder.data_utils import load_data_tensor, load_metadata, normalize_tensor
from variational_autoencoder.preprocess import shuffle_bacteria, split_tensor
from variational_autoencoder.training.model import SplitVAE
from variational_autoencoder.training.dataset import create_dataloaders
from variational_autoencoder.training.train import train_model

def objective(trial):
    # --- 1. Hyperparameters Search Space ---
    
    # Architecture & Optimizer
    embedding_dim = trial.suggest_categorical("embedding_dim", [16, 32, 64, 128])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # Loss Weights Optimization (CRITICAL for improving Pearson)
    # w_recon is fixed at 1.0 as an anchor
    w_recon = 1.0
    w_bact = trial.suggest_float("w_bact", 0.1, 10.0)
    w_sample = trial.suggest_float("w_sample", 0.1, 5.0)
    
    lambda_weight = [w_recon, w_bact, w_sample]

    num_epochs = 100 

    # --- 2. Data Loading & Splitting ---
    data = load_data_tensor(config.GENE_FAMILIES_TENSOR_PATH)
    unannotated_data = load_data_tensor(config.GENE_FAMILIES_COMPLEMENTRARY_TENSOR_PATH)
    
    samples, bacteria, unannotated_bacteria, gene_families = load_metadata(
        config.SAMPLE_LIST_PATH, config.BACTERIA_LIST_PATH,
        config.UNANNOTATED_BACTERIA_LIST_PATH, config.GENE_LIST_PATH
    )

    data_norm = normalize_tensor(data)
    unannotated_data_norm = normalize_tensor(unannotated_data)

    data_tensor = torch.tensor(data_norm, dtype=torch.float32)
    unannotated_data_tensor = torch.tensor(unannotated_data_norm, dtype=torch.float32)

    data_tensor_shuffled, bacteria_shuffled = shuffle_bacteria(data_tensor, bacteria)

    # Split (Train / Val / Test)
    split = split_tensor(
        data_tensor_shuffled, 
        bacteria_shuffled, 
        unannotated_data_tensor, 
        unannotated_bacteria, 
        train_ratio=0.7, 
        val_ratio=0.15 
    )

    train_loader = create_dataloaders(split["train_tensor"], batch_size=batch_size)
    val_loader = create_dataloaders(split["val_tensor"], batch_size=batch_size)

    # --- 3. Model Initialization ---
    gene_dim = data.shape[-1]
    
    # *** SET GPU 1 HERE ***
    device_str = "cuda:1" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    
    model = SplitVAE(gene_dim=gene_dim, embedding_dim=embedding_dim).to(device)

    # --- 4. Training ---
    run_name = f"trial_{trial.number}_dim{embedding_dim}_wB{w_bact:.1f}"

    try:
        _, best_val_loss = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            name=run_name,
            lambda_weight=lambda_weight, # Passing the optimized weights
            weight_decay=weight_decay,          
            trial=trial                         
        )
    except Exception as e:
        print(f"Trial pruned or failed: {e}")
        # WandB is handled in train.py finally block, but safe to ensure cleanup
        if wandb.run is not None:
            wandb.finish()
        raise optuna.exceptions.TrialPruned()

    return best_val_loss

if __name__ == "__main__":
    # Create Study
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    
    print("Starting optimization on GPU 1...")
    
    # Increase trials to 50 because we added more parameters (loss weights)
    study.optimize(objective, n_trials=50)

    # --- Finish ---
    print("\n--- Optimization Finished ---")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Best Val Loss): {trial.value}")
    print("  Best Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    with open("best_hyperparams.txt", "w") as f:
        f.write(f"Best Val Loss: {trial.value}\n")
        f.write("Best Parameters:\n")
        f.write(str(trial.params))
    
    print("Results saved to 'best_hyperparams.txt'")