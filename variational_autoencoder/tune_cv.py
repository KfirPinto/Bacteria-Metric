import os
import sys
import optuna
import torch
import wandb
import numpy as np
from sklearn.model_selection import KFold
from torch import optim

# Setup path to import config from parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import config
from variational_autoencoder.data_utils import load_data_tensor, load_metadata, normalize_tensor
from variational_autoencoder.preprocess import shuffle_bacteria
from variational_autoencoder.training.model import SplitVAE
from variational_autoencoder.training.dataset import create_dataloaders
from variational_autoencoder.training.train import custom_loss

def train_one_fold(model, train_loader, val_loader, device, num_epochs, learning_rate, fold_idx, trial_number, lambda_weight, weight_decay):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # We want to find the lowest bacteria consistency loss achieved during this fold
    best_bact_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            batch_tensor = batch[0].to(device).squeeze(0)
            optimizer.zero_grad()
            encoded, decoded, mu, logvar = model(batch_tensor)
            
            # Calculate all losses
            recon, bact, sample, wd, kl = custom_loss(batch_tensor, decoded, encoded, mu, logvar, model=model, weight_decay=weight_decay)
            
            # Weighted sum for optimization (Backward)
            if lambda_weight:
                loss = recon * lambda_weight[0] + bact * lambda_weight[1] + sample * lambda_weight[2] + wd + kl
            else:
                loss = recon + bact + sample + wd + kl
                
            loss.backward()
            optimizer.step()

        # Validation Step
        model.eval()
        val_bact_accum = 0.0
        batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_tensor = batch[0].to(device).squeeze(0)
                encoded, decoded, mu, logvar = model(batch_tensor)
                
                # Calculate raw losses again to get the pure bact_loss
                _, bact, _, _, _ = custom_loss(batch_tensor, decoded, encoded, mu, logvar, model=model, weight_decay=0)
                val_bact_accum += bact.item()
                batches += 1
        
        avg_bact = val_bact_accum / batches
        
        # Update best score for this fold
        if avg_bact < best_bact_loss:
            best_bact_loss = avg_bact
            
        # Log to WandB
        if wandb.run is not None:
            wandb.log({
                f"trial_{trial_number}_fold_{fold_idx}_val_bact": avg_bact,
                "epoch": epoch
            })

    return best_bact_loss

def objective(trial):
    # --- 1. Refined Search Space (טווחים מצומצמים) ---
    embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True)

    num_epochs = 100
    k_folds = 5

    # --- 2. Data Preparation ---
    data = load_data_tensor(config.GENE_FAMILIES_TENSOR_PATH)
    unannotated_data = load_data_tensor(config.GENE_FAMILIES_COMPLEMENTRARY_TENSOR_PATH)
    samples, bacteria, unannotated_bacteria, _ = load_metadata(
        config.SAMPLE_LIST_PATH, config.BACTERIA_LIST_PATH,
        config.UNANNOTATED_BACTERIA_LIST_PATH, config.GENE_LIST_PATH
    )

    # Normalize
    data_norm = normalize_tensor(data)
    unannotated_norm = normalize_tensor(unannotated_data)
    
    data_tensor = torch.tensor(data_norm, dtype=torch.float32)
    unannotated_tensor = torch.tensor(unannotated_norm, dtype=torch.float32)

    # Shuffle bacteria
    data_shuffled, bacteria_shuffled = shuffle_bacteria(data_tensor, bacteria)

    # --- 3. Data Splitting for CV ---
    # Total bacteria count
    num_bacteria = data_shuffled.shape[1]
    
    # We reserve 15% for the final TEST set (held out completely)
    # The remaining 85% (Train + Val from original split) is used for Cross Validation
    idx_test_start = int(0.85 * num_bacteria) 
    
    # The "Development Set" (Annotated bacteria only)
    dev_data_annotated = data_shuffled[:, :idx_test_start, :]
    
    # K-Fold Cross Validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_bact_losses = []

    # Initialize WandB for this trial
    run = wandb.init(
        project="BacteriaMetric_CV_Optimization", 
        name=f"trial_{trial.number}",
        reinit=True,
        config=trial.params
    )

    print(f"Starting Trial {trial.number} with params: {trial.params}")

    # --- 4. K-Fold Loop ---
    # Iterate over 5 folds
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.arange(dev_data_annotated.shape[1]))):
        
        # Split Dev set into Train/Val for this fold
        fold_train_annotated = dev_data_annotated[:, train_idx, :]
        fold_val_tensor = dev_data_annotated[:, val_idx, :]
        
        # Important: Unannotated bacteria are ALWAYS added to the Training set of every fold
        fold_train_tensor = torch.cat((fold_train_annotated, unannotated_tensor), dim=1)

        # Create DataLoaders
        train_loader = create_dataloaders(fold_train_tensor, batch_size=batch_size)
        val_loader = create_dataloaders(fold_val_tensor, batch_size=batch_size)

        # Initialize Model (fresh for each fold, on GPU 3)
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        model = SplitVAE(gene_dim=data.shape[-1], embedding_dim=embedding_dim).to(device)

        # Train the Fold
        try:
            bact_loss = train_one_fold(
                model, train_loader, val_loader, device, num_epochs, 
                learning_rate, fold_idx, trial.number, config.LAMBDA_WEIGHT, weight_decay
            )
            fold_bact_losses.append(bact_loss)
            print(f"  Fold {fold_idx}: Best Bact Loss = {bact_loss:.4f}")
        except Exception as e:
            print(f"  Fold {fold_idx} Failed: {e}")
            run.finish()
            raise optuna.exceptions.TrialPruned()

    run.finish()
    
    # --- 5. Trial Result ---
    # The objective to minimize is the Average Bacterial Consistency Loss across 5 folds
    avg_bact_loss = np.mean(fold_bact_losses)
    return avg_bact_loss

if __name__ == "__main__":
    # Create Study
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    
    # Run 20 Trials (20 * 5 = 100 full trainings)
    print("Starting CV Optimization on GPU 3...")
    study.optimize(objective, n_trials=20)

    print("\n--- CV Optimization Finished ---")
    trial = study.best_trial
    print(f"Best Avg Bact Loss: {trial.value}")
    print("Best Params:", trial.params)

    # Save best params
    with open("best_params_cv.txt", "w") as f:
        f.write(f"Best Avg Bact Loss: {trial.value}\n")
        f.write(str(trial.params))