import torch
import numpy as np
import os
import random

def shuffle_bacteria(data, bacteria):
    torch.manual_seed(42) # fix the random seed for const partitioning to train and test sets
    perm = torch.randperm(data.size(1))
    return data[:, perm, :], bacteria[perm]

def split_tensor(data, bacteria, unannotated_data, unannotated_bacteria, train_ratio=0.7, val_ratio=0.15):
    """
    Splits data into Train, Validation, and Test sets.
    """
    num_bacteria = data.shape[1]
    
    # Calculate split indices
    idx_train = int(train_ratio * num_bacteria)
    idx_val = int((train_ratio + val_ratio) * num_bacteria)

    # --- 1. Train Set ---
    # Includes the first chunk of annotated bacteria + ALL unannotated bacteria
    train_tensor = torch.cat((data[:, :idx_train, :], unannotated_data), dim=1)
    train_bacteria = np.concatenate((bacteria[:idx_train], unannotated_bacteria))

    # --- 2. Validation Set ---
    # The middle chunk, used for Hyperparameter Tuning
    val_tensor = data[:, idx_train:idx_val, :]
    val_bacteria = bacteria[idx_train:idx_val]

    # --- 3. Test Set ---
    # The final chunk, held out for final evaluation
    test_tensor = data[:, idx_val:, :]
    test_bacteria = bacteria[idx_val:]

    print(f"Data Split Stats:")
    print(f"  Train: {train_tensor.shape[1]} bacteria (including unannotated)")
    print(f"  Val:   {val_tensor.shape[1]} bacteria")
    print(f"  Test:  {test_tensor.shape[1]} bacteria")

    split = {
        "train_tensor": train_tensor,
        "val_tensor": val_tensor,
        "test_tensor": test_tensor,
        "train_bacteria": train_bacteria,
        "val_bacteria": val_bacteria,
        "test_bacteria": test_bacteria,
    }
    return split

def save_eval_data(split, samples, gene_families, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    # Save Test Set (Final Evaluation)
    np.save(f"{out_dir}/test_tensor.npy", split["test_tensor"].numpy())
    np.save(f"{out_dir}/test_bacteria.npy", split["test_bacteria"])
    
    # Save Validation Set (Optimization)
    np.save(f"{out_dir}/val_tensor.npy", split["val_tensor"].numpy())
    np.save(f"{out_dir}/val_bacteria.npy", split["val_bacteria"])
    
    # Save Metadata
    np.save(f"{out_dir}/samples.npy", samples)
    np.save(f"{out_dir}/genes.npy", gene_families)