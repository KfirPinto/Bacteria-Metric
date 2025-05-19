import torch
import numpy as np
import os

def shuffle_bacteria(data, bacteria):
    perm = torch.randperm(data.size(1))
    return data[:, perm, :], bacteria[perm]

def split_tensor(data, bacteria, train_ratio=0.7, val_ratio=0.15):
    num_bacteria = data.shape[1]
    idx_train = int(train_ratio * num_bacteria)
    idx_val = int((train_ratio + val_ratio) * num_bacteria)

    split = {
        "train_tensor": data[:, :idx_train, :],
        "val_tensor": data[:, idx_train:idx_val, :],
        "test_tensor": data[:, idx_val:, :],
        "train_bacteria": bacteria[:idx_train],
        "val_bacteria": bacteria[idx_train:idx_val],
        "test_bacteria": bacteria[idx_val:]
    }
    return split

def save_eval_data(split, samples, gene_families, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/test_tensor.npy", split["test_tensor"].numpy())
    np.save(f"{out_dir}/test_bacteria.npy", split["test_bacteria"])
    np.save(f"{out_dir}/val_tensor.npy", split["val_tensor"].numpy())
    np.save(f"{out_dir}/val_bacteria.npy", split["val_bacteria"])
    np.save(f"{out_dir}/samples.npy", samples)
    np.save(f"{out_dir}/genes.npy", gene_families)