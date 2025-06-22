import torch
import numpy as np
import os
import random

def shuffle_bacteria(data, bacteria):
    torch.manual_seed(42) # fix the random seed for const partitioning to train and test sets
    perm = torch.randperm(data.size(1))
    return data[:, perm, :], bacteria[perm]

def split_tensor(data, bacteria, unannotated_data, unannotated_bacteria, train_ratio=0.7):
    """
    Parameters:
        data (torch.Tensor): 3D tensor of shape (samples, bacteria, genes)
        bacteria (np.ndarray): Array of bacteria names corresponding to the second dimension of data
        unannotated_data (torch.Tensor): 3D tensor of unannotated bacteria 
        unannotated_bacteria (np.ndarray): Array of unannotated bacteria names (bacteria with no corresponding pathways)
        train_ratio (float): Ratio of training data to total data
    """

    num_bacteria = data.shape[1]
    idx_train = int(train_ratio * num_bacteria)

    # The train tensor is the concatentation 
    train_tensor = torch.cat((data[:, :idx_train, :], unannotated_data), dim=1)
    test_tensor = data[:, idx_train:, :]

    train_bacteria = np.concatenate((bacteria[:idx_train], unannotated_bacteria))
    test_bacteria = bacteria[idx_train:]

    split = {
        "train_tensor": train_tensor,
        "test_tensor": test_tensor,
        "train_bacteria": train_bacteria,
        "test_bacteria": test_bacteria,
    }
    return split

def save_eval_data(split, samples, gene_families, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/test_tensor.npy", split["test_tensor"].numpy())
    np.save(f"{out_dir}/test_bacteria.npy", split["test_bacteria"])
    np.save(f"{out_dir}/samples.npy", samples)
    np.save(f"{out_dir}/genes.npy", gene_families)