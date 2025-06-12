import torch
import numpy as np
import os

def shuffle_bacteria(gene_families, pathways, bacteria):
    perm = torch.randperm(gene_families.size(1))
    return gene_families[:, perm, :], pathways[:, perm, :], bacteria[perm]

def split_tensor(gene_families, pathways, bacteria, train_ratio=0.7, val_ratio=0.15):
    num_bacteria = gene_families.shape[1]
    idx_train = int(train_ratio * num_bacteria)
    idx_val = int((train_ratio + val_ratio) * num_bacteria)

    split = {
        "train_gene_families": gene_families[:, :idx_train, :],
        "train_pathways": pathways[:, :idx_train, :],

        "val_gene_families": gene_families[:, idx_train:idx_val, :],
        "val_pathways": pathways[:, idx_train:idx_val, :],

        "test_gene_families": gene_families[:, idx_val:, :],
        "test_pathways": pathways[:, idx_val:, :],

        "train_bacteria": bacteria[:idx_train],
        "val_bacteria": bacteria[idx_train:idx_val],
        "test_bacteria": bacteria[idx_val:]
    }
    return split

def save_eval_data(split, samples, gene_families, pathways, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/metadata", exist_ok=True)
    os.makedirs(f"{out_dir}/test", exist_ok=True)
    os.makedirs(f"{out_dir}/validation", exist_ok=True)
    os.makedirs(f"{out_dir}/train", exist_ok=True)

    np.save(f"{out_dir}/train/train_gene_families.npy", split["train_gene_families"].numpy())
    np.save(f"{out_dir}/train/train_pathways.npy", split["train_pathways"].numpy())
    np.save(f"{out_dir}/train/train_bacteria.npy", split["train_bacteria"])

    np.save(f"{out_dir}/test/test_gene_families.npy", split["test_gene_families"].numpy())
    np.save(f"{out_dir}/test/test_pathways.npy", split["test_pathways"].numpy())
    np.save(f"{out_dir}/test/test_bacteria.npy", split["test_bacteria"])

    np.save(f"{out_dir}/validation/val_gene_families.npy", split["val_gene_families"].numpy())
    np.save(f"{out_dir}/validation/val_pathways.npy", split["val_pathways"].numpy())
    np.save(f"{out_dir}/validation/val_bacteria.npy", split["val_bacteria"])

    np.save(f"{out_dir}/metadata/samples.npy", samples)
    np.save(f"{out_dir}/metadata/genes.npy", gene_families)
    np.save(f"{out_dir}/metadata/pathways.npy", pathways)