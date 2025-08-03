import numpy as np
import pickle
import torch

def load_npy(npy_path):
    return np.load(npy_path, allow_pickle=True)

def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def load_tensor(tensor_path):
    if tensor_path.endswith(".pt"):
        return torch.load(tensor_path)
    elif tensor_path.endswith(".npy"):
        return torch.tensor(np.load(tensor_path))
    else:
        raise ValueError("Unsupported tensor format. Use .pt or .npy")
    
def normalize_taxonomy_string(tax_string):
    """
    Converts taxonomy from plain format (e.g. 'Bacteria;Firmicutes;...') to
    labeled format (e.g. 'k__Bacteria|p__Firmicutes|...')
    """
    levels = ['k', 'p', 'c', 'o', 'f', 'g', 's']
    parts = tax_string.strip().split(';')
    if len(parts) != len(levels):
        return tax_string
    return '|'.join([f"{prefix}__{name}" for prefix, name in zip(levels, parts)])

def main(npy_file, pkl_file, tensor_file):
    # Load data
    query_entries = load_pkl(pkl_file)
    reference_entries = load_npy(npy_file)
    tensor = load_tensor(tensor_file)

    reference_set = set(reference_entries)
    query = [normalize_taxonomy_string(q) for q in query_entries]

    # Match entries
    indices = [i for i, item in enumerate(query) if item in reference_set]

    # Filter tensor based on indices
    filtered_tensor = tensor[indices]
    filtered_query_entries = [query[i] for i in indices]

    # Save filtered tensor
    filtered_tensor_path = "/home/bcrlab/barsapi1/metric/Bacteria-Metric/data/HMP_2012_stool/raw/filtered_tensor.npy"
    np.save(filtered_tensor_path, filtered_tensor)

    # save filtered query entries
    filtered_query_entries_path = "/home/bcrlab/barsapi1/metric/Bacteria-Metric/data/HMP_2012_stool/raw/filtered_query_entries.npy"
    np.save(filtered_query_entries_path, filtered_query_entries)


# === Example Usage ===
if __name__ == "__main__":
    npy_file = "/home/bcrlab/barsapi1/metric/Bacteria-Metric/data/HMP_2012_stool/raw/bacteria_names_full_taxonomy.npy"
    pkl_file = "/home/bcrlab/barsapi1/metric/Bacteria-Metric/data/HMP_2012_stool/raw/embedding_16_bacteria_train_Sapir_[1].pkl"
    tensor_file = "/home/bcrlab/barsapi1/metric/Bacteria-Metric/data/HMP_2012_stool/raw/embedding_16_dim_trained_embeddings_Sapir_[1].pt"
    main(npy_file, pkl_file, tensor_file)
