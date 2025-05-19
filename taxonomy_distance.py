import torch
import pandas as pd
from scipy.spatial.distance import pdist, squareform

# Function to calculate the distance between two bacterium names
def calc_dist_pair_split(l1, l2) -> int:
    len_l1, len_l2 = len(l1), len(l2)
    min_len = min(len_l1, len_l2)
    
    for k, (s1, s2) in enumerate(zip(l1[:min_len], l2[:min_len])):
        if s1 != s2:
            return (len_l1 - k) + (len_l2 - k)
    return abs(len_l1 - len_l2)

# Function to calculate the distance matrix
def calc_distance_in_sample(file_loc, device):
    df = pd.read_csv(file_loc)
    bact_names = df.iloc[:, 0]  # First column (sample names)
    
    # Pre-split each bacteria name by ';'
    split_names = [tuple(name.split(';')) for name in bact_names]
    N = len(split_names)

    # Initialize result tensor directly on the specified device
    C = torch.zeros((N, N), dtype=torch.uint8, device=device)

    # Calculate upper triangle indices directly on the device
    row_idx, col_idx = torch.triu_indices(N, N, 1, device=device)

    # Calculate distances using list comprehension and transfer results to device
    distances = [
        calc_dist_pair_split(split_names[i], split_names[j])
        for i, j in zip(row_idx.tolist(), col_idx.tolist())
    ]
    C[row_idx, col_idx] = torch.tensor(distances, dtype=torch.uint8, device=device)

    # Mirror upper triangle to lower triangle to make C symmetric
    C += C.T.clone()

    return C
