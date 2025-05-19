import numpy as np

def load_data_tensor(path):
    return np.load(path, mmap_mode='r')

def load_metadata(sample_path, bacteria_path, gene_path):
    samples = np.load(sample_path, allow_pickle=True)
    bacteria = np.load(bacteria_path, allow_pickle=True)
    genes = np.load(gene_path, allow_pickle=True)
    return samples, bacteria, genes

def normalize_tensor(X):
    """
    Normalize a 3D tensor of shape (samples, bacteria, genes).
    Log-transform and then Z-normalize per bacteria across samples:
    1. y_ij = log_10(x_ij+epsilon)
    2. z_ij = (y_ij - mean_j)/std_j

    mean_j defined as the mean of bacteria j across all n samples. 
    std_j defined as the std of bacteria j across all n samples, given mean_j. 
    epsilon - min value across all dataset. 

    Parameters:
        X (np.ndarray): 3D array (samples, bacteria, genes)

    Returns:
        Z (np.ndarray): normalized tensor of same shape
    """
    
    # Ensure the input is float
    X = X.astype(np.float64)

    # Step 1: add ε = min of all elements to avoid log(0)
    epsilon = np.min(X[X > 0]) if np.any(X > 0) else 1e-8
    X_log = np.log10(X + epsilon)

    # Step 2: Compute mean and std over samples (axis=0)
    mean = np.mean(X_log, axis=0)   # shape: (bacteria, genes)
    std = np.std(X_log, axis=0)     # shape: (bacteria, genes)

    # Avoid division by zero
    std[std == 0] = 1e-8
    
    # Step 3: Normalize
    Z = (X_log - mean) / std

    return Z