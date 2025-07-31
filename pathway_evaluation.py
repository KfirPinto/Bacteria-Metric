import argparse
import sys
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances
from sklearn.preprocessing import normalize, MinMaxScaler
from scipy.stats import spearmanr
from scipy.spatial import distance
from skbio.stats.distance import mantel
from skbio import DistanceMatrix

def load_embeddings(file_path="embeddings.npy"):
    embeddings_tensor = np.load(file_path, allow_pickle=True)
    embeddings_tensor = torch.tensor(embeddings_tensor, dtype=torch.float32)
    return embeddings_tensor

def load_embeddings_labels(file_path="embeddings_labels.csv"):
    df = pd.read_csv(file_path)
    labels = df.iloc[:, 0].tolist()
    return labels

def load_pathway_data(pathway_data_path, pathway_bacteria_path):
    pathway_abundance_tensor = np.load(pathway_data_path)  
    all_bacteria = np.load(pathway_bacteria_path, allow_pickle=True)  
    return pathway_abundance_tensor, all_bacteria

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute correlation between embeddings and metabollic pathways.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input files
    parser.add_argument("--embeddings", type=str, default="embeddings.npy", help="Path to external embeddings numpy file (.npy).")
    parser.add_argument("--embeddings_labels", type=str, default="embeddings_labels.csv", help="Path to embeddings bacteria labels numpy file (.csv)")
    parser.add_argument("--pathway_data", type=str, default="pathway_data.csv",help="Path to pathway data CSV file")
    parser.add_argument("--pathway_labels", type=str, default="pathway_labels.csv",help="Path to pathway labels CSV file")

    # Output directory
    parser.add_argument("--output_dir", type=str, default="./plots",help="Directory to save output plots and results")

    # Similarity calculation
    parser.add_argument('--method_embeddings', type=str, default='l2',
                           choices=['l1', 'l2', 'cosine', 'mahalanobis'],help='Method for calculating embedding similarities.')
    parser.add_argument('--method_pathways', type=str, default='l2',
                           choices=['l1', 'l2', 'cosine', 'mahalanobis'], help='Method for calculating pathway similarities.')
    parser.add_argument('--normalized_embeddings', action='store_true',help='If set, normalize embedding vectors.')

    return parser.parse_args()

# Convert pathway scores to binary vectors (participation or not)
def compute_binary_pathway_vectors(pathway_abundance, bacteria_names, threshold=0):
    """
    Converts pathway scores to binary vectors (1 if mean score > threshold, else 0)
    for each bacterium. Returns a dict: {bacterium_name: binary_vector}
    """
    if isinstance(bacteria_names, np.ndarray):
            bacteria_names = bacteria_names.tolist()

    binary_pathway_vectors = {} # Dictionary to hold binary vectors for each bacterium
    # Iterate over each bacterium in the test labels and convert pathway scores to binary
    for i, bacterium in enumerate(bacteria_names):
        sub_matrix = pathway_abundance[:, i, :]  # (num_samples, num_pathways)
        binary_vector = (sub_matrix.mean(axis=0) > threshold).astype(int)  # 1 if pathway score > threshold, else 0
        binary_pathway_vectors[bacterium] = binary_vector

    return binary_pathway_vectors

def normalize_vectors(vectors):
    return normalize(vectors, norm='l2', axis=1)

def organize_embeddings(embeddings, embeddings_labels, device="cuda"):
    embeddings = embeddings[1:]  # Exclude the first row

    # Convert embeddings_data to numpy if it's a tensor
    if hasattr(embeddings, 'cpu'):
        embeddings_data = embeddings.cpu().numpy()
    
    num_bacteria, embedding_dim = embeddings_data.shape
    print(f"Embeddings shape: ({num_bacteria}, {embedding_dim}) - one embedding per bacterium")
    
    # Verify that we have the same number of bacteria in both files
    if num_bacteria != len(embeddings_labels):
        raise ValueError(f"Mismatch: {num_bacteria} bacteria in embeddings but {len(embeddings_labels)} in labels")
    
    print(f"Processing {num_bacteria} bacteria...")

    # Prepare embeddings for all bacteria
    embeddings_dict = {}
    # itertate over each bacterium in the test labels
    for i in range(num_bacteria):
        # Average of encodings across samples for each bacterium
        single_bacterium_encoding = embeddings_data[:, i] # shape: (embedding_dim,)
        bacterium_name = embeddings_labels[i]
        embeddings_dict[bacterium_name] = single_bacterium_encoding  # Store the encoding for this bacterium

    return embeddings_dict

# Similarity calculations between embedding vectors
def calculate_similarities(method, embeddings):
    """
    Calculate similarity matrix for given embeddings by the selected method. Returns:
        np.ndarray: similarity or distance matrix (num_bacteria x num_bacteria)
    """
    names = embeddings.keys()
    embedding_vectors = np.array([embeddings[name] for name in names])

    if method == 'cosine':
        return cosine_similarity(embedding_vectors)
    elif method == 'l1':
        return 1 / (1 + manhattan_distances(embedding_vectors))
    elif method == 'l2':
        return 1 / (1 + euclidean_distances(embedding_vectors))
    elif method == 'mahalanobis':
        # Compute inverse covariance matrix
        VI = np.linalg.pinv(np.cov(embedding_vectors.T))
        # Compute pairwise Mahalanobis distances
        dist_matrix = distance.cdist(embedding_vectors, embedding_vectors, metric='mahalanobis', VI=VI)
        # Convert distances to similarity-like scale
        similarity_matrix = 1 / (1 + dist_matrix)
        return similarity_matrix

# Normalize the similarities to the same scale
def normalize_similarities(similarity_matrix):
    min_val = similarity_matrix.min()
    max_val = similarity_matrix.max()
    return (similarity_matrix - min_val) / (max_val - min_val)

def calculate_correlations(embedding_sim, pathway_sim):
    """
    Calculate Pearson, Spearman, and Mantel correlations between two similarity matrices (embedding sim and pathway sim).
    """
    # Sanity check
    if embedding_sim.shape != pathway_sim.shape:
        raise ValueError("Embedding and pathway similarity matrices must have the same shape.")

    n = embedding_sim.shape[0]

    # Flatten upper triangular part (excluding diagonal)
    embed_flat = embedding_sim[np.triu_indices(n, k=1)]
    pathway_flat = pathway_sim[np.triu_indices(n, k=1)]

    # Pearson correlation
    pearson_corr = np.corrcoef(embed_flat, pathway_flat)[0, 1]

    # Spearman correlation
    spearman_corr, _ = spearmanr(embed_flat, pathway_flat)

    # Mantel test
    # Convert similarity to distance (assuming similarity in [0,1])
    
    embed_dist = (1 / embedding_sim) - 1 # convert similarity to distance
    embed_dist = (embed_dist + embed_dist.T) / 2 # make it symmetric
    np.fill_diagonal(embed_dist, 0) # fill diagonal with 0
    embed_dist = embed_dist.astype(np.float32) # skbio expect to float32

    pathway_dist = (1 / pathway_sim) - 1
    pathway_dist = (pathway_dist + pathway_dist.T) / 2
    np.fill_diagonal(pathway_dist, 0)
    pathway_dist = pathway_dist.astype(np.float32)

    # Convert to scikit-bio DistanceMatrix
    embed_dm = DistanceMatrix(embed_dist)
    pathway_dm = DistanceMatrix(pathway_dist)

    mantel_corr, mantel_p_value, _ = mantel(embed_dm, pathway_dm, method='pearson', permutations=999)

    return pearson_corr, spearman_corr, mantel_corr, mantel_p_value

def get_save_path(output_dir, method_embeddings: str, method_pathways:str, normalized_embeddings: bool):
    method = f"{method_embeddings}_{method_pathways}"
    if normalized_embeddings:
        embeddings_norm = "_normalized_embeddings"
    else:
        embeddings_norm = "_unnormalized_embeddings"

    # Create a directory for saving plots   
    path = os.path.join(output_dir, method, embeddings_norm)
    os.makedirs(path, exist_ok=True)
    return path

def plot_similarities(embedding_similarities, pathway_similarities,
                         pearson_corr, spearman_corr, mantel_corr, mantel_p_value, save_dir):
    sns.set_theme()      
    n = embedding_similarities.shape[0]
    # Scatter plot with identity line and correlation annotation
    x = embedding_similarities[np.triu_indices(n, k=1)]
    y = pathway_similarities[np.triu_indices(n, k=1)]
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.7)
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    plt.plot(lims, lims, '--', linewidth=1, label='y = x')
    ax = plt.gca()
    ax.text(0.05, 0.95, f'Pearson = {pearson_corr:.4f}', transform=ax.transAxes, va='top')
    ax.text(0.05, 0.9, f'Spearman = {spearman_corr:.4f}', transform=ax.transAxes, va='top')
    ax.text(0.05, 0.85, f'Mantel = {mantel_corr:.4f}', transform=ax.transAxes, va='top')
    ax.text(0.05, 0.8, f'Mantel p_value = {mantel_p_value:.4f}', transform=ax.transAxes, va='top')
   
    plt.xlabel('Embedding Similarity')
    plt.ylabel('Binary pathway Similarity')
    plt.title('Pairwise Similarity Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scatter_with_identity.png'))
    plt.close()

def sub_matrix(gene_abundance_metadata, pathway_abundance, pathway_abundance_metadata):
    """
    Filters and reorders pathway_abundance matrix to include only bacteria appearing in gene_abundance_metadata,
    and reorders them to match the gene_abundance_metadata order.

    Args:
        gene_abundance_metadata (array-like): ordered list of bacteria names in gene_abundance 
        pathway_abundance (np.ndarray): matrix of shape (samples, bacteria, pathways)
        pathway_metadata (array-like): bacteria names order in matrix pathway_abundance

    Returns:
        np.ndarray: filtered and reordered pathway_abundance
    """

    # Convert pathway_metadata to list for .index() lookup
    if isinstance(gene_abundance_metadata, np.ndarray):
        gene_abundance_metadata_list = gene_abundance_metadata.tolist()
    if isinstance(pathway_abundance_metadata, np.ndarray):
        pathway_abundance_metadata_list = pathway_abundance_metadata.tolist()
    
    # Find indices of bacteria in pathway_metadata that match gene_abundance_metadata
    indices = []
    for bacteria in gene_abundance_metadata_list:
        if bacteria in pathway_abundance_metadata_list:
            idx = pathway_abundance_metadata_list.index(bacteria)
            indices.append(idx)
        else:
            raise ValueError(f"{bacteria} not found in pathway_metadata")

    # Reorder and subset pathway_abundance accordingly
    reordered_tensor = pathway_abundance[:, indices, :]
    reordered_metadata = pathway_abundance_metadata[indices]
    return reordered_tensor, reordered_metadata

def normalize_embeddings_dict(embeddings):
    # Stack to a matrix: shape (num_bacteria, embedding_dim)
    names = list(embeddings.keys())
    vectors = np.vstack([embeddings[name] for name in names])
    
    # Normalize each vector (row) to L2 norm=1
    normalized_vectors = normalize(vectors, norm='l2', axis=1)
    
    # Rebuild dictionary
    normalized_embeddings = {name: normalized_vectors[i] for i, name in enumerate(names)}
    
    return normalized_embeddings

def main():
    args = parse_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings = load_embeddings(args.embeddings)
    embeddings_labels = load_embeddings_labels(args.embeddings_labels)
    pathway_abundance, pathway_metadata = load_pathway_data(args.pathway_data, args.pathway_bacteria)

    # Extract relevant pathways 
    pathway_abundance_reordered, pathway_abundance_metadata_reordered = sub_matrix(embeddings_labels,
                                                                                   pathway_abundance, pathway_metadata)

    # Get embeddings from model - dict: {bacterium_name: embedding}
    embeddings = organize_embeddings(embeddings, embeddings_labels, device=device)

    # Get binary pathway vectors - dict: {bacterium_name: binary vector correspoing to participation at metabolic pathways}
    binary_pathway_vectors = compute_binary_pathway_vectors(pathway_abundance_reordered,
                                                            pathway_abundance_metadata_reordered, threshold=0)
    
    if args.normalized_embeddings:
        embeddings = normalize_embeddings_dict(embeddings)

    # Calculate similarities
    embedding_sim = calculate_similarities(args.method_embeddings, embeddings)
    pathway_sim = calculate_similarities(args.method_pathways, binary_pathway_vectors)

    bacteria_names = list(embeddings.keys())
    n = len(bacteria_names)
    print("Pairs with embedding similarity > 0.6 or pathway similarity > 0.4:")
    for i in range(n):
        for j in range(i + 1, n):
            emb_sim = embedding_sim[i, j]
            path_sim = pathway_sim[i, j]
            if emb_sim > 0.6 or path_sim > 0.4:
                print(f"{bacteria_names[i]} - {bacteria_names[j]}: embedding_sim={emb_sim:.3f}, pathway_sim={path_sim:.3f}")

    # Calculate correlations
    pearson_corr, spearman_corr, mantel_corr, mantel_p_value = calculate_correlations(embedding_sim, pathway_sim)
    print(f"Pearson Correlation: {pearson_corr:.4f}, Spearman Correlation: {spearman_corr:.4f}, Mantel Correlation: {mantel_corr:.4f}, Mantel p-value: {mantel_p_value:.4f}")

    # Save paths for plotting
    save_path = get_save_path(args.output_dir, args.method_embeddings, args.method_pathways, args.normalized_embeddings)
    plot_similarities(embedding_sim, pathway_sim, pearson_corr, spearman_corr, mantel_corr, mantel_p_value, save_path)

if __name__ == "__main__":
    main()
