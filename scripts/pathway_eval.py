import argparse
import sys
import torch
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
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

from autoencoder_model.training.model import SplitAutoencoder
from variational_autoencoder.training.model import SplitVAE

def load_embedding(test_data_path):
    embedding_tensor = np.load(test_data_path)
    #embedding_tensor = torch.tensor(embedding_tensor, dtype=torch.float32)
    return embedding_tensor

def load_embedding_metadata(test_labels_path): 
    test_labels = np.load(test_labels_path, allow_pickle=True)
    return test_labels

def load_pathway_data(pathway_data_path, pathway_bacteria_path):
    pathway_abundance_tensor = np.load(pathway_data_path)  
    all_bacteria = np.load(pathway_bacteria_path, allow_pickle=True)  
    return pathway_abundance_tensor, all_bacteria

# Convert pathway scores to binary vectors (participation or not)
def compute_binary_pathway_vectors(pathway_abundance, bacteria_names, threshold=0):
    """
    Converts pathway scores to binary vectors (1 if mean score > threshold, else 0)
    for each bacterium.

    Args:
        pathway_abundance (np.ndarray): shape (samples, bacteria, pathways)
        bacteria_names (array-like): list of bacteria names (length matches axis=1)
        threshold (float): threshold for binary conversion

    Returns:
        dict: {bacterium_name: binary_vector}
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

# Similarity calculations between embedding vectors
def calculate_similarities(method, embeddings):
    """
    Calculate similarity matrix for given embeddings by the selected method.

    Args:
        method (str): 'cosine', 'l1', or 'l2' or 'mahalanobis'
        embeddings (dict): {bacterium_name: embedding_vector}

    Returns:
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
    Calculate Pearson, Spearman, and Mantel correlations between two similarity matrices.

    Args:
        embedding_sim (np.ndarray): embedding similarity matrix (square)
        pathway_sim (np.ndarray): pathway similarity matrix (square)

    Returns:
        tuple: (pearson_corr, spearman_corr, mantel_corr, mantel_p_value)
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

def plot_similarities(embedding_similarities, pathway_similarities, pearson_corr, spearman_corr, mantel_corr, mantel_p_value, save_dir):
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

def get_parser():
    parser = argparse.ArgumentParser(
        description="Compute similarities between bacterial embeddings and pathway participation vectors."
    )
    
    # -----------------------------
    # Input Data Arguments
    # -----------------------------
    data_group = parser.add_argument_group("Input Data")
    data_group.add_argument('--test_embedding', type=str, required=True,
                            help='Path to the embedding tensor (.npy file).')
    data_group.add_argument('--test_metadata', type=str, required=True,
                            help='Path to the tensor metadata (bacteria names, .npy file).')
    data_group.add_argument('--pathway_data', type=str, required=True,
                            help='Path to the pathway abundance tensor (.npy file).')
    data_group.add_argument('--pathway_bacteria', type=str, required=True,
                            help='Path to the pathway abundance metadata (bacteria names, .npy file).')
    
    # -----------------------------
    # Model Arguments
    # -----------------------------
    model_group = parser.add_argument_group("Model")
    model_group.add_argument('--embedding_dim', type=int, required=True,
                             help='Dimension of the embedding layer.')

    # -----------------------------
    # Similarity Calculation Arguments
    # -----------------------------
    sim_group = parser.add_argument_group("Similarity Calculation")
    sim_group.add_argument('--method_embeddings', type=str, default='l2',
                           choices=['l1', 'l2', 'cosine', 'mahalanobis'],
                           help='Method for calculating embedding similarities.')
    sim_group.add_argument('--method_pathways', type=str, default='l2',
                           choices=['l1', 'l2', 'cosine', 'mahalanobis'],
                           help='Method for calculating pathway similarities.')
    sim_group.add_argument('--normalized_embeddings', action='store_true',
                           help='If set, normalize embedding vectors.')

    # -----------------------------
    # Output
    # -----------------------------
    output_group = parser.add_argument_group("Output")
    output_group.add_argument('--output_dir', type=str, required=True,
                              help='Directory to save the output plots and results.')

    return parser

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
    
    parser = get_parser()
    args = parser.parse_args()

    # Load gene abundance (test only), metadata (test only), model
    # and pathway abundance (train and test, filtering is required)
    embedding = load_embedding(args.test_embedding).squeeze()
    embedding_metadata = load_embedding_metadata(args.test_metadata)
    pathway_abundance, pathway_metadata = load_pathway_data(args.pathway_data, args.pathway_bacteria)


    # Extract relevant pathways 
    pathway_abundance_reordered, pathway_abundance_metadata_reordered = sub_matrix(embedding_metadata,
                                                                                   pathway_abundance, pathway_metadata)

    # Get embeddings from model - dict: {bacterium_name: embedding}
    embeddings = {name: embedding[i] for i, name in enumerate(embedding_metadata)}

    # Get binary pathway vectors - dict: {bacterium_name: binary vector correspoing to participation at metabolic pathways}
    binary_pathway_vectors = compute_binary_pathway_vectors(pathway_abundance_reordered,
                                                            pathway_abundance_metadata_reordered, threshold=0)
    
    if args.normalized_embeddings:
        normalize_embeddings_dict(embeddings)

    # Calculate similarities
    embedding_sim = calculate_similarities(args.method_embeddings, embeddings)
    pathway_sim = calculate_similarities(args.method_pathways, binary_pathway_vectors)

    #embedding_sim = normalize_similarities(embedding_sim)
    #pathway_sim = normalize_similarities(pathway_sim)

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