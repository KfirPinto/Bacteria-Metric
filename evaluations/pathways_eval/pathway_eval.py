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

# נסיון לייבוא המודל (לא קריטי לסקריפט הזה אך נשמר לתאימות)
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    sys.path.append(parent_dir)
    from variational_autoencoder.training.model import SplitVAE
except ImportError:
    pass

def load_embedding(test_data_path):
    embedding_tensor = np.load(test_data_path)
    return embedding_tensor

def load_embedding_metadata(test_labels_path): 
    test_labels = np.load(test_labels_path, allow_pickle=True)
    return test_labels

def load_pathway_data(pathway_data_path, pathway_bacteria_path):
    pathway_abundance_tensor = np.load(pathway_data_path)  
    all_bacteria = np.load(pathway_bacteria_path, allow_pickle=True)  
    return pathway_abundance_tensor, all_bacteria

def compute_binary_pathway_vectors(pathway_abundance, bacteria_names, threshold=0):
    if isinstance(bacteria_names, np.ndarray):
            bacteria_names = bacteria_names.tolist()

    binary_pathway_vectors = {}
    for i, bacterium in enumerate(bacteria_names):
        sub_matrix = pathway_abundance[:, i, :] 
        binary_vector = (sub_matrix.mean(axis=0) > threshold).astype(int)
        binary_pathway_vectors[bacterium] = binary_vector

    return binary_pathway_vectors

def normalize_vectors(vectors):
    return normalize(vectors, norm='l2', axis=1)

# --- הפונקציה המעודכנת: מקבלת background_embeddings ---
def calculate_similarities(method, embeddings, background_embeddings=None):
    """
    Calculate similarity matrix for given embeddings by the selected method.
    If method is 'mahalanobis' and background_embeddings is provided, 
    it uses the background to calculate the covariance matrix.
    """
    names = list(embeddings.keys())
    embedding_vectors = np.array([embeddings[name] for name in names], dtype=np.float64)

    print(f"DEBUG: Calculating similarities using method: {method}")

    if method == 'cosine':
        return cosine_similarity(embedding_vectors)
    elif method == 'l1':
        return 1 / (1 + manhattan_distances(embedding_vectors))
    elif method == 'l2':
        return 1 / (1 + euclidean_distances(embedding_vectors))
    elif method == 'mahalanobis':
        print("DEBUG: Computing Robust Mahalanobis...")
        
        # שימוש בדאטה מלא לחישוב שונות משותפת אם קיים
        if background_embeddings is not None:
            print(f"DEBUG: Using Background Data for Covariance (N={background_embeddings.shape[0]})")
            cov_data = background_embeddings
        else:
            print("WARNING: Using Test Data for Covariance (Might be unstable!)")
            cov_data = embedding_vectors
            
        cov_matrix = np.cov(cov_data.T)
        
        # Regularization (Epsilon)
        epsilon = 1e-5
        cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * epsilon
        
        try:
            VI = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            print("Error: Covariance matrix is singular. Adding larger epsilon.")
            cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-3
            VI = np.linalg.inv(cov_matrix)
        
        # חישוב מרחקים על ה-Vectors המקוריים (Test Set)
        dist_matrix = distance.cdist(embedding_vectors, embedding_vectors, metric='mahalanobis', VI=VI)
        
        similarity_matrix = 1 / (1 + dist_matrix)
        return similarity_matrix
    else:
        raise ValueError(f"Unknown method: {method}")

def normalize_similarities(similarity_matrix):
    min_val = similarity_matrix.min()
    max_val = similarity_matrix.max()
    return (similarity_matrix - min_val) / (max_val - min_val)

def calculate_correlations(embedding_sim, pathway_sim):
    if embedding_sim.shape != pathway_sim.shape:
        raise ValueError("Embedding and pathway similarity matrices must have the same shape.")

    n = embedding_sim.shape[0]
    embed_flat = embedding_sim[np.triu_indices(n, k=1)]
    pathway_flat = pathway_sim[np.triu_indices(n, k=1)]

    pearson_corr = np.corrcoef(embed_flat, pathway_flat)[0, 1]
    spearman_corr, _ = spearmanr(embed_flat, pathway_flat)

    # Mantel test prep
    with np.errstate(divide='ignore', invalid='ignore'):
        embed_dist = (1 / embedding_sim) - 1
        embed_dist[np.isinf(embed_dist)] = 0 
        embed_dist = np.nan_to_num(embed_dist)
        
    embed_dist = (embed_dist + embed_dist.T) / 2
    np.fill_diagonal(embed_dist, 0)
    embed_dist = embed_dist.astype(np.float32)

    with np.errstate(divide='ignore', invalid='ignore'):
        pathway_dist = (1 / pathway_sim) - 1
        pathway_dist[np.isinf(pathway_dist)] = 0
        pathway_dist = np.nan_to_num(pathway_dist)

    pathway_dist = (pathway_dist + pathway_dist.T) / 2
    np.fill_diagonal(pathway_dist, 0)
    pathway_dist = pathway_dist.astype(np.float32)

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

    path = os.path.join(output_dir, method, embeddings_norm)
    os.makedirs(path, exist_ok=True)
    return path

def plot_similarities(embedding_similarities, pathway_similarities, pearson_corr, spearman_corr, mantel_corr, mantel_p_value, save_dir):
    sns.set_theme()      
    n = embedding_similarities.shape[0]
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
    if isinstance(gene_abundance_metadata, np.ndarray):
        gene_abundance_metadata_list = gene_abundance_metadata.tolist()
    if isinstance(pathway_abundance_metadata, np.ndarray):
        pathway_abundance_metadata_list = pathway_abundance_metadata.tolist()
    
    indices = []
    for bacteria in gene_abundance_metadata_list:
        if bacteria in pathway_abundance_metadata_list:
            idx = pathway_abundance_metadata_list.index(bacteria)
            indices.append(idx)
        else:
            raise ValueError(f"{bacteria} not found in pathway_metadata")

    reordered_tensor = pathway_abundance[:, indices, :]
    reordered_metadata = pathway_abundance_metadata[indices]
    return reordered_tensor, reordered_metadata

def get_parser():
    parser = argparse.ArgumentParser(
        description="Compute similarities between bacterial embeddings and pathway participation vectors."
    )
    
    data_group = parser.add_argument_group("Input Data")
    data_group.add_argument('--test_embedding', type=str, required=True, help='Path to the embedding tensor (.npy file).')
    data_group.add_argument('--test_metadata', type=str, required=True, help='Path to the tensor metadata (bacteria names, .npy file).')
    data_group.add_argument('--pathway_data', type=str, required=True, help='Path to the pathway abundance tensor (.npy file).')
    data_group.add_argument('--pathway_bacteria', type=str, required=True, help='Path to the pathway abundance metadata (bacteria names, .npy file).')
    
    model_group = parser.add_argument_group("Model")
    model_group.add_argument('--embedding_dim', type=int, required=True, help='Dimension of the embedding layer.')

    sim_group = parser.add_argument_group("Similarity Calculation")
    sim_group.add_argument('--method_embeddings', type=str, default='l2',
                           choices=['l1', 'l2', 'cosine', 'mahalanobis'],
                           help='Method for calculating embedding similarities.')
    sim_group.add_argument('--method_pathways', type=str, default='l2',
                           choices=['l1', 'l2', 'cosine', 'mahalanobis'],
                           help='Method for calculating pathway similarities.')
    sim_group.add_argument('--normalized_embeddings', action='store_true', help='If set, normalize embedding vectors.')

    output_group = parser.add_argument_group("Output")
    output_group.add_argument('--output_dir', type=str, required=True, help='Directory to save the output plots and results.')

    return parser

def normalize_embeddings_dict(embeddings):
    names = list(embeddings.keys())
    vectors = np.vstack([embeddings[name] for name in names])
    normalized_vectors = normalize(vectors, norm='l2', axis=1)
    normalized_embeddings = {name: normalized_vectors[i] for i, name in enumerate(names)}
    return normalized_embeddings

def main():
    parser = get_parser()
    args = parser.parse_args()

    embedding = load_embedding(args.test_embedding).squeeze()
    embedding_metadata = load_embedding_metadata(args.test_metadata)
    pathway_abundance, pathway_metadata = load_pathway_data(args.pathway_data, args.pathway_bacteria)

    # --- טעינת דאטה רקע (Full Embeddings) ---
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    full_emb_path = os.path.join(project_root, "full_embeddings_after_train", "full_embeddings_run5.npy")
    
    background_emb = None
    if os.path.exists(full_emb_path):
        print(f"DEBUG: Found background embeddings at: {full_emb_path}")
        background_emb = np.load(full_emb_path)
    else:
        print(f"WARNING: Could not find background embeddings at {full_emb_path}")
    # ----------------------------------------

    pathway_abundance_reordered, pathway_abundance_metadata_reordered = sub_matrix(embedding_metadata,
                                                                                   pathway_abundance, pathway_metadata)

    embeddings = {name: embedding[i] for i, name in enumerate(embedding_metadata)}

    binary_pathway_vectors = compute_binary_pathway_vectors(pathway_abundance_reordered,
                                                            pathway_abundance_metadata_reordered, threshold=0)
    
    if args.normalized_embeddings:
        embeddings = normalize_embeddings_dict(embeddings)

    # שליחת background_embeddings לפונקציה
    embedding_sim = calculate_similarities(args.method_embeddings, embeddings, background_embeddings=background_emb)
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

    pearson_corr, spearman_corr, mantel_corr, mantel_p_value = calculate_correlations(embedding_sim, pathway_sim)
    print(f"Pearson Correlation: {pearson_corr:.4f}, Spearman Correlation: {spearman_corr:.4f}, Mantel Correlation: {mantel_corr:.4f}, Mantel p-value: {mantel_p_value:.4f}")

    save_path = get_save_path(args.output_dir, args.method_embeddings, args.method_pathways, args.normalized_embeddings)
    plot_similarities(embedding_sim, pathway_sim, pearson_corr, spearman_corr, mantel_corr, mantel_p_value, save_path)

if __name__ == "__main__":
    main()

# python evaluations/pathways_eval/pathway_eval.py \
#     --test_embedding /home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/eval_results/HMP_Kfir/Run_5/test_tensor_embeddings.npy \
#     --test_metadata /home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/eval_results/HMP_Kfir/Run_5/test_bacteria.npy \
#     --pathway_data /home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/processed_data/pathways_processed/after_intersection/tensor.npy \
#     --pathway_bacteria /home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/processed_data/pathways_processed/after_intersection/bacteria_list.npy \
#     --embedding_dim 64 \
#     --output_dir /home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/eval_results/HMP_Kfir/Run_5/plots_pathways_eval/pathway_correlation_mahalanobis_l2_COV_1/ \
#     --method_embeddings mahalanobis \
#     --method_pathways l2 \
#     --normalized_embeddings