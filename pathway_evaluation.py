import argparse
import sys
import torch
import os
import numpy as np
# Add the autoencoder_model directory to Python path so torch.load can find the training module
sys.path.append(os.path.join(os.path.dirname(__file__), 'autoencoder_model'))
from autoencoder_model.training.model import SplitAutoencoder
# from training.model import SplitAutoencoder  
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances
from sklearn.preprocessing import normalize, MinMaxScaler
from scipy.stats import rankdata, spearmanr

def load_test_data(test_data_path):
    test_tensor = np.load(test_data_path, allow_pickle=True)
    test_tensor = torch.tensor(test_tensor, dtype=torch.float32)
    return test_tensor

def load_test_labels(test_labels_path):
    test_labels = np.load(test_labels_path, allow_pickle=True)
    return test_labels

def load_model(model_path="split_autoencoder.pt", device="cuda"):
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()  
    return model

def load_pathway_data(pathway_data_path, pathway_bacteria_path):
    pathway_data = np.load(pathway_data_path, allow_pickle=True)  
    pathway_bacteria = np.load(pathway_bacteria_path, allow_pickle=True)  
    return pathway_data, pathway_bacteria

# returns a dictionary with the bacteria names from the test data and their relevant indices from the pathway data
def extract_relevant_pathway_indices(pathway_bacteria_names, bacteria_names_in_test):
    relevant_pathways_indices = {}
    for i, bacteria in enumerate(bacteria_names_in_test):
        if bacteria in pathway_bacteria_names:
            relevant_pathways_indices[bacteria] = pathway_bacteria_names.tolist().index(bacteria)  # Get the index directly
    return relevant_pathways_indices

# Convert pathway scores to binary vectors (participation or not)
def compute_binary_pathway_vectors(pathway_data, relevant_pathways_indices, test_labels, threshold=0):
    binary_pathway_vectors = {} # Dictionary to hold binary vectors for each bacterium
    valid_bacteria = [] # List to hold valid bacteria names that have pathway data
    # Iterate over each bacterium in the test labels and convert pathway scores to binary
    for i, bacterium in enumerate(test_labels):
        if bacterium not in relevant_pathways_indices:
            continue
        valid_bacteria.append(bacterium)
        b_idx = relevant_pathways_indices[bacterium]
        pathway_scores_2d = pathway_data[:, b_idx, :]  # (num_samples, num_pathways)
        
        # Convert pathway scores to binary based on the threshold
        binary_vector = (pathway_scores_2d.mean(axis=0) > threshold).astype(int)  # 1 if pathway score > threshold else 0
        binary_pathway_vectors[bacterium] = binary_vector

    return binary_pathway_vectors, valid_bacteria

def normalize_vectors(vectors):
    return normalize(vectors, norm='l2', axis=1)

def encode_data(model, test_data, test_labels,relevant_pathways_indices, device="cuda"):
    test_data = test_data.to(device)
    with torch.no_grad():
        # Apply the encoder part of the model to the test data
        x_encoded = model.encoder(test_data)  # shape: (num_samples, num_bacteria, 2b)
        x_encoded = model.activation(x_encoded)  
        b = x_encoded.shape[-1] // 2
        bacteria_encoding = x_encoded[:, :, :b]  # first half of embedding
        print(f"Encoded Hi (bacteria matrix) shape: {bacteria_encoding.shape}")
    
    # Get the number of bacteria and the embedding dimension
    num_bacteria = bacteria_encoding.shape[1]  
    embedding_dim = bacteria_encoding.shape[2]  
    # Prepare embeddings for all bacteria
    embeddings = {}

    # itertate over each bacterium in the test labels
    for i in range(num_bacteria):
        # Average of encodings across samples for each bacterium
        single_bacterium_encoding = bacteria_encoding[:, i, :].mean(dim=0).cpu().numpy()  # shape: (embedding_dim,)
        bacterium_name = test_labels[i]
        print(f"Processing bacterium: {bacterium_name}")
        print(f"single bacteria encoding shape: {single_bacterium_encoding.shape}")
        print(f"single bacterium encoding: {single_bacterium_encoding}")
        embeddings[bacterium_name] = single_bacterium_encoding  # Store the encoding for this bacterium
    return embeddings

# Similarity calculations between embedding vectors
def calculate_embedding_similarities(method, embeddings, valid_bacteria):
    embedding_vectors = [embeddings[bacterium] for bacterium in valid_bacteria]
    embedding_vectors = np.array(embedding_vectors) 
    if method == 'cosine':
        return cosine_similarity(embedding_vectors)
    elif method == 'l1':
        return 1 / (1 + manhattan_distances(embedding_vectors))
    elif method == 'l2':
        return 1 / (1 + euclidean_distances(embedding_vectors))

# Similarity calculations between binary pathway vectors (Jaccard distance)
def calculate_pathway_similarities(method, ranked_pathway_vectors, valid_bacteria):
    pathway_vectors = np.array([ranked_pathway_vectors[bacterium] for bacterium in valid_bacteria])
    if method == 'cosine':
        return cosine_similarity(pathway_vectors)
    elif method == 'jaccard':
        # Jaccard distance: 1 - Jaccard similarity
        num_bacteria = len(valid_bacteria)
        jaccard_dist = np.zeros((num_bacteria, num_bacteria))
        for i in range(num_bacteria):
            for j in range(i + 1, num_bacteria):
                intersection = np.sum(np.logical_and(pathway_vectors[i], pathway_vectors[j]))
                union = np.sum(np.logical_or(pathway_vectors[i], pathway_vectors[j]))
                jaccard_dist[i, j] = 1 - (intersection / union) if union != 0 else 1
                jaccard_dist[j, i] = jaccard_dist[i, j]
        return jaccard_dist
    elif method == 'l1':
        return 1 / (1 + manhattan_distances(pathway_vectors))
    elif method == 'l2':
        return 1 / (1 + euclidean_distances(pathway_vectors))

# Normalize the similarities to the same scale
def normalize_similarities(similarity_matrix):
    scaler = MinMaxScaler()
    return scaler.fit_transform(similarity_matrix)

def calculate_correlations(embedding_sim, pathway_sim, valid_bacteria):
    # Flatten the upper triangular part of the matrices (excluding diagonal) for comparison
    n = len(valid_bacteria)
    embed_flat = embedding_sim[np.triu_indices(n, k=1)]
    pathway_flat = pathway_sim[np.triu_indices(n, k=1)]
    
    # Create a DataFrame for visualization
    pairs = [(valid_bacteria[i], valid_bacteria[j]) for i, j in zip(*np.triu_indices(n, k=1))]
    labels = [f"{a} vs {b}" for a, b in pairs]
    df = pd.DataFrame({
        'Embedding Similarity': embed_flat,
        'Pathway Similarity': pathway_flat,
        'Pair': labels
    })
    # Pearson and Spearman correlation
    pearson_corr = np.corrcoef(embed_flat, pathway_flat)[0, 1]
    spearman_corr, _ = spearmanr(embed_flat, pathway_flat)
    return pearson_corr, spearman_corr

def get_save_path(method_embeddings: str, method_pathways:str, normalized_embeddings: bool, normalized_similarities: bool):
    method = f"{method_embeddings}_{method_pathways}"
    if normalized_embeddings:
        embeddings_norm = "_normalized_embeddings"
    else:
        embeddings_norm = "_unnormalized_embeddings"
    if normalized_similarities:
        similarities_norm = "_normalized_similarities"
    else:   
        similarities_norm= "_unnormalized_similarities"
    # Create a directory for saving plots   
    path = os.path.join("similarity_plots", method, embeddings_norm, similarities_norm)
    os.makedirs(path, exist_ok=True)
    return path

def plot_similarities(embedding_similarities, pathway_similarities, valid_bacteria, pearson_corr, spearman_corr, save_dir):
    sns.set_theme()      
    # Scatter plot with identity line and correlation annotation
    x = embedding_similarities[np.triu_indices(len(valid_bacteria), k=1)]
    y = pathway_similarities[np.triu_indices(len(valid_bacteria), k=1)]
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.7)
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    plt.plot(lims, lims, '--', linewidth=1, label='y = x')
    ax = plt.gca()
    ax.text(0.05, 0.95, f'Pearson r = {pearson_corr:.2f}', transform=ax.transAxes, va='top')
    ax.text(0.05, 0.88, f'Spearman r = {spearman_corr:.2f}', transform=ax.transAxes, va='top')
    plt.xlabel('Embedding Similarity')
    plt.ylabel('Binary pathway Similarity')
    plt.title('Pairwise Similarity Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scatter_with_identity.png'))
    plt.close()

def main():
    
    parser = argparse.ArgumentParser(description="Process bacterial data for similarity analysis.")
    parser.add_argument('--test_data', type=str, required=True, help='Path to the test tensor data')
    parser.add_argument('--test_labels', type=str, required=True, help='Path to the test labels (bacteria)')
    parser.add_argument('--model_path', type=str, default="split_autoencoder.pt", help='Path to the trained model')
    parser.add_argument('--pathway_data', type=str, required=True, help='Path to the pathway data tensor')
    parser.add_argument('--pathway_bacteria', type=str, required=True, help='Path to the pathway bacteria list')
    parser.add_argument('--method_embeddings', type=str, default='l2', choices=['l1', 'l2', 'cosine'], help='Method for embedding similarity calculation')
    parser.add_argument('--method_pathways', type=str, default='cosine', choices=['l1', 'l2', 'cosine', 'jaccard'], help='Method for pathway similarity calculation')
    parser.add_argument('--normalized_embeddings', type=bool, default=False, help='Normalize embeddings before calculation')
    parser.add_argument('--normalized_similarities', type=bool, default=True, help='Normalize similarities before plotting')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output plots')

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load test data, labels, model, and pathway data
    test_data = load_test_data(args.test_data)
    test_labels = load_test_labels(args.test_labels)
    model = load_model(model_path=args.model_path, device=device)
    pathway_data, pathway_bacteria = load_pathway_data(args.pathway_data, args.pathway_bacteria)


    # Extract relevant pathways for the test bacteria
    relevant_pathways_indices = extract_relevant_pathway_indices(pathway_bacteria, test_labels)

    # Get embeddings from model
    embeddings = encode_data(
        model, test_data, test_labels, relevant_pathways_indices, device=device
    )

    # Get binary pathway vectors (participation in pathways as binary vectors)
    binary_pathway_vectors, valid_bacteria = compute_binary_pathway_vectors(
        pathway_data, relevant_pathways_indices, test_labels, threshold=0
    )

    # method_embeddings = 'l2'  # Change to 'l1' or 'l2' or 'cosine' for different similarity metrics
    # method_pathways = 'cosine'  # Change to 'l1', 'l2', 'cosine', or 'jaccard' for different similarity metrics
    # normalized_embeddings = False  # Change to False for unnormalized embeddings or True for normalized embeddings
    # normalized_similarities = True  # Change to False for unnormalized similarities or True for normalized similarities
    
    if args.normalized_embeddings:
        # Extract the valid bacteria embeddings and normalize if necessary
        embeddings_values = np.array([embeddings[bacterium] for bacterium in valid_bacteria])
        embeddings_values = normalize_vectors(embeddings_values) 
        embeddings = {bacterium: embeddings_values[i] for i, bacterium in enumerate(valid_bacteria)}

    # Calculate similarities
    embedding_sim = calculate_embedding_similarities(method_embeddings, embeddings, valid_bacteria)
    pathway_sim = calculate_pathway_similarities(method_pathways, binary_pathway_vectors, valid_bacteria)

    if args.normalized_similarities:
        embedding_sim = normalize_similarities(embedding_sim)
        pathway_sim = normalize_similarities(pathway_sim)

    # Calculate correlations
    pearson_corr, spearman_corr = calculate_correlations(embedding_sim, pathway_sim, valid_bacteria)

    # Save paths for plotting
    save_path = get_save_path(args.method_embeddings, args.method_pathways, args.normalized_embeddings, args.normalized_similarities)
    # save_path = get_save_path(method_embeddings, method_pathways, normalized_embeddings, normalized_similarities)
    plot_similarities(embedding_sim, pathway_sim, valid_bacteria, pearson_corr, spearman_corr, save_path)

    # print correlation results 
    print(f"Method for embeddings : {args.method_embeddings}, Method for pathways : {args.method_pathways}, Normalized: {args.normalized_embeddings}, Similarities normalized: {args.normalized_similarities}")
    print(f"Pearson Correlation: {pearson_corr:.4f}, Spearman Correlation: {spearman_corr:.4f}")

if __name__ == "__main__":
    main()