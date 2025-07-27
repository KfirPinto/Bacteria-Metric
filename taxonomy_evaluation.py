import torch
import numpy as np
import sys
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text
import collections
from pathlib import Path
import argparse
import importlib.util
import umap
from sklearn.manifold import MDS, TSNE
from scipy.spatial.distance import pdist, squareform
sys.path.append(".")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_test_data(file_path="test_tensor.npy"):
    test_tensor = np.load(file_path, allow_pickle=True)
    test_tensor = torch.tensor(test_tensor, dtype=torch.float32)
    return test_tensor

def load_test_labels(file_path="test_bacteria.npy"):
    test_labels = np.load(file_path, allow_pickle=True)
    return test_labels

def load_model(model_path, model_class, gene_dim, embedding_dim):
    model = model_class(gene_dim=gene_dim, embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_taxonomy(taxonomy_file):

    taxonomy_df = pd.read_csv(taxonomy_file)
    # Detect missing family values
    missing_family_df = taxonomy_df[taxonomy_df['Family'].isna() | (taxonomy_df['Family'] == '')]

    if not missing_family_df.empty:
        print("The following bacteria have missing 'Family' information and will be dropped:")
        print(missing_family_df["Original Name"].tolist())

    # Drop rows with missing or empty Family column
    taxonomy_df = taxonomy_df[~taxonomy_df['Family'].isna() & (taxonomy_df['Family'] != '')]

    return taxonomy_df

def apply_dimensionality_reduction(data, method='pca', n_components=2, random_state=42):
    """
    methods: 'pca', 'tsne', 'umap', or 'pcoa'
    """
    print(f"Applying {method.upper()} dimensionality reduction...")
    
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
        reduced_data = reducer.fit_transform(data)
        
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=random_state, 
                      perplexity=min(30, len(data)-1))
        reduced_data = reducer.fit_transform(data)
        
    elif method.lower() == 'umap':
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        reduced_data = reducer.fit_transform(data)
        
    elif method.lower() == 'pcoa':
        if not PCOA_AVAILABLE:
            raise ImportError("PCoA dependencies not available.")
        # Compute pairwise distances
        distances = pdist(data, metric='euclidean')
        distance_matrix = squareform(distances)
        
        # Apply classical MDS (which is equivalent to PCoA)
        reducer = MDS(n_components=n_components, dissimilarity='precomputed', 
                     random_state=random_state)
        reduced_data = reducer.fit_transform(distance_matrix)
        
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    return reduced_data

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate clustering performance of bacterial embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input files
    parser.add_argument("--test_data", 
                        type=str, 
                        default="test_tensor.npy",
                        help="Path to test data numpy file (.npy)"
    )
    
    parser.add_argument("--test_labels", 
                        type=str, 
                        default="test_bacteria.npy",
                        help="Path to test bacteria labels numpy file (.npy)"
    )
    
    parser.add_argument("--model_path", 
                        type=str, 
                        default="split_autoencoder.pt",
                        help="Path to trained model file (.pt)"
    )
    
    parser.add_argument("--taxonomy_file", 
                        type=str, 
                        default="bacterial_lineage.csv",
                        help="Path to bacterial taxonomy CSV file"
    )

    # External embeddings option
    parser.add_argument("--external_embeddings", 
                        type=str, 
                        default=None,
                        help="Path to external embeddings numpy file (.npy). If provided, will use these instead of model encoding."
    )
    
    parser.add_argument("--projection_matrix", 
                        type=str, 
                        default=None,
                        help="Path to projection matrix numpy file (.npy). Applied to external embeddings if provided."
    )

    # Output directory
    parser.add_argument("--output_dir", 
                        type=str, 
                        default="./plots",
                        help="Directory to save output plots and results"
    )
    
    # Clustering parameters
    parser.add_argument("--min_k", 
                        type=int, 
                        default=2,
                        help="Minimum number of clusters to test"
    )
    
    parser.add_argument("--max_k", 
                        type=int, 
                        default=15,
                        help="Maximum number of clusters to test"
    )

    # Dimensionality reduction method
    parser.add_argument("--reduction_method", 
                        type=str, 
                        default="pca",
                        choices=['pca', 'tsne', 'umap', 'pcoa'],
                        help="Dimensionality reduction method for visualization"
    )


    # Sapir Additions
    parser.add_argument("--model-file",
                        type=str,
                        required=True,
                        help="Path to the model.py file that defines the model architecture.")

    parser.add_argument("--model-class",
                        type=str,
                        default="SplitAutoencoder",
                        help="Name of the model class to import from the model file (default: SplitAutoencoder)")

    parser.add_argument("--embedding-dim",
                        type=int,
                        default=32,
                        help="Dimensionality of the embedding space (default: 32)")


    return parser.parse_args()

def setup_output_directory(output_dir):
    """Create output directory if it doesn't exist"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    return output_dir  

def load_external_embeddings(embeddings_path, projection_matrix_path=None, test_labels=None, taxonomy_df=None):
    """
    Load external embeddings and optionally apply a projection matrix
    
    Args:
        embeddings_path: Path to numpy file containing embeddings
        projection_matrix_path: Optional path to projection matrix numpy file
        test_labels: List of bacteria names corresponding to embeddings
        taxonomy_df: Taxonomy dataframe for filtering
        
    Returns:
        Dictionary with embeddings and taxonomy info, similar to encode_data output
    """
    print(f"Loading external embeddings from: {embeddings_path}")
    
    # Load embeddings
    embeddings = np.load(embeddings_path, allow_pickle=True)
    print(f"Loaded embeddings shape: {embeddings.shape}")
    
    # Apply projection matrix if provided
    if projection_matrix_path is not None:
        print(f"Loading projection matrix from: {projection_matrix_path}")
        projection_matrix = np.load(projection_matrix_path, allow_pickle=True)
        print(f"Projection matrix shape: {projection_matrix.shape}")
        
        # Apply projection: embeddings @ projection_matrix
        if len(embeddings.shape) == 2 and len(projection_matrix.shape) == 2:
            embeddings = embeddings @ projection_matrix
            print(f"Projected embeddings shape: {embeddings.shape}")
        else:
            raise ValueError(f"Invalid shapes for matrix multiplication: embeddings {embeddings.shape}, projection {projection_matrix.shape}")
    
    # Create encoded_dict similar to encode_data function
    encoded_dict = {}
    
    # Create lookup from bacterium name to taxonomy
    if taxonomy_df is not None:
        taxonomy_map = taxonomy_df.set_index("Original Name")[["Family", "Order", "Class"]].to_dict(orient="index")
    else:
        taxonomy_map = {}
    
    # Check if embeddings and test_labels have compatible dimensions
    if test_labels is not None:
        if len(embeddings) != len(test_labels):
            raise ValueError(f"Mismatch between embeddings length ({len(embeddings)}) and test_labels length ({len(test_labels)})")
    
    # Process each bacterium
    for i in range(len(embeddings)):
        if test_labels is not None:
            bacterium_name = test_labels[i]
        else:
            bacterium_name = f"bacterium_{i}"
        
        bacterium_encoding = embeddings[i]
        
        # Get taxonomy info if available
        if taxonomy_df is not None and bacterium_name in taxonomy_map:
            tax_info = taxonomy_map[bacterium_name]
            if tax_info.get("Family") in [None, ""]:
                print(f"[WARNING] Skipping bacterium with missing family: {bacterium_name}")
                continue
                
            encoded_dict[bacterium_name] = {
                "encoding": bacterium_encoding,
                "family": tax_info.get("Family"),
                "Order": tax_info.get("Order"),
                "Class": tax_info.get("Class")
            }
        else:
            # Use placeholder taxonomy if not available
            encoded_dict[bacterium_name] = {
                "encoding": bacterium_encoding,
                "family": f"Unknown_Family_{i % 5}",  # Create some artificial families for visualization
                "Order": f"Unknown_Order_{i % 3}",
                "Class": f"Unknown_Class_{i % 2}"
            }
    
    print(f"Successfully loaded {len(encoded_dict)} bacterial embeddings")

def encode_data(model, test_data, test_labels, taxonomy_df, device, model_type):
    # apply the model to the test data to get the encoded representation of all bacteria
    with torch.no_grad():
        # Apply the encoder part of the model
        if model_type == "SplitAutoencoder":
            x_encoded, _ = model.encoder(test_data.to(device))  # shape: (num_samples, num_bacteria, 2b)
            x_encoded = model.activation(x_encoded)
        elif model_type == "SplitVAE":
            x_encoded, x_reconstruction, mu, logvar = model.forward(test_data.to(device))  # shape: (num_samples, num_bacteria, 2b)
        
        # Now split the embeddings as in the original forward method
        b = x_encoded.shape[-1] // 2
        encoded_data = x_encoded[:, :, :b] 
        print(f"Encoded Hi (bacteria matrix) shape: {encoded_data.shape}")
    
    # a dictionary to store the encoding and family of each bacterium
    encoded_dict = {}
    num_samples, num_bacteria, embedding_dim = encoded_data.shape    
    # Create lookup from bacterium name in test data to family
    taxonomy_map = taxonomy_df.set_index("Original Name")[["Family", "Order", "Class"]].to_dict(orient="index")

    # Iterate over each bacterium in the test data
    for i in range(num_bacteria):
        # Average of encodings across samples for each bacterium
        bacterium_encoding = encoded_data[:, i, :].mean(dim=0).cpu().numpy()  # shape: (embedding_dim,)
        bacterium_name = test_labels[i] # test_labels are bacterium names
        
        tax_info = taxonomy_map.get(bacterium_name)
        if tax_info is None or tax_info.get("Family") in [None, ""]:
            print(f"[WARNING] Skipping bacterium with missing taxonomy: {bacterium_name}")
            continue

        encoded_dict[bacterium_name] = {
            "encoding": bacterium_encoding,
            "family": tax_info.get("Family"),
            "Order": tax_info.get("Order"),
            "Class": tax_info.get("Class")
        }

        print(f"Taxonomy for {bacterium_name}: Family={tax_info.get('Family')}, "
          f"Order={tax_info.get('Order')}, Class={tax_info.get('Class')}")

    return encoded_dict

def apply_kmeans(reduced_data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(reduced_data)
    return kmeans.labels_

def calculate_purity_score(cluster_labels, true_labels):
    # Convert to numpy arrays for easier manipulation
    cluster_labels = np.array(cluster_labels)
    true_labels = np.array(true_labels)
    
    # Total number of points
    n_samples = len(cluster_labels)
    
    # Get unique clusters
    unique_clusters = np.unique(cluster_labels)
    
    # Calculate purity for each cluster
    total_pure_points = 0
    
    for cluster in unique_clusters:
        # Get indices of points in this cluster
        cluster_mask = cluster_labels == cluster
        cluster_true_labels = true_labels[cluster_mask]
        
        # Count occurrences of each true label in this cluster
        unique_labels, counts = np.unique(cluster_true_labels, return_counts=True)
        
        # The number of points that belong to the most frequent true label in this cluster
        max_count = np.max(counts)
        total_pure_points += max_count
    
    # Purity is the fraction of points that are in the "correct" cluster
    purity = total_pure_points / n_samples
    
    return purity

def test_purity_significance(encoded_data, cluster_labels, true_labels, n_permutations=100, random_seed=42):
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Calculate actual purity score
    actual_purity = calculate_purity_score(cluster_labels, true_labels)
    
    # Generate null distribution by permuting labels
    null_purities = []
    
    print(f"Testing significance with {n_permutations} random permutations...")
    
    for i in range(n_permutations):
        if (i + 1) % 100 == 0:
            print(f"  Permutation {i + 1}/{n_permutations}")
        
        # Randomly shuffle the true labels
        shuffled_labels = np.random.permutation(true_labels)
        
        # Calculate purity with shuffled labels
        null_purity = calculate_purity_score(cluster_labels, shuffled_labels)
        null_purities.append(null_purity)
    
    null_purities = np.array(null_purities)
    
    # Calculate p-value (fraction of null purities >= actual purity)
    p_value = (null_purities >= actual_purity).sum() / n_permutations
    
    # Calculate statistics
    null_mean = np.mean(null_purities)
    null_std = np.std(null_purities)
    z_score = (actual_purity - null_mean) / null_std if null_std > 0 else 0
    
    # Calculate percentile rank
    percentile = (null_purities < actual_purity).sum() / n_permutations * 100
    
    results = {
        'actual_purity': actual_purity,
        'null_mean': null_mean,
        'null_std': null_std,
        'null_purities': null_purities,
        'p_value': p_value,
        'z_score': z_score,
        'percentile': percentile,
        'n_permutations': n_permutations,
        'is_significant_05': p_value < 0.05,
        'is_significant_01': p_value < 0.01,
        'is_significant_001': p_value < 0.001
    }
    
    return results

def plot_significance_test(sig_results, label_type="Family", save_path="purity_significance.png"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot histogram of null distribution
    ax.hist(sig_results['null_purities'], bins=50, alpha=0.7, color='lightblue', 
            edgecolor='black', label='Null distribution')
    
    # Add vertical line for actual purity
    ax.axvline(sig_results['actual_purity'], color='red', linestyle='--', linewidth=2,
               label=f'Actual purity = {sig_results["actual_purity"]:.3f}')
    
    # Add vertical line for null mean
    ax.axvline(sig_results['null_mean'], color='blue', linestyle=':', linewidth=2,
               label=f'Null mean = {sig_results["null_mean"]:.3f}')
    
    ax.set_xlabel('Purity Score')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Purity Score Significance Test ({label_type})\n'
                f'p-value = {sig_results["p_value"]:.4f}, '
                f'Percentile = {sig_results["percentile"]:.1f}%')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_clustering(encoded_data, cluster_labels, true_labels, label_type="Family"):
    purity = calculate_purity_score(cluster_labels, true_labels)
    silhouette_avg = silhouette_score(encoded_data, cluster_labels)
    
    # Create a summary
    evaluation_results = {
        'Purity': purity,
        'Silhouette_Score': silhouette_avg,
        'Label_Type': label_type,
        'Num_Clusters': len(np.unique(cluster_labels)),
        'Num_True_Classes': len(set(true_labels))
    }
    
    return evaluation_results

def print_evaluation_results(results):
    """Print evaluation results in a formatted way"""
    print(f"\n{'='*50}")
    print(f"CLUSTERING EVALUATION RESULTS ({results['Label_Type']})")
    print(f"{'='*50}")
    print(f"Number of clusters: {results['Num_Clusters']}")
    print(f"Number of true {results['Label_Type'].lower()} classes: {results['Num_True_Classes']}")
    print(f"")
    print(f"Purity Score: {results['Purity']:.4f}")
    print(f"")
    print(f"Silhouette Score: {results['Silhouette_Score']:.4f}")
    print(f"{'='*50}")

def evaluate_multiple_k(encoded_data, true_labels, k_range, label_type="Family"):
    # Evaluate clustering for multiple values of k to find optimal number of clusters
    results = {}
    
    print(f"Evaluating clustering for k values: {list(k_range)}")
    
    for k in k_range:
        cluster_labels = apply_kmeans(encoded_data, k)
        eval_results = evaluate_clustering(encoded_data, cluster_labels, true_labels, label_type)
        results[k] = eval_results

    return results

def plot_clustering_metrics(k_results, save_path="clustering_metrics.png"):

    k_values = list(k_results.keys())
    purity_scores = [k_results[k]['Purity'] for k in k_values]
    silhouette_scores = [k_results[k]['Silhouette_Score'] for k in k_values]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Purity plot
    ax1.plot(k_values, purity_scores, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Purity Score')
    ax1.set_title('Purity Score vs Number of Clusters')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette plot
    ax2.plot(k_values, silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score vs Number of Clusters')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_embeddings_with_cluster_boundaries(reduced_dict, kmeans_labels, evaluation_results,
                                           taxonomic_level="family", reduction_method="PCA",
                                           boundary_type="convex_hull", save_path="plot_cluster_boundaries.png"):

    from scipy.spatial import ConvexHull
    from matplotlib.patches import Polygon, Circle
    from sklearn.cluster import DBSCAN
    import matplotlib.patches as patches
    
    # Extract data
    reduced_data = np.array([entry['reduced_encoding'] for entry in reduced_dict.values()])
    tax_labels = [entry[taxonomic_level] for entry in reduced_dict.values()]
    
    # Set up colors for taxonomic groups
    unique_taxa = sorted(set(tax_labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_taxa)))
    taxa_to_color = {taxa: colors[i] for i, taxa in enumerate(unique_taxa)}
    point_colors = [taxa_to_color[tax] for tax in tax_labels]
    
    # Create the plot
    plt.figure(figsize=(16, 12))
    
    # Plot cluster boundaries first (so they appear behind points)
    unique_clusters = sorted(set(kmeans_labels))
    boundary_color = 'lightgrey'  # All boundaries will be light grey
    
    for i, cluster in enumerate(unique_clusters):
        cluster_mask = np.array(kmeans_labels) == cluster
        cluster_points = reduced_data[cluster_mask]
        
        if len(cluster_points) < 3:
            # For clusters with fewer than 3 points, draw circles around each point
            for point in cluster_points:
                circle = Circle(point, radius=0.1, fill=False, 
                              edgecolor=boundary_color, linewidth=1.5, alpha=0.8)
                plt.gca().add_patch(circle)
            continue
        
        if boundary_type == "convex_hull":
            try:
                # Draw convex hull around cluster points
                hull = ConvexHull(cluster_points)
                hull_points = cluster_points[hull.vertices]
                hull_polygon = Polygon(hull_points, fill=False, 
                                     edgecolor=boundary_color, linewidth=1.5, alpha=0.6)
                plt.gca().add_patch(hull_polygon)
            except:
                # Fallback to circle if convex hull fails
                center = np.mean(cluster_points, axis=0)
                radius = np.max(np.linalg.norm(cluster_points - center, axis=1)) * 1.1
                circle = Circle(center, radius=radius, fill=False,
                              edgecolor=boundary_color, linewidth=1.5, alpha=0.6)
                plt.gca().add_patch(circle)
                
        elif boundary_type == "circle":
            # Draw circle around cluster centroid
            center = np.mean(cluster_points, axis=0)
            radius = np.max(np.linalg.norm(cluster_points - center, axis=1)) * 1.2
            circle = Circle(center, radius=radius, fill=False,
                          edgecolor=boundary_color, linewidth=1.5, alpha=0.6)
            plt.gca().add_patch(circle)
            
    # Plot the points on top of boundaries
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1],
                         c=point_colors, s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Create legend for taxonomic groups (colors)
    taxa_handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=taxa_to_color[taxa], markersize=10,
                              label=taxa, markeredgecolor='black')
                   for taxa in unique_taxa]
    
  
    # Add legends
    taxa_legend = plt.legend(handles=taxa_handles, title=f"{taxonomic_level.capitalize()}", 
                            bbox_to_anchor=(1.05, 1), loc='upper left')
        
    # Set title and labels
    k = evaluation_results['Num_Clusters']
    boundary_name = boundary_type.replace('_', ' ').title()
    plt.title(f"2D {reduction_method.upper()} of Bacteria (k = {k})\n"
             f"Color = {taxonomic_level.capitalize()}, {boundary_name} Boundaries = Clusters", fontsize=16)
    
    plt.xlabel(f"{reduction_method.upper()} Component 1", fontsize=14)
    plt.ylabel(f"{reduction_method.upper()} Component 2", fontsize=14)
    
    # Add clustering evaluation scores
    true_k = evaluation_results['Num_True_Classes']
    purity = evaluation_results['Purity']
    silhouette = evaluation_results['Silhouette_Score']
    
    score_text = f"Purity: {purity:.3f} | Silhouette: {silhouette:.3f}\n" \
                f"Chosen k = {k} | True #{taxonomic_level.capitalize()}s = {true_k}"
    
    plt.text(0.99, 0.01, score_text,
            transform=plt.gca().transAxes,
            fontsize=10, color='black',
            ha='right', va='bottom',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster boundary plot to: {save_path}")

def plot_cluster_labels_by_family(reduced_dict, kmeans_labels, evaluation_results, 
                                 reduction_method="PCA", save_path="plot_clustered_family.png"):
    
    family_labels = [entry['family'] for entry in reduced_dict.values()]
    reduced_data = np.array([entry['reduced_encoding'] for entry in reduced_dict.values()])
    
    # Assign colors based on family
    unique_families = sorted(set(family_labels))
    family_to_color = {fam: idx for idx, fam in enumerate(unique_families)}
    colors = [family_to_color[fam] for fam in family_labels]
    
    plt.figure(figsize=(14, 12))
    scatter = plt.scatter(
        reduced_data[:, 0], reduced_data[:, 1],
        c=colors, cmap='tab20', edgecolors='k', s=100, alpha=0.8
    )
    # Add cluster number as text label (adjusted for overlap)
    texts = []
    for i, cluster in enumerate(kmeans_labels):
        texts.append(
            plt.text(reduced_data[i, 0], reduced_data[i, 1], str(cluster),
                     fontsize=10, weight='bold', alpha=0.9)
        )
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    # Legend for families
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          label=fam,
                          markerfacecolor=scatter.cmap(family_to_color[fam] / len(unique_families)), markersize=10)
               for fam in unique_families]
    plt.legend(handles=handles, title="Family", bbox_to_anchor=(1.05, 1), loc='upper left')
    k = evaluation_results['Num_Clusters']
    plt.title(f"2D {reduction_method.upper()} of Bacteria (k = {k})\nColor = Family, Label = Cluster ID", fontsize=16)
    
    plt.xlabel(f"{reduction_method.upper()} Component 1", fontsize=14)
    plt.ylabel(f"{reduction_method.upper()} Component 2", fontsize=14)

    # Add clustering evaluation scores and k
    k = evaluation_results['Num_Clusters']
    true_k = evaluation_results['Num_True_Classes']
    purity = evaluation_results['Purity']
    silhouette = evaluation_results['Silhouette_Score']

    score_text = f"Purity: {purity:.3f} | Silhouette: {silhouette:.3f}\n" \
                f"Chosen k = {k} | True #Classes = {true_k}"

    plt.text(0.99, 0.01, score_text,
            transform=plt.gca().transAxes,
            fontsize=10, color='black',
            ha='right', va='bottom',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_cluster_labels_by_order(reduced_dict, kmeans_labels_order, evaluation_results, 
                                reduction_method="PCA", save_path="plot_clustered_order.png"):
    # Map from bacterium name to order
    order_labels = [entry['Order'] for entry in reduced_dict.values()]
    reduced_data = np.array([entry['reduced_encoding'] for entry in reduced_dict.values()])

    # Assign colors based on order
    unique_orders = sorted(o for o in set(order_labels) if isinstance(o, str))
    order_to_color = {order: idx for idx, order in enumerate(unique_orders)}
    colors = [order_to_color.get(order, -1) for order in order_labels]

    plt.figure(figsize=(14, 12))
    scatter = plt.scatter(
        reduced_data[:, 0], reduced_data[:, 1],
        c=colors, cmap='tab20', edgecolors='k', s=100, alpha=0.8
    )

    texts = []
    for i, cluster in enumerate(kmeans_labels_order):
        texts.append(
            plt.text(reduced_data[i, 0], reduced_data[i, 1], str(cluster),
                     fontsize=10, weight='bold', alpha=0.9)
        )
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # Legend for orders
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          label=order,
                          markerfacecolor=scatter.cmap(order_to_color[order] / len(unique_orders)), markersize=10)
               for order in unique_orders]
    plt.legend(handles=handles, title="Order", bbox_to_anchor=(1.05, 1), loc='upper left')

    k = evaluation_results['Num_Clusters']
    plt.title(f"2D {reduction_method.upper()} of Bacteria (k = {k})\nColor = Order, Label = Cluster ID", fontsize=16)

    plt.xlabel(f"{reduction_method.upper()} Component 1", fontsize=14)
    plt.ylabel(f"{reduction_method.upper()} Component 2", fontsize=14)

    # Add clustering evaluation scores and k
    k = evaluation_results['Num_Clusters']
    true_k = evaluation_results['Num_True_Classes']
    purity = evaluation_results['Purity']
    silhouette = evaluation_results['Silhouette_Score']

    score_text = f"Purity: {purity:.3f} | Silhouette: {silhouette:.3f}\n" \
                f"Chosen k = {k} | True #Classes = {true_k}"

    plt.text(0.99, 0.01, score_text,
            transform=plt.gca().transAxes,
            fontsize=10, color='black',
            ha='right', va='bottom',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_family_distribution(reduced_dict, save_path="family_distribution.png"):
    # Count how many bacteria are in each family
    family_counts = collections.Counter([entry['family'] for entry in reduced_dict.values()])
    
    # Sort families by count (descending)
    sorted_families = sorted(family_counts.items(), key=lambda x: x[1], reverse=True)
    families, counts = zip(*sorted_families)

    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(families, counts, color='skyblue', edgecolor='k')

    # Rotate labels for readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=12)
    plt.xlabel("Family", fontsize=14)
    plt.ylabel("Number of Bacteria", fontsize=14)
    plt.title("Bacteria Count per Family", fontsize=16)

    # Annotate counts on top of bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count),
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def load_model_class(model_file_path, class_name):
    spec = importlib.util.spec_from_file_location("model_module", model_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

def main():
    # Parse command line arguments
    args = parse_arguments()
    output_dir = setup_output_directory(args.output_dir)

    # Check if using external embeddings or model-based encoding
    use_external_embeddings = args.external_embeddings is not None
    
    if use_external_embeddings:
        print("Using external embeddings mode")
        # Validate external embeddings files
        if not os.path.exists(args.external_embeddings):
            print(f"Error: External embeddings file not found: {args.external_embeddings}")
            return
        
        if args.projection_matrix and not os.path.exists(args.projection_matrix):
            print(f"Error: Projection matrix file not found: {args.projection_matrix}")
            return

        # Load test labels and taxonomy (still needed for analysis)
        if os.path.exists(args.test_labels):
            test_labels = load_test_labels(args.test_labels)
        else:
            print(f"Warning: Test labels file not found: {args.test_labels}. Using placeholder labels.")
            test_labels = None
            
        if os.path.exists(args.taxonomy_file):
            taxonomy_df = load_taxonomy(args.taxonomy_file)
        else:
            print(f"Warning: Taxonomy file not found: {args.taxonomy_file}. Using placeholder taxonomy.")
            taxonomy_df = None
            
        print(f"Loading files:")
        print(f"  External embeddings: {args.external_embeddings}")
        if args.projection_matrix:
            print(f"  Projection matrix: {args.projection_matrix}")
        print(f"  Test labels: {args.test_labels}")
        print(f"  Taxonomy: {args.taxonomy_file}")
        print(f"  Dimensionality reduction method: {args.reduction_method}")
        
        # Load external embeddings
        encoded_dict = load_external_embeddings(
            args.external_embeddings, 
            args.projection_matrix, 
            test_labels, 
            taxonomy_df
        )

    else:
        print("Using model-based encoding mode")
        # Validate input files exist
        required_files = [args.test_data, args.test_labels, args.model_path, args.taxonomy_file]
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"Error: File not found: {file_path}")
                return
        
        # Check if model file is provided for model-based encoding
        if not args.model_file:
            print("Error: --model-file is required when not using external embeddings")
            return
            
        print(f"Loading files:")
        print(f"  Test data: {args.test_data}")
        print(f"  Test labels: {args.test_labels}")
        print(f"  Model: {args.model_path}")
        print(f"  Taxonomy: {args.taxonomy_file}")
        print(f"  Dimensionality reduction method: {args.reduction_method}")

        # Load test data, model, and taxonomy table
        test_data = load_test_data(args.test_data)
        test_labels = load_test_labels(args.test_labels)
        ModelClass = load_model_class(args.model_file, args.model_class)
        model = load_model(args.model_path, ModelClass, gene_dim=test_data.shape[-1], embedding_dim=args.embedding_dim)
        taxonomy_df = load_taxonomy(args.taxonomy_file)
      
        # Encode using the model
        encoded_dict = encode_data(model, test_data, test_labels, taxonomy_df, device=device, model_type=args.model_class)

    # From here, the analysis is the same regardless of embedding source
    all_encoded = np.vstack([entry['encoding'] for entry in encoded_dict.values()])

    # Extract true labels for evaluation
    family_labels = [entry['family'] for entry in encoded_dict.values()]
    order_labels = [entry['Order'] for entry in encoded_dict.values()]
    
    # Test multiple k values to find optimal clustering
    k_range = range(args.min_k, args.max_k + 1)
    
    family_results = evaluate_multiple_k(all_encoded, family_labels, k_range, "Family")
    order_results = evaluate_multiple_k(all_encoded, order_labels, k_range, "Order")
    
    # Find best k for each taxonomic level 
    best_k_family = max(family_results.keys(), 
                       key=lambda k: family_results[k]['Purity'])
    best_k_order = max(order_results.keys(), 
                      key=lambda k: order_results[k]['Purity'])
    
    print(f"\nBest k for Family clustering: {best_k_family}")
    print(f"Best k for Order clustering: {best_k_order}")

    # Apply KMeans in original (high-dimensional) space
    kmeans_labels_family = apply_kmeans(all_encoded, num_clusters=best_k_family)
    kmeans_labels_order = apply_kmeans(all_encoded, num_clusters=best_k_order)

    # Evaluate family clustering
    family_eval = evaluate_clustering(all_encoded, kmeans_labels_family, family_labels, "Family")
    print_evaluation_results(family_eval)
    
    # Evaluate order clustering
    order_eval = evaluate_clustering(all_encoded, kmeans_labels_order, order_labels, "Order")
    print_evaluation_results(order_eval)

    # Test statistical significance of purity scores
    print("\n" + "="*60)
    print("TESTING STATISTICAL SIGNIFICANCE OF CLUSTERING RESULTS")
    print("="*60)
    
    # Test family clustering significance
    family_sig_results = test_purity_significance(
        all_encoded, kmeans_labels_family, family_labels, 100
    )
    
    # Test order clustering significance
    order_sig_results = test_purity_significance(
        all_encoded, kmeans_labels_order, order_labels, 100
    )

    # Plot significance results
    plot_significance_test(family_sig_results, "Family", os.path.join(output_dir, "family_purity_significance.png"))
    plot_significance_test(order_sig_results, "Order", os.path.join(output_dir, "order_purity_significance.png"))

    # Generate clustering metrics plots
    plot_clustering_metrics(family_results, os.path.join(output_dir, "family_clustering_metrics.png"))
    plot_clustering_metrics(order_results, os.path.join(output_dir, "order_clustering_metrics.png"))

    # Apply dimensionality reduction for visualization
    print(f"\nApplying {args.reduction_method.upper()} for 2D visualization...")
    reduced_data = apply_dimensionality_reduction(all_encoded, method=args.reduction_method, n_components=2)

    # Update reduced_dict with 2D encodings
    reduced_dict = {}
    for i, (bacterium_name, entry) in enumerate(encoded_dict.items()):
        reduced_dict[bacterium_name] = {
            "reduced_encoding": reduced_data[i],
            "family": entry["family"],
            "Order": entry["Order"],
            "Class": entry["Class"]
        }

    # Generate all plots with the selected dimensionality reduction method
    method_name = args.reduction_method.upper()

    # Original plots with cluster numbers
    family_plot_path = os.path.join(output_dir, f"plot_clustered_family_{args.reduction_method}.png")
    order_plot_path = os.path.join(output_dir, f"plot_clustered_order_{args.reduction_method}.png")
    family_dist_path = os.path.join(output_dir, "family_distribution.png")

    plot_cluster_labels_by_family(reduced_dict, kmeans_labels_family, family_eval, method_name, family_plot_path)
    plot_cluster_labels_by_order(reduced_dict, kmeans_labels_order, order_eval, method_name, order_plot_path)
    plot_family_distribution(reduced_dict, family_dist_path)

    # Cluster boundary plots (dots colored by taxonomy with cluster boundaries)
    family_boundary_path = os.path.join(output_dir, f"plot_family_cluster_boundaries_{args.reduction_method}.png")
    order_boundary_path = os.path.join(output_dir, f"plot_order_cluster_boundaries_{args.reduction_method}.png")
    
    plot_embeddings_with_cluster_boundaries(reduced_dict, kmeans_labels_family, family_eval,
                                           "family", method_name, "circle", family_boundary_path)
    plot_embeddings_with_cluster_boundaries(reduced_dict, kmeans_labels_order, order_eval,
                                           "Order", method_name, "circle", order_boundary_path)


    print(f"\nAll plots saved to: {output_dir}")
    print(f"Dimensionality reduction method used: {args.reduction_method.upper()}")

    if use_external_embeddings:
        print(f"Analysis completed using external embeddings from: {args.external_embeddings}")
        if args.projection_matrix:
            print(f"Projection matrix applied from: {args.projection_matrix}")
    else:
        print(f"Analysis completed using model-based encoding from: {args.model_path}")

if __name__ == "__main__":
    main()