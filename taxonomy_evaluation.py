import torch
import numpy as np
import pandas as pd
import sys, os, collections, argparse
import importlib.util
import umap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.manifold import MDS, TSNE
import matplotlib.pyplot as plt
from adjustText import adjust_text
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
sys.path.append(".")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_embeddings(file_path="embeddings.npy")
    embeddings_tensor = np.load(file_path, allow_pickle=True)
    embeddings_tensor = torch.tensor(embeddings_tensor, dtype=torch.float32)
    return embeddings_tensor

def load_embeddings_labels(file_path="embeddings_labels.npy"):
    embeddings_labels = np.load(file_path, allow_pickle=True)
    return embeddings_labels

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
    parser.add_argument("--embeddings", 
                        type=str, 
                        default="embeddings.npy",
                        help="Path to external embeddings numpy file (.npy)."
    )

    parser.add_argument("--embeddings_labels", 
                        type=str, 
                        default="embeddings_labels.npy",
                        help="Path to embeddings bacteria labels numpy file (.npy)"
    )
    
    parser.add_argument("--taxonomy_file", 
                        type=str, 
                        default="bacterial_lineage.csv",
                        help="Path to bacterial taxonomy CSV file"
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

    return parser.parse_args()

def setup_output_directory(output_dir):
    """Create output directory if it doesn't exist"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    return output_dir  

def organize_embeddings(embeddings_data, embeddings_labels, taxonomy_df):
    """Process embeddings and organize them with taxonomy information
    embeddings_data: numpy array of shape (num_bacteria, embedding_dim) 
    embeddings_labels: list/array where embeddings_labels[i] is the name of the bacteria corresponding to embeddings_data[i]
    taxonomy_df: DataFrame with taxonomy information
    """
    # Create a dictionary to store the encoding and family of each bacterium
    encoded_dict = {}
    
    if len(embeddings_data.shape) == 2: # Shape: (num_bacteria, embedding_dim)
        num_bacteria, embedding_dim = embeddings_data.shape
        print(f"Embeddings shape: ({num_bacteria}, {embedding_dim}) - one embedding per bacterium")
    else:
        raise ValueError(f"Unexpected embeddings shape: {embeddings_data.shape}")
    
    # Verify that we have the same number of bacteria in both files
    if num_bacteria != len(embeddings_labels):
        raise ValueError(f"Mismatch: {num_bacteria} bacteria in embeddings but {len(embeddings_labels)} in labels")
    
    print(f"Processing {num_bacteria} bacteria...")
    
    # Create lookup from bacterium name to taxonomy info
    taxonomy_map = taxonomy_df.set_index("Original Name")[["Family", "Order", "Class"]].to_dict(orient="index")

    # Iterate over each bacterium - embeddings_data[i] corresponds to embeddings_labels[i]
    for i in range(num_bacteria):
        bacterium_name = embeddings_labels[i]
        
        if len(embeddings_data.shape) == 2:
            # Single embedding per bacterium
            bacterium_embedding = embeddings_data[i]  # shape: (embedding_dim,)
        else:
            # Multiple samples per bacterium - take the mean
            bacterium_embedding = embeddings_data[i].mean(axis=0)  # shape: (embedding_dim,)
        
        # Convert to numpy if it's a tensor
        if hasattr(bacterium_embedding, 'cpu'):
            bacterium_embedding = bacterium_embedding.cpu().numpy()
        
        # Get taxonomy information
        tax_info = taxonomy_map.get(bacterium_name)
        if tax_info is None or tax_info.get("Family") in [None, ""]:
            print(f"[WARNING] Skipping bacterium with missing taxonomy: {bacterium_name}")
            continue

        encoded_dict[bacterium_name] = {
            "encoding": bacterium_embedding,
            "family": tax_info.get("Family"),
            "Order": tax_info.get("Order"),
            "Class": tax_info.get("Class")
        }

        if i < 5:  # Print first 5 for verification
            print(f"Bacterium {i}: {bacterium_name} -> Family={tax_info.get('Family')}, "
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
    np.random.seed(random_seed)
    # Calculate actual purity score
    actual_purity = calculate_purity_score(cluster_labels, true_labels)
    # Generate null distribution by permuting labels
    null_purities = []
        
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
                                           save_path="plot_cluster_boundaries_{taxonomic_level}.png"):

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

def plot_by_taxonomy(reduced_dict, evaluation_results, 
                            taxonomic_level, reduction_method="PCA", 
                            save_path="plot_{taxonomic_level}.png"):

    # Extract taxonomic labels and reduced data
    taxonomic_labels = [entry.get(taxonomic_level, 'Unknown') for entry in reduced_dict.values()]
    reduced_data = np.array([entry['reduced_encoding'] for entry in reduced_dict.values()])
    
    # Get unique taxonomic groups (filter out None/NaN values)
    unique_taxa = sorted([taxa for taxa in set(taxonomic_labels) 
                         if taxa is not None and str(taxa) != 'nan' and str(taxa) != 'Unknown'])
    
    # Add 'Unknown' category if it exists in the data
    if 'Unknown' in taxonomic_labels or None in taxonomic_labels:
        unique_taxa.append('Unknown')
    
    # Create color mapping
    taxa_to_color = {taxa: idx for idx, taxa in enumerate(unique_taxa)}
    colors = [taxa_to_color.get(taxa, taxa_to_color.get('Unknown', 0)) for taxa in taxonomic_labels]
    
    # Create the plot
    plt.figure(figsize=(14, 12))
    scatter = plt.scatter(
        reduced_data[:, 0], reduced_data[:, 1],
        c=colors, cmap='tab20', edgecolors='k', s=100, alpha=0.8
    )
    
    # Create legend
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          label=taxa,
                          markerfacecolor=scatter.cmap(taxa_to_color[taxa] / max(len(unique_taxa)-1, 1)), 
                          markersize=10)
               for taxa in unique_taxa]
    plt.legend(handles=handles, title=taxonomic_level.capitalize(), 
               bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set title and labels
    k = evaluation_results['Num_Clusters']
    plt.title(f"2D {reduction_method.upper()} of Bacteria (k = {k})\nColored by {taxonomic_level.capitalize()}", 
              fontsize=16)
    
    plt.xlabel(f"{reduction_method.upper()} Component 1", fontsize=14)
    plt.ylabel(f"{reduction_method.upper()} Component 2", fontsize=14)
    
    # Add clustering evaluation scores
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

def main():
    # Parse command line arguments
    args = parse_arguments()
    output_dir = setup_output_directory(args.output_dir)

    # Load files
    embeddings_data = load_embeddings(args.embeddings)
    embeddings_labels = load_embeddings_labels(args.embeddings_labels)
    taxonomy_df = load_taxonomy(args.taxonomy_file)
    
    # Organize embeddings with taxonomy information
    encoded_dict = organize_embeddings(embeddings_data, embeddings_labels, taxonomy_df)

    # Extract all encodings for clustering analysis
    all_encoded = np.array([entry['encoding'] for entry in encoded_dict.values()])

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

    # Test family clustering significance
    family_sig_results = test_purity_significance(
        all_encoded, kmeans_labels_family, family_labels, 100
    )
    
    # Test order clustering significance
    order_sig_results = test_purity_significance(
        all_encoded, kmeans_labels_order, order_labels, 100
    )

    family_output_dir = os.path.join(output_dir, 'Family')
    order_output_dir = os.path.join(output_dir, 'Order')
    
    os.makedirs(family_output_dir, exist_ok=True)
    os.makedirs(order_output_dir, exist_ok=True)

    # Plot significance results
    plot_significance_test(family_sig_results, "Family", os.path.join(family_output_dir, "family_purity_significance.png"))
    plot_significance_test(order_sig_results, "Order", os.path.join(order_output_dir, "order_purity_significance.png"))

    # Generate clustering metrics plots
    plot_clustering_metrics(family_results, os.path.join(family_output_dir, "family_clustering_metrics.png"))
    plot_clustering_metrics(order_results, os.path.join(order_output_dir, "order_clustering_metrics.png"))

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
    
    plot_family_distribution(reduced_dict, os.path.join(family_output_dir, "family_distribution.png"))

    # Generate all plots with the selected dimensionality reduction method
    method_name = args.reduction_method.upper()
    # Define plot file paths
    family_plot_path = os.path.join(family_output_dir, f"plot_family_{args.reduction_method}.png")
    order_plot_path = os.path.join(order_output_dir, f"plot_order_{args.reduction_method}.png")
    # Plot embeddings colored by taxonomy
    plot_by_taxonomy(reduced_dict, family_eval, "family", method_name, family_plot_path)
    plot_by_taxonomy(reduced_dict, order_eval, "Order", method_name, order_plot_path)
    
    # Cluster boundary plots (dots colored by taxonomy with cluster boundaries)
    family_boundary_path = os.path.join(family_output_dir, f"plot_family_cluster_boundaries_{args.reduction_method}.png")
    order_boundary_path = os.path.join(order_output_dir, f"plot_order_cluster_boundaries_{args.reduction_method}.png")
    
    plot_embeddings_with_cluster_boundaries(reduced_dict, kmeans_labels_family, family_eval,
                                           "family", method_name, family_boundary_path)
    plot_embeddings_with_cluster_boundaries(reduced_dict, kmeans_labels_order, order_eval,
                                           "Order", method_name, order_boundary_path)

    print(f"\nAll plots saved to: {output_dir}")
    print(f"Dimensionality reduction method used: {args.reduction_method.upper()}")

if __name__ == "__main__":
    main()