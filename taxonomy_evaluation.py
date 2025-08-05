import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os, collections, argparse, umap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import MDS, TSNE
from scipy.spatial.distance import pdist, squareform
from adjustText import adjust_text
from collections import Counter
from pathlib import Path
sys.path.append(".")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_embeddings(file_path="embeddings.npy"):
    embeddings_tensor = np.load(file_path, allow_pickle=True)
    embeddings_tensor = torch.tensor(embeddings_tensor, dtype=torch.float32)
    return embeddings_tensor

def load_embeddings_labels(file_path="embeddings_labels.npy"):
    """
    Load .npy file containing full taxonomy lineage strings and extract taxonomy levels.
    """
    full_lineages = np.load(file_path, allow_pickle=True)[1:]

    parsed = []
    for lineage in full_lineages:
        parts = lineage.split('|')
        taxonomy = {p[0]: p[3:] for p in parts if '__' in p}  
        parsed.append({
            "Original Name": lineage,
            "Phylum": taxonomy.get('p', ''),
            "Class": taxonomy.get('c', ''),
            "Order": taxonomy.get('o', ''),
            "Family": taxonomy.get('f', ''),
        })

    taxonomy_df = pd.DataFrame(parsed)
    labels = taxonomy_df["Original Name"].tolist()
    return labels, taxonomy_df

def load_variance_file(variance_file, num_bacteria):
    if variance_file is None:
        return None
    try:
        if variance_file.endswith('.npy'):
            variances = np.load(variance_file)
        elif variance_file.endswith('.csv'):
            variances = pd.read_csv(variance_file).iloc[:, 0].to_numpy()
        else:
            raise ValueError("Unsupported variance file format. Use .npy or .csv")

        if len(variances) != num_bacteria:
            raise ValueError(f"Variance file has {len(variances)} entries, but {num_bacteria} embeddings exist")

        return variances
    except Exception as e:
        print(f"[ERROR] Failed to load variance file: {e}")
        return None

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate clustering performance of bacterial embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input files
    parser.add_argument("--embeddings", type=str, default="embeddings.npy", help="Path to external embeddings numpy file (.npy).")
    parser.add_argument("--embeddings_labels", type=str, default="embeddings_labels.csv", help="Path to embeddings bacteria labels numpy file (.csv)")
    parser.add_argument("--variance_file", type=str, default=None, 
    help="Optional path to variance file (npy or csv) where each entry corresponds to variance of a bacterium embedding")

    # Output directory
    parser.add_argument("--output_dir", type=str, default="./plots",help="Directory to save output plots and results")
    
    # Clustering parameters
    parser.add_argument("--min_k", type=int, default=2,help="Minimum number of clusters to test")
    parser.add_argument("--max_k", type=int, default=15, help="Maximum number of clusters to test")

    # Dimensionality reduction method
    parser.add_argument("--reduction_method", type=str, default="pca",choices=['pca', 'tsne', 'umap', 'pcoa'],
                        help="Dimensionality reduction method for visualization"
    )
    # Taxonomic levels to process
    parser.add_argument("--taxonomic_levels", type=str, default="family,order,class,phylum", 
        help="Comma-separated list of taxonomic levels to process. Options: 'family', 'order', 'class', 'phylum'")

    return parser.parse_args()

def setup_output_directory(output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    return output_dir  

def organize_embeddings(embeddings_data, embeddings_labels, taxonomy_df,  variances=None):
    """
    Process embeddings and organize them with taxonomy information
    """    
    embeddings_data = embeddings_data[1:]  # Exclude the first row
    # Create a dictionary to store the encoding and family of each bacterium
    encoded_dict = {}

    # Convert embeddings_data to numpy if it's a tensor
    if hasattr(embeddings_data, 'cpu'):
        embeddings_data = embeddings_data.cpu().numpy()
    
    num_bacteria, embedding_dim = embeddings_data.shape
    print(f"Embeddings shape: ({num_bacteria}, {embedding_dim}) - one embedding per bacterium")
    
    # Verify that we have the same number of bacteria in both files
    if num_bacteria != len(embeddings_labels):
        raise ValueError(f"Mismatch: {num_bacteria} bacteria in embeddings but {len(embeddings_labels)} in labels")
    
    print(f"Processing {num_bacteria} bacteria...")
    
    # Create lookup from bacterium name to taxonomy info
    taxonomy_map = taxonomy_df.set_index("Original Name")[["Family", "Order", "Class", "Phylum"]].to_dict(orient="index")

    # Iterate over each bacterium - embeddings_data[i] corresponds to embeddings_labels[i]
    for i in range(num_bacteria):
        bacterium_name = embeddings_labels[i]
        bacterium_embedding = embeddings_data[i]  # shape: (embedding_dim,)

        # Convert to numpy if it's a tensor
        if hasattr(bacterium_embedding, 'cpu'):
            bacterium_embedding = bacterium_embedding.cpu().numpy()
        
        # Get taxonomy information
        tax_info = taxonomy_map.get(bacterium_name)
        if tax_info is None or tax_info.get("Family") in [None, ""]:
            print(f"[WARNING] Skipping bacterium with missing taxonomy: {bacterium_name}")
            continue

        # Use consistent lowercase keys for all taxonomic levels
        encoded_dict[bacterium_name] = {
            "encoding": bacterium_embedding,
            "family": tax_info.get("Family"),
            "order": tax_info.get("Order"),      # Changed from "Order" to "order"
            "class": tax_info.get("Class"),      # Changed from "Class" to "class" 
            "phylum": tax_info.get("Phylum")     # Changed from "Phylum" to "phylum"
        }

        if variances is not None:
            encoded_dict[bacterium_name]['variance'] = variances[i]
        else:
            encoded_dict[bacterium_name]['variance'] = None

        if i < 5:  # Print first 5 for verification
            print(f"Bacterium {i}: {bacterium_name} -> Family={tax_info.get('Family')}, "
                    f"Order={tax_info.get('Order')}, Class={tax_info.get('Class')}, Phylum={tax_info.get('Phylum')}")

    return encoded_dict

def filter_top_taxonomic_levels(encoded_dict, taxonomic_level, top_n=15):
    """Filter data by the top N most frequent taxonomic labels."""
    # Extract taxonomic labels
    tax_labels = [entry[taxonomic_level] for entry in encoded_dict.values()]
    
    # Count occurrences and get top N
    label_counts = Counter(tax_labels)
    most_frequent_labels = [label for label, _ in label_counts.most_common(top_n)]
    
    # Filter the encoded_dict to keep only bacteria with top N taxonomic labels
    filtered_encoded_dict = {}
    for bacterium_name, entry in encoded_dict.items():
        if entry[taxonomic_level] in most_frequent_labels:
            filtered_encoded_dict[bacterium_name] = entry
    
    return filtered_encoded_dict

def apply_kmeans(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans.labels_

def calculate_purity_score(cluster_labels, true_labels):
    # Convert to numpy arrays for easier manipulation
    cluster_labels = np.array(cluster_labels)
    true_labels = np.array(true_labels)
    
    n_samples = len(cluster_labels)
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

def test_purity_significance(data, cluster_labels, true_labels, n_permutations=100, random_seed=42):
    np.random.seed(random_seed)
    actual_purity = calculate_purity_score(cluster_labels, true_labels)
    # Generate null distribution by permuting labels
    null_purities = []
        
    for i in range(n_permutations):        
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
    percentile = (null_purities < actual_purity).sum() / n_permutations * 100
    
    results = {
        'actual_purity': actual_purity,
        'null_mean': null_mean,
        'null_std': null_std,
        'null_purities': null_purities,
        'p_value': p_value,
        'z_score': z_score,
        'percentile': percentile,
        'n_permutations': n_permutations
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

def evaluate_clustering(data, cluster_labels, true_labels, label_type="Family"):
    purity = calculate_purity_score(cluster_labels, true_labels)
    silhouette_avg = silhouette_score(data, cluster_labels)
    
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

def evaluate_multiple_k(data, true_labels, k_range, label_type="Family"):
    results = {}
    print(f"Evaluating clustering for k values: {list(k_range)}")
    
    for k in k_range:
        cluster_labels = apply_kmeans(data, k)
        # Evaluate clustering 
        purity = calculate_purity_score(cluster_labels, true_labels)
        silhouette_avg = silhouette_score(data, cluster_labels)
        
        eval_results = {
            'Purity': purity,
            'Silhouette_Score': silhouette_avg,
            'Label_Type': label_type,
            'Num_Clusters': len(np.unique(cluster_labels)),
            'Num_True_Classes': len(set(true_labels))
        }
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

def apply_global_dimensionality_reduction(encoded_dict, method='pca', n_components=2, random_state=42):
    """
    Apply dimensionality reduction to the full dataset and store coordinates in encoded_dict.
    ensures consistent coordinates across all taxonomic level plots.
    """
    print(f"Applying global {method.upper()} dimensionality reduction to full dataset...")
    
    # Extract all embeddings
    bacterium_names = list(encoded_dict.keys())
    all_data = np.array([encoded_dict[name]['encoding'] for name in bacterium_names])
    
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
        reduced_data = reducer.fit_transform(all_data)
        
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=random_state, 
                      perplexity=min(30, len(all_data)-1))
        reduced_data = reducer.fit_transform(all_data)
        
    elif method.lower() == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        reduced_data = reducer.fit_transform(all_data)
        
    elif method.lower() == 'pcoa':
        # Compute pairwise distances
        distances = pdist(all_data, metric='euclidean')
        distance_matrix = squareform(distances)
        
        # Apply classical MDS (which is equivalent to PCoA)
        reducer = MDS(n_components=n_components, dissimilarity='precomputed', 
                     random_state=random_state)
        reduced_data = reducer.fit_transform(distance_matrix)
        
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    # Store the reduced coordinates in the encoded_dict
    for i, bacterium_name in enumerate(bacterium_names):
        encoded_dict[bacterium_name]['global_reduced_encoding'] = reduced_data[i]
    
    print(f"Global {method.upper()} reduction complete. Coordinates stored for {len(bacterium_names)} bacteria.")
    return encoded_dict

def plot_by_taxonomy(filtered_encoded_dict, taxonomic_level, evaluation_results, reduction_method="PCA", 
                    save_path="plot_taxonomy.png"):

    # Extract taxonomic labels and reduced data
    taxonomic_labels = [entry.get(taxonomic_level, 'Unknown') for entry in filtered_encoded_dict.values()]
    # reduced_data = np.array([entry['reduced_encoding'] for entry in filtered_encoded_dict.values()])
    reduced_data = np.array([entry['global_reduced_encoding'] for entry in filtered_encoded_dict.values()])
    
    variances = [entry.get('variance', None) for entry in filtered_encoded_dict.values()]
    
    # Normalize variances to a reasonable radius scale, or set constant
    if any(v is not None for v in variances):
        variance_array = np.array([v if v is not None else 0 for v in variances])
        # Scale variance to size range (e.g., 60–300)
        min_radius, max_radius = 60, 300
        norm_variance = (variance_array - variance_array.min()) / (variance_array.ptp() + 1e-6)
        radii = min_radius + norm_variance * (max_radius - min_radius)
    else:
        radii = 100  # default size

    # Create color mapping for the taxonomic groups
    unique_taxa = sorted(set(taxonomic_labels))
    taxa_to_color = {taxa: idx for idx, taxa in enumerate(unique_taxa)}
    colors = [taxa_to_color.get(taxa, taxa_to_color.get('Unknown', 0)) for taxa in taxonomic_labels]

    # Create the plot
    plt.figure(figsize=(14, 12))
    scatter = plt.scatter(
        reduced_data[:, 0], reduced_data[:, 1],
        c=colors, cmap='tab20', edgecolors='k', s=radii, alpha=0.8
    )
    
    # Create legend
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          label=taxa,
                          markerfacecolor=scatter.cmap(taxa_to_color[taxa] / max(len(unique_taxa)-1, 1)), 
                          markersize=10)
               for taxa in unique_taxa]
    plt.legend(handles=handles, title=f"{taxonomic_level.capitalize()}", 
               bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set title and labels
    plt.title(f"2D {reduction_method.upper()} of Bacteria \nColored by {taxonomic_level.capitalize()}", 
              fontsize=16)
    
    plt.xlabel(f"{reduction_method.upper()} Component 1", fontsize=14)
    plt.ylabel(f"{reduction_method.upper()} Component 2", fontsize=14)
    
    k = evaluation_results['Num_Clusters']
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
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_tax_distribution(filtered_encoded_dict, taxonomic_level, save_path="tax_distribution.png"):
    # Count how many bacteria are in each family (or other taxonomic level)
    counts = collections.Counter([entry[taxonomic_level] for entry in filtered_encoded_dict.values()])
    
    # Sort taxonomic labels by count (descending)
    sorted_tax = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    tax, counts = zip(*sorted_tax)

    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(tax, counts, color='skyblue', edgecolor='k')

    # Rotate labels for readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=12)
    plt.xlabel(f"{taxonomic_level}", fontsize=14)  # Use f-string to insert taxonomic_level
    plt.ylabel("Number of Bacteria", fontsize=14)
    plt.title(f"Bacteria Count per {taxonomic_level.capitalize()}", fontsize=16)  # Capitalize the taxonomic level in title

    # Annotate counts on top of bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count),
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def process_taxonomic_level(taxonomic_level, encoded_dict, k_range, output_dir, reduction_method="PCA", top_n=15):
    """
    Process the embeddings for a given taxonomic level (e.g., 'family' or 'order').
    This includes clustering, dimensionality reduction, and plotting.
    """
    # Create output directory for the current taxonomic level
    taxonomic_output_dir = os.path.join(output_dir, taxonomic_level.capitalize())
    os.makedirs(taxonomic_output_dir, exist_ok=True)

    # Filter by the top N most frequent taxonomic labels
    filtered_dict = filter_top_taxonomic_levels(encoded_dict, taxonomic_level, top_n)
    filtered_data = np.array([entry['encoding'] for entry in filtered_dict.values()])
    filtered_labels = [entry[taxonomic_level] for entry in filtered_dict.values()]
    print(f"{taxonomic_level.capitalize()} filtered data shape: {filtered_data.shape}")
    
    # Test multiple k values to find optimal clustering
    results = evaluate_multiple_k(filtered_data, filtered_labels, k_range, taxonomic_level.capitalize())
    # Find the best k based on purity
    best_k = max(results.keys(), key=lambda k: results[k]['Purity'])
    # Plot clustering metrics
    plot_clustering_metrics(results, os.path.join(taxonomic_output_dir, f"{taxonomic_level}_clustering_metrics.png"))
    print(f"\nBest k for {taxonomic_level.capitalize()} clustering: {best_k}")

    # Apply KMeans with the optimal k
    kmeans_labels = apply_kmeans(filtered_data, num_clusters=best_k)

    # Evaluate clustering performance
    eval_results = evaluate_clustering(filtered_data, kmeans_labels, filtered_labels, taxonomic_level.capitalize())
    print_evaluation_results(eval_results)

    # Test clustering significance
    sig_results = test_purity_significance(filtered_data, kmeans_labels, filtered_labels, n_permutations=100)
    
    # Plot significance results
    plot_significance_test(sig_results, taxonomic_level.capitalize(),
                           os.path.join(taxonomic_output_dir, f"{taxonomic_level}_purity_significance.png"))
    
    # Plot taxonomy distribution
    plot_tax_distribution(filtered_dict, taxonomic_level, os.path.join(taxonomic_output_dir, 
                                                                       f"{taxonomic_level}_distribution.png"))

    # now uses global coordinates stored in 'global_reduced_encoding'
    tax_plot_path = os.path.join(taxonomic_output_dir, f"plot_{taxonomic_level}_{reduction_method}.png")
    plot_by_taxonomy(filtered_dict, taxonomic_level, eval_results, reduction_method, tax_plot_path)

    print(f"\nAll plots saved to: {taxonomic_output_dir}")
    print(f"Dimensionality reduction method used: {reduction_method.upper()}")

def main():
    # Parse command line arguments
    args = parse_arguments()
    output_dir = setup_output_directory(args.output_dir)

    # Load files
    embeddings_data = load_embeddings(args.embeddings)
    embeddings_labels, taxonomy_df = load_embeddings_labels(args.embeddings_labels)
    variances = load_variance_file(args.variance_file, len(embeddings_labels))
    
    # Organize embeddings with taxonomy information
    encoded_dict = organize_embeddings(embeddings_data, embeddings_labels, taxonomy_df, variances)

    # Apply global dimensionality reduction once for consistent coordinates
    encoded_dict = apply_global_dimensionality_reduction(encoded_dict, method=args.reduction_method, 
                                                        n_components=2, random_state=42)
    
    # Define the k range for clustering
    k_range = range(args.min_k, args.max_k + 1)

    # Get the list of taxonomic levels to process from the command line argument
    taxonomic_levels = args.taxonomic_levels.split(',')

    # Process the taxonomic levels (family, order, class, phylum)
    for taxonomic_level in taxonomic_levels:
        process_taxonomic_level(taxonomic_level, encoded_dict, k_range, output_dir,
                                     reduction_method=args.reduction_method, top_n=15)
    print("\nAll evaluations and plots completed successfully!")
    
if __name__ == "__main__":
    main()