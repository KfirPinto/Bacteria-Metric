import torch
import numpy as np
import sys
import os
# Add the autoencoder_model directory to Python path so torch.load can find the training module
sys.path.append(os.path.join(os.path.dirname(__file__), 'autoencoder_model'))
from autoencoder_model.training.model import SplitAutoencoder
# from training.model import SplitAutoencoder  # new import path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text
import collections
from io import StringIO
from pathlib import Path
import argparse

def load_test_data(file_path="test_tensor.npy"):
    test_tensor = np.load(file_path, allow_pickle=True)
    test_tensor = torch.tensor(test_tensor, dtype=torch.float32)
    return test_tensor

def load_test_labels(file_path="test_bacteria.npy"):
    test_labels = np.load(file_path, allow_pickle=True)
    return test_labels

def load_model(model_path="split_autoencoder.pt", device="cuda"):
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()  #set the model to evaluation mode
    return model

def load_taxonomy(taxonomy_file="bacterial_lineage_formatted.csv"):
    # Load the taxonomy table into a pandas DataFrame
    taxonomy_df = pd.read_csv(taxonomy_file)

    # Detect missing family values
    missing_family_df = taxonomy_df[taxonomy_df['Family'].isna() | (taxonomy_df['Family'] == '')]

    if not missing_family_df.empty:
        print("The following bacteria have missing 'Family' information and will be dropped:")
        print(missing_family_df["Original Name"].tolist())

    # Drop rows with missing or empty Family column
    taxonomy_df = taxonomy_df[~taxonomy_df['Family'].isna() & (taxonomy_df['Family'] != '')]

    return taxonomy_df

def parse_arguments():
    """Parse command line arguments for input files and parameters"""
    parser = argparse.ArgumentParser(
        description="Evaluate clustering performance of bacterial embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input files
    parser.add_argument(
        "--test_data", 
        type=str, 
        default="test_tensor.npy",
        help="Path to test data numpy file (.npy)"
    )
    
    parser.add_argument(
        "--test_labels", 
        type=str, 
        default="test_bacteria.npy",
        help="Path to test bacteria labels numpy file (.npy)"
    )
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="split_autoencoder.pt",
        help="Path to trained model file (.pt)"
    )
    
    parser.add_argument(
        "--taxonomy_file", 
        type=str, 
        default="bacterial_lineage_formatted.csv",
        help="Path to bacterial taxonomy CSV file"
    )
    
    # Output directory
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./plots",
        help="Directory to save output plots and results"
    )
    
    # Clustering parameters
    parser.add_argument(
        "--min_k", 
        type=int, 
        default=2,
        help="Minimum number of clusters to test"
    )
    
    parser.add_argument(
        "--max_k", 
        type=int, 
        default=15,
        help="Maximum number of clusters to test"
    )
    
    # Device
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for computation"
    )
    
    return parser.parse_args()

def setup_device(device_arg):
    """Setup computation device"""
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    return device

def setup_output_directory(output_dir):
    """Create output directory if it doesn't exist"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    return output_dir  

def encode_data(model, test_data, test_labels, taxonomy_df, device="cuda"):
    # Move test data to the same device as the model (GPU or CPU)
    test_data = test_data.to(device)
    # apply the model to the test data to get the encoded representation of all bacteria
    with torch.no_grad():
        # Apply the encoder part of the model to the test data
        x_encoded = model.encoder(test_data)  # shape: (num_samples, num_bacteria, 2b)
        x_encoded = model.activation(x_encoded)  
        
        # Now split the embeddings as in the original forward method
        b = x_encoded.shape[-1] // 2
        encoded_data = x_encoded[:, :, :b]  # first half of embedding
        print(f"Encoded Hi (bacteria matrix) shape: {encoded_data.shape}")
    
    # a dictionary to store the encoding and family of each bacterium
    encoded_dict = {}
    # the model output is a tensor of shape (samples, bacteria, embedding dim) 
    num_samples, num_bacteria, embedding_dim = encoded_data.shape    
    # Create lookup from bacterium name in test data to family
    # taxonomy_map = taxonomy_df.set_index("Original Name")["Family"].to_dict()
    taxonomy_map = taxonomy_df.set_index("Original Name")[["Family", "Order", "Class"]].to_dict(orient="index")

    # Iterate over each bacterium in the test data
    for i in range(num_bacteria):
        # Average of encodings across samples for each bacterium
        bacterium_encoding = encoded_data[:, i, :].mean(dim=0).cpu().numpy()  # shape: (embedding_dim,)
        # Extract the corresponding bacterium name from test_bacteria labels
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
        # print(f"Family for bacterium {bacterium_name}: {family}")

    return encoded_dict

def apply_kmeans(reduced_data, num_clusters):
    # Apply k-means clustering
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

def evaluate_clustering(encoded_data, cluster_labels, true_labels, label_type="Family"):
    
    # Calculate Purity Score - measures how many points in each cluster belong to the most common true label
    purity = calculate_purity_score(cluster_labels, true_labels)

    # Calculate Silhouette Score - measures how well separated the clusters are
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
    """
    Evaluate clustering for multiple values of k to find optimal number of clusters
    """
    results = {}
    
    print(f"Evaluating clustering for k values: {list(k_range)}")
    
    for k in k_range:
        cluster_labels = apply_kmeans(encoded_data, k)
        eval_results = evaluate_clustering(encoded_data, cluster_labels, true_labels, label_type)
        results[k] = eval_results

    return results

def plot_clustering_metrics(k_results, save_path="clustering_metrics.png"):
    """Plot clustering metrics vs number of clusters"""
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

def plot_cluster_labels_by_family(reduced_dict, kmeans_labels, evaluation_results, save_path="plot_clustered_family.png"):
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
    plt.title(f"2D PCA of Bacteria (k = {k})\nColor = Family, Label = Cluster ID", fontsize=16)

    
    #plt.title("2D PCA of Bacteria\nColor = Family, Label = Cluster ID from High-Dimensional KMeans", fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)

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

def plot_cluster_labels_by_order(reduced_dict, kmeans_labels_order, evaluation_results, save_path="plot_clustered_order.png"):
    # Map from bacterium name to order
    # order_map = taxonomy_df.set_index("Original Name")["Order"].to_dict()
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
    plt.title(f"2D PCA of Bacteria (k = {k})\nColor = Order, Label = Cluster ID", fontsize=16)

    
    # plt.title("2D PCA of Bacteria\nColor = Order, Label = Cluster ID from High-Dimensional KMeans", fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)

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

def plot_family_only(reduced_dict, save_path="plot_family_only.png"):
    # Extract family labels and reduced data
    family_labels = [entry['family'] for entry in reduced_dict.values()]
    reduced_data = np.array([entry['reduced_encoding'] for entry in reduced_dict.values()])

    # Plot plain 2D scatter
    plt.figure(figsize=(14, 12))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], color='steelblue', edgecolor='k', s=80, alpha=0.8)

    # Create label texts and adjust to avoid overlapping
    texts = []
    for i, family in enumerate(family_labels):
        texts.append(
            plt.text(reduced_data[i, 0], reduced_data[i, 1], family, fontsize=9, weight='bold')
        )

    # Use adjust_text to move overlapping labels
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.title("2D PCA Embeddings of Bacteria\nLabel = Taxonomic Family", fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

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

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup device and output directory
    device = setup_device(args.device)
    output_dir = setup_output_directory(args.output_dir)
    
    # Validate input files exist
    required_files = [args.test_data, args.test_labels, args.model_path, args.taxonomy_file]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return
    
    print(f"Loading files:")
    print(f"  Test data: {args.test_data}")
    print(f"  Test labels: {args.test_labels}")
    print(f"  Model: {args.model_path}")
    print(f"  Taxonomy: {args.taxonomy_file}")

    # Load test data, model, and taxonomy table
    test_data = load_test_data(args.test_data)
    test_labels = load_test_labels(args.test_labels)
    model = load_model(args.model_path, device=device)
    taxonomy_df = load_taxonomy(args.taxonomy_file)

    # Encode the test data and include family information
    encoded_dict = encode_data(model, test_data, test_labels, taxonomy_df, device=device)

    # Prepare data for clustering (high-dimensional encodings)
    all_encoded = np.vstack([entry['encoding'] for entry in encoded_dict.values()])

    # Extract true labels for evaluation
    family_labels = [entry['family'] for entry in encoded_dict.values()]
    order_labels = [entry['Order'] for entry in encoded_dict.values()]
    
    # Test multiple k values to find optimal clustering
    k_range = range(args.min_k, args.max_k + 1)
    
    print("\nEvaluating clustering against FAMILY labels")
    family_results = evaluate_multiple_k(all_encoded, family_labels, k_range, "Family")
    
    print("\nEvaluating clustering against ORDER labels")
    order_results = evaluate_multiple_k(all_encoded, order_labels, k_range, "Order")
    
    # Find best k for each taxonomic level (using combination of purity and silhouette)
    best_k_family = max(family_results.keys(), 
                       key=lambda k: family_results[k]['Purity'])
    best_k_order = max(order_results.keys(), 
                      key=lambda k: order_results[k]['Purity'])
    
    print(f"\nBest k for Family clustering: {best_k_family} (Purity: {family_results[best_k_family]['Purity']:.4f})")
    print(f"Best k for Order clustering: {best_k_order} (Purity: {order_results[best_k_order]['Purity']:.4f})")

    # Apply KMeans in original (high-dimensional) space
    kmeans_labels_family = apply_kmeans(all_encoded, num_clusters=best_k_family)
    kmeans_labels_order = apply_kmeans(all_encoded, num_clusters=best_k_order)

    # Evaluate family clustering
    family_eval = evaluate_clustering(all_encoded, kmeans_labels_family, family_labels, "Family")
    print_evaluation_results(family_eval)
    
    # Evaluate order clustering
    order_eval = evaluate_clustering(all_encoded, kmeans_labels_order, order_labels, "Order")
    print_evaluation_results(order_eval)
    
    # Generate plots
    plot_clustering_metrics(family_results, "family_clustering_metrics.png")
    plot_clustering_metrics(order_results, "order_clustering_metrics.png")

    # Perform PCA for 2D visualization
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(all_encoded)

    # Update reduced_dict with 2D encodings
    reduced_dict = {}
    for i, (bacterium_name, entry) in enumerate(encoded_dict.items()):
        reduced_dict[bacterium_name] = {
            "reduced_encoding": reduced_data[i],
            "family": entry["family"],
            "Order": entry["Order"],
            "Class": entry["Class"]
        }

    # Plot the 2D embeddings with output directory paths
    family_plot_path = os.path.join(output_dir, "plot_clustered_family.png")
    order_plot_path = os.path.join(output_dir, "plot_clustered_order.png")
    family_only_path = os.path.join(output_dir, "plot_family_only.png")
    family_dist_path = os.path.join(output_dir, "family_distribution.png")
    
    plot_cluster_labels_by_family(reduced_dict, kmeans_labels_family, family_eval, family_plot_path)
    plot_cluster_labels_by_order(reduced_dict, kmeans_labels_order, order_eval, order_plot_path)
    plot_family_only(reduced_dict, family_only_path)
    plot_family_distribution(reduced_dict, family_dist_path)


if __name__ == "__main__":
    main()