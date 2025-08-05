import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import sys
import importlib.util
from collections import Counter

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
    full_lineages = np.load(file_path, allow_pickle=True)

    parsed = []
    for lineage in full_lineages:
        parts = lineage.split('|')
        taxonomy = {p[0]: p[3:] for p in parts if '__' in p}  # e.g., {'p': 'Firmicutes'}
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

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate nearest-neighbor taxonomic agreement.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input files
    parser.add_argument("--embeddings", type=str, default="embeddings.npy", help="Path to external embeddings numpy file (.npy).")
    parser.add_argument("--embeddings_labels", type=str, default="embeddings_labels.csv", help="Path to embeddings bacteria labels numpy file (.csv)")

    # Output directory
    parser.add_argument("--output_dir", type=str, default="./plots", help="Directory to save output plots and results")

    # Calculation args
    parser.add_argument('--taxonomic_level', type=str, default='Class', choices=['Order', 'Family', 'Class', 'Phylum'], help="Taxonomic level")
    parser.add_argument('--distance', type=str, default='cosine', choices=['cosine', 'euclidean'], help="Distance metric")

    return parser.parse_args()

def setup_output_directory(output_dir, taxonomic_level):
    """
    Create output directory and a subfolder based on the taxonomic level (label_type).
    """
    taxonomic_level_dir = os.path.join(output_dir, taxonomic_level)  # Create a subfolder based on taxonomic level
    os.makedirs(taxonomic_level_dir, exist_ok=True)  # Create subfolder if it doesn't exist
    return taxonomic_level_dir

def organize_embeddings(embeddings_data, embeddings_labels, taxonomy_df):
    """
    Process embeddings and organize them with taxonomy information
    """    
    embeddings_data = embeddings_data[1:]  # Exclude the first row

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

        encoded_dict[bacterium_name] = {
            "encoding": bacterium_embedding,
            "family": tax_info.get("Family"),
            "Order": tax_info.get("Order"),
            "Class": tax_info.get("Class"),
            "Phylum": tax_info.get("Phylum") 
        }

        if i < 5:  # Print first 5 for verification
            print(f"Bacterium {i}: {bacterium_name} -> Family={tax_info.get('Family')}, "
                    f"Order={tax_info.get('Order')}, Class={tax_info.get('Class')}, Phylum={tax_info.get('Phylum')}")

    return encoded_dict

def evaluate_taxonomic_brothers(embeddings_dict, distance_metric='cosine', taxonomic_level='Family'):
    n_bacteria = len(embeddings_dict)
    
    # Extract embeddings from the embeddings_dict
    embeddings = np.array([embeddings_dict[name]["encoding"] for name in embeddings_dict])
    # Get bacterium names in the same order
    bacterium_names = list(embeddings_dict.keys())
    # Get taxonomic labels based on the specified level 
    taxonomic_labels = np.array([embeddings_dict[name][taxonomic_level] for name in bacterium_names])

    # Compute pairwise distances or similarities
    if distance_metric == 'cosine':
        sims = cosine_similarity(embeddings)
        # to exclude self-matches
        np.fill_diagonal(sims, -np.inf)
        nearest_indices = np.argmax(sims, axis=1) 
        # array that stores the index of the nearest neighbor for each bacterium based on the maximum similarity.
    else:
        dists = euclidean_distances(embeddings)
        # to exclude self-matches
        np.fill_diagonal(dists, np.inf)
        nearest_indices = np.argmin(dists, axis=1)
        # array that stores the index of the nearest neighbor for each bacterium based on the minimum distance.

    pairs = []
    pair_results = []
    for i in range(n_bacteria):
        j = nearest_indices[i] # j is the nearest neighbor index for bacterium i
        # Check if they are taxonomic brothers (same taxonomic label)
        are_brothers = taxonomic_labels[i] == taxonomic_labels[j]

        # Only include each unique pair once (avoid counting both i->j and j->i)
        pair_key = tuple(sorted([i, j]))
        if pair_key not in pairs:
            pairs.append(pair_key)
            pair_results.append({
                'bacterium_1': i,
                'bacterium_2': j,
                'bacterium_1_name': bacterium_names[i],
                'bacterium_2_name': bacterium_names[j],
                'label_1': taxonomic_labels[i],
                'label_2': taxonomic_labels[j],
                'are_brothers': are_brothers,
                'distance': sims[i, j] if distance_metric == 'cosine' else dists[i, j]
            })

    # Calculate accuracy
    correct_pairs = sum(1 for result in pair_results if result['are_brothers'])
    total_pairs = len(pair_results)
    accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct_pairs': correct_pairs,
        'total_pairs': total_pairs,
        'pair_results': pair_results,
        'nearest_indices': nearest_indices,
        'bacterium_names': bacterium_names,
        'taxonomic_labels': taxonomic_labels
    }

def plot_nn_accuracy_per_taxon(bacterium_names, nearest_indices, taxonomic_labels, taxonomic_level, save_path=None):
    """
    Plot nearest neighbor accuracy for each taxonomic group
    """
    # Calculate per-taxon accuracy
    taxon_stats = {}
    
    for i, bacterium in enumerate(bacterium_names):
        true_label = taxonomic_labels[i]
        nn_label = taxonomic_labels[nearest_indices[i]]
        
        if true_label not in taxon_stats:
            taxon_stats[true_label] = {'correct': 0, 'total': 0}
        
        taxon_stats[true_label]['total'] += 1
        if true_label == nn_label:
            taxon_stats[true_label]['correct'] += 1
    
    # Calculate accuracy for each taxon
    taxon_accuracies = {taxon: stats['correct'] / stats['total'] 
                       for taxon, stats in taxon_stats.items()}
    
    # Sort by accuracy for better visualization
    sorted_taxa = sorted(taxon_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Accuracy per taxon
    taxa_names = [item[0] for item in sorted_taxa]
    accuracies = [item[1] for item in sorted_taxa]
    sample_counts = [taxon_stats[taxon]['total'] for taxon in taxa_names]
    
    bars = ax1.bar(range(len(taxa_names)), accuracies, 
                   color=['green' if acc >= 0.5 else 'red' for acc in accuracies])
    ax1.set_xlabel(f'{taxonomic_level}')
    ax1.set_ylabel('Nearest Neighbor Accuracy')
    ax1.set_title(f'Nearest Neighbor Accuracy by {taxonomic_level}')
    ax1.set_xticks(range(len(taxa_names)))
    ax1.set_xticklabels(taxa_names, rotation=45, ha='right')
    ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Add sample count annotations
    for i, (bar, count) in enumerate(zip(bars, sample_counts)):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'n={count}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Sample size distribution
    ax2.bar(range(len(taxa_names)), sample_counts, alpha=0.7, color='steelblue')
    ax2.set_xlabel(f'{taxonomic_level}')
    ax2.set_ylabel('Number of Bacteria')
    ax2.set_title(f'Sample Size Distribution by {taxonomic_level}')
    ax2.set_xticks(range(len(taxa_names)))
    ax2.set_xticklabels(taxa_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy per taxon plot saved to: {save_path}")
    
    plt.close()
    
    return taxon_accuracies

def plot_nn_confusion(true_labels, predicted_labels, taxonomic_level, save_path=None):
    """
    Plot confusion matrix for nearest neighbor predictions
    """
    # Get unique labels
    unique_labels = sorted(list(set(true_labels) | set(predicted_labels)))
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)
    
    # Calculate accuracy and other metrics
    accuracy = np.trace(cm) / np.sum(cm)
    
    plt.figure(figsize=(max(8, len(unique_labels) * 0.5), max(6, len(unique_labels) * 0.4)))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_labels, yticklabels=unique_labels,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Nearest Neighbor Confusion Matrix ({taxonomic_level})\nOverall Accuracy: {accuracy:.3f}')
    plt.xlabel(f'Predicted {taxonomic_level}')
    plt.ylabel(f'True {taxonomic_level}')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.close()

def plot_distance_distribution(pair_results, distance_metric, save_path=None):
    """
    Plot distribution of distances for matching vs non-matching pairs
    """
    matching_distances = [r['distance'] for r in pair_results if r['are_brothers']]
    non_matching_distances = [r['distance'] for r in pair_results if not r['are_brothers']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Histogram comparison
    ax1.hist(matching_distances, bins=30, alpha=0.7, label='Same taxonomic group', 
             color='green', density=True)
    ax1.hist(non_matching_distances, bins=30, alpha=0.7, label='Different taxonomic group', 
             color='red', density=True)
    ax1.set_xlabel(f'{distance_metric.capitalize()} Distance/Similarity')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Distance Distribution: Matching vs Non-matching Pairs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot comparison
    data_to_plot = [matching_distances, non_matching_distances]
    labels = ['Same Group', 'Different Group']
    colors = ['green', 'red']
    
    bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel(f'{distance_metric.capitalize()} Distance/Similarity')
    ax2.set_title('Distance Distribution Box Plot')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    if matching_distances and non_matching_distances:
        match_mean = np.mean(matching_distances)
        non_match_mean = np.mean(non_matching_distances)
        ax1.axvline(match_mean, color='green', linestyle='--', alpha=0.8, 
                   label=f'Same group mean: {match_mean:.3f}')
        ax1.axvline(non_match_mean, color='red', linestyle='--', alpha=0.8,
                   label=f'Different group mean: {non_match_mean:.3f}')
        ax1.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distance distribution plot saved to: {save_path}")
    
    plt.close()

def main():
    args = parse_arguments()
    output_dir = setup_output_directory(args.output_dir, args.taxonomic_level)

    # Load embeddings, labels, taxonomic data
    embeddings = load_embeddings(args.embeddings)
    labels, taxonomy_df = load_embeddings_labels(args.embeddings_labels)

    # Organize embeddings with taxonomy information
    encoded_dict = organize_embeddings(embeddings, labels, taxonomy_df)

    # Evaluate taxonomic brothers (nearest neighbors)
    results = evaluate_taxonomic_brothers(
        encoded_dict,  
        distance_metric=args.distance,
        taxonomic_level=args.taxonomic_level  
    )

    # Print summary results
    print(f"\nTaxonomic Brothers Evaluation @ {args.taxonomic_level}:")
    print(f"  Distance metric: {args.distance}")
    print(f"  Total bacteria: {len(results['bacterium_names'])}")  # Fixed: use correct variable
    print(f"  Total pairs evaluated: {results['total_pairs']}")
    print(f"  Correct taxonomic brother pairs: {results['correct_pairs']}")
    print(f"  Accuracy: {results['accuracy']:.4f}")

    # Print example pairs for review
    print(f"\nExample pairs:")
    for i, pair_result in enumerate(results['pair_results'][:5]):
        name1 = pair_result['bacterium_1_name'] 
        name2 = pair_result['bacterium_2_name']  
        status = "✓" if pair_result['are_brothers'] else "✗"
        print(f"  {status} {name1} ({pair_result['label_1']}) <-> {name2} ({pair_result['label_2']})")

    # Create and save visualizations
    plot_nn_accuracy_per_taxon(
        results['bacterium_names'], 
        results['nearest_indices'], 
        results['taxonomic_labels'],  
        args.taxonomic_level,
        save_path=os.path.join(output_dir, "nn_accuracy_per_taxon.png")
    )

    # Get predicted labels for confusion matrix
    predicted_labels = [results['taxonomic_labels'][idx] for idx in results['nearest_indices']]
    
    plot_nn_confusion(
        results['taxonomic_labels'], 
        predicted_labels,  
        args.taxonomic_level,
        save_path=os.path.join(output_dir, "nn_confusion.png")
    )

    plot_distance_distribution(
        results['pair_results'],
        args.distance,
        save_path=os.path.join(output_dir, "distance_distribution.png")
    )
    
    # Save detailed results to CSV
    results_df = pd.DataFrame(results['pair_results'])
    results_df.to_csv(os.path.join(output_dir, "taxonomic_brothers_results.csv"), index=False)
    
    print(f"\nDetailed results saved to: {os.path.join(output_dir, 'taxonomic_brothers_results.csv')}")

if __name__ == "__main__":
    main()