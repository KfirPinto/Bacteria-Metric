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

sys.path.append(".")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, model_class, gene_dim, embedding_dim):
    model = model_class(gene_dim=gene_dim, embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def encode_data(model, input_data, model_type):
    with torch.no_grad():
        # Apply the encoder part of the model
        if model_type == "SplitAutoencoder":
            x_encoded, _ = model.encoder(input_data.to(device))  # shape: (num_samples, num_bacteria, 2b)
            x_encoded = model.activation(x_encoded)
        elif model_type == "SplitVAE":
            x_encoded, x_reconstruction, mu, logvar = model.forward(input_data.to(device))  # shape: (num_samples, num_bacteria, 2b)

        b = x_encoded.shape[-1] // 2
        encoded_data = x_encoded[:, :, :b]  # first half of embedding
        return encoded_data.mean(dim=0).cpu().numpy()

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate nearest-neighbor taxonomic agreement.")
    parser.add_argument('--test_data', type=str, required=True, help="Path to .npy test tensor")
    parser.add_argument('--labels_file', type=str, required=True, help="Path to .npy file with bacterium names")
    parser.add_argument('--taxonomy_csv', type=str, required=True, help="Path to taxonomy CSV file")
    parser.add_argument('--model_file', type=str, required=True, help="Path to model .py file")
    parser.add_argument('--model_class', type=str, default='SplitAutoencoder', help="Model class name")
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained model .pt")
    parser.add_argument('--embedding_dim', type=int, default=32, help="Embedding dimension")
    parser.add_argument('--label_type', type=str, default='Order', choices=['Order', 'Family', 'Class'], help="Taxonomic level")
    parser.add_argument('--distance', type=str, default='cosine', choices=['cosine', 'euclidean'], help="Distance metric")
    parser.add_argument('--output_dir', type=str, default='.', help="Directory to save plots")
    parser.add_argument('--avoid_duplicates', action='store_true', help="Avoid duplicate pairs in evaluation")
    return parser.parse_args()

# Load model class from file
def load_model_class(model_file_path, class_name):
    spec = importlib.util.spec_from_file_location("model_module", model_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

def evaluate_taxonomic_brothers(embeddings, labels, distance_metric='cosine', avoid_duplicates=False):
    """
    Evaluate if nearest neighbors are 'taxonomic brothers' (same taxonomic label).
    
    Args:
        embeddings: numpy array of shape (n_bacteria, embedding_dim)
        labels: numpy array of taxonomic labels
        distance_metric: 'cosine' or 'euclidean'
        avoid_duplicates: if True, only evaluate each unique pair once
    
    Returns:
        results dict with accuracy, correct pairs, total pairs, and detailed results
    """
    n_bacteria = len(embeddings)
    
    # Compute pairwise distances or similarities
    if distance_metric == 'cosine':
        sims = cosine_similarity(embeddings)
        # Set diagonal to -inf to exclude self-matches
        np.fill_diagonal(sims, -np.inf)
        nearest_indices = np.argmax(sims, axis=1)
    else:
        dists = euclidean_distances(embeddings)
        # Set diagonal to inf to exclude self-matches
        np.fill_diagonal(dists, np.inf)
        nearest_indices = np.argmin(dists, axis=1)
    
    # Collect all bacterium-nearest_neighbor pairs
    pairs = []
    pair_results = []
    
    for i in range(n_bacteria):
        j = nearest_indices[i]
        
        # Check if they are taxonomic brothers (same taxonomic label)
        are_brothers = labels[i] == labels[j]
        
        if avoid_duplicates:
            # Only include each unique pair once (avoid counting both i->j and j->i)
            pair_key = tuple(sorted([i, j]))
            if pair_key not in pairs:
                pairs.append(pair_key)
                pair_results.append({
                    'bacterium_1': i,
                    'bacterium_2': j,
                    'label_1': labels[i],
                    'label_2': labels[j],
                    'are_brothers': are_brothers,
                    'distance': sims[i, j] if distance_metric == 'cosine' else dists[i, j]
                })
        else:
            # Include all directed pairs
            pairs.append((i, j))
            pair_results.append({
                'bacterium_1': i,
                'bacterium_2': j,
                'label_1': labels[i],
                'label_2': labels[j],
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
        'nearest_indices': nearest_indices
    }

def plot_nn_accuracy_per_taxon(labels, nearest_indices, label_type='Order', save_path='nn_accuracy_per_taxon.png'):
    label_df = pd.DataFrame({
        'True': labels,
        'NN': labels[nearest_indices]
    })
    label_df['Correct'] = label_df['True'] == label_df['NN']
    accuracy_per_taxon = label_df.groupby('True')['Correct'].mean().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=accuracy_per_taxon.index, y=accuracy_per_taxon.values, palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('NN Match Accuracy')
    plt.title(f'Nearest Neighbor Match Accuracy per {label_type}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_nn_confusion(labels, nearest_labels, label_type='Order', save_path='nn_confusion.png'):
    unique_labels = sorted(list(set(labels)))
    cm = confusion_matrix(labels, nearest_labels, labels=unique_labels)
    cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, xticklabels=unique_labels, yticklabels=unique_labels,
                annot=True, fmt=".2f", cmap="Blues")
    plt.xlabel("Nearest Neighbor's Taxon")
    plt.ylabel("Bacterium's True Taxon")
    plt.title(f'Nearest Neighbor Confusion Matrix ({label_type})')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_distance_distribution(pair_results, distance_metric, save_path='distance_distribution.png'):
    """Plot distribution of distances for correct vs incorrect pairs"""
    correct_distances = [r['distance'] for r in pair_results if r['are_brothers']]
    incorrect_distances = [r['distance'] for r in pair_results if not r['are_brothers']]
    
    plt.figure(figsize=(10, 6))
    plt.hist(correct_distances, alpha=0.7, label='Taxonomic Brothers', bins=30, color='green')
    plt.hist(incorrect_distances, alpha=0.7, label='Different Taxa', bins=30, color='red')
    plt.xlabel(f'{distance_metric.capitalize()} Distance/Similarity')
    plt.ylabel('Frequency')
    plt.title('Distance Distribution: Taxonomic Brothers vs Different Taxa')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data and labels
    test_tensor = torch.tensor(np.load(args.test_data, allow_pickle=True), dtype=torch.float32).to(device)
    bacteria_names = np.load(args.labels_file, allow_pickle=True)
    taxonomy_df = pd.read_csv(args.taxonomy_csv)

    # Build name → taxon map
    taxonomy_map = taxonomy_df.set_index("Original Name")[args.label_type].to_dict()

    # Load the model
    ModelClass = load_model_class(args.model_file, args.model_class)
    model = load_model(args.model_path, ModelClass, gene_dim=test_tensor.shape[-1], embedding_dim=args.embedding_dim)
    
    # Encode the data
    embeddings = encode_data(model, test_tensor, args.model_class)

    # Filter by known taxonomy
    valid_embeddings, valid_labels, valid_names = [], [], []
    for i, name in enumerate(bacteria_names):
        taxon = taxonomy_map.get(name)
        if isinstance(taxon, str) and taxon.strip() != '':
            valid_embeddings.append(embeddings[i])
            valid_labels.append(taxon)
            valid_names.append(name)
            print(f"Encoded {name} -> {taxon}")

    embeddings = np.stack(valid_embeddings)
    labels = np.array(valid_labels)
    names = np.array(valid_names)

    # Evaluate taxonomic brothers
    results = evaluate_taxonomic_brothers(
        embeddings, 
        labels, 
        distance_metric=args.distance,
        avoid_duplicates=args.avoid_duplicates
    )

    # Print results
    print(f"\nTaxonomic Brothers Evaluation @ {args.label_type}:")
    print(f"  Distance metric: {args.distance}")
    print(f"  Avoid duplicates: {args.avoid_duplicates}")
    print(f"  Total bacteria: {len(labels)}")
    print(f"  Total pairs evaluated: {results['total_pairs']}")
    print(f"  Correct taxonomic brother pairs: {results['correct_pairs']}")
    print(f"  Accuracy: {results['accuracy']:.4f}")

    # Print some example pairs
    print(f"\nExample pairs:")
    for i, pair_result in enumerate(results['pair_results'][:5]):
        name1 = names[pair_result['bacterium_1']]
        name2 = names[pair_result['bacterium_2']]
        status = "✓" if pair_result['are_brothers'] else "✗"
        print(f"  {status} {name1} ({pair_result['label_1']}) <-> {name2} ({pair_result['label_2']})")

    # Create visualizations
    plot_nn_accuracy_per_taxon(
        labels, 
        results['nearest_indices'], 
        args.label_type,
        save_path=os.path.join(args.output_dir, "nn_accuracy_per_taxon.png")
    )

    plot_nn_confusion(
        labels, 
        labels[results['nearest_indices']], 
        args.label_type,
        save_path=os.path.join(args.output_dir, "nn_confusion.png")
    )

    plot_distance_distribution(
        results['pair_results'],
        args.distance,
        save_path=os.path.join(args.output_dir, "distance_distribution.png")
    )

    # Save detailed results to CSV
    results_df = pd.DataFrame(results['pair_results'])
    results_df['bacterium_1_name'] = names[results_df['bacterium_1']]
    results_df['bacterium_2_name'] = names[results_df['bacterium_2']]
    results_df.to_csv(os.path.join(args.output_dir, "taxonomic_brothers_results.csv"), index=False)
    
    print(f"\nDetailed results saved to: {os.path.join(args.output_dir, 'taxonomic_brothers_results.csv')}")

if __name__ == "__main__":
    main()