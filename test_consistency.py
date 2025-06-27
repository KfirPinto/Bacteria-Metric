import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr, pearsonr
import seaborn as sns
from skbio.stats.ordination import pcoa
import pandas as pd
import argparse
import importlib.util
import sys
sys.path.append(".")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_input_data(input_file):
    # bacteria data is saved as a numpy file with shape (num_samples, num_bacteria, gene_exp_dim)
    input_tensor = np.load(input_file, allow_pickle=True)
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
    return input_tensor

def load_model(model_path, model_class, gene_dim, embedding_dim):
    model = model_class(gene_dim=gene_dim, embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Apply a model to the input bacteria to get the encoded representation of them by that model
def encode_data(model, input_data, model_type="SplitAutoencoder"):
    with torch.no_grad():
        # Apply the encoder part of the model
        if model_type == "SplitAutoencoder":
            x_encoded, _ = model.encoder(input_data.to(device))  # shape: (num_samples, num_bacteria, 2b)
            x_encoded = model.activation(x_encoded)
        elif model_type == "SplitVAE":
            x_encoded, x_reconstruction, mu, logvar = model.forward(input_data.to(device))  # shape: (num_samples, num_bacteria, 2b)

        b = x_encoded.shape[-1] // 2
        encoded_data = x_encoded[:, :, :b]  # first half of embedding
        
        # For each bacterium (i.e., the first dimension of the input)
        bacterium_encodings = []
        for i in range(input_data.shape[1]):  # Iterate through each bacterium from the input
        # Calculate the mean encoding for each bacterium across all samples
            bacterium_encoding = encoded_data[:, i, :].mean(dim=0).cpu().numpy()  
            bacterium_encodings.append(bacterium_encoding) # shape: (embedding_dim,)
        return np.array(bacterium_encodings)

def plot_distribution_separate(distance_matrices, output_dir):
    """
    Plot the distribution of pairwise distances for each model, each on its own plot,
    and arrange them in a grid of subplots.
    """
    num_models = len(distance_matrices)
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))  # 2 rows, 5 columns grid
    axes = axes.flatten()  # Flatten the 2x5 grid to make indexing easier

    # Process each model and plot the distribution of distances on its own subplot
    for model_idx in range(num_models):
        distances = distance_matrices[model_idx].flatten()
        axes[model_idx].hist(distances, bins=100, alpha=0.7, density=True)
        axes[model_idx].set_title(f"Model {model_idx + 1}")
        axes[model_idx].set_xlabel("Distance")
        axes[model_idx].set_ylabel("Density")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot to a file
    output_path = os.path.join(output_dir, "distance_matrix_distributions_separate.png")
    plt.savefig(output_path)
    plt.close()

def calculate_consistency(distance_matrices, normality_decision, output_dir):
    """
    Calculate consistency between distance matrices using either Pearson or Spearman
    based on the normality decision.
    """
    num_models = len(distance_matrices)
    consistency_scores = np.zeros((num_models, num_models))
    
    # Set diagonal elements to 1.0 (perfect correlation with itself)
    np.fill_diagonal(consistency_scores, 1.0)
    
    for i in range(num_models):
        for j in range(i + 1, num_models):
            # Flatten the distance matrices for correlation calculation
            dist_i_flat = distance_matrices[i].flatten()
            dist_j_flat = distance_matrices[j].flatten()
            
            # Calculate the correlation based on the normality decision
            if normality_decision == "normal":
                # Pearson correlation if both are normal
                correlation, _ = pearsonr(dist_i_flat, dist_j_flat)
            else:
                # Spearman correlation if one or both are not normal
                correlation, _ = spearmanr(dist_i_flat, dist_j_flat)
            
            # Store the consistency score (symmetric matrix)
            consistency_scores[i, j] = correlation
            consistency_scores[j, i] = correlation
    
    # Plot consistency scores as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(consistency_scores, annot=True, cmap='coolwarm', fmt=".2f", 
                xticklabels=[f"Model {i+1}" for i in range(num_models)],
                yticklabels=[f"Model {i+1}" for i in range(num_models)])
    plt.title("Consistency Between Distance Matrices of Different Models")
    
    # Save the plot to a file
    output_path = os.path.join(output_dir, "consistency_heatmap.png")
    plt.savefig(output_path)
    plt.close()  

    return consistency_scores

def plot_coordinate_distributions(all_encodings, output_dir):
    """
    Plot the distribution of the average value of each coordinate (dimension) of the encoding
    across all models. Each model will calculate the average of each element in the encodings
    through all the bacteria it encoded.
    """
    num_models = all_encodings.shape[0]  # 10 models
    embedding_dim = all_encodings.shape[2]  # 16 dimensions
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))  # 4 rows, 4 columns grid for 16 subplots
    axes = axes.flatten()  # Flatten the 4x4 grid to make indexing easier

    # Loop through each coordinate (dimension) of the embedding
    for dim in range(embedding_dim):
        avg_values = []

        # Calculate the average of the dim-th coordinate across all bacteria for each model
        for model_idx in range(num_models):
            model_encodings = all_encodings[model_idx]  # Shape: (num_bacteria, embedding_dim)
            avg_value = model_encodings[:, dim].mean()  # Average across all bacteria for the dim-th coordinate
            avg_values.append(avg_value)

        # Plot the distribution of average values for the current coordinate
        axes[dim].hist(avg_values, bins=10, alpha=0.7, label=f"Coordinate {dim + 1}")
        axes[dim].set_title(f"Distribution of Average Value for Coordinate {dim + 1}")
        axes[dim].set_xlabel("Average Value")
        axes[dim].set_ylabel("Frequency")
        axes[dim].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot to a file
    output_path = os.path.join(output_dir, "coordinate_distributions.png")
    plt.savefig(output_path)
    plt.close()  # Close the plot to avoid it being shown

def plot_pcoa_per_model(distance_matrices, output_dir):
    """
    Perform PCoA on each distance matrix and plot the 2D projection.
    Each subplot corresponds to one model's PCoA projection.
    """
    num_models = len(distance_matrices)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))  # 2 rows, 5 columns for 10 models
    axes = axes.flatten()

    for idx, dist_matrix in enumerate(distance_matrices):
        # Convert the distance matrix to a DataFrame with index and columns (needed for skbio)
        dist_df = pd.DataFrame(dist_matrix)
        ordination_result = pcoa(dist_df)
        
        # Get first two principal coordinates
        pc1 = ordination_result.samples.iloc[:, 0]
        pc2 = ordination_result.samples.iloc[:, 1]

        # Scatter plot in 2D
        axes[idx].scatter(pc1, pc2, alpha=0.8)
        axes[idx].set_title(f"Model {idx + 1}")
        axes[idx].set_xlabel("PC1")
        axes[idx].set_ylabel("PC2")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "pcoa_2d_projections.png")
    plt.savefig(output_path)
    plt.close()

def plot_pcoa_per_model_w_names(distance_matrices, bacteria_names, output_dir):
    """
    Perform PCoA on each distance matrix and plot the 2D projection.
    Each subplot corresponds to one model's PCoA projection.
    Labels are added only for outlier bacteria that are far from the main cluster.
    """
    num_models = len(distance_matrices)
    fig, axes = plt.subplots(2, 5, figsize=(25, 12))  # 2 rows, 5 columns for 10 models
    axes = axes.flatten()

    for idx, dist_matrix in enumerate(distance_matrices):
        # Convert the distance matrix to a DataFrame with index and columns (needed for skbio)
        dist_df = pd.DataFrame(dist_matrix)
        ordination_result = pcoa(dist_df)
        
        # Get first two principal coordinates
        pc1 = ordination_result.samples.iloc[:, 0]
        pc2 = ordination_result.samples.iloc[:, 1]
        
        # Calculate distances from centroid to identify outliers
        centroid_x = pc1.mean()
        centroid_y = pc2.mean()
        distances_from_centroid = np.sqrt((pc1 - centroid_x)**2 + (pc2 - centroid_y)**2)
        
        # Get the top 5 most distant bacteria from the centroid
        num_outliers_to_label = 5
        outlier_indices = np.argsort(distances_from_centroid)[-num_outliers_to_label:]

        # Scatter plot in 2D
        axes[idx].scatter(pc1, pc2, alpha=0.8)
        
        # Add labels only for outlier bacteria
        for outlier_idx in outlier_indices:
            # Extract only the species name after "s__"
            full_name = bacteria_names[outlier_idx]
            if "s__" in full_name:
                species_name = full_name.split("s__")[1]
            else:
                species_name = full_name
                
            axes[idx].annotate(species_name, 
                             (pc1.iloc[outlier_idx], pc2.iloc[outlier_idx]),
                             xytext=(3, 3), textcoords='offset points',
                             fontsize=6, alpha=0.9,
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
        
        axes[idx].set_title(f"Model {idx + 1}")
        axes[idx].set_xlabel("PC1")
        axes[idx].set_ylabel("PC2")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "pcoa_2d_projections_w_names.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Analyze bacteria autoencoder models and generate visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python test_consistency.py --input-data data/bacteria.npy --bacteria-names data/names.npy --models-dir models/ --output-dir results/
        python test_consistency.py -i bacteria.npy -n names.npy -m ./models --num-models 5 -o ./output
        """
    )
    
    # Input files
    parser.add_argument("--input-data", "-i", 
                       type=str, 
                       default="input_bacteria.npy",
                       help="Path to input bacteria data file (.npy) (default: input_bacteria.npy)")
    
    parser.add_argument("--bacteria-names", "-n",
                       type=str,
                       default="input_bacteria_names.npy", 
                       help="Path to bacteria names file (.npy) (default: input_bacteria_names.npy)")
    
    # Model paths
    parser.add_argument("--models-dir", "-m",
                       type=str,
                       default="./",
                       help="Directory containing model files (default: current directory)")
    
    parser.add_argument("--model-prefix", 
                       type=str,
                       default="split_autoencoder_",
                       help="Prefix for model filenames (default: split_autoencoder_)")
    
    parser.add_argument("--model-suffix",
                       type=str, 
                       default=".pt",
                       help="Suffix for model filenames (default: .pt)")
    
    parser.add_argument("--num-models",
                       type=int,
                       default=10,
                       help="Number of models to process (default: 10)")
    
    # Output options
    parser.add_argument("--output-dir", "-o",
                       type=str,
                       default="./output",
                       help="Directory to save output files and plots (default: ./output)")
    
    parser.add_argument("--encoded-output",
                       type=str,
                       default="encoded_bacteria.npy",
                       help="Filename for saved encoded bacteria data (default: encoded_bacteria.npy)")
    
    # Analysis options  
    parser.add_argument("--normality", 
                       choices=["normal", "non-normal"],
                       default="non-normal",
                       help="Assumption about data normality for correlation analysis (default: non-normal)")

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


def load_model_class(model_file_path, class_name):
    spec = importlib.util.spec_from_file_location("model_module", model_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading input data from: {args.input_data}")
    print(f"Loading bacteria names from: {args.bacteria_names}")
    print(f"Models directory: {args.models_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of models: {args.num_models}")
    
    # Load input data
    input_data = load_input_data(args.input_data)

    # Load bacteria names
    bacteria_names = np.load(args.bacteria_names, allow_pickle=True)
    print(f"Loaded {len(bacteria_names)} bacteria names")

    # Generate list of model paths
    model_paths = []
    for i in range(args.num_models):
        model_filename = f"{args.model_prefix}{i+1}{args.model_suffix}"
        model_path = os.path.join(args.models_dir, model_filename)
        model_paths.append(model_path)
        
    print(f"Model paths: {model_paths}")

    # To store encodings of bacteria for each model
    all_encodings = []

    # Process each model
    for i, model_path in enumerate(model_paths):
        print(f"Processing model {i+1}/{args.num_models}: {model_path}")
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found, skipping...")
            continue
        ModelClass = load_model_class(args.model_file, args.model_class)
        model = load_model(model_path, ModelClass, gene_dim=input_data.shape[-1], embedding_dim=args.embedding_dim)  
        encodings = encode_data(model, input_data, model_type=args.model_class)
        all_encodings.append(encodings)

    if not all_encodings:
        print("Error: No valid models found!")
        return
    
    # Convert the list to a numpy array: (num_models, num_bacteria, embedding_dim)
    all_encodings = np.array(all_encodings)
    print(f"Encodings shape: {all_encodings.shape}")  

    print("Generating coordinate distributions plot...")
    plot_coordinate_distributions(all_encodings, args.output_dir)

    # Calculate pairwise distance matrices for each model
    print("Calculating pairwise distance matrices...")
    distance_matrices = []
    for model_idx in range(all_encodings.shape[0]):
        distances = pairwise_distances(all_encodings[model_idx])
        distance_matrices.append(distances)

    # Plot the distribution of pairwise distances for each model
    print("Generating distance distribution plots...")
    plot_distribution_separate(distance_matrices, args.output_dir)

    # Calculate and plot consistency between distance matrices
    print("Calculating consistency scores...")
    consistency_scores = calculate_consistency(distance_matrices, args.normality, args.output_dir)
    print(f"Consistency Scores: \n{consistency_scores}")

    print("Generating simple PCoA plots...")
    plot_pcoa_per_model(distance_matrices, args.output_dir)
        
    print("Generating PCoA plots with species names...")
    plot_pcoa_per_model_w_names(distance_matrices, bacteria_names, args.output_dir)

    # Save the encodings to a file for later inspection
    encoded_output_path = os.path.join(args.output_dir, args.encoded_output)
    print(f"Saving encoded bacteria data to: {encoded_output_path}")
    np.save(encoded_output_path, all_encodings)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()