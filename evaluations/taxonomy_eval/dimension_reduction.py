from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def parse_taxonomy(tax_string):
    """Parse full taxonomy string into a dictionary with all ranks."""
    parts = tax_string.split('|')
    levels = {'kingdom': '', 'phylum': '', 'class': '', 'order': '', 'family': '', 'genus': '', 'species': ''}
    for part in parts:
        for level in levels:
            if part.startswith(level[0] + '__'):
                levels[level] = part.split('__')[1]
    return levels

def plot_by_level(df, level, method, save_dir=None):
    """
    Create and save a plot colored by the specified taxonomic level.
    """
    plt.figure(figsize=(10, 8))

    # Drop missing values for that level
    df_plot = df.dropna(subset=[level])

    # Generate distinct color palette
    unique_vals = df_plot[level].unique()
    palette = sns.color_palette("hls", len(unique_vals))

    # Column names depend on method
    if method == "PCA":
        x_col, y_col = "PC1", "PC2"
    elif method == "PCoA":
        x_col, y_col = "PC1", "PC2"
    else:  # t-SNE
        x_col, y_col = "x", "y"

    sns.scatterplot(
        data=df_plot,
        x=x_col, y=y_col,
        hue=level,
        palette=palette,
        s=60,
        linewidth=0,
        alpha=1.0,
        edgecolor=None
    )

    plt.title(f"{method} colored by {level}", fontsize=14)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small', title=level)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{method.lower()}_by_{level}.png", dpi=300)
    plt.show()

def main(embedding_path, metadata_path, color_level, method="PCA", save_dir="plots"):
    print("🔹 Loading embeddings and metadata...")
    embeddings = np.load(embedding_path)
    metadata = np.load(metadata_path, allow_pickle=True)

    print("🔹 Parsing taxonomy strings...")
    taxonomy_df = pd.DataFrame([parse_taxonomy(t) for t in metadata])

    if method == "PCA":
        print("🔹 Running PCA...")
        pca = PCA(n_components=2, random_state=42)
        result = pca.fit_transform(embeddings)
        
        # Add PCA results
        taxonomy_df["PC1"] = result[:, 0]
        taxonomy_df["PC2"] = result[:, 1]
        
        print(f"🔹 PCA explained variance ratio: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")
        
    elif method == "PCoA":
        print("🔹 Computing distance matrix...")
        distance_matrix = pairwise_distances(embeddings, metric='euclidean')
        
        print("🔹 Running PCoA...")
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        result = mds.fit_transform(distance_matrix)
        
        # Add PCoA results
        taxonomy_df["PC1"] = result[:, 0]
        taxonomy_df["PC2"] = result[:, 1]
        
    else:  # t-SNE
        from sklearn.manifold import TSNE
        print("🔹 Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        result = tsne.fit_transform(embeddings)
        
        # Add t-SNE results
        taxonomy_df["x"] = result[:, 0]
        taxonomy_df["y"] = result[:, 1]

    print(f"🔹 Plotting by taxonomy level: {color_level}")
    plot_by_level(taxonomy_df, level=color_level, method=method, save_dir=save_dir)

    print(f"✅ Done! Plot saved to: {save_dir}/{method.lower()}_by_{color_level}.png")

# === Edit these paths and parameters before running ===
if __name__ == "__main__":
    embedding_file = "/home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/eval_results/HMP_Kfir/Run_5/test_tensor_embeddings.npy"
    metadata_file = "/home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/eval_results/HMP_Kfir/Run_5/bacteria_names_full_taxonomy.npy"

    # Run all three methods
    for method in ["PCA", "PCoA", "tSNE"]:
        for level in ["phylum", "class", "order", "family", "genus"]:
            main(embedding_file, metadata_file, level, method=method, 
                 save_dir=f"/home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/eval_results/HMP_Kfir/Run_5/plots/{method.lower()}/{level}")