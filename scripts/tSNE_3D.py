import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
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

def plot_tsne_by_level(df, level, save_dir=None):
    """
    Create and save a 3D t-SNE plot colored by the specified taxonomic level.
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Drop missing values for that level
    df_plot = df.dropna(subset=[level])

    # Generate distinct color palette
    unique_vals = df_plot[level].unique()
    palette = sns.color_palette("hls", len(unique_vals))
    
    # Create color map
    color_map = dict(zip(unique_vals, palette))
    colors = [color_map[val] for val in df_plot[level]]

    ax.scatter(df_plot["x"], df_plot["y"], df_plot["z"], 
              c=colors, s=60, alpha=0.8)

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')
    ax.set_title(f"3D t-SNE colored by {level}", fontsize=14)

    # Create legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color_map[val], markersize=8, label=val)
                      for val in unique_vals]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/tsne_3d_by_{level}.png", dpi=300)
    plt.show()

def main(embedding_path, metadata_path, color_level, save_dir="tsne_plots"):
    print("🔹 Loading embeddings and metadata...")
    embeddings = np.load(embedding_path)
    metadata = np.load(metadata_path, allow_pickle=True)

    print("🔹 Parsing taxonomy strings...")
    taxonomy_df = pd.DataFrame([parse_taxonomy(t) for t in metadata])

    print("🔹 Running t-SNE...")
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(embeddings)

    # Add t-SNE results
    taxonomy_df["x"] = tsne_result[:, 0]
    taxonomy_df["y"] = tsne_result[:, 1]
    taxonomy_df["z"] = tsne_result[:, 2]

    print(f"🔹 Plotting by taxonomy level: {color_level}")
    plot_tsne_by_level(taxonomy_df, level=color_level, save_dir=save_dir)

    print(f"✅ Done! Plot saved to: {save_dir}/tsne_by_{color_level}.png")

# === Edit these paths and parameters before running ===
if __name__ == "__main__":
    embedding_file = "/home/bcrlab/barsapi1/metric/Bacteria-Metric/eval_data/HMP_2012_Stool/Run_0/test_tensor_embeddings.npy"     # path to your embeddings file
    metadata_file = "/home/bcrlab/barsapi1/metric/Bacteria-Metric/eval_data/HMP_2012_Stool/Run_0/bacteria_names_full_taxonomy.npy"  # path to your metadata file (bacteria names with full taxonomy)
    taxonomy_level = "phylum"                   # level to color by (e.g., 'phylum', 'genus', 'species')
    main(embedding_file, metadata_file, taxonomy_level)
