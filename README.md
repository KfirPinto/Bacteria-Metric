# Bacterial Taxonomic Clustering Analysis

## Overview
This tool performs clustering analysis on bacterial embeddings and evaluates how well the clusters align with taxonomic classifications. It supports multiple dimensionality reduction methods and provides various visualization options to assess clustering quality.

## How to Run

### 1. Prepare the Files
Make sure you have the following files ready:
- `test_tensor.npy`: The test data containing bacterial embeddings.
- `test_bacteria.npy`: The test labels, representing the bacteria names corresponding to the embeddings.
- `split_autoencoder.pt`: A pre-trained model to load the encoder.
- `bacterial_lineage_formatted.csv`: A CSV file containing the taxonomy information for bacteria (columns: Original Name, Family, Order, Class).

You can either place these files in the same directory as the script or specify their paths using command-line arguments.

### 2. Install Dependencies
Make sure you have the required packages installed:
```bash
pip install torch numpy scikit-learn matplotlib pandas adjustText umap-learn
```

### 3. Run the Script
To run the script and evaluate clustering performance, use the following command:
```bash
python script_name.py --test_data <path-to-test-data> --test_labels <path-to-test-labels> --model_path <path-to-model> --taxonomy_file <path-to-taxonomy-file> --output_dir <output-directory> --model-file <path-to-model-file> --model-class <model-class-name> --embedding-dim <embedding-dimension> --reduction_method <method>
```

**Example:**
```bash
python taxonomy_evaluation.py --test_data test_tensor.npy --test_labels test_bacteria.npy --model_path split_autoencoder.pt --taxonomy_file bacteria_lineage.csv --output_dir ./plots --model-file variational_autoencoder/training/model.py --model-class SplitVAE --embedding-dim 32 --min_k 2 --max_k 20 --reduction_method umap
```

### 4. Command-Line Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--test_data` | Path to the test data numpy file (.npy) | `test_tensor.npy` | - |
| `--test_labels` | Path to the test bacteria labels numpy file (.npy) | `test_bacteria.npy` | - |
| `--model_path` | Path to the pre-trained model (.pt) | `split_autoencoder.pt` | - |
| `--taxonomy_file` | Path to the bacterial taxonomy CSV file | `bacterial_lineage_formatted.csv` | - |
| `--output_dir` | Directory to save output plots and results | `./plots` | - |
| `--model-file` | Path to the model file that defines the model architecture | Required | - |
| `--model-class` | Name of the model class inside the model file | `SplitAutoencoder` | - |
| `--embedding-dim` | Dimensionality of the embedding space | `32` | - |
| `--min_k` | Minimum number of clusters to test | `2` | - |
| `--max_k` | Maximum number of clusters to test | `15` | - |
| `--reduction_method` | Dimensionality reduction method for visualization | `pca` | `pca`, `tsne`, `umap`, `pcoa` |

### 5. Dimensionality Reduction Methods

The tool supports four different dimensionality reduction methods for 2D visualization:

- **PCA** (`pca`): Principal Component Analysis - Linear method, preserves global structure
- **t-SNE** (`tsne`): t-Distributed Stochastic Neighbor Embedding - Non-linear, good for local structure
- **UMAP** (`umap`): Uniform Manifold Approximation and Projection - Preserves both local and global structure
- **PCoA** (`pcoa`): Principal Coordinate Analysis - Distance-based method

### 6. Output Files

After running the script, the following output files will be generated in the specified output directory:

#### Clustering Performance Analysis
- **`family_clustering_metrics.png`**: Purity and silhouette scores vs. number of clusters for family-level clustering
- **`order_clustering_metrics.png`**: Purity and silhouette scores vs. number of clusters for order-level clustering
- **`family_purity_significance.png`**: Statistical significance test for family clustering purity
- **`order_purity_significance.png`**: Statistical significance test for order clustering purity

#### Visualization Plots (with selected dimensionality reduction method)
- **`plot_clustered_family_{method}.png`**: 2D visualization with family colors and cluster number labels
- **`plot_clustered_order_{method}.png`**: 2D visualization with order colors and cluster number labels
- **`plot_family_cluster_boundaries_{method}.png`**: - Family-colored dots with light grey cluster boundaries
- **`plot_order_cluster_boundaries_{method}.png`**: - Order-colored dots with light grey cluster boundaries

#### Distribution Analysis
- **`family_distribution.png`**: Bar plot showing the distribution of bacteria across different families

### 8. Statistical Analysis

The tool performs statistical significance testing for clustering purity:

| Metric          | Based On        | Purpose                                       |
| --------------- | --------------- | --------------------------------------------- |
| `actual_purity` | real labels     | What purity do I get from my actual taxonomy? |
| `null_purities` | shuffled labels | What purity do I get just by random chance?   |

The significance test uses permutation testing with 100 random shuffles to determine if the observed clustering purity is significantly better than random chance.

