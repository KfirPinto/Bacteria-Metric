# Bacterial Taxonomic Clustering Analysis

## Overview
This tool performs clustering analysis on bacterial embeddings and evaluates how well the clusters align with taxonomic classifications. It supports multiple dimensionality reduction methods and provides various visualization options to assess clustering quality. The tool can work with either model-based embeddings or external pre-computed embeddings.

## How to Run

### 1. Prepare the Files

#### For Model-Based Analysis
Make sure you have the following files ready:
- `test_tensor.npy`: The test data containing bacterial gene expression data
- `test_bacteria.npy`: The test labels, representing the bacteria names corresponding to the data
- `split_autoencoder.pt`: A pre-trained model to load the encoder
- `bacterial_lineage_formatted.csv`: A CSV file containing the taxonomy information for bacteria (columns: Original Name, Family, Order, Class). This is a file that is gotten as an output from running the script `create_lineage.py`
- `model.py`: Python file containing the model architecture definition

#### For External Embeddings Analysis
Make sure you have the following files ready:
- `embeddings.npy`: Pre-computed bacterial embeddings
- `projection_matrix.npy` (optional): Projection matrix to apply to embeddings
- `test_bacteria.npy`: The bacteria names corresponding to the embeddings
- `bacterial_lineage_formatted.csv`: A CSV file containing the taxonomy information

You can either place these files in the same directory as the script or specify their paths using command-line arguments.

### 2. Install Dependencies
Make sure you have the required packages installed:
```bash
pip install torch numpy scikit-learn matplotlib pandas adjustText umap-learn scipy
```

### 3. Run the Script

#### Model-Based Analysis
To run the script using a trained model to generate embeddings:
```bash
python taxonomy_evaluation.py --test_data <path-to-test-data> --test_labels <path-to-test-labels> --model_path <path-to-model> --taxonomy_file <path-to-taxonomy-file> --output_dir <output-directory> --model-file <path-to-model-file> --model-class <model-class-name> --embedding-dim <embedding-dimension> --reduction_method <method>
```

**Example:**
```bash
python taxonomy_evaluation.py --test_data test_tensor.npy --test_labels test_bacteria.npy --model_path split_autoencoder.pt --taxonomy_file bacterial_lineage_formatted.csv --output_dir ./plots --model-file variational_autoencoder/training/model.py --model-class SplitVAE --embedding-dim 32 --min_k 2 --max_k 20 --reduction_method umap
```

#### External Embeddings Analysis
To run the script using pre-computed embeddings:
```bash
python taxonomy_evaluation.py --external_embeddings <path-to-embeddings> --test_labels <path-to-test-labels> --taxonomy_file <path-to-taxonomy-file> --output_dir <output-directory> --reduction_method <method>
```

**Example:**
```bash
python taxonomy_evaluation.py --external_embeddings bacterial_embeddings.npy --projection_matrix projection_matrix.npy --test_labels test_bacteria.npy --taxonomy_file bacterial_lineage_formatted.csv --output_dir ./plots --min_k 2 --max_k 20 --reduction_method pca
```

### 4. Command-Line Arguments

#### Input Files
| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `--test_data` | Path to test data numpy file (.npy) | `test_tensor.npy` | For model-based analysis |
| `--test_labels` | Path to test bacteria labels numpy file (.npy) | `test_bacteria.npy` | Yes |
| `--model_path` | Path to trained model file (.pt) | `split_autoencoder.pt` | For model-based analysis |
| `--taxonomy_file` | Path to bacterial taxonomy CSV file | `bacterial_lineage.csv` | Yes |
| `--external_embeddings` | Path to external embeddings numpy file (.npy) | `None` | For external embeddings analysis |
| `--projection_matrix` | Path to projection matrix numpy file (.npy) | `None` | Optional with external embeddings |

#### Model Configuration (for model-based analysis)
| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `--model-file` | Path to model.py file defining model architecture | - | For model-based analysis |
| `--model-class` | Name of model class to import from model file | `SplitAutoencoder` | No |
| `--embedding-dim` | Dimensionality of the embedding space | `32` | No |

#### Analysis Parameters
| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--output_dir` | Directory to save output plots and results | `./plots` | - |
| `--min_k` | Minimum number of clusters to test | `2` | - |
| `--max_k` | Maximum number of clusters to test | `15` | - |
| `--reduction_method` | Dimensionality reduction method for visualization | `pca` | `pca`, `tsne`, `umap`, `pcoa` |

### 5. Analysis Modes

#### Model-Based Analysis
- Uses a trained neural network model to encode bacterial gene expression data
- Requires test data, model file, and model architecture definition
- Generates embeddings by passing data through the model's encoder
- Supports both `SplitAutoencoder` and `SplitVAE` model types

#### External Embeddings Analysis
- Uses pre-computed embeddings from any source
- Optionally applies a projection matrix to transform embeddings
- Useful for analyzing embeddings from different models or methods
- Requires only the embeddings file and taxonomy information

### 6. Dimensionality Reduction Methods

The tool supports four different dimensionality reduction methods for 2D visualization:

- **PCA** (`pca`): Principal Component Analysis - Linear method, preserves global structure
- **t-SNE** (`tsne`): t-Distributed Stochastic Neighbor Embedding - Non-linear, good for local structure  
- **UMAP** (`umap`): Uniform Manifold Approximation and Projection - Preserves both local and global structure
- **PCoA** (`pcoa`): Principal Coordinate Analysis - Distance-based method

### 7. Output Files

After running the script, the following output files will be generated in the specified output directory:

#### Clustering Performance Analysis
- **`family_clustering_metrics.png`**: Purity and silhouette scores vs. number of clusters for family-level clustering
- **`order_clustering_metrics.png`**: Purity and silhouette scores vs. number of clusters for order-level clustering
- **`family_purity_significance.png`**: Statistical significance test for family clustering purity
- **`order_purity_significance.png`**: Statistical significance test for order clustering purity

#### Visualization Plots (with selected dimensionality reduction method)
- **`plot_clustered_family_{method}.png`**: 2D visualization with family colors and cluster number labels
- **`plot_clustered_order_{method}.png`**: 2D visualization with order colors and cluster number labels
- **`plot_family_cluster_boundaries_{method}.png`**: Family-colored dots with light grey cluster boundaries
- **`plot_order_cluster_boundaries_{method}.png`**: Order-colored dots with light grey cluster boundaries

#### Distribution Analysis
- **`family_distribution.png`**: Bar plot showing the distribution of bacteria across different families

### 8. Statistical Analysis

The tool performs statistical significance testing for clustering purity:

| Metric          | Based On        | Purpose                                       |
| --------------- | --------------- | --------------------------------------------- |
| `actual_purity` | real labels     | What purity do I get from my actual taxonomy? |
| `null_purities` | shuffled labels | What purity do I get just by random chance?   |

The significance test uses permutation testing with 100 random shuffles to determine if the observed clustering purity is significantly better than random chance.

