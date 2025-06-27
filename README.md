# Bacteria Autoencoder Consistency Analysis

This script analyzes the consistency of multiple trained autoencoder models by comparing their learned representations of bacterial gene expression data. It generates various visualizations including distance matrix distributions, consistency heatmaps, coordinate distributions, and PCoA projections.

## How to Run

### 1. Prepare the Files

Make sure you have the following files ready:
- `input_bacteria.npy`: The input bacterial gene expression data with shape (num_samples, num_bacteria, gene_exp_dim).
- `input_bacteria_names.npy`: The bacteria names corresponding to the input data.
- Multiple trained model files (e.g., `split_autoencoder_1.pt`, `split_autoencoder_2.pt`, etc.): Pre-trained autoencoder models.
- `model.py`: A Python file containing the model class definition (e.g., SplitAutoencoder, SplitVAE).

You can either place these files in the same directory as the script or specify their paths using command-line arguments.

### 2. Run the Script

To run the script and analyze model consistency, use the following command:

```bash
python test_consistency.py --input-data <path-to-input-data> --bacteria-names <path-to-bacteria-names> --models-dir <path-to-models-directory> --output-dir <output-directory> --model-file <path-to-model-file> --model-class <model-class-name> --embedding-dim <embedding-dimension>
```

**Example:**

```bash
python test_consistency.py --input-data data/bacteria.npy --bacteria-names data/names.npy --models-dir models/ --output-dir results/ --model-file variational_autoencoder/training/model.py --model-class SplitAutoencoder --embedding-dim 32 --num-models 10
```

### 3. Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input-data`, `-i` | Path to input bacteria data file (.npy) | `input_bacteria.npy` |
| `--bacteria-names`, `-n` | Path to bacteria names file (.npy) | `input_bacteria_names.npy` |
| `--models-dir`, `-m` | Directory containing model files | `./` |
| `--model-prefix` | Prefix for model filenames | `split_autoencoder_` |
| `--model-suffix` | Suffix for model filenames | `.pt` |
| `--num-models` | Number of models to process | `10` |
| `--output-dir`, `-o` | Directory to save output files and plots | `./output` |
| `--encoded-output` | Filename for saved encoded bacteria data | `encoded_bacteria.npy` |
| `--normality` | Assumption about data normality for correlation analysis | `non-normal` |
| `--model-file` | Path to the model.py file that defines the model architecture | **Required** |
| `--model-class` | Name of the model class to import from the model file | `SplitAutoencoder` |
| `--embedding-dim` | Dimensionality of the embedding space | `32` |

### 4. Output

After running the script, the following output files will be generated in the specified output directory:

- **Distance Matrix Distributions** (`distance_matrix_distributions_separate.png`): Histograms showing the distribution of pairwise distances for each model.
- **Consistency Heatmap** (`consistency_heatmap.png`): A heatmap showing correlation scores between different models' distance matrices.
- **Coordinate Distributions** (`coordinate_distributions.png`): Distribution plots for each coordinate dimension across all models.
- **PCoA Projections** (`pcoa_2d_projections.png`): 2D Principal Coordinate Analysis plots for each model.
- **PCoA with Species Names** (`pcoa_2d_projections_w_names.png`): Enhanced PCoA plots with species names labeled for outlier bacteria.
- **Encoded Bacteria Data** (`encoded_bacteria.npy`): The encoded representations from all models saved as a numpy array with shape (num_models, num_bacteria, embedding_dim).

### 5. Analysis Results

The script provides several types of analysis:

| Analysis Type | Purpose | Output |
|---------------|---------|--------|
| **Distance Distributions** | Compare how each model spreads bacteria in embedding space | Separate histograms for each model |
| **Model Consistency** | Measure correlation between models' distance matrices | Heatmap with correlation scores |
| **Coordinate Analysis** | Examine how each embedding dimension is utilized | Distribution plots per coordinate |
| **PCoA Visualization** | Visualize bacterial relationships in 2D space | Scatter plots with optional species labels |

### 6. Model Requirements

Your model file must contain a class that implements:
- An `encoder` method (for SplitAutoencoder)
- A `forward` method that returns encoded data (for SplitVAE)
- The model should be compatible with PyTorch's `load_state_dict()` method

The script supports both `SplitAutoencoder` and `SplitVAE` model types, with the encoding extraction handled appropriately for each type.