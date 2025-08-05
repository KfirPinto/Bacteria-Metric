# Bacterial Taxonomic Clustering Analysis

## Overview
This tool performs clustering analysis on bacterial embeddings and evaluates how well the clusters align with taxonomic classifications. It supports multiple dimensionality reduction methods and provides various visualization options to assess clustering quality. The tool works with pre-computed embeddings and their corresponding bacterial labels. Optionally, you can supply a variance file to control the dot size in the taxonomy plots, making the visualization reflect per-bacterium uncertainty or drift.

## How to Run

### 1. Prepare the Input Files

Before running the clustering analysis, you need to generate the required input files from your raw data.

#### Raw Data Requirements
You should have the following raw files:
- `embeddings.npy`: Pre-computed bacterial embeddings (numpy array)
- `embeddings_labels.npy`: Numpy file containing bacterial names corresponding to the embeddings (filtered to include only relevant bacteria)


### 2. Install Dependencies
Make sure you have the required packages installed:
```bash
pip install torch numpy scikit-learn matplotlib pandas adjustText umap-learn scipy
```

### 3. Run the Clustering Analysis

**Basic Usage:**
```bash
python taxonomy_evaluation.py
```

**With Custom Parameters:**
```bash
python taxonomy_evaluation.py --embeddings <path-to-embeddings> --embeddings_labels <path-to-labels> --taxonomy_file <path-to-taxonomy-file> --output_dir <output-directory> --reduction_method <method> --taxonomic_levels <levels>
```

**Example with variance file:**
```bash
python taxonomy_evaluation.py --embeddings embeddings.npy --embeddings_labels embeddings_labels.csv --taxonomy_file bacterial_lineage.csv --output_dir ./results --min_k 2 --max_k 20 --reduction_method umap --taxonomic_levels "family,order,class" --variance_file embedding_variances.npy
```

**Example without variance file:**
```bash
python taxonomy_evaluation.py --embeddings embeddings.npy --embeddings_labels embeddings_labels.csv --taxonomy_file bacterial_lineage.csv --output_dir ./results --min_k 2 --max_k 20 --reduction_method umap --taxonomic_levels "family,order,class"
```

### 4. Command-Line Arguments

#### Input Files
| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `--embeddings` | Path to embeddings numpy file (.npy) | `embeddings.npy` | No |
| `--embeddings_labels` | Path to embeddings labels CSV file | `embeddings_labels.csv` | No |
| `--taxonomy_file` | Path to bacterial taxonomy CSV file | `bacterial_lineage.csv` | No |
| `--variance_file` | (Optional) Path to file containing variance per embedding (.npy or .csv) | None | No

#### Analysis Parameters
| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--output_dir` | Directory to save output plots and results | `./plots` | - |
| `--min_k` | Minimum number of clusters to test | `2` | - |
| `--max_k` | Maximum number of clusters to test | `15` | - |
| `--reduction_method` | Dimensionality reduction method for visualization | `pca` | `pca`, `tsne`, `umap`, `pcoa` |
| `--taxonomic_levels` | Comma-separated taxonomic levels to analyze | `family,order,class,phylum` | Any combination of: `family`, `order`, `class`, `phylum` |

### 5. Data Processing

#### Taxonomic Level Processing
For each specified taxonomic level, the script:
1. Filters data to include only the top 15 most frequent taxonomic groups
2. Tests clustering with k values from `min_k` to `max_k`
3. Selects the optimal k based on highest purity score
4. Applies dimensionality reduction for visualization
5. Generates comprehensive plots and statistical analysis

### 6. Output Structure

The script creates separate subdirectories for each taxonomic level within the main output directory:

```
output_dir/
├── Family/
│   ├── family_clustering_metrics.png
│   ├── family_purity_significance.png
│   ├── family_distribution.png
│   ├── plot_family_{reduction_method}.png
│   └── plot_family_cluster_boundaries_{reduction_method}.png
├── Order/
│   ├── order_clustering_metrics.png
│   ├── order_purity_significance.png
│   ├── order_distribution.png
│   ├── plot_order_{reduction_method}.png
│   └── plot_order_cluster_boundaries_{reduction_method}.png
└── [Additional taxonomic levels...]
```

### 7. Generated Output Files

For each taxonomic level, the following files are created:

#### Clustering Performance Analysis
- **`{level}_clustering_metrics.png`**: Purity and silhouette scores vs. number of clusters
- **`{level}_purity_significance.png`**: Statistical significance test results with null distribution histogram

#### Visualization Plots
- **`plot_{level}_{method}.png`**: 2D scatter plot colored by taxonomic groups
- **`plot_{level}_cluster_boundaries_{method}.png`**: Taxonomic groups with cluster boundaries overlay
- **`{level}_distribution.png`**: Bar chart showing frequency distribution of taxonomic groups

### 8. Statistical Analysis

#### Clustering Evaluation Metrics
- **Purity Score**: Measures how "pure" each cluster is with respect to true taxonomic labels
- **Silhouette Score**: Measures how well-separated the clusters are

#### Significance Testing
The tool performs permutation testing (100 iterations) to determine if clustering purity is significantly better than random chance:

| Metric | Description |
|--------|-------------|
| `actual_purity` | Purity score using real taxonomic labels |
| `null_purities` | Distribution of purity scores from randomly shuffled labels |
| `p_value` | Probability that observed purity occurred by chance |
| `z_score` | Standard deviations above null distribution mean |
| `percentile` | Percentile rank of actual purity in null distribution |

## Complete Workflow Summary

1. **Prepare raw data**: Ensure you have `embeddings.npy` and `embeddings_labels.npy`
2. **Generate required files**: Run `process_embedding_labels.py all` to create formatted input files
3. **Install dependencies**: Install required Python packages
4. **Run clustering analysis**: Execute `taxonomy_evaluation.py` with desired parameters
5. **Analyze results**: Review generated plots and statistical analyses in the output directory