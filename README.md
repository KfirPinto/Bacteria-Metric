# Bacterial Embedding-Pathway Similarity Analysis

## Overview
The script addresses the question: "Do bacteria that are similar in the learned embedding space also have similar metabolic pathway profiles?" 

## Requirements

### Dependencies
```bash
pip install torch numpy matplotlib seaborn pandas scikit-learn scipy scikit-bio
```

### Required Files
- **Embeddings data**: A npy file conatining learnt embeddings (`embeddings.npy`)
- **Embeddings metadata**: Bacterial names for the embeddings (`embeddings_labels.npy`)
- **Pathway data**: Metabolic pathway participation scores (`pathway_data.npy`)
- **Pathway metadata**: Bacterial names for pathway data (`pathway_bacteria.npy`)

## Usage

### Example for run command
```bash
python pathway_evaluation.py --embeddings embeddings.npy --embeddings_labels embeddings_labels.npy --pathway_data pathway_data.npy --pathway_bacteria pathway_bacteria.npy --method_embeddings cosine --method_pathways mahalanobis --normalized_embeddings  --output_dir ./results
```

## Command-Line Arguments

### Input Data (Required)
| Argument | Description |
|----------|-------------|
| `--embeddings` | Path to embeddings tensor (.npy file) |
| `--embeddings_labels` | Path to embeddings metadata - bacteria names (.npy file) |
| `--pathway_data` | Path to pathway abundance tensor (.npy file) |
| `--pathway_bacteria` | Path to pathway abundance metadata - bacteria names (.npy file) |

### Similarity Calculation (Optional)
| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--method_embeddings` | Similarity metric for embeddings | `l2` | `l1`, `l2`, `cosine`, `mahalanobis` |
| `--method_pathways` | Similarity metric for pathways | `l2` | `l1`, `l2`, `cosine`, `mahalanobis` |
| `--normalized_embeddings` | Normalize embedding vectors (flag) | `False` | Flag (include to enable) |

### Output (Required)
| Argument | Description |
|----------|-------------|
| `--output_dir` | Directory to save output plots and results |

## Similarity Metrics

### Available Methods
- **L1 (Manhattan)**: `1 / (1 + manhattan_distance)`
- **L2 (Euclidean)**: `1 / (1 + euclidean_distance)`
- **Cosine**: Cosine similarity between vectors
- **Mahalanobis**: Mahalanobis distance with inverse covariance matrix

All methods return similarity matrices (higher values = more similar).

## Statistical Analysis

### Correlation Measures
1. **Pearson Correlation**: Linear relationship between similarity matrices
2. **Spearman Correlation**: Monotonic (rank-based) relationship
3. **Mantel Test**: Statistical test for correlation between distance matrices
   - Includes p-value from permutation test (999 permutations)
   - Tests significance of matrix correlation

### High Similarity Detection
The script automatically identifies and reports bacteria pairs with:
- Embedding similarity > 0.6, OR
- Pathway similarity > 0.4

## Output Files

### Directory Structure
```
output_dir/
└── {method_embeddings}_{method_pathways}/
    └── {normalized_embeddings}/
        └── scatter_with_identity.png
```


## Plot Interpretation

### Scatter Plot Features
- **X-axis**: Embedding similarity between bacteria pairs
- **Y-axis**: Pathway similarity between bacteria pairs
- **Identity Line**: Perfect correlation (y = x)
- **Annotations**: 
  - Pearson correlation coefficient
  - Spearman correlation coefficient
  - Mantel correlation coefficient
  - Mantel test p-value
