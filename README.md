# Taxonomic Brothers Evaluation

## Overview

This tool evaluates how well bacterial embeddings capture taxonomic relationships by performing nearest-neighbor analysis. It determines whether bacteria that are closest in embedding space also belong to the same taxonomic group (taxonomic "brothers"), providing insights into the quality of learned representations.

## Requirements

```bash
pip install numpy pandas scikit-learn matplotlib seaborn torch
```

## Input Files

The tool requires three input files:

1. **Embeddings file** (`embeddings.npy`): 
   - NumPy array of bacterial embeddings
   - Shape: `(n_bacteria, embedding_dim)`
   - First row is typically excluded during processing

2. **Labels file** (`embeddings_labels.csv`):
   - CSV file with bacterial names corresponding to embeddings
   - First column should contain bacterial names
   - Must match the order of embeddings

3. **Taxonomy file** (`taxonomy.csv`):
   - CSV file with taxonomic information
   - Required columns: `Original Name`, `Family`, `Order`, `Class`, `Phylum`
   - Bacteria with missing `Family` information will be automatically dropped

## Usage

### Custom Parameters

```bash
python taxonomic_evaluation.py --embeddings my_embeddings.npy --embeddings_labels my_labels.csv --taxonomy_file my_taxonomy.csv --taxonomic_level Class --distance cosine --output_dir ./results
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--embeddings` | `embeddings.npy` | Path to embeddings numpy file |
| `--embeddings_labels` | `embeddings_labels.csv` | Path to bacterial labels CSV file |
| `--taxonomy_file` | `taxonomy.csv` | Path to taxonomy data CSV file |
| `--taxonomic_level` | `Class` | Taxonomic level for evaluation (`Order`, `Family`, `Class`, `Phylum`) |
| `--distance` | `cosine` | Distance metric (`cosine`, `euclidean`) |
| `--output_dir` | `./plots` | Directory for output files |

## Output

### Visualizations

1. **`nn_accuracy_per_taxon.png`**: 
   - Bar chart showing nearest-neighbor accuracy for each taxonomic group
   - Includes sample size information
   - Groups with accuracy ≥ 0.5 are highlighted in green

2. **`nn_confusion.png`**: 
   - Confusion matrix showing true vs predicted taxonomic assignments
   - Includes overall accuracy metric

3. **`distance_distribution.png`**: 
   - Histogram and box plot comparing distances between:
     - Same taxonomic group pairs
     - Different taxonomic group pairs

### Data Files

- **`taxonomic_brothers_results.csv`**: Detailed results for all bacterial pairs including:
  - Bacterial names and indices
  - Taxonomic labels
  - Distance/similarity scores
  - Whether pairs are taxonomic brothers

## Example Output Structure

```
results/
├── Class/                          # Taxonomic level subdirectory
│   ├── nn_accuracy_per_taxon.png   # Accuracy by taxonomic group
│   ├── nn_confusion.png            # Confusion matrix
│   ├── distance_distribution.png   # Distance analysis
│   └── taxonomic_brothers_results.csv  # Detailed results
```
