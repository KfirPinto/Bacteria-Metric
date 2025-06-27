# Bacterial Taxonomic Clustering Analysis

## How to Run

### 1. Prepare the Files

Make sure you have the following files ready:

- `test_tensor.npy`: The test data containing bacterial embeddings.
- `test_bacteria.npy`: The test labels, representing the bacteria names corresponding to the embeddings.
- `split_autoencoder.pt`: A pre-trained model to load the encoder.
- `bacterial_lineage_formatted.csv`: A CSV file containing the taxonomy information for bacteria (columns: Original Name, Family, Order, Class).

You can either place these files in the same directory as the script or specify their paths using command-line arguments.

### 3. Run the Script

To run the script and evaluate clustering performance, use the following command:

```bash
python script_name.py --test_data <path-to-test-data> --test_labels <path-to-test-labels> --model_path <path-to-model> --taxonomy_file <path-to-taxonomy-file> --output_dir <output-directory> --model-file <path-to-model-file> --model-class <model-class-name> --embedding-dim <embedding-dimension>
```

**Example:**

```bash
python taxonomy_evaluation.py --test_data test_tensor.npy --test_labels test_bacteria.npy --model_path split_autoencoder.pt --taxonomy_file bacteria_lineage.csv --output_dir ./plots --model-file variational_autoencoder/training/model.py --model-class SplitVAE --embedding-dim 32 --min_k 2 --max_k 20
```

### 4. Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--test_data` | Path to the test data numpy file (.npy) | `test_tensor.npy` |
| `--test_labels` | Path to the test bacteria labels numpy file (.npy) | `test_bacteria.npy` |
| `--model_path` | Path to the pre-trained model (.pt) | `split_autoencoder.pt` |
| `--taxonomy_file` | Path to the bacterial taxonomy CSV file | `bacterial_lineage_formatted.csv` |
| `--output_dir` | Directory to save output plots and results | `./plots` |
| `--model-file` | Path to the model file that defines the model architecture | Required |
| `--model-class` | Name of the model class inside the model file | `SplitAutoencoder` |
| `--embedding-dim` | Dimensionality of the embedding space | `32` |
| `--min_k` | Minimum number of clusters to test | `2` |
| `--max_k` | Maximum number of clusters to test | `15` |

### 5. Output

After running the script, the following output files will be generated in the specified output directory:

- **Clustering Metrics**: Plots of purity and silhouette scores for different values of k are saved to the output directory (`./plots` by default).
- **Cluster Visualizations**: Visualizations of bacterial clusters based on taxonomic family and order will be saved as PNG files (`plot_clustered_family.png`, `plot_clustered_order.png`, etc.).
- **Purity Significance**: Plots of purity significance saved to the output directory (`./plots` by default).

| Metric          | Based On        | Purpose                                       |
| --------------- | --------------- | --------------------------------------------- |
| `actual_purity` | real labels     | What purity do I get from my actual taxonomy? |
| `null_purities` | shuffled labels | What purity do I get just by random chance?   |

- **Family Distribution**: A bar plot showing the distribution of bacteria across different families is also saved (`family_distribution.png`).