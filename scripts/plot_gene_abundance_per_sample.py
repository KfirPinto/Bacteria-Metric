import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_gene_abundance_per_sample(input_file):
    df = pd.read_csv(input_file, index_col=0)

    # Sum each column (total gene abundance per sample)
    gene_abundance = df.sum(axis=0) / 2  # division by 2 is needed since the dataset structure

    # Bar plot: one bar per sample
    plt.figure(figsize=(12, 6))
    plt.bar(gene_abundance.index, gene_abundance.values, color='blue', alpha=0.7)
    plt.xticks([])  # Rotate x-axis labels if sample names are long
    plt.title('Total Gene Abundance per Sample')
    plt.xlabel('Sample')
    plt.ylabel('Total Gene Abundance')
    plt.tight_layout()
    plt.savefig('gene_abundance_per_sample.png')

if __name__ == "__main__":
    input_file = "data/PRJEB53403_168_samples/genefamilies/humann_2_genefamilies.csv"
    plot_gene_abundance_per_sample(input_file)
