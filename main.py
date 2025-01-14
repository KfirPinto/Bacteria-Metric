
from data_loader_asnicarF import union
from data_loader_asnicarF import filter_zero_samples
from data_loader_asnicarF import load_gene_families_data_from_csv
from data_loader_asnicarF import process_large_csv
from plot import bacteria_count_across_samples
from plot import create_heatmap


if __name__ == "__main__":

   input_path = "data/data_files/raw_data/2021-03-31.AsnicarF_2021.gene_families.csv"
   output_dir = "data/data_files/gene_families/AsnicarF_2021_march"
   process_large_csv(input_path, output_dir)