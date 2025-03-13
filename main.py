
from data_loader_asnicarF import load_gene_families_data_from_csv
from data_loader_asnicarF import process_big_data

if __name__ == "__main__":

   input_path = "data/data_files/raw_data/2021-03-31.AsnicarF_2021_gene_families.csv"
   output_dir = "trial-2021"
   process_big_data(input_path, output_dir)