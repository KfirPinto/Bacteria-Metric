
from data_loader_asnicarF import load_gene_families_data_from_csv

if __name__ == "__main__":

   input_path = "data/data_files/raw_data/2021-03-31.AsnicarF_2021_gene_families.csv"
   output_dir = "data/data_files/gene_families/trial/"
   load_gene_families_data_from_csv(input_path, output_dir)
