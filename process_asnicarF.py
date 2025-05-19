from data_loader_asnicarF import load_pathway_data
from data_loader_asnicarF import load_gene_families_threshold
from data_loader_asnicarF import download_UniRef90_mapping
from data_loader_asnicarF import filter_based_uniprotkb
from data_loader_asnicarF import union
from data_loader_asnicarF import intersect
from plot import data_distribution

def preprocess_asnicarF():
    """
    Preprocess the data from AsnicarF_2017 and AsnicarF_2021 datasets.
    The preprocessing steps include:
    1. Downloading UniRef90 mapping using API calls.
    2. Filtering raw gene families data based on UniRef90 entries.
    3. Filtering gene families based on frequency.
    4. Loading appropriate pathways.
    5. Performing union of tensors for pathways and gene families.
    6. Intersecting pathway abundances and gene families through samples and bacteria dimensions.
    7. Loading the intersection tensors and plotting the data distribution.
   
   # Stage 1: download UniRef90 mapping using API call.

   api_list = [
   "https://rest.uniprot.org/idmapping/uniprotkb/results/stream/59ab6ec858c643c5be1832a4151f1a2653eb198a?fields=accession%2Creviewed%2Cid%2Cprotein_name%2Cgene_names%2Corganism_name%2Clength&format=xlsx",
   "https://rest.uniprot.org/idmapping/uniprotkb/results/stream/9a1699728f0f905b1144884f700d132f54a24198?fields=accession%2Creviewed%2Cid%2Cprotein_name%2Cgene_names%2Corganism_name%2Clength&format=xlsx",
   "https://rest.uniprot.org/idmapping/uniprotkb/results/stream/9700ca79fe52cb1904772d44b899e6fc8595373c?fields=accession%2Creviewed%2Cid%2Cprotein_name%2Cgene_names%2Corganism_name%2Clength&format=xlsx",
   "https://rest.uniprot.org/idmapping/uniprotkb/results/stream/fdead7e8bf94f78392226110673cedf6c9a553e2?fields=accession%2Creviewed%2Cid%2Cprotein_name%2Cgene_names%2Corganism_name%2Clength&format=xlsx",
   "https://rest.uniprot.org/idmapping/uniprotkb/results/stream/aa1311fd34b11816761d332362267740cc60d140?fields=accession%2Creviewed%2Cid%2Cprotein_name%2Cgene_names%2Corganism_name%2Clength&format=xlsx",
   "https://rest.uniprot.org/idmapping/uniprotkb/results/stream/32a935e48a6014c64e62089b3413a1fddda400f5?fields=accession%2Creviewed%2Cid%2Cprotein_name%2Cgene_names%2Corganism_name%2Clength&format=xlsx",
   "https://rest.uniprot.org/idmapping/uniprotkb/results/stream/5c935eb79a69c6b418fdb23305dd4b3c668aa60c?fields=accession%2Creviewed%2Cid%2Cprotein_name%2Cgene_names%2Corganism_name%2Clength&format=xlsx"
   ]
   output_dir = "Bacteria-Metric/data/data_files/gene_families/AsnicarF_2017_march"
   download_UniRef90_mapping(api_list, output_dir)

   # Stage 2: filter the raw gene families data based on UniRef90 entries founded in stage 1.

   input_path = "data/data_files/raw_data/gene_families/2021-03-31.AsnicarF_2017_gene_families.csv"
   output_dir = "data/data_files/gene_families/AsnicarF_2017_march"
   uniprotkb_path = "uniref_entries_AsnicarF_2017_march.csv"
   filter_based_uniprotkb(input_path, output_dir, uniprotkb_path)

   # Stage 3: for each dataset, filter gene families based on frequency.

   input_path = "data/data_files/gene_families/AsnicarF_2021_march/filtered_data.csv"
   output_dir = "data/data_files/gene_families/AsnicarF_2021_march/"

   load_gene_families_threshold(input_path, output_dir, top_k=1000)
    

   # Stage 4: load appropirate pathways.

   input_path = "data/data_files/raw_data/pathways/2021-03-31.AsnicarF_2021_pathway_abundance.csv"
   output_dir = "data/data_files/pathways/AsnicarF_2021_march"
   load_pathway_data(input_path, output_dir)
    
   # Stage 5: Given n tensors from the shape: [samples, bacteria, pathways] perform an union.

   input_path = ["data/data_files/pathways/AsnicarF_2017_march", "data/data_files/pathways/AsnicarF_2021_march"]
   output_dir = "data/data_files/pathways/Union/"
   union(input_path, output_dir, is_pathway=True)

   # Stage 6: Given n tensors from the shape: [samples, bacteria, gene_families] perform an union.

   input_path = ["data/data_files/gene_families/AsnicarF_2017_march", "data/data_files/gene_families/AsnicarF_2021_march"]
   output_dir = "data/data_files/gene_families/Union/"
   union(input_path, output_dir, is_pathway=False)

   # Stage 7: intersect between pathway abundances and gene families through the samples and bacteria dimensions."
   
   raw_gene_families="data/data_files/gene_families/Union"
   raw_pathways="data/data_files/pathways/Union"
   intersected_gene_families="data/data_files/gene_families/Intersection"
   intersected_pathways="data/data_files/pathways/Intersection"
   intersect(raw_gene_families, raw_pathways, intersected_gene_families, intersected_pathways)
   
   # Stage 8: load the intersection tensors and plot the data distribution.
   gene_tensor_path = "data/data_files/gene_families/Intersection/tensor.npy"
   pathway_tensor_path = "data/data_files/pathways/Intersection/tensor.npy"
   bacteria_list_path = "data/data_files/gene_families/Intersection/bacteria_list.npy"
   
   data_distribution(gene_tensor_path, pathway_tensor_path, bacteria_list_path)
   """