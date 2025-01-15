import pandas as pd
import numpy as np
import torch as torch
import os

def intersect(gene_families_path, pathways_path, output_gene_families, output_pathways):

    # Load gene families data
    gene_families_tensor = np.load(os.path.join(gene_families_path, "tensor.npy"))
    gene_families_samples = np.load(os.path.join(gene_families_path, "sample_list.npy"))
    gene_families_bacteria = np.load(os.path.join(gene_families_path, "bacteria_list.npy"))
    gene_families_genes = np.load(os.path.join(gene_families_path, "gene_families_list.npy"))

    # Load pathways data
    pathways_tensor = np.load(os.path.join(pathways_path, "tensor.npy"))
    pathways_samples = np.load(os.path.join(pathways_path, "sample_list.npy"))
    pathways_bacteria = np.load(os.path.join(pathways_path, "bacteria_list.npy"))
    pathways_pathways = np.load(os.path.join(gene_families_path, "pathway_list.npy"))

    # Find intersections
    intersected_samples = np.intersect1d(gene_families_samples, pathways_samples)
    intersected_bacteria = np.intersect1d(gene_families_bacteria, pathways_bacteria)

    # Get indices for intersected samples and bacteria
    gene_families_sample_indices = np.isin(gene_families_samples, intersected_samples)
    gene_families_bacteria_indices = np.isin(gene_families_bacteria, intersected_bacteria)

    pathways_sample_indices = np.isin(pathways_samples, intersected_samples)
    pathways_bacteria_indices = np.isin(pathways_bacteria, intersected_bacteria)

    # Filter tensors
    updated_gene_families_tensor = gene_families_tensor[
        np.ix_(gene_families_sample_indices, gene_families_bacteria_indices, np.arange(gene_families_tensor.shape[2]))
    ]

    updated_pathways_tensor = pathways_tensor[
        np.ix_(pathways_sample_indices, pathways_bacteria_indices, np.arange(pathways_tensor.shape[2]))
    ]

    # Filter and update lists
    updated_gene_families_samples = gene_families_samples[gene_families_sample_indices]
    updated_gene_families_bacteria = gene_families_bacteria[gene_families_bacteria_indices]

    updated_pathways_samples = pathways_samples[pathways_sample_indices]
    updated_pathways_bacteria = pathways_bacteria[pathways_bacteria_indices]

    # Save updated tensors and lists
    np.save(os.path.join(output_gene_families, "tensor.npy"), updated_gene_families_tensor)
    np.save(os.path.join(output_gene_families, "sample_list.npy"), updated_gene_families_samples)
    np.save(os.path.join(output_gene_families, "bacteria_list.npy"), updated_gene_families_bacteria)
    np.save(os.path.join(output_gene_families, "gene_families_list.npy"), gene_families_genes)

    np.save(os.path.join(output_pathways, "tensor.npy"), updated_pathways_tensor)
    np.save(os.path.join(output_pathways, "sample_list.npy"), updated_pathways_samples)
    np.save(os.path.join(output_pathways, "bacteria_list.npy"), updated_pathways_bacteria)
    np.save(os.path.join(output_pathways, "pathway_list.npy"), pathways_pathways)

def filter_zero_samples(input_path, output_dir):

    tensor = torch.tensor(np.load(os.path.join(input_path, "tensor.npy")))
    samples = np.load(os.path.join(input_path, "sample_list.npy"))
    bacteria = np.load(os.path.join(input_path, "bacteria_list.npy"))
    third_axis = np.load(os.path.join(input_path, "pathway_list.npy")) # change accordingly!

    flattened_tensor = torch.sum(tensor, dim=2) # tensor shape: (samples, bacteria)
    counts = torch.sum(flattened_tensor != 0, dim=1) # tensor shape: (samples, )
    non_zero_indices = (counts > 0).nonzero(as_tuple=True)[0]
    filtered_samples = samples[non_zero_indices]
    filtered_tensor = tensor[non_zero_indices] # filter the first dim of the tensor
    print(filtered_tensor.shape)

    np.save(os.path.join(output_dir, "tensor.npy"), filtered_tensor.numpy())
    np.save(os.path.join(output_dir, "sample_list.npy"), filtered_samples)
    np.save(os.path.join(output_dir, "bacteria_list.npy"), bacteria)
    np.save(os.path.join(output_dir, "pathway_list.npy"), third_axis)

def union(input_paths, output_dir):

    # Load tensors and metadata
    tensors = []
    bacteria_lists = []
    pathway_lists = []
    samples_lists = []

    for input_path in input_paths:
        tensors.append(np.load(os.path.join(input_path, "tensor.npy")))
        bacteria_lists.append(np.load(os.path.join(input_path, "bacteria_list.npy"), allow_pickle=True))
        pathway_lists.append(np.load(os.path.join(input_path, "pathway_list.npy"), allow_pickle=True))
        samples_lists.append(np.load(os.path.join(input_path, "sample_list.npy"), allow_pickle=True))

    # Step 1: Rename duplicates in sample lists
    renamed_samples_lists = []
    for i, samples in enumerate(samples_lists):
        previous_samples = np.concatenate(renamed_samples_lists) if renamed_samples_lists else np.array([])
        renamed_samples_lists.append(rename_duplicates(samples, previous_samples))

    # Step 2: Create unified lists
    bacteria_union = np.unique(np.concatenate(bacteria_lists))
    pathway_union = np.unique(np.concatenate(pathway_lists))
    samples_union = np.unique(np.concatenate(renamed_samples_lists))

    # Step 3: Align tensors w.r.t to the unified lists from step 2
    aligned_tensors = []
    for tensor, bacteria_list, pathway_list in zip(tensors, bacteria_lists, pathway_lists):
        aligned_tensors.append(align_tensor(tensor, bacteria_list, pathway_list, bacteria_union, pathway_union))

    combined_tensor = np.concatenate(aligned_tensors, axis=0)
    print(combined_tensor.shape)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save outputs
    np.save(os.path.join(output_dir, "tensor.npy"), combined_tensor)
    np.save(os.path.join(output_dir, "bacteria_list.npy"), bacteria_union)
    np.save(os.path.join(output_dir, "pathway_list.npy"), pathway_union)
    np.save(os.path.join(output_dir, "sample_list.npy"), samples_union)

def rename_duplicates(people_list, existing_samples):
    new_people_list = []
    existing_sample_set = set(existing_samples)
    for i, person in enumerate(people_list):
        new_name = person
        count = 1
        while new_name in existing_sample_set:
            new_name = f"{person}_{count}"
            count += 1
        new_people_list.append(new_name)
        existing_sample_set.add(new_name)
    return np.array(new_people_list, dtype = object)

def align_tensor(tensor, bacteria_list, pathway_list, bacteria_union, pathway_union):
    aligned_tensor = np.zeros((tensor.shape[0], len(bacteria_union), len(pathway_union)))
    bacteria_indices = {b: i for i, b in enumerate(bacteria_union)}
    pathway_indices = {p: i for i, p in enumerate(pathway_union)}
    for b_idx, b in enumerate(bacteria_list):
        for p_idx, p in enumerate(pathway_list):
            new_b_idx = bacteria_indices[b]
            new_p_idx = pathway_indices[p]
            aligned_tensor[:, new_b_idx, new_p_idx] = tensor[:, b_idx, p_idx]
    return aligned_tensor   

def load_gene_families_data_from_csv(input_path, output_dir):

    pre_df = pd.read_csv(input_path)

    # Filter rows based on the regex pattern
    regex_pattern = r"UniRef90_.+\|g__.+\.s__.+"
    df = pre_df[pre_df.iloc[:, 0].str.contains(regex_pattern, regex=True, na=False)].copy()

    # Split the first column into 'gene_family' and 'Bacteria'
    df[['gene_family', 'Bacteria']] = df.iloc[:, 0].str.split('|', expand=True)

    # Extract unique lists for bacteria, gene families, and people (samples)
    bacteria_list = df['Bacteria'].unique()
    gene_families_list = df['gene_family'].unique()
    people_list = df.columns[1:-2]

    # Create mapping dictionaries for indexing
    bacteria_idx_map = {bacteria: idx for idx, bacteria in enumerate(bacteria_list)}
    gene_family_idx_map = {gene_family: idx for idx, gene_family in enumerate(gene_families_list)}

    # Convert the DataFrame to a NumPy array
    df_array = df.to_numpy()

    # Initialize a tensor with zeros
    tensor = np.zeros((len(people_list), len(bacteria_list), len(gene_families_list)))

    # Populate the tensor with values from the DataFrame
    for row in df_array:
        bacteria_idx = bacteria_idx_map[row[-1]]  # 'Bacteria' is the last column
        gene_family_idx = gene_family_idx_map[row[-2]]  # 'gene_family' is in the second-to-last column
        for person_idx, person in enumerate(people_list):
            tensor[person_idx, bacteria_idx, gene_family_idx] = row[person_idx + 1]

    # Print the shape of the tensor for verification
    print(tensor.shape)

    # Save the outputs to the specified directory
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "tensor.npy"), tensor) 
    np.save(os.path.join(output_dir, "sample_list.npy"), people_list)
    np.save(os.path.join(output_dir, "bacteria_list.npy"), bacteria_list)
    np.save(os.path.join(output_dir, "gene_families_list.npy"), gene_families_list)

    """
    t1 = np.where(people_list == 'MV_FEI2_t1Q14')[0][0]  
    t2 = bacteria_idx_map['g__Klebsiella.s__Klebsiella_pneumoniae']
    t3 = gene_family_idx_map['UniRef90_J7QIY4']
    print(tensor[t1][t2][t3])
    """

def process_large_csv(input_path, output_dir, chunksize=10_000):

    os.makedirs(output_dir, exist_ok=True)

    all_bacteria = set()
    all_gene_families = set()
    people_list = None
    regex_pattern = r"UniRef90_.+\|g__.+\.s__.+"
    tensor_data = []

    for pre_chunk in pd.read_csv(input_path, chunksize=chunksize):

        chunk = pre_chunk[pre_chunk.iloc[:, 0].str.contains(regex_pattern, regex=True, na=False)].copy()
        chunk[['gene_family', 'Bacteria']] = chunk.iloc[:, 0].str.split('|', expand=True)

def load_pathway_data_from_csv(input_file, output_dir):
    
    pre_df = pd.read_csv(input_file)
    regex_pattern = r".+\|g__.+\.s__.+"
    df = pre_df[pre_df.iloc[:, 0].str.contains(regex_pattern, regex=True, na=False)].copy()

    df[['Pathway', 'Bacteria']] = df.iloc[:, 0].str.split('|', expand=True)
    #df.to_csv("data/data_files/pathways/AsnicarF_2021_march/pathways.csv") 

    bacteria_list = df['Bacteria'].unique()
    pathway_list = df['Pathway'].unique()
    people_list = df.columns[1:-2]  

    array = np.zeros((len(people_list), len(bacteria_list), len(pathway_list)))

    for _, row in df.iterrows():
        # Find the position (index) where the current bacterium appears in bacteria_list
        bacteria_idx = np.where(bacteria_list == row['Bacteria'])[0][0]
        # Find the position (index) where the current pathway appears in pathway_list
        pathway_idx = np.where(pathway_list == row['Pathway'])[0][0]
        for person_idx, person in enumerate(people_list):
            array[person_idx, bacteria_idx, pathway_idx] = row[person]

    tensor = torch.from_numpy(array)
    print(tensor.shape)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "tensor.npy"), tensor)
    np.save(os.path.join(output_dir, "bacteria_list.npy"), bacteria_list)
    np.save(os.path.join(output_dir, "pathway_list.npy"), pathway_list)
    np.save(os.path.join(output_dir, "sample_list.npy"), people_list)

    """
    sanity check
    t1 = np.where(people_list =='MV_FEM5_t3Q15')[0][0]
    t2 = np.where(bacteria_list == 'g__Bifidobacterium.s__Bifidobacterium_bifidum')[0][0]
    t3 = np.where(pathway_list == 'UNINTEGRATED')[0][0]
    print(tensor[t1][t2][t3])
    """
