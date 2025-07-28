import pandas as pd
import numpy as np
import torch as torch
import os
import csv
from tqdm import tqdm
from collections import defaultdict
#import requests
from io import BytesIO

def intersect(gene_families_path, pathways_path, output_gene_families, output_pathways):

    # Load gene families data
    gene_families_tensor = np.load(os.path.join(gene_families_path, "tensor.npy"))
    gene_families_samples = np.load(os.path.join(gene_families_path, "sample_list.npy"), allow_pickle=True)
    gene_families_bacteria = np.load(os.path.join(gene_families_path, "bacteria_list.npy"), allow_pickle=True)
    gene_families_genes = np.load(os.path.join(gene_families_path, "gene_families_list.npy"), allow_pickle=True)

    # Load pathways data
    pathways_tensor = np.load(os.path.join(pathways_path, "tensor.npy"))
    pathways_samples = np.load(os.path.join(pathways_path, "sample_list.npy"), allow_pickle=True)
    pathways_bacteria = np.load(os.path.join(pathways_path, "bacteria_list.npy"), allow_pickle=True)
    pathways_pathways = np.load(os.path.join(pathways_path, "pathway_list.npy"), allow_pickle=True)

    # Find intersections
    intersected_samples = np.intersect1d(gene_families_samples, pathways_samples)
    intersected_bacteria = np.intersect1d(gene_families_bacteria, pathways_bacteria)

    # Get indices for intersected samples and bacteria
    gene_families_sample_indices = np.isin(gene_families_samples, intersected_samples)
    gene_families_bacteria_indices = np.isin(gene_families_bacteria, intersected_bacteria)

    pathways_sample_indices = np.isin(pathways_samples, intersected_samples)
    pathways_bacteria_indices = np.isin(pathways_bacteria, intersected_bacteria)

    # Find bacteria indices at gene families abundance table whose are *not* in the intersection
    bacteria_complementary_indices = np.where(~np.isin(gene_families_bacteria, intersected_bacteria))[0]

    # Filter tensors
    updated_gene_families_tensor = gene_families_tensor[
        np.ix_(gene_families_sample_indices, gene_families_bacteria_indices, np.arange(gene_families_tensor.shape[2]))
    ]

    updated_pathways_tensor = pathways_tensor[
        np.ix_(pathways_sample_indices, pathways_bacteria_indices, np.arange(pathways_tensor.shape[2]))
    ]

    complementary_tensor = gene_families_tensor[
        np.ix_(gene_families_sample_indices, bacteria_complementary_indices, np.arange(gene_families_tensor.shape[2]))]

    # Filter and update lists
    updated_gene_families_samples = gene_families_samples[gene_families_sample_indices]
    updated_gene_families_bacteria = gene_families_bacteria[gene_families_bacteria_indices]

    updated_pathways_samples = pathways_samples[pathways_sample_indices]
    updated_pathways_bacteria = pathways_bacteria[pathways_bacteria_indices]

    bacteria_complementary = gene_families_bacteria[bacteria_complementary_indices]

    print(f"Intersected gene families tensor shape: {updated_gene_families_tensor.shape}")
    print(f"Complementary gene families tensor shape: {complementary_tensor.shape}")
    print(f"Intersected pathways tensor shape: {updated_pathways_tensor.shape}")

    os.makedirs(output_gene_families, exist_ok=True)
    os.makedirs(output_pathways, exist_ok=True)

    # Save updated tensors and lists
    np.save(os.path.join(output_gene_families, "tensor.npy"), updated_gene_families_tensor)
    np.save(os.path.join(output_gene_families, "sample_list.npy"), updated_gene_families_samples)
    np.save(os.path.join(output_gene_families, "bacteria_list.npy"), updated_gene_families_bacteria)
    np.save(os.path.join(output_gene_families, "gene_families_list.npy"), gene_families_genes)

    np.save(os.path.join(output_pathways, "tensor.npy"), updated_pathways_tensor)
    np.save(os.path.join(output_pathways, "sample_list.npy"), updated_pathways_samples)
    np.save(os.path.join(output_pathways, "bacteria_list.npy"), updated_pathways_bacteria)
    np.save(os.path.join(output_pathways, "pathway_list.npy"), pathways_pathways)

    np.save(os.path.join(output_gene_families, "bacteria_complementary_list.npy"), bacteria_complementary)
    np.save(os.path.join(output_gene_families, "complementary_tensor.npy"), complementary_tensor)


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

def union(input_paths, output_dir, is_pathway=True):

    """
    Union data from multiple input paths. Based on `is_pathway`, handles either pathway or gene family data.
    """

    # Load tensors and metadata
    tensors = []
    samples_lists = []
    bacteria_lists = []
    third_axis_lists = []

    for input_path in input_paths:
        tensors.append(np.load(os.path.join(input_path, "tensor.npy"))) 
        samples_lists.append(np.load(os.path.join(input_path, "sample_list.npy"), allow_pickle=True))
        bacteria_lists.append(np.load(os.path.join(input_path, "bacteria_list.npy"), allow_pickle=True))
        if is_pathway:
            third_axis_lists.append(np.load(os.path.join(input_path, "pathway_list.npy"), allow_pickle=True))
        else:
            third_axis_lists.append(np.load(os.path.join(input_path, "gene_families_list.npy"), allow_pickle=True))

    # Step 1: Since some people participated in more than one experiment,
    # duplicates should be eliminated by creating copies: ID_1, ID_2, etc.
    renamed_samples_lists = []
    for i, samples in enumerate(samples_lists):
        previous_samples = np.concatenate(renamed_samples_lists) if renamed_samples_lists else np.array([])
        renamed_samples_lists.append(rename_duplicates(samples, previous_samples))

    # Step 2: Create unified lists
    bacteria_union = np.unique(np.concatenate(bacteria_lists))
    samples_union = np.unique(np.concatenate(renamed_samples_lists))
    third_axis_union = np.unique(np.concatenate(third_axis_lists))

    # Step 3: Align tensors w.r.t to the unified lists from step 2
    aligned_tensors = []
    for tensor, bacteria_list, third_axis_list in zip(tensors, bacteria_lists, third_axis_lists):
        aligned_tensors.append(align_tensor(tensor, bacteria_list, third_axis_list, bacteria_union, third_axis_union))

    combined_tensor = np.concatenate(aligned_tensors, axis=0)
    print(combined_tensor.shape)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save outputs
    np.save(os.path.join(output_dir, "tensor.npy"), combined_tensor) 
    np.save(os.path.join(output_dir, "sample_list.npy"), samples_union)
    np.save(os.path.join(output_dir, "bacteria_list.npy"), bacteria_union)
    if is_pathway:
        np.save(os.path.join(output_dir, "pathway_list.npy"), third_axis_union)
    else:
        np.save(os.path.join(output_dir, "gene_families_list.npy"), third_axis_union)

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

def align_tensor(tensor, bacteria_list, third_axis_list, bacteria_union, third_axis_union):
    aligned_tensor = np.zeros((tensor.shape[0], len(bacteria_union), len(third_axis_union)))
    bacteria_indices = {b: i for i, b in enumerate(bacteria_union)}
    third_axis_indices = {p: i for i, p in enumerate(third_axis_union)}
    for b_idx, b in enumerate(bacteria_list):
        for p_idx, p in enumerate(third_axis_list):
            new_b_idx = bacteria_indices[b]
            new_p_idx = third_axis_indices[p]
            aligned_tensor[:, new_b_idx, new_p_idx] = tensor[:, b_idx, p_idx]
    return aligned_tensor   

def filter_based_uniprotkb(input_path, output_dir, uniprotkb_path):
       
    regex_pattern = r"UniRef90_.+\|g__.+\.s__.+"
    chunk_size = 10000
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "filtered_data.csv")

    # Load UniProtKB entries
    uniprotkb_set = set(pd.read_csv(uniprotkb_path, header=0).iloc[:, 0])

     # Read header from input file
    header = pd.read_csv(input_path, nrows=0).columns.tolist()

    with pd.read_csv(input_path, chunksize=chunk_size) as reader:
        for i, chunk in enumerate(reader):
            print(f"Processing chunk {i}")
            chunk_filtered = chunk[chunk.iloc[:, 0].str.contains(regex_pattern, regex=True, na=False)].copy()
            gene_family = []
            bacteria = []
            # Remove rows which the split result not contain 2 arguments
            rows_to_keep = []
            for idx, value in enumerate(chunk_filtered.iloc[:, 0]):
                split_result = value.split('|')
                if len(split_result) == 2:
                    rows_to_keep.append(idx)
                    gene_family.append(split_result[0])
                    bacteria.append(split_result[1])

            # Filter chunk_filtered to keep only valid rows
            chunk_filtered = chunk_filtered.iloc[rows_to_keep].reset_index(drop=True)
            chunk_filtered['gene_family'] = pd.Series(gene_family, index=chunk_filtered.index)
            chunk_filtered['bacteria'] = pd.Series(bacteria, index=chunk_filtered.index)

            #chunk_filtered[['gene_family', 'Bacteria']] = chunk_filtered.iloc[:, 0].str.split('|', expand=True, n=1)

            # Remove 'UniRef90_' prefix from gene_family before intersection
            chunk_filtered['gene_family'] = chunk_filtered['gene_family'].str.replace(r'^UniRef90_', '', regex=True)
            chunk_filtered = chunk_filtered[chunk_filtered['gene_family'].isin(uniprotkb_set)]

            # Append to output file with header only on the first write
            chunk_filtered.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file), columns=header)
        
def load_gene_families(input_path, output_dir, threshold=False, top_k=1000):

    pre_df = pd.read_csv(input_path)
    #regex_pattern = r"UniRef90_.+\|g__.+\.s__.+"
    #regex_pattern = r"^GO:\d+\|g__[^.]+\.s__[^.]+$"
    regex_pattern = r"^\d+\.\d+\.\d+\.\d+\|g__.+\.s__.+$"

    df = pre_df[pre_df.iloc[:, 0].str.contains(regex_pattern, regex=True, na=False)].copy()

    df[['gene_family', 'Bacteria']] = df.iloc[:, 0].str.split('|', expand=True)

    # Trim sample column names (1:-2) at the first underscore (ERR9830179_Abundance-RPKs → ERR9830179)
    original_people = df.columns[1:-2]
    trimmed_people = [col.split('_')[0] for col in original_people]
    df.columns = list(df.columns[:1]) + trimmed_people + list(df.columns[-2:])

    people_list = trimmed_people

    # Count (sample, bacteria) occurrences for each gene_family
    gene_family_counter = defaultdict(set)

    if threshold:
        for row in tqdm(df.itertuples(index=False, name='Pandas'), total=len(df), desc="Counting occurrences"):
            gene_family = row[-2]
            bacteria = row[-1]
            for person_idx, person in enumerate(people_list):
                val = getattr(row, person)
                if val != 0:
                    gene_family_counter[gene_family].add((person, bacteria))

        # Optional: Keep only the top_k gene families with the most (sample, bacteria) occurrences
        sorted_gene_families = sorted(gene_family_counter.items(), key=lambda x: len(x[1]), reverse=True)
        valid_gene_families = set([gf for gf, _ in sorted_gene_families[:top_k]])

        # Filter the DataFrame
        df = df[df['gene_family'].isin(valid_gene_families)].copy()

    # Compute index mappings after filtering
    bacteria_list = df['Bacteria'].unique()
    gene_families_list = df['gene_family'].unique()
    
    bacteria_idx_map = {bacteria: idx for idx, bacteria in enumerate(bacteria_list)}
    gene_family_idx_map = {gene_family: idx for idx, gene_family in enumerate(gene_families_list)}

    # Initialize a tensor with zeros
    array = np.zeros((len(people_list), len(bacteria_list), len(gene_families_list)))

    for row in tqdm(df.itertuples(index=False, name='Pandas'), total=len(df), desc="Processing rows"):
        bacteria_idx = bacteria_idx_map[row[-1]]  # 'Bacteria' is the last column
        gene_family_idx = gene_family_idx_map[row[-2]]  # 'gene_family' is in the second-to-last column

        for person_idx, person in enumerate(people_list):
            array[person_idx, bacteria_idx, gene_family_idx] = getattr(row, person)

    # Print the shape of the tensor for verification
    tensor = torch.from_numpy(array)
    print(f"Tensor shape before normalization: {tensor.shape}")
    
    # Normalize each sample (first dimension) so that all values sum to 1
    # Sum across bacteria and gene families dimensions (dim=1 and dim=2)
    sample_sums = tensor.sum(dim=(1, 2), keepdim=True)
    
    # Avoid division by zero - set sum to 1 for samples with zero total
    sample_sums = torch.where(sample_sums == 0, torch.ones_like(sample_sums), sample_sums)
    
    # Normalize to proportional abundances
    tensor = tensor / sample_sums
    
    print(f"Tensor shape after normalization: {tensor.shape}")
    print(f"Sample sums after normalization (should all be 1.0): {tensor.sum(dim=(1, 2))[:5]}")  # Show first 5 samples

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "tensor.npy"), tensor) 
    np.save(os.path.join(output_dir, "sample_list.npy"), people_list)
    np.save(os.path.join(output_dir, "bacteria_list.npy"), bacteria_list)
    np.save(os.path.join(output_dir, "gene_families_list.npy"), gene_families_list)

    # Sanity check
    
    #t1 = np.where(np.array(people_list) == 'ERR9830187')[0][0]
    #t2 = bacteria_idx_map['g__Escherichia.s__Escherichia_coli']
    #t3 = gene_family_idx_map['GO:0000006']
    #print(f"Sanity check for {people_list[t1]}, {bacteria_list[t2]}, {gene_families_list[t3]}")
    #torch.set_printoptions(precision=10)
    #print(tensor[t1][t2][t3])
    

def load_pathway_data(input_file, output_dir):
    
    pre_df = pd.read_csv(input_file, quotechar='"')
    regex_pattern = r".+\|g__.+\.s__.+"

    # iloc[:,0] used for index based selection
    # : (before the comma) → Selects all rows
    # 0 (after the comma) → Selects only the first column
    df = pre_df[pre_df.iloc[:, 0].str.contains(regex_pattern, regex=True, na=False)].copy()

    # Split the first column into 'Pathway' and 'Bacteria'
    # expand=True: Splits the string and expands the result into separate columns.
    # Each new column will contain one of the split parts.
    df[['Pathway', 'Bacteria']] = df.iloc[:, 0].str.split('|', expand=True)

    # unique() returns the unique values as a NumPy array.
    bacteria_list = df['Bacteria'].unique()
    pathway_list = df['Pathway'].unique()

    # Trim sample column names (1:-2) at the first underscore (ERR9830179_Abundance-RPKs → ERR9830179)
    original_people = df.columns[1:-2]
    trimmed_people = [col.split('_')[0] for col in original_people]
    df.columns = list(df.columns[:1]) + trimmed_people + list(df.columns[-2:])

    # [start_index:end_index]: list slicing syntax. It specifies a range of indices to extract a subset of items.
    # Inclusive start_index, exclusive end_index.
    people_list = trimmed_people 

    array = np.zeros((len(people_list), len(bacteria_list), len(pathway_list)))

    # df.iterrows() - generator that allows you to iterate over the rows of a DataFrame as pairs of index and Series
    for _, row in df.iterrows():
        # np.where() returns tuples of arrays (in the case of multiple dimensions input array).
        # [i][j] - access the i-th element at the j-th array
        # Find the position (index) where the current bacteria appear in bacteria_list
        bacteria_idx = np.where(bacteria_list == row['Bacteria'])[0][0]

        # Find the position (index) where the current pathway appears in pathway_list
        pathway_idx = np.where(pathway_list == row['Pathway'])[0][0]

        for person_idx, person in enumerate(people_list):
            array[person_idx, bacteria_idx, pathway_idx] = row.iloc[person_idx+1]

    tensor = torch.from_numpy(array)
    print(tensor.shape)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "tensor.npy"), tensor)
    np.save(os.path.join(output_dir, "bacteria_list.npy"), bacteria_list)
    np.save(os.path.join(output_dir, "pathway_list.npy"), pathway_list)
    np.save(os.path.join(output_dir, "sample_list.npy"), people_list)

    # Sanity check 
    """
    t1 = np.where(people_list =='ERR9830182')[0][0]
    t2 = np.where(bacteria_list == 'g__Akkermansia.s__Akkermansia_muciniphila')[0][0]
    t3 = np.where(pathway_list == '1CMET2-PWY: folate transformations III (E. coli)')[0][0]
    torch.set_printoptions(precision=10)
    print(tensor[t1][t2][t3])
    """
    
def extract_UniRef90_entries(input_path, output_dir):
    """
    A preliminary step aimed at extracting gene families in the raw data
    in order to produce a list of genes that appear in UniprotKB.
    """
    UniRef_90_entries = set()
    regex_pattern = r"UniRef90_.+\|g__.+\.s__.+"

    os.makedirs(output_dir, exist_ok=True)
    chunk_size = 10000
    max_entries_per_file = 100000
    file_index = 0

    def write_partial_set(entries_set, file_index):
        output_file = os.path.join(output_dir, f"Uniref90_entries_{file_index}.csv")
        df = pd.DataFrame({'Value': list(entries_set)})
        df.to_csv(output_file, index=False)
        print(f"Wrote {len(entries_set)} entries to {output_file}")

    with pd.read_csv(input_path, chunksize=chunk_size) as reader:
        for i, chunk in enumerate(reader):
            print(f"Processing chunk {i}")
            chunk_filtered = chunk[chunk.iloc[:, 0].str.contains(regex_pattern, regex=True, na=False)].copy()
            for value in chunk_filtered.iloc[:, 0]:
                split_result = value.split('|')
                if len(split_result) == 2:
                    UniRef_90_entries.add(split_result[0].removeprefix('UniRef90_'))

            # Write out every 100,000 entries and reset the set
            if len(UniRef_90_entries) >= max_entries_per_file:
                batch = set(list(UniRef_90_entries)[:max_entries_per_file])
                UniRef_90_entries -= batch
                write_partial_set(batch, file_index)
                file_index += 1

    # Write any remaining entries
    if UniRef_90_entries:
        write_partial_set(UniRef_90_entries, file_index)

def extract_all_UniRef90_entries(input_path, output_dir):
    UniRef_90_entries = set()
    regex_pattern = r"UniRef90_.+\|g__.+\.s__.+"

    os.makedirs(output_dir, exist_ok=True)
    chunk_size = 10000

    with pd.read_csv(input_path, chunksize=chunk_size) as reader:
        for i, chunk in enumerate(reader):
            print(f"Processing chunk {i}")
            chunk_filtered = chunk[chunk.iloc[:, 0].str.contains(regex_pattern, regex=True, na=False)].copy()
            for value in chunk_filtered.iloc[:, 0]:
                split_result = value.split('|')
                if len(split_result) == 2:
                    UniRef_90_entries.add(split_result[0].removeprefix('UniRef90_'))

    # Write all entries to a single file
    output_file = os.path.join(output_dir, "UniRef90_entries_all.csv")
    df = pd.DataFrame({'Value': sorted(UniRef_90_entries)})
    df.to_csv(output_file, index=False)
    print(f"Wrote {len(df)} entries to {output_file}")

def download_UniRef90_mapping(api_urls, output_dir):
    """
    Downloads UniProt Excel files from provided API URLs, filters them, and saves a merged CSV.

    Parameters:
        api_urls (list): List of UniProt API URLs returning XLSX files.
        output_csv_path (str): Path to save the filtered CSV file.

    Returns:
        pd.DataFrame: The final filtered dataframe.
    """
    filtered_dfs = []

    for url in api_urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
            df = pd.read_excel(BytesIO(response.content))

            # Normalize column name just in case
            protein_col = 'Protein names'
            if protein_col not in df.columns:
                raise ValueError(f"'Protein names' column not found in data from {url}")

            # Filter conditions
            df = df[df[protein_col].notna()]
            df = df[~df[protein_col].str.strip().str.lower().eq("deleted")]
            df = df[~df[protein_col].str.contains(r"\bUncharacterized\b", case=False, na=False)]

            filtered_dfs.append(df)

        except Exception as e:
            print(f"Error processing URL {url}: {e}")

    output_file = os.path.join(output_dir, "Uniref90_mapping.csv")

    if filtered_dfs:
        final_df = pd.concat(filtered_dfs, ignore_index=True)
        final_df.to_csv(output_file, index=False)
        print(f"Filtered data saved to: {output_file}")
        return final_df
    else:
        print("No data was processed successfully.")
        return pd.DataFrame()

def extract_bacteria(input_path, output_dir):

    bacteria = np.load(input_path, allow_pickle=True)
    df = pd.DataFrame(bacteria, columns=["Bacteria"])
    output_file = os.path.join(output_dir, "bacteria_list.csv")
    df.to_csv(output_file, index=False)