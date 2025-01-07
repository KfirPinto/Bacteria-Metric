import pandas as pd
import numpy as np
import torch as torch

def concatenate():

    tensor1 = np.load("data/data_files/pathways/AsnicarF_2017_march/tensor.npy")
    tensor2 = np.load("data/data_files/pathways/AsnicarF_2021_march/tensor.npy")

    bacteria_list1 = np.load("data/data_files/pathways/AsnicarF_2017_march/bacteria_list.npy", allow_pickle=True)
    bacteria_list2 = np.load("data/data_files/pathways/AsnicarF_2021_march/bacteria_list.npy", allow_pickle=True)

    people_list1 = np.load("data/data_files/pathways/AsnicarF_2017_march/samples.npy", allow_pickle=True)
    people_list2 = np.load("data/data_files/pathways/AsnicarF_2021_march/samples.npy", allow_pickle=True)

    bacteria_union = np.unique(np.concatenate([bacteria_list1, bacteria_list2]))
    samples_union = np.unique(np.concatenate([people_list1, people_list2]))

    aligned_tensor1 = align_tensor(tensor1, bacteria_list1, pathway_list1, bacteria_union, pathway_union)
    aligned_tensor2 = align_tensor(tensor2, bacteria_list2, pathway_list2, bacteria_union, pathway_union)

    combined_tensor = np.concatenate([aligned_tensor1, aligned_tensor2], axis=0)
    print(combined_tensor.shape)

    np.save("data/data_files/pathways/Union/tensor.npy", combined_tensor)
    np.save("data/data_files/pathways/Union/bacteria_list.npy", bacteria_union)
    np.save("data/data_files/pathways/Union/pathway_list.npy", pathway_union)
    np.save("data/data_files/pathways/Union/samples.npy", samples_union)


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


def load_gene_families_data_from_csv(file_path):

    # Load the CSV file
    print("Loading data...")
    pre_df = pd.read_csv(file_path, index_col=0)
    # Replace NaN or missing values with 0
    pre_df = pre_df.fillna(0)

    # Filter only valid rows
    regex_pattern = r"UniRef90_.+\|g__.+\.s__.+"
    df = pre_df[pre_df.iloc[:, 0].str.contains(regex_pattern, regex=True, na=False)].copy()

    df[['gene_family', 'Bacteria']] = df.iloc[:, 0].str.split('|', expand=True)

    bacteria_list = df['Bacteria'].unique()
    gene_families_list = df['gene_family'].unique()
    people_list = df.columns[1:-2]

    bacteria_idx_map = {bacteria: idx for idx, bacteria in enumerate(bacteria_list)}
    gene_family_idx_map = {gene_family: idx for idx, gene_family in enumerate(gene_families_list)}

    # Create an empty tensor with the calculated dimensions
    data_tensor = torch.zeros((distinct_bacteria, distinct_gene_families, num_samples), dtype=torch.float32)

    # Fill the tensor with data
    for idx, row in valid_rows.iterrows():
        gene = row['Gene_Family']
        bacteria = row['Bacteria']
        gene_idx = gene_map[gene]
        bacteria_idx = bacteria_map[bacteria]
        # Convert row to float before inserting into the tensor
        float_values = torch.tensor(row.iloc[:-2].values.astype(float), dtype=torch.float32)
        data_tensor[bacteria_idx, gene_idx, :] = float_values

    print("Tensor construction complete.")

    # df_array = df.to_numpy()

    # tensor = np.zeros((len(people_list), len(bacteria_list), len(gene_families_list)))

    # for row in df_array:
    #     bacteria_idx = bacteria_idx_map[row[-1]]  # 'Bacteria' בעמודה האחרונה
    #     gene_family_idx = gene_family_idx_map[row[-2]]  # 'gene_family' בעמודה הלפני אחרונה
    #     for person_idx, person in enumerate(people_list):
    #         tensor[person_idx, bacteria_idx, gene_family_idx] = row[person_idx + 1]

    print(data_tensor.shape)

    np.save("data/data_files/gene_families/tensor.npy", tensor)
    np.save("data/data_files/gene_families/bacteria_list.npy", bacteria_list)
    np.save("data/data_files/gene_families/gene_families.npy", gene_families_list)
    np.save("data/data_files/gene_families/samples.npy", people_list)

    return data_tensor, bacteria_list, gene_families_list, people_list

    """
    t1 = np.where(people_list == 'MV_FEI2_t1Q14')[0][0]  
    t2 = bacteria_idx_map['g__Klebsiella.s__Klebsiella_pneumoniae']
    t3 = gene_family_idx_map['UniRef90_J7QIY4']
    print(tensor[t1][t2][t3])
    """

def load_pathway_data_from_csv():
    
    pathway_abundance_file = "data/data_files/raw_data/2021-03-31.AsnicarF_2021_pathway_abundance.csv"
    pre_df = pd.read_csv(pathway_abundance_file)
    regex_pattern = r".+\|g__.+\.s__.+"
    df = pre_df[pre_df.iloc[:, 0].str.contains(regex_pattern, regex=True, na=False)].copy()

    df[['Pathway', 'Bacteria']] = df.iloc[:, 0].str.split('|', expand=True)
    df.to_csv("data/data_files/pathways/AsnicarF_2021_march/pathways.csv") 

    bacteria_list = df['Bacteria'].unique()
    pathway_list = df['Pathway'].unique()
    people_list = df.columns[1:-2]  

    array = np.zeros((len(people_list), len(bacteria_list), len(pathway_list)))

    for _, row in df.iterrows():
        # Find the position (index) where the current bacterium appears in bacteria_list.
        bacteria_idx = np.where(bacteria_list == row['Bacteria'])[0][0]
        # Find the position (index) where the current pathway appears in pathway_list
        pathway_idx = np.where(pathway_list == row['Pathway'])[0][0]
        for person_idx, person in enumerate(people_list):
            array[person_idx, bacteria_idx, pathway_idx] = row[person]

    tensor = torch.from_numpy(array)
    print(tensor.shape)

    np.save("data/data_files/pathways/AsnicarF_2021_march/tensor.npy", tensor)
    np.save("data/data_files/pathways/AsnicarF_2021_march/bacteria_list.npy", bacteria_list)
    np.save("data/data_files/pathways/AsnicarF_2021_march/pathway_list.npy", pathway_list)
    np.save("data/data_files/pathways/AsnicarF_2021_march/samples.npy", people_list)

    """
    t1 = np.where(people_list =='MV_FEM5_t3Q15')[0][0]
    t2 = np.where(bacteria_list == 'g__Bifidobacterium.s__Bifidobacterium_bifidum')[0][0]
    t3 = np.where(pathway_list == 'UNINTEGRATED')[0][0]
    print(tensor[t1][t2][t3])
    """
