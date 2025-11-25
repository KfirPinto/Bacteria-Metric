import pandas as pd
import numpy as np

def intersect_by_taxid(source, relative_abunndance_long, relative_abundance_NCBI, test_set):
    # read txt file
    with open(source, 'r') as file:
        source_NCBI = file.read().splitlines()

    # Upload relative abundance of reference
    rel_abundance_long = pd.read_csv(relative_abunndance_long, index_col=0)
    rel_abundance_NCBI = pd.read_csv(relative_abundance_NCBI, index_col=0)

    # Map long to NCBI format
    long_NCBI = dict(zip(rel_abundance_long.columns, rel_abundance_NCBI.columns))
    NCBI_long = dict(zip(rel_abundance_NCBI.columns, rel_abundance_long.columns))

    # Upload test set
    test_set_long = np.load(test_set, allow_pickle=True) # Assuming test_set is in long format
    test_set_NCBI = [long_NCBI[t] for t in test_set_long if t in long_NCBI]

    rel_abundance_long = rel_abundance_long.loc[:, rel_abundance_long.columns.isin(test_set_long)]
    rel_abundance_NCBI = rel_abundance_NCBI.loc[:, rel_abundance_NCBI.columns.isin(test_set_NCBI)]

    # Intersect between source and reference (both in NCBI format)
    source_NCBI_intersect = [t for t in rel_abundance_NCBI.columns if t in source_NCBI]

    # Map to long format
    source_long_intersect = [NCBI_long[t] for t in source_NCBI_intersect if t in NCBI_long]

    # Write to txt file
    with open("intersected_taxids_long.txt", 'w') as file:
        for t in source_long_intersect:
            file.write(f"{t}\n")

    with open("intersected_taxids_NCBI.txt", 'w') as file:
        for t in source_NCBI_intersect:
            file.write(f"{t}\n")

def map_taxid_to_embed(blast, taxa_ncbi, taxa_long, embedding, output_prefix="output"):
    """
    Map embeddings to taxonomic IDs and save as .npy files.

    Parameters:
    ----------
    blast : str
        Path to blast_results.tsv (col 1 = embed ID, col 7 = NCBI taxon ID)
    taxa_ncbi : str
        Path to intersected_taxids_NCBI.txt (NCBI taxon IDs).
    taxa_long : str
        Path to intersected_taxids_long.txt (long-format taxon names).
    embedding : str
        Path to GloVe embedding file (e.g., embed_.07_100dim.txt).
    output_prefix : str
        Prefix for output files.

    Returns:
    -------
    Saves:
        - {output_prefix}_embeddings.npy  (numpy array of embeddings)
        - {output_prefix}_metadata.npy    (numpy array of long-format taxon names)
    """

    # 1 - Read BLAST results (no header, tab-separated)
    blast_results = pd.read_csv(blast, sep="\t", header=None)

    # 2 - Read taxa files (simple text lists)
    taxa_ncbi_list = [line.strip() for line in open(taxa_ncbi, 'r')]
    taxa_long_list = [line.strip() for line in open(taxa_long, 'r')]

    # 3 - Create mapping dict: NCBI taxid -> Long-format taxon
    ncbi_to_long = dict(zip(taxa_ncbi_list, taxa_long_list))

    # 4 - Read embeddings (GloVe-style: first column ID, then 100-d vector)
    embed_dict = {}
    with open(embedding, 'r') as f:
        for line in f:
            # skip the first line
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            embed_id = parts[0]
            vector = np.array(parts[1:], dtype=float)
            embed_dict[embed_id] = vector

    # 5 - Build final dataframe: embed_id | ncbi_taxid | long_taxid | embedding
    data = []
    for _, row in blast_results.iterrows():
        embed_id = str(row[0])               # Embedding ID
        ncbi_taxid = str(row[6])            # NCBI taxon ID (7th column)
        if ncbi_taxid in ncbi_to_long and embed_id in embed_dict:
            long_taxid = ncbi_to_long[ncbi_taxid]
            vector = embed_dict[embed_id]
            data.append((embed_id, ncbi_taxid, long_taxid, vector))

    # 6 - Convert to structured output
    if not data:
        raise ValueError("No matching taxa found between BLAST results and embeddings.")

    # Separate metadata and embedding matrix
    metadata = [row[2] for row in data]                 # long-format taxon names
    embedding_matrix = np.stack([row[3] for row in data])  # stack embedding vectors

    # 7 - Save outputs as .npy
    np.save(f"{output_prefix}_embeddings.npy", embedding_matrix)
    np.save(f"{output_prefix}_metadata.npy", np.array(metadata))

    print(f"Saved embeddings: {output_prefix}_embeddings.npy")
    print(f"Saved metadata: {output_prefix}_metadata.npy")

    return embedding_matrix, metadata

if __name__ == "__main__":
    
    source = "/home/barsapi1/metric/Bacteria-Metric/data/GloVe/taxids.txt"
    relative_abunndance_long = "/home/barsapi1/metric/Bacteria-Metric/data/HMP/raw/relative_abundance.csv"
    relative_abundance_NCBI = "/home/barsapi1/metric/Bacteria-Metric/data/HMP/raw/relative_abundance_NCBI.csv"
    test_set = "/home/barsapi1/metric/Bacteria-Metric/data/HMP/test_set/bacteria_list_expanded.npy"
    intersect_by_taxid(source, relative_abunndance_long, relative_abundance_NCBI, test_set)

    blast = "/home/barsapi1/metric/Bacteria-Metric/data/GloVe/blast_results.tsv"
    taxa_ncbi = "/home/barsapi1/metric/Bacteria-Metric/data/GloVe/intersected_taxids_NCBI.txt"
    taxa_long = "/home/barsapi1/metric/Bacteria-Metric/data/GloVe/intersected_taxids_long.txt"
    embedding = "/home/barsapi1/metric/Bacteria-Metric/data/GloVe/embed_.07_100dim.txt"
    output_prefix = "GloVe_embeddings"
    map_taxid_to_embed(blast, taxa_ncbi, taxa_long, embedding, output_prefix)
