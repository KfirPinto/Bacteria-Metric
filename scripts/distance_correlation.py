import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from training.model import PathwayReg
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import spearmanr, pearsonr
import config

def evaluate(bacteria, gene_families, pathways, model_weights):

    # ----- load the information -----
    val_bacteria = np.load(bacteria, allow_pickle=True)  # shape: (d,)
    val_gene_families = np.load(gene_families)  # shape: (n, d, g)
    val_pathways = np.load(pathways)  # shape: (n, d, p)

    gene_dim= val_gene_families.shape[-1]  # g
    pathway_dim = val_pathways.shape[-1]  # p
    embedding_dim = config.EMBEDDING_DIM  

    model = PathwayReg(gene_dim, embedding_dim, pathway_dim)
    model.load_state_dict(torch.load(model_weights))
    model.eval()  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():  # disables gradient tracking to save memory
        x = torch.tensor(val_gene_families, dtype=torch.float32).to(device)  # shape: (n, d, g)
        H_ij, _ = model(x)  # shape: (n, d, 2b)

    # take the first half of the embedding and average over d samples
    b = H_ij.shape[-1] // 2
    H_i = H_ij[..., :b]  # (n, d, b)
    latent_embeddings = H_i.mean(dim=0).cpu().numpy()  # shape: (d, b)

    # ----- produce binary functional pathways -----
    pathways_bin = (val_pathways > 0).astype(int)  # (n, d, p)
    functional_vectors = pathways_bin.mean(axis=0)  # (d, p)

    # Print embeddings vectors
    #print("Latent embeddings per bacterium:\n")
    #for name, emb in zip(val_bacteria, latent_embeddings):
    #    print(f"{name}: {np.round(emb, 4)}")

    #print("\nFunctional vectors per bacterium:\n")
    #for name, vec in zip(val_bacteria, functional_vectors):
    #    print(f"{name}: {np.round(vec, 4)}")
    
    # ----- calculate distances -----
    latent_distances = squareform(pdist(latent_embeddings, metric='cosine'))
    functional_distances = squareform(pdist(functional_vectors, metric='cosine'))

    latent_distances = latent_distances.flatten()
    functional_distances = functional_distances.flatten()

    # ----- visualization of the distribution -----

    plt.figure(figsize=(10, 6))

    # Plot latent distances
    sns.histplot(latent_distances, kde=True, color="skyblue", label="Latent", stat="density", bins=50)

    # Plot functional distances
    sns.histplot(functional_distances, kde=True, color="salmon", label="Functional", stat="density", bins=50)

    plt.xlabel("Pairwise Distance")
    plt.ylabel("Density")
    plt.title("Distributions of Pairwise Distances")
    plt.legend()  # show only one legend entry per curve
    plt.tight_layout()
    plt.savefig("pairwise_distances_Run16.png")

    r_s, r_s_p_value = spearmanr(latent_distances, functional_distances)
    r_p, r_p_p_value = pearsonr(latent_distances, functional_distances)
    print("Spearman correlation:", r_s, "p-value:", r_s_p_value)
    print("Pearson correlation:", r_p, "p-value:", r_p_p_value)

if __name__ == "__main__":
    # example
    bacteria_path = "eval_data/Run_16/test/test_bacteria.npy"
    gene_families_path = "eval_data/Run_16/test/test_gene_families.npy"
    pathways_path = "eval_data/Run_16/test/test_pathways.npy"
    checkpoint_path = "eval_data/Run_16/model_weights.pt"
    evaluate(bacteria_path, gene_families_path, pathways_path, checkpoint_path)
