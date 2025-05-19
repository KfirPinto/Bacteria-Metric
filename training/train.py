import torch
from torch import nn, optim
import torch.nn.functional as F
import wandb

def sample_constituency_loss(H):
    n, d, b = H.shape
    total_loss = 0.0
    count = 0

    # Pre-generate a different sample index for each i
    random_offsets = torch.randint(1, n, (n,))
    random_indices = (torch.arange(n) + random_offsets) % n  # shape: (n,)
    random_indices = random_indices.to(H.device)

    for i in range(n):
        sample = H[i]                 # (d, b)
        other = H[random_indices[i]]  # (d, b)

        for j in range(d):
            for k in range(j + 1, d):
                dist_same = F.pairwise_distance(sample[j], sample[k], p=2)

                # Distance to bacterium j in a different sample
                dist_diff = F.pairwise_distance(sample[j], other[k], p=2)

                ratio = dist_same / (dist_diff + 1e-8)
                total_loss += ratio
                count += 1

    return total_loss / count

def vectorize_sample_constituency_loss(H):
    n, d, b = H.shape

    # Step 1: Pre-select a different sample index for each sample
    random_offsets = torch.randint(1, n, (n,))
    random_indices = (torch.arange(n) + random_offsets) % n
    random_indices = random_indices.to(H.device)  # shape: (n,)

    # Step 2: Gather the corresponding "other" samples → shape: (n, d, b)
    other_H = H[random_indices]

    # Step 3: Get upper-triangular indices for bacteria pairs (j < k)
    j_idx, k_idx = torch.triu_indices(d, d, offset=1)  # shape: (num_pairs,)

    # Step 4: Gather bacteria pairs for all samples
    H_j = H[:, j_idx, :]           # shape: (n, num_pairs, b)
    H_k = H[:, k_idx, :]           # same

    other_H_k = other_H[:, k_idx, :]  # same as H_k but from different sample

    # Step 5: Compute distances (batch-wise pairwise distances)
    dist_same = torch.norm(H_j - H_k, dim=-1)         # shape: (n, num_pairs)
    dist_diff = torch.norm(H_j - other_H_k, dim=-1)   # shape: (n, num_pairs)

    # Step 6: Avoid division by zero
    dist_diff = torch.where(dist_diff == 0, torch.ones_like(dist_diff) * 1e-8, dist_diff)

    # Step 7: Compute normalized loss and return average
    loss = (dist_same / dist_diff).mean()
    return loss

def bacteria_constituency_loss(H):
    """
    Computes a loss item intended to enforce that representations
    of the same bacterium across different samples are closer than representations
    of different bacteria.

    Parameters:
        H (torch.Tensor): shape (num_samples, num_bacteria, embedding_dim)
                          low dimensional representations of bacteria

    Returns:
        torch.Tensor: scalar loss value
    """
    num_samples, num_bacteria, embedding_dim = H.shape

    # Choose a different (random) bacterium index for each bacterium
    random_offsets = torch.randint(low=1, high=num_bacteria, size=(num_bacteria,))
    random_indices = (torch.arange(num_bacteria) + random_offsets) % num_bacteria
    random_indices = random_indices.to(H.device)  # Ensure device compatibility

    total_loss = 0.0
    num_pairs = 0

    for b in range(num_bacteria):
        # Representations of bacterium `b` across all samples
        same_bacterium = H[:, b, :]  # shape: (num_samples, embedding_dim)

        # Representations of a *random other bacterium* across all samples
        other_bacterium = H[:, random_indices[b], :]  # shape: (num_samples, embedding_dim)

        # Compare all sample pairs j < k
        for j in range(num_samples):
            for k in range(j + 1, num_samples):
                dist_same = F.pairwise_distance(same_bacterium[j], same_bacterium[k], p=2)
                dist_diff = F.pairwise_distance(same_bacterium[j], other_bacterium[k], p=2)

                ratio = dist_same / (dist_diff + 1e-8)  # Add epsilon to prevent div by zero
                total_loss += ratio
                num_pairs += 1

    return total_loss / num_pairs

def vectorize_bacteria_constituency_loss(H):
    n, d, b = H.shape  # samples, bacteria, embedding_dim

    # Step 1: For each bacterium, choose a different random bacterium
    rand_offsets = torch.randint(1, d, (d,))
    rand_indices = (torch.arange(d) + rand_offsets) % d
    rand_indices = rand_indices.to(H.device)

    # Step 2: Get upper-triangular indices for sample pairs (j < k)
    j_idx, k_idx = torch.triu_indices(n, n, offset=1)  # shape: (num_pairs,)
    num_pairs = j_idx.shape[0]  # number of sample pairs

    # Step 3: Gather same-bacterium representations
    # shape: (d, num_pairs, b)
    H_j = H[j_idx, torch.arange(d).unsqueeze(1)]       # (d, num_pairs, b)
    H_k = H[k_idx, torch.arange(d).unsqueeze(1)]       # (d, num_pairs, b)

    # Step 4: Gather comparison bacterium (random) for the same sample pairs
    H_k_other = H[k_idx, rand_indices.unsqueeze(1)]    # (d, num_pairs, b)

    # Step 5: Compute distances
    dist_same = torch.norm(H_j - H_k, dim=-1)          # (d, num_pairs)
    dist_diff = torch.norm(H_j - H_k_other, dim=-1)    # (d, num_pairs)

    # Step 6: Avoid division by zero
    dist_diff = torch.where(dist_diff == 0, torch.ones_like(dist_diff) * 1e-8, dist_diff)

    # Step 7: Compute loss
    ratio = dist_same / dist_diff                      # (d, num_pairs)
    loss = ratio.mean()
    return loss

def vectorize_bacteria_constituency_loss_strict(H):
    """
    More challenging bacteria constituency loss:
    For each bacterium i and sample pair (j, k),
    compare it to a different (random) bacterium per pair.

    Parameters:
        H (torch.Tensor): shape (n, d, b)

    Returns:
        torch.Tensor: scalar loss
    """
    n, d, b = H.shape  # samples, bacteria, embedding_dim

    # Step 1: Get all upper-triangular sample pair indices (j < k)
    j_idx, k_idx = torch.triu_indices(n, n, offset=1)  # shape: (num_pairs,)
    num_pairs = j_idx.shape[0]

    # Step 2: For each bacterium and sample pair, pick a different bacterium (≠ i)
    rand_offsets = torch.randint(1, d, (d, num_pairs), device=H.device)
    rand_other_indices = (torch.arange(d, device=H.device).unsqueeze(1) + rand_offsets) % d  # shape: (d, num_pairs)

    # Step 3: Gather same-bacterium embeddings
    # For each bacterium i: gather all H[j,k] sample pairs
    H_j = H[j_idx, torch.arange(d, device=H.device).unsqueeze(1)]  # (d, num_pairs, b)
    H_k = H[k_idx, torch.arange(d, device=H.device).unsqueeze(1)]  # (d, num_pairs, b)

    # Step 4: Gather random different bacterium from sample k (different per pair)
    H_k_other = H[k_idx.unsqueeze(0), rand_other_indices]  # shape: (d, num_pairs, b)

    # Step 5: Compute distances
    dist_same = torch.norm(H_j - H_k, dim=-1)          # (d, num_pairs)
    dist_diff = torch.norm(H_j - H_k_other, dim=-1)    # (d, num_pairs)

    # Avoid division by zero
    dist_diff = torch.where(dist_diff == 0, torch.ones_like(dist_diff) * 1e-8, dist_diff)

    # Final loss
    loss = (dist_same / dist_diff).mean()
    return loss


def custom_loss(X, X_reconstructed, H_ij):
    """
    X: original input tensor (n, d, gene_dim)
    X_reconstructed: model output (n, d, gene_dim)
    H_ij: encoded tensor (n, d, 2b)
    """

    n, d, gene_dim = X.shape
    b = H_ij.shape[-1] // 2

    # Split H_ij to H_i, H_j
    H_i = H_ij[..., :b]  # shape: (n, d, b)
    H_j = H_ij[..., b:]  # shape: (n, d, b)

    # ---- part A - Normalized reconstruction error ----

    # Calculate mean over samples (axis=0) per bacteria
    X_mean = X.mean(dim=0, keepdim=True)  # shape: (1, d, gene_dim)

    # Numerator: L2 norm of reconstruction error
    numerator = torch.norm(X - X_reconstructed, dim=-1)  # shape: (n, d)

    # Denominator: L2 norm of difference from mean
    denominator = torch.norm(X - X_mean, dim=-1)  # shape: (n, d)

    # Avoid division by zero
    denominator = torch.where(denominator == 0, torch.ones_like(denominator) * 1e-8, denominator)

    # Final normalized reconstruction loss
    #recon_loss = (numerator / denominator).mean()
    recon_loss = torch.clamp((numerator / denominator), max=1e3).mean()

    # ---- part B - Bacteria consistency ----
    bacteria_consistency_loss = vectorize_bacteria_constituency_loss_strict(H_i)

    # ---- part C- Sample consistency ----
    sample_consistency_loss = vectorize_sample_constituency_loss(H_j)
    
    return recon_loss, bacteria_consistency_loss, sample_consistency_loss

def train_model(model, train_loader, val_loader, device, num_epochs, learning_rate, name):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize wandb
    wandb.init(project="SplitAutoencoder", config={
        "name": name,
        "epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": train_loader.batch_size if hasattr(train_loader, "batch_size") else None
    })

    for epoch in range(num_epochs):

        model.train()
        running_recon, running_bact, running_sample, running_total = 0.0, 0.0, 0.0, 0.0

        for batch in train_loader:
            batch_tensor = batch[0].to(device)                    # Move data tensor to the same device as the model
            batch_tensor = batch_tensor.squeeze(0)         
            print(f"train batch shape: {batch_tensor.shape}")  # Debugging line to check batch shape
            optimizer.zero_grad()

            # Forward pass
            encoded, decoded = model(batch_tensor) 

            recon_loss, bact_loss, sample_loss = custom_loss(batch_tensor, decoded, encoded)
            total_loss = recon_loss + bact_loss + sample_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

            running_recon += recon_loss.item()
            running_bact += bact_loss.item()
            running_sample += sample_loss.item()
            running_total += total_loss.item()

        avg_train_recon = running_recon / len(train_loader)
        avg_train_bact = running_bact / len(train_loader)
        avg_train_sample = running_sample / len(train_loader)
        avg_train_total = running_total / len(train_loader)

        model.eval()
        val_recon, val_bact, val_sample, val_total = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch_tensor = batch[0].to(device)                    
                batch_tensor = batch_tensor.squeeze(0)   
                encoded, decoded = model(batch_tensor)
                recon_loss, bact_loss, sample_loss = custom_loss(batch_tensor, decoded, encoded)  
                total_loss = recon_loss + bact_loss + sample_loss

                val_recon += recon_loss.item()
                val_bact += bact_loss.item()
                val_sample += sample_loss.item()
                val_total += total_loss.item()

        avg_val_recon = val_recon / len(val_loader)
        avg_val_bact = val_bact / len(val_loader)
        avg_val_sample = val_sample / len(val_loader)
        avg_val_total = val_total / len(val_loader)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_recon_loss": avg_train_recon,
            "train_bact_loss": avg_train_bact,
            "train_sample_loss": avg_train_sample,
            "train_total_loss": avg_train_total,

            "val_recon_loss": avg_val_recon,
            "val_bact_loss": avg_val_bact,
            "val_sample_loss": avg_val_sample,
            "val_total_loss": avg_val_total,

            "epoch": epoch + 1
        })

    wandb.finish()    
    return model