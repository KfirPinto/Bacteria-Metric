import torch
from torch import nn, optim
import torch.nn.functional as F
import wandb

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

    # Step 5: Normalize embeddings for cosine similarity
    H_j = F.normalize(H_j, dim=-1)
    H_k = F.normalize(H_k, dim=-1)
    other_H_k = F.normalize(other_H_k, dim=-1)

    # Step 6: Cosine similarity (dot product of unit vectors)
    sim_same = torch.sum(H_j * H_k, dim=-1)           # shape: (n, num_pairs)
    sim_diff = torch.sum(H_j * other_H_k, dim=-1)     # shape: (n, num_pairs)

    # Step 7: Convert to distance-like and compute ratio
    loss = (1 - sim_same) / (1 - sim_diff + 1e-8)     # avoid division by zero
    return loss.mean()

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
    H_j = H[j_idx, torch.arange(d).unsqueeze(1)]       # (d, num_pairs, b)
    H_k = H[k_idx, torch.arange(d).unsqueeze(1)]       # (d, num_pairs, b)

    # Step 4: Gather comparison bacterium (random) for the same sample pairs
    H_k_other = H[k_idx, rand_indices.unsqueeze(1)]    # (d, num_pairs, b)

    # Step 5: Normalize vectors along the embedding dimension
    H_j = F.normalize(H_j, dim=-1)
    H_k = F.normalize(H_k, dim=-1)
    H_k_other = F.normalize(H_k_other, dim=-1)

    # Step 6: Cosine similarity
    sim_same = torch.sum(H_j * H_k, dim=-1)            # (d, num_pairs)
    sim_diff = torch.sum(H_j * H_k_other, dim=-1)      # (d, num_pairs)

    # Step 7: Convert to "distance"
    loss = (1 - sim_same) / (1 - sim_diff + 1e-8)       # avoid divide-by-zero
    return loss.mean()

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

def custom_loss(X, X_reconstructed, latent, mu, logvar, model=None, weight_decay=0.0):
    """
    Parameters:
        X: original input tensor (n, d, gene_dim)
        X_reconstructed: model output (n, d, gene_dim)
        latent: encoded tensor (n, d, 2b)
        mu: mean of the latent distribution (n, d, 2b)
        logvar: log-variance of the latent distribution (n, d, 2b)
        model: optional
        weight_decay: hyperparameter for weight decay (L2-regularization)
    """

    n, d, gene_dim = X.shape
    b = latent.shape[-1] // 2

    # Split H_ij to H_i, H_j
    H_i = latent[..., :b]  # shape: (n, d, b)
    H_j = latent[..., b:]  # shape: (n, d, b)

    # ---- part A - mean MSE ----
    #MSE = F.mse_loss(X_reconstructed, X, reduction='mean')
    MSE = torch.norm(X - X_reconstructed, dim=-1).mean()
    
    # ---- part B - Bacteria consistency ----
    bacteria_consistency_loss = vectorize_bacteria_constituency_loss(H_i)

    # ---- part C - Sample consistency ----
    sample_consistency_loss = vectorize_sample_constituency_loss(H_j)

    # ---- Part D - Weight Decay (L2 regularization) ----
    l2_reg = 0.0
    if model is not None and weight_decay > 0:
        for param in model.parameters():
            if param.requires_grad:
                l2_reg += torch.sum(param ** 2)
        l2_reg = weight_decay * l2_reg
    else:
        l2_reg = torch.tensor(0.0, device=X.device)

    # ---- Part E - KL Divergence ----
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # ---- Total Loss ----
    return MSE, bacteria_consistency_loss, sample_consistency_loss, l2_reg, kl_div

def balanced_loss(loss_history, eps=1e-8):
    """
    Computes balanced weights for loss components from their values across multiple epochs.
    
    Args:
        loss_history (list or Tensor): shape (num_epochs, num_components), e.g., (10, 3)
        eps (float): small constant to avoid division by zero

    Returns:
        list of float: normalized inverse weights per component
    """
    
    loss_history = torch.tensor(loss_history) # shape (10, 3)

    # Mean loss per component across epochs → shape (3,)
    mean_losses = loss_history.mean(dim=0) 

    # Compute inverse weights 
    inverse_weights = 1.0 / (mean_losses + eps)  # shape (3,)

    # Normalize 
    normalized_weights = inverse_weights / inverse_weights.sum()  # shape (3,)

    return normalized_weights.tolist()  # Convert to list

def train_model(model, train_loader, device, num_epochs, learning_rate, name, lambda_weight=None, weight_decay=0.0):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize wandb
    wandb.init(
        project="SplitAutoencoder",
        config={
        "name": name,
        "epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": train_loader.batch_size if hasattr(train_loader, "batch_size") else None,
        })

    for epoch in range(num_epochs):

        model.train()
        running_recon, running_bact, running_sample, running_kl, running_total = 0.0, 0.0, 0.0, 0.0, 0.0

        for batch in train_loader:
            batch_tensor = batch[0].to(device)                    # Move data tensor to the same device as the model
            batch_tensor = batch_tensor.squeeze(0)         
            #print(f"train batch shape: {batch_tensor.shape}")    # Debugging line to check batch shape
            #print the max and min values of the batch tensor
            #print(f"train batch max: {batch_tensor.max()}, min: {batch_tensor.min()}")  # Debugging line to check values
            optimizer.zero_grad()
            
            # Forward pass
            encoded, decoded, mu, logvar = model(batch_tensor)

            recon_loss, bact_loss, sample_loss, wd, kl_loss = custom_loss(batch_tensor, decoded, encoded, mu, logvar, model=model, weight_decay=weight_decay)
            if lambda_weight is not None:
                # Use balanced weights if provided
                recon_loss *= lambda_weight[0]
                bact_loss *= lambda_weight[1]
                sample_loss *= lambda_weight[2]
            total_loss = recon_loss + bact_loss + sample_loss + wd + kl_loss

            # Backward pass
            total_loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
            optimizer.step()

            #for name, param in model.named_parameters():
                #print(name, param.data.norm())

            running_recon += recon_loss.item()
            running_bact += bact_loss.item()
            running_sample += sample_loss.item()
            running_kl += kl_loss.item()
            running_total += total_loss.item()

        avg_train_recon = running_recon / len(train_loader)
        avg_train_bact = running_bact / len(train_loader)
        avg_train_sample = running_sample / len(train_loader)
        avg_train_kl = running_kl / len(train_loader)
        avg_train_total = running_total / len(train_loader)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_recon_loss": avg_train_recon,
            "train_bact_loss": avg_train_bact,
            "train_sample_loss": avg_train_sample,
            "train_kl_loss": avg_train_kl,
            "train_total_loss": avg_train_total,
        })

    wandb.finish()    
    return model