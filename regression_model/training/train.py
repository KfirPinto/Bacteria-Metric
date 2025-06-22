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

def custom_loss(embeddings, pred, target, model=None, weight_decay=0.0):
    """
    Parameters:
        embeddings: low-dimensional representations (n, d, 2b)
        pred: pathway predictions (n, d, pathway_dim)
        target: pathway ground truth (n, d, pathway_dim)
        model: optional
        weight_decay: hyperparameter for weight decay (L2-regularization)
    """
    # Split H_ij to H_i(bacteria representation), H_j(sample representation)

    b = embeddings.shape[-1] // 2
    H_i = embeddings[..., :b]  # shape: (n, d, b)
    H_j = embeddings[..., b:]  # shape: (n, d, b)

    # ---- part A - MSE ----
    mse_loss = F.mse_loss(pred, target, reduction='mean')  # shape: scalar

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
        l2_reg = torch.tensor(0.0, device=target.device)
    
    return mse_loss, bacteria_consistency_loss, sample_consistency_loss, l2_reg

def train_model(model, train_loader, val_loader, device, num_epochs, learning_rate, name, lambda_weight=None):

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
        i = 0
        model.train()
        running_mse, running_bact, running_sample, running_total = 0.0, 0.0, 0.0, 0.0

        for gene_batch, pathway_batch in train_loader:
            i += 1
            print(f"Processing batch {i} in epoch {epoch+1}/{num_epochs}")  # Debugging line to track progress
            gene_batch = gene_batch.to(device)  
            pathway_batch = pathway_batch.to(device)  

            #print(f"train batch shape: {batch_tensor.shape}")    # Debugging line to check batch shape
            optimizer.zero_grad()

            # Forward pass
            embeddings, predictions = model(gene_batch)  # shape: (n, d, embedding_dim), (n, d, pathway_dim)

            mse_loss, bact_loss, sample_loss, wd = custom_loss(embeddings, predictions, pathway_batch, model=model, weight_decay=1e-4)
            if lambda_weight is not None:
                # Use balanced weights if provided
                mse_loss *= lambda_weight[0]
                bact_loss *= lambda_weight[1]
                sample_loss *= lambda_weight[2]
            total_loss = mse_loss + bact_loss + sample_loss + wd
                       
            # Backward pass
            total_loss.backward()
            optimizer.step()

            running_mse += mse_loss.item()
            running_bact += bact_loss.item()
            running_sample += sample_loss.item()
            running_total += total_loss.item()

        # Average loss per batch
        avg_train_mse = running_mse / len(train_loader) 
        avg_train_bact = running_bact / len(train_loader)
        avg_train_sample = running_sample / len(train_loader)
        avg_train_total = running_total / len(train_loader)

        model.eval()
        val_mse, val_bact, val_sample, val_total = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for gene_batch, pathway_batch  in val_loader:
                gene_batch = gene_batch.to(device)  
                pathway_batch = pathway_batch.to(device)  

                embeddings, predictions = model(gene_batch)
                mse_loss, bact_loss, sample_loss, wd = custom_loss(embeddings, predictions, pathway_batch, model=model, weight_decay=1e-4) 
                if lambda_weight is not None:
                    # Use balanced weights if provided
                    mse_loss *= lambda_weight[0]
                    bact_loss *= lambda_weight[1]
                    sample_loss *= lambda_weight[2]
                total_loss = mse_loss + bact_loss + sample_loss + wd

                val_mse += mse_loss.item()
                val_bact += bact_loss.item()
                val_sample += sample_loss.item()
                val_total += total_loss.item()

        # Average loss per batch
        avg_val_mse = val_mse / len(val_loader)
        avg_val_bact = val_bact / len(val_loader)
        avg_val_sample = val_sample / len(val_loader)
        avg_val_total = val_total / len(val_loader)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_mse_loss": avg_train_mse,
            "train_bact_loss": avg_train_bact,
            "train_sample_loss": avg_train_sample,
            "train_total_loss": avg_train_total,

            "val_mse_loss": avg_val_mse,
            "val_bact_loss": avg_val_bact,
            "val_sample_loss": avg_val_sample,
            "val_total_loss": avg_val_total,
        })

    wandb.finish()    
    return model