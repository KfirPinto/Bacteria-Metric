import torch
from torch import nn, optim
import torch.nn.functional as F
import wandb
import optuna

def vectorize_sample_constituency_loss(H):
    n, d, b = H.shape
    
    # Step 1: Pre-select a different sample index for each sample
    random_offsets = torch.randint(1, n, (n,), device=H.device)
    random_indices = (torch.arange(n, device=H.device) + random_offsets) % n
    
    other_H = H[random_indices]

    # Step 3: Get upper-triangular indices for bacteria pairs (j < k)
    j_idx, k_idx = torch.triu_indices(d, d, offset=1, device=H.device)

    # Step 4: Gather bacteria pairs
    H_j = H[:, j_idx, :]           
    H_k = H[:, k_idx, :]           
    other_H_k = other_H[:, k_idx, :] 

    # Step 5: Normalize
    H_j = F.normalize(H_j, dim=-1, eps=1e-8)
    H_k = F.normalize(H_k, dim=-1, eps=1e-8)
    other_H_k = F.normalize(other_H_k, dim=-1, eps=1e-8)

    # Step 6: Cosine similarity
    sim_same = torch.sum(H_j * H_k, dim=-1)           
    sim_diff = torch.sum(H_j * other_H_k, dim=-1)     

    # Step 7: Loss calculation
    loss = (1 - sim_same) / (1 - sim_diff + 1e-8)     
    return loss.mean()

def vectorize_bacteria_constituency_loss(H):
    n, d, b = H.shape 

    rand_offsets = torch.randint(1, d, (d,), device=H.device)
    rand_indices = (torch.arange(d, device=H.device) + rand_offsets) % d
    
    j_idx, k_idx = torch.triu_indices(n, n, offset=1, device=H.device)
    
    H_j = H[j_idx, torch.arange(d, device=H.device).unsqueeze(1)]       
    H_k = H[k_idx, torch.arange(d, device=H.device).unsqueeze(1)]       
    H_k_other = H[k_idx, rand_indices.unsqueeze(1)]    

    H_j = F.normalize(H_j, dim=-1, eps=1e-8)
    H_k = F.normalize(H_k, dim=-1, eps=1e-8)
    H_k_other = F.normalize(H_k_other, dim=-1, eps=1e-8)

    sim_same = torch.sum(H_j * H_k, dim=-1)            
    sim_diff = torch.sum(H_j * H_k_other, dim=-1)      

    loss = (1 - sim_same) / (1 - sim_diff + 1e-8)       
    return loss.mean()

def custom_loss(X, X_reconstructed, latent, mu, logvar, model=None, weight_decay=0.0):
    n, d, gene_dim = X.shape
    b = latent.shape[-1] // 2

    H_i = latent[..., :b]  
    H_j = latent[..., b:]  

    # Part A - MSE
    MSE = torch.norm(X - X_reconstructed, dim=-1).mean()
    
    # Part B & C - Consistency
    bacteria_consistency_loss = vectorize_bacteria_constituency_loss(H_i)
    sample_consistency_loss = vectorize_sample_constituency_loss(H_j)

    # Part D - Weight Decay
    l2_reg = torch.tensor(0.0, device=X.device)
    if model is not None and weight_decay > 0:
        for param in model.parameters():
            if param.requires_grad:
                l2_reg += torch.sum(param ** 2)
        l2_reg = weight_decay * l2_reg

    # Part E - KL Divergence
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE, bacteria_consistency_loss, sample_consistency_loss, l2_reg, kl_div

def train_model(model, train_loader, val_loader, device, num_epochs, learning_rate, name, lambda_weight=None, weight_decay=0.0, trial=None):
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # --- WandB Initialization (FIXED) ---
    # reinit=True מבטיח פתיחת גרף חדש לכל ניסוי
    run = wandb.init(
        project="SplitAutoencoder_Tuning",
        name=name,
        reinit=True,
        group="optuna_optimization", # כדי שתוכל לסנן ב-dashboard
        config={
            "epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": train_loader.batch_size,
            "weight_decay": weight_decay,
            "lambda_weights": lambda_weight
        }
    )

    best_val_loss = float('inf')

    try:
        for epoch in range(num_epochs):
            # --- Training ---
            model.train()
            running_total = 0.0

            for batch in train_loader:
                batch_tensor = batch[0].to(device)
                batch_tensor = batch_tensor.squeeze(0)         
                
                optimizer.zero_grad()
                
                encoded, decoded, mu, logvar = model(batch_tensor)

                recon_loss, bact_loss, sample_loss, wd, kl_loss = custom_loss(batch_tensor, decoded, encoded, mu, logvar, model=model, weight_decay=weight_decay)
                
                # Apply weights
                if lambda_weight is not None:
                    recon_loss   *= lambda_weight[0]
                    bact_loss    *= lambda_weight[1]
                    sample_loss  *= lambda_weight[2]
                    
                total_loss = recon_loss + bact_loss + sample_loss + wd + kl_loss

                total_loss.backward()
                optimizer.step()

                running_total += total_loss.item()

            avg_train_total = running_total / len(train_loader)

            # --- Validation ---
            model.eval()
            val_running_total = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch_tensor = batch[0].to(device)
                    batch_tensor = batch_tensor.squeeze(0)

                    encoded, decoded, mu, logvar = model(batch_tensor)
                    
                    recon_loss, bact_loss, sample_loss, wd, kl_loss = custom_loss(batch_tensor, decoded, encoded, mu, logvar, model=model, weight_decay=weight_decay)
                    
                    if lambda_weight is not None:
                        recon_loss   *= lambda_weight[0]
                        bact_loss    *= lambda_weight[1]
                        sample_loss  *= lambda_weight[2]

                    total_val_loss = recon_loss + bact_loss + sample_loss + wd + kl_loss
                    val_running_total += total_val_loss.item()

            avg_val_total = val_running_total / len(val_loader)

            # Update best loss
            if avg_val_total < best_val_loss:
                best_val_loss = avg_val_total

            # Log to WandB
            wandb.log({
                "epoch": epoch,
                "train_total_loss": avg_train_total,
                "val_total_loss": avg_val_total,
                "best_val_loss": best_val_loss
            })

            # --- Optuna Pruning ---
            if trial:
                trial.report(avg_val_total, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

    finally:
        # CRITICAL: Ensure the run is closed so the next trial starts fresh
        run.finish()
    
    return model, best_val_loss