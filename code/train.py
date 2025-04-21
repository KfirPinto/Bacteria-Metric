import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

    # 1. reconstruction
    recon_loss = F.mse_loss(X_reconstructed, X, reduction='mean')  # MSE mean

    # 2. sample consistency 
    # H_j: (n, d, b)
    H_j_diff = H_j.unsqueeze(2) - H_j.unsqueeze(1)  # shape (n, d, d, b)
    H_j_dist = torch.norm(H_j_diff, dim=-1)  # shape (n, d, d)
    # Keep only upper triangle (no duplicate pairs, no self-pairs)
    mask = torch.triu(torch.ones(d, d, device=H_j.device), diagonal=1)
    sample_consistency_loss = (H_j_dist * mask).sum() / (n * d * (d - 1) / 2)
    """
    for i in range(n):
        for j in range(d):
            for k in range(j+1, d):  
                sample_consistency_loss += torch.norm(H_j[i, j] - H_j[i, k], p=2)
    """

    # 3. bacteria consistency
    # H_i: (n, d, b)
    H_i_diff = H_i.unsqueeze(1) - H_i.unsqueeze(0)  # (n, n, d, b)
    H_i_dist = torch.norm(H_i_diff, dim=-1)  # (n, n, d)
    mask = torch.triu(torch.ones(n, n, device=H_i.device), diagonal=1)
    bacteria_consistency_loss = (H_i_dist * mask.unsqueeze(-1)).sum() / (n * (n - 1) / 2 * d)
    """
    for j in range(d):
        for i in range(n):
            for k in range(i+1, n):
                bacteria_consistency_loss += torch.norm(H_i[i, j] - H_i[k, j], p=2)
    """

    # normalize by amount of arguments at the summation
    sample_consistency_loss /= (n * d * (d-1) / 2)  # d choose 2 = d * (d-1)/2
    bacteria_consistency_loss /= (n * (n-1) / 2 * d)  # n choose 2 = n * (n-1)/2

    # combine all parts of loss function
    total_loss = recon_loss + sample_consistency_loss + bacteria_consistency_loss
    return total_loss


def train_model(model, train_loader, val_loader, device, num_epochs=100, learning_rate=0.001):
    """
    Training loop for the SplitAutoencoder model.

    Args:
    - model: the neural network model
    - train_loader: DataLoader for the training data
    - val loader: DataLoader for the validation data 
    - num_epochs: number of training epochs
    - learning_rate: optimizer learning rate
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch_tensor = batch.to(device)              # Move data tensor to the same device as the model
            batch_tensor = batch_tensor.squeeze(0)         
            print(f"train batch shape: {batch_tensor.shape}")  # Debugging line to check batch shape
            optimizer.zero_grad()

            # Forward pass
            X_encoded, X_decoded = model(batch_tensor) 

            # Compute the loss
            loss = custom_loss(batch_tensor, X_decoded, X_encoded)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data in val_loader:
                X_tensor = data.to(device)  
                X_tensor = X_tensor.squeeze(0) 
                print(f"val batch shape: {X_tensor.shape}")
                X_encoded, X_decoded = model(X_tensor)
                loss = custom_loss(X_tensor, X_decoded, X_encoded)
                val_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, " 
              f"Validatation Loss: {avg_val_loss:.4f}") 
        
    plot_losses(train_losses, val_losses)

    return model

# Plotting function to visualize train and validation losses
def plot_losses(train_losses, val_losses, filename="loss_plot.png"):

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss per Epoch')
    plt.legend()
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()