import torch
from torch import nn, optim
import torch.nn.functional as F


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
    sample_consistency_loss = 0.0
    for i in range(n):
        for j in range(d):
            for k in range(j+1, d):  
                sample_consistency_loss += torch.norm(H_j[i, j] - H_j[i, k], p=2)

    # 3. bacteria consistency
    bacteria_consistency_loss = 0.0
    for j in range(d):
        for i in range(n):
            for k in range(i+1, n):
                bacteria_consistency_loss += torch.norm(H_i[i, j] - H_i[k, j], p=2)

    # normalize by amount of arguments at the summation
    sample_consistency_loss /= (n * d * (d-1) / 2)  # d choose 2 = d * (d-1)/2
    bacteria_consistency_loss /= (n * (n-1) / 2 * d)  # n choose 2 = n * (n-1)/2

    # combine all parts of loss function
    total_loss = recon_loss + sample_consistency_loss + bacteria_consistency_loss
    return total_loss


def train_model(model, data_tensor, num_epochs=100, learning_rate=0.001):
    """
    Training loop for the SplitAutoencoder model.

    Args:
    - model: the neural network model
    - data_tensor: input tensor of gene expressions
    - num_epochs: number of training epochs
    - learning_rate: optimizer learning rate
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()

        # Forward pass
        X = data_tensor
        X_encoded, X_reconstructed = model(X)

        # Compute the loss
        loss = custom_loss(X, X_reconstructed, X_encoded)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    return model

