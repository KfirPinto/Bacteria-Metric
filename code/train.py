import torch
from torch import nn, optim

def custom_loss(data_tensor, person_data, bacteria_data, model):
    """
    data_tensor: Original data tensor (bacteria_dim × gene_dim × people_dim)
    person_data: Tensor of person input data (people_dim × bacteria_dim × gene_dim)
    bacteria_data: Tensor of bacteria input data (bacteria_dim × people_dim × gene_dim)
    model: Instance of the SplitAutoencoder
    """
    total_loss = 0.0

    for person_idx in range(person_data.size(0)):
        # Get person and bacteria inputs
        person_input = person_data[person_idx]  # Shape: (bacteria_dim × gene_dim)
        bacteria_input = bacteria_data[:, person_idx, :]  # Shape: (bacteria_dim × gene_dim)

        # Forward pass
        reconstructed, person_embedding, bacteria_embedding = model(person_input, bacteria_input)

        # Calculate loss over all bacteria
        for bacteria_idx in range(bacteria_data.size(0)):
            gene_expression = data_tensor[bacteria_idx, :, person_idx]  # Original gene expression
            prediction = reconstructed[bacteria_idx]  # Reconstructed gene expression
            loss = torch.sum((gene_expression - prediction) ** 2)  # MSE for bacteria i in person j
            total_loss += loss

    return total_loss

def train_model(model, data_tensor, person_data, bacteria_data, num_epochs=100, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Compute loss
        loss = custom_loss(data_tensor, person_data, bacteria_data, model)
        loss.backward()
        optimizer.step()

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    return model

