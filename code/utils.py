import torch

def save_embeddings(person_embeddings, bacteria_embeddings, person_path="person_embeddings.pt", bacteria_path="bacteria_embeddings.pt"):
    torch.save(person_embeddings, person_path)
    torch.save(bacteria_embeddings, bacteria_path)
    print(f"Embeddings saved to {person_path} and {bacteria_path}")

def load_embeddings(person_path="person_embeddings.pt", bacteria_path="bacteria_embeddings.pt"):
    person_embeddings = torch.load(person_path)
    bacteria_embeddings = torch.load(bacteria_path)
    return person_embeddings, bacteria_embeddings
