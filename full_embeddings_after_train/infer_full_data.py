import torch
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config
from variational_autoencoder.data_utils import load_data_tensor, normalize_tensor
from variational_autoencoder.training.model import SplitVAE

def generate_full_embeddings():
    print("Loading full dataset...")
    data = load_data_tensor(config.GENE_FAMILIES_TENSOR_PATH)
    data_norm = normalize_tensor(data)
    data_tensor = torch.tensor(data_norm, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # טוענים את המודל שכבר אימנו ב-Run 5
    model_path = f"{config.EVAL_OUTPUT_DIR}/split_autoencoder.pt"
    print(f"Loading model from: {model_path}")
    
    gene_dim = data.shape[-1]
    model = SplitVAE(gene_dim=gene_dim, embedding_dim=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Running inference on full dataset...")
    with torch.no_grad():
        z, _, _, _ = model(data_tensor.to(device))
        
        # לוקחים רק את החלק של זהות החיידק
        b = z.shape[-1] // 2
        H_i = z[..., :b]
        
        # ממוצע כדי לקבל וקטור אחד לכל חיידק
        full_embeddings = H_i.mean(dim=0).cpu().numpy()

    # שומרים את הקובץ הגדול
    output_path = os.path.join("/home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/full_embeddings_after_train", "full_embeddings_run5.npy")
    np.save(output_path, full_embeddings)
    print(f"Saved full embeddings to: {output_path}")

if __name__ == "__main__":
    generate_full_embeddings()