import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score

def parse_full_taxonomy_string(tax_string):
    """מפרק מחרוזת טקסונומיה ומחלץ את רמת ה-Genus."""
    parts = tax_string.split('|')
    levels = {'genus': '', 'species': ''}
    
    for part in parts:
        if part.startswith('g__'):
            levels['genus'] = part.split('__')[1]
        elif part.startswith('s__'):
            levels['species'] = part.split('__')[1]
            
    # יצירת מפתח וזיהוי Genus
    genus = levels['genus']
    species = levels['species']
    
    if genus and species:
        lookup_key = f"g__{genus}.s__{species}"
        return lookup_key, genus
    return None, None

def build_genus_mapping(full_taxonomy_path):
    """בונה מילון: שם חיידק -> Genus."""
    print(f"🔹 Loading taxonomy from: {full_taxonomy_path}")
    full_strings = np.load(full_taxonomy_path, allow_pickle=True)
    
    mapping = {}
    for tax_str in full_strings:
        key, genus = parse_full_taxonomy_string(tax_str)
        if key and genus:
            mapping[key] = genus
    return mapping

def evaluate_brothers(embedding_path, test_names_path, taxonomy_ref_path, save_dir):
    print("\n🚀 Starting 'Are Brothers?' Evaluation (Genus Level Resolution)")
    
    # 1. טעינת נתונים
    embeddings = np.load(embedding_path)
    test_names = np.load(test_names_path, allow_pickle=True)
    genus_map = build_genus_mapping(taxonomy_ref_path)
    
    # 2. שיוך Genus לכל נקודה בטסט
    valid_indices = []
    genus_labels = []
    names_clean = []
    
    for i, name in enumerate(test_names):
        genus = genus_map.get(name)
        # ניסיון חילוץ נוסף אם השם בטסט הוא הפורמט הארוך
        if not genus and name.startswith('k__'):
             _, genus = parse_full_taxonomy_string(name)
             
        if genus:
            valid_indices.append(i)
            genus_labels.append(genus)
            names_clean.append(name)
    
    X = embeddings[valid_indices]
    y_genus = np.array(genus_labels)
    names_clean = np.array(names_clean)
    
    print(f"🔹 Analyzed {len(X)} bacteria with valid Genus info.")

    # 3. מציאת השכן הקרוב ביותר (לא כולל את עצמו)
    # k=2 כי הראשון הוא תמיד החיידק עצמו (מרחק 0)
    nbrs = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # 4. בדיקת "אחים"
    correct_count = 0
    total_count = len(X)
    
    print("\n🔍 Checking Neighbors:")
    # רשימה לשמירת דוגמאות
    examples = []
    
    for i in range(total_count):
        # האינדקס של השכן הכי קרוב (שהוא לא אני)
        neighbor_idx = indices[i][1] 
        
        my_genus = y_genus[i]
        neighbor_genus = y_genus[neighbor_idx]
        
        is_brother = (my_genus == neighbor_genus)
        if is_brother:
            correct_count += 1
            
        # שמירת דוגמאות להדפסה
        if i < 5: # נשמור 5 דוגמאות ראשונות
            status = "✅ Brother" if is_brother else "❌ Stranger"
            examples.append(f"{status}: {names_clean[i]} ({my_genus}) <--> Neighbor: {names_clean[neighbor_idx]} ({neighbor_genus})")

    accuracy = correct_count / total_count
    
    # 5. הדפסת תוצאות
    for ex in examples:
        print(ex)
        
    print(f"\n{'='*40}")
    print(f"🧬 Are Brothers? Accuracy (Genus Level): {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*40}")
    
    # שמירת התוצאה לקובץ טקסט
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/brothers_score.txt", "w") as f:
        f.write(f"Genus Level Accuracy: {accuracy:.4f}\n")

if __name__ == "__main__":
    # --- הגדרות (Run 5) ---
    BASE_DIR = "/home/dsi/pintokf/Projects/Microbium/Bacteria-Metric"
    RUN_DIR = f"{BASE_DIR}/eval_results/HMP_Kfir/Run_0"
    
    evaluate_brothers(
        embedding_path=f"{RUN_DIR}/test_tensor_embeddings.npy",
        test_names_path=f"{RUN_DIR}/test_bacteria.npy",
        taxonomy_ref_path=f"{RUN_DIR}/bacteria_names_full_taxonomy.npy",
        save_dir=f"{RUN_DIR}/plots_brothers_eval"
    )