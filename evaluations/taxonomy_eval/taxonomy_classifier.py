import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def parse_full_taxonomy_string(tax_string):
    """
    מפרק מחרוזת טקסונומיה מלאה למילון רמות ולשם מקוצר.
    לדוגמה: 'k__Bacteria|...|g__Lactobacillus|s__Lactobacillus_gasseri'
    """
    parts = tax_string.split('|')
    levels = {'kingdom': '', 'phylum': '', 'class': '', 'order': '', 'family': '', 'genus': '', 'species': ''}
    
    for part in parts:
        for level in levels:
            if part.startswith(level[0] + '__'):
                levels[level] = part.split('__')[1]
    
    # יצירת המפתח לחיפוש (כפי שמופיע בקובץ הטסט: g__Genus.s__Species)
    genus = levels['genus']
    species = levels['species']
    
    # טיפול במקרים של שמות חסרים
    if genus and species:
        lookup_key = f"g__{genus}.s__{species}"
    else:
        lookup_key = None
        
    return lookup_key, levels

def build_taxonomy_mapping(full_taxonomy_path):
    """
    בונה מילון מיפוי: שם מקוצר -> רמות טקסונומיה.
    """
    print(f"🔹 Loading taxonomy reference from: {full_taxonomy_path}")
    full_strings = np.load(full_taxonomy_path, allow_pickle=True)
    
    mapping = {}
    for tax_str in full_strings:
        key, levels = parse_full_taxonomy_string(tax_str)
        if key:
            mapping[key] = levels
            
    print(f"🔹 Built taxonomy mapping for {len(mapping)} bacteria.")
    return mapping

def plot_confusion_matrix(y_true, y_pred, labels, level, accuracy, save_dir):
    """מצייר ושומר את מטריצת הבלבול."""
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_labels, yticklabels=unique_labels, 
                cbar=False)
    
    plt.title(f"Nearest Neighbor Confusion Matrix ({level})\nOverall Accuracy: {accuracy:.3f}", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/confusion_matrix_{level}.png", dpi=300)
        print(f"📷 Confusion Matrix saved to: {save_dir}/confusion_matrix_{level}.png")
    plt.show()

def evaluate_classification(embedding_path, test_names_path, taxonomy_ref_path, level='phylum', metric='euclidean', save_dir="classifier_results"):
    print(f"\n🚀 Starting Classification Evaluation for Level: {level.upper()}")
    
    # 1. Load Embeddings and Names
    embeddings = np.load(embedding_path)
    test_names = np.load(test_names_path, allow_pickle=True)
    
    # 2. Build Mapping and Resolve Labels
    tax_mapping = build_taxonomy_mapping(taxonomy_ref_path)
    
    valid_indices = []
    y_raw = []
    missing_count = 0
    
    for i, name in enumerate(test_names):
        info = tax_mapping.get(name)
        if not info and name.startswith('k__'):
             _, info = parse_full_taxonomy_string(name)
        
        if info and info[level]: 
            valid_indices.append(i)
            y_raw.append(info[level])
        else:
            missing_count += 1

    if len(valid_indices) == 0:
        print("❌ Error: No valid labels found!")
        return

    X = embeddings[valid_indices]
    y_raw = np.array(y_raw)
    
    print(f"🔹 Matched {len(X)} samples (filtered {missing_count} missing/unmapped).")
    
    # 3. 1-Nearest Neighbor Classification
    clf = KNeighborsClassifier(n_neighbors=1, metric=metric)
    
    print("🔹 Running Nearest Neighbor Classifier...")
    
    # ★★★ התיקון הקריטי כאן ★★★
    # במקום מספר שלם, אנו מעבירים אובייקט LeaveOneOut מפורש
    if len(X) > 500:
        cv_strategy = 10  # לדאטה גדול נשתמש ב-10-Fold (ברירת מחדל Stratified)
        print("   Using: 10-Fold Cross Validation")
    else:
        cv_strategy = LeaveOneOut() # לדאטה קטן נשתמש ב-Leave-One-Out אמיתי
        print(f"   Using: Leave-One-Out Cross Validation (on {len(X)} samples)")
    
    try:
        y_pred = cross_val_predict(clf, X, y_raw, cv=cv_strategy, n_jobs=-1)
    except ValueError as e:
        print(f"⚠️ Standard CV failed: {e}")
        print("   Switching to non-stratified KFold as fallback...")
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=min(len(X), 10), shuffle=True, random_state=42)
        y_pred = cross_val_predict(clf, X, y_raw, cv=kf, n_jobs=-1)

    # 4. Metrics
    acc = accuracy_score(y_raw, y_pred)
    print(f"\n🏆 Overall Accuracy ({level}): {acc:.4f} ({acc*100:.2f}%)")
    
    # Report
    print(classification_report(y_raw, y_pred, zero_division=0))
    
    # 5. Plot
    plot_confusion_matrix(y_raw, y_pred, sorted(list(set(y_raw))), level, acc, save_dir)

if __name__ == "__main__":
    # --- הגדרות הנתיבים (Run 5) ---
    BASE_DIR = "/home/dsi/pintokf/Projects/Microbium/Bacteria-Metric"
    RUN_DIR = f"{BASE_DIR}/eval_results/HMP_Kfir/Run_5"
    
    EMBEDDING_FILE = f"{RUN_DIR}/test_tensor_embeddings.npy"
    
    # קובץ השמות של הטסט (מכיל שמות מקוצרים)
    TEST_NAMES_FILE = f"{RUN_DIR}/test_bacteria.npy"
    
    # קובץ הרפרנס המלא (המילון) - וודא שהנתיב הזה נכון!
    TAXONOMY_REF_FILE = f"{RUN_DIR}/bacteria_names_full_taxonomy.npy"
    
    OUTPUT_DIR = f"{RUN_DIR}/plots_taxonomy_eval"

    # הרצה על רמת ה-Phylum
    evaluate_classification(
        EMBEDDING_FILE, 
        TEST_NAMES_FILE, 
        TAXONOMY_REF_FILE, 
        level='phylum', 
        metric='euclidean', 
        save_dir=OUTPUT_DIR
    )