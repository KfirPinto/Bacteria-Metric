import os
import sys
import optuna
import torch
import wandb
import numpy as np
from sklearn.model_selection import KFold
from torch import optim
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import cdist

# הגדרת נתיבים לייבוא קבצים מהתיקייה הראשית
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import config
from variational_autoencoder.data_utils import load_data_tensor, load_metadata, normalize_tensor
from variational_autoencoder.preprocess import shuffle_bacteria
from variational_autoencoder.training.model import SplitVAE
from variational_autoencoder.training.dataset import create_dataloaders
from variational_autoencoder.training.train import custom_loss

# --- פונקציית עזר לחישוב פירסון ב-3 שיטות שונות ---
def calculate_val_pearson_multi(model, val_loader, val_pathways, device):
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        # הרצה במנות (Batches) למניעת קריסת זיכרון
        for batch in val_loader:
            batch_tensor = batch[0].to(device).squeeze(0)
            encoded, _, _, _ = model(batch_tensor)
            
            # לוקחים את החצי הראשון (חיידק) וממצעים על הדגימות
            b = encoded.shape[-1] // 2
            bacteria_emb = encoded[..., :b].mean(dim=0).cpu().numpy() 
            all_embeddings.append(bacteria_emb)
            
        # איחוד למטריצה אחת גדולה
        full_bacteria_emb = np.concatenate(all_embeddings, axis=0)
        
        # --- 1. הכנת ה-Ground Truth (מסלולים) ---
        # משתמשים ב-Cosine למסלולים (זה הסטנדרט)
        pathway_sim = cosine_similarity(val_pathways)
        n = pathway_sim.shape[0]
        triu_indices = np.triu_indices(n, k=1) # לוקחים רק את המשולש העליון
        path_flat = pathway_sim[triu_indices]

        # --- 2. חישוב דמיון ב-3 מטריקות שונות ---
        pearsons = []

        # שיטה א': Cosine Similarity (הקלאסי)
        try:
            emb_sim_cos = cosine_similarity(full_bacteria_emb)
            corr_cos, _ = pearsonr(emb_sim_cos[triu_indices], path_flat)
            pearsons.append(corr_cos)
        except: pearsons.append(0)

        # שיטה ב': Euclidean (L2) Similarity
        try:
            emb_dist_l2 = euclidean_distances(full_bacteria_emb)
            emb_sim_l2 = 1 / (1 + emb_dist_l2) # המרה ממרחק לדמיון
            corr_l2, _ = pearsonr(emb_sim_l2[triu_indices], path_flat)
            pearsons.append(corr_l2)
        except: pearsons.append(0)

        # שיטה ג': Mahalanobis Similarity (הנשק הסודי מהדו"ח)
        try:
            # חישוב מטריצת השונות המשותפת ההופכית
            # מוסיפים רעש אפסי לאלכסון (1e-6) כדי למנוע קריסה מתמטית (Singular Matrix)
            cov_matrix = np.cov(full_bacteria_emb.T) + np.eye(full_bacteria_emb.shape[1]) * 1e-6
            VI = np.linalg.pinv(cov_matrix)
            
            emb_dist_mah = cdist(full_bacteria_emb, full_bacteria_emb, metric='mahalanobis', VI=VI)
            emb_sim_mah = 1 / (1 + emb_dist_mah)
            
            corr_mah, _ = pearsonr(emb_sim_mah[triu_indices], path_flat)
            pearsons.append(corr_mah)
        except: pearsons.append(0)

        # --- 3. בחירת המנצח ---
        # אנחנו לוקחים את הציון הגבוה ביותר מבין ה-3
        pearsons = [p if not np.isnan(p) else -1.0 for p in pearsons]
        max_pearson = max(pearsons)
        
        return max_pearson

def train_one_fold(model, train_loader, val_loader, val_pathways, device, num_epochs, learning_rate, fold_idx, trial_number, lambda_weight, weight_decay, trial):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # משתנה לשמירת הציון הכי טוב ב-Fold הזה
    best_stable_pearson = -1.0 
    
    # הגנה: לא שופטים מודל לפני שהתייצב (אפוק 40)
    min_epoch_to_consider = 40 
    
    for epoch in range(num_epochs):
        # --- שלב האימון ---
        model.train()
        running_train_loss = 0.0
        
        for batch in train_loader:
            batch_tensor = batch[0].to(device).squeeze(0)
            optimizer.zero_grad()
            encoded, decoded, mu, logvar = model(batch_tensor)
            recon, bact, sample, wd, kl = custom_loss(batch_tensor, decoded, encoded, mu, logvar, model=model, weight_decay=weight_decay)
            
            if lambda_weight:
                loss = recon * lambda_weight[0] + bact * lambda_weight[1] + sample * lambda_weight[2] + wd + kl
            else:
                loss = recon + bact + sample + wd + kl
            
            # הגנה מקריסה: אם ה-Loss מתפוצץ, עוצרים מיד
            if torch.isnan(loss): 
                raise ValueError("NaN Loss detected")

            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        # --- שלב הבדיקה (Pearson) ---
        current_pearson = calculate_val_pearson_multi(model, val_loader, val_pathways, device)
        
        # עדכון השיא (רק אם עברנו את שלב החימום)
        if epoch >= min_epoch_to_consider:
            if current_pearson > best_stable_pearson:
                best_stable_pearson = current_pearson
        
        if wandb.run is not None:
            wandb.log({
                f"trial_{trial_number}_fold_{fold_idx}_pearson": current_pearson,
                f"trial_{trial_number}_fold_{fold_idx}_train_loss": avg_train_loss,
                "epoch": epoch
            })

        # --- מנגנון ה-Pruning (חיתוך ניסויים כושלים) ---
        if trial:
            # 1. בדיקת שפיות: אם ה-Loss בשמיים באפוק 20, אין טעם להמשיך
            if epoch == 20 and avg_train_loss > 1000: 
                 raise optuna.exceptions.TrialPruned("Loss too high")
            
            # 2. בדיקת איכות: אם הפירסון נמוך משמעותית מאחרים (רק אחרי שהתייצב)
            if epoch > min_epoch_to_consider:
                trial.report(current_pearson, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

    # אם המודל סיים אבל היה גרוע מאוד, מחזירים את הציון האחרון
    if best_stable_pearson == -1.0:
        return current_pearson

    return best_stable_pearson

def objective(trial):
    # --- 1. הגדרת טווחי החיפוש (ממוקדים) ---
    embedding_dim = trial.suggest_categorical("embedding_dim", [24, 32, 48, 64]) 
    learning_rate = trial.suggest_float("learning_rate", 5e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    num_epochs = 100
    k_folds = 5

    # --- 2. הכנת הדאטה ---
    data = load_data_tensor(config.GENE_FAMILIES_TENSOR_PATH)
    unannotated_data = load_data_tensor(config.GENE_FAMILIES_COMPLEMENTRARY_TENSOR_PATH)
    pathways_tensor = load_data_tensor(config.PATHWAYS_TENSOR_PATH)
    
    samples, bacteria, unannotated_bacteria, _ = load_metadata(
        config.SAMPLE_LIST_PATH, config.BACTERIA_LIST_PATH,
        config.UNANNOTATED_BACTERIA_LIST_PATH, config.GENE_LIST_PATH
    )

    # נרמול
    data_norm = normalize_tensor(data)
    pathways_norm = normalize_tensor(pathways_tensor)
    unannotated_norm = normalize_tensor(unannotated_data)
    
    data_tensor = torch.tensor(data_norm, dtype=torch.float32)
    pathways_tensor = torch.tensor(pathways_norm, dtype=torch.float32)
    unannotated_tensor = torch.tensor(unannotated_norm, dtype=torch.float32)

    # ערבוב (Seed קבוע!)
    torch.manual_seed(42)
    perm = torch.randperm(data_tensor.size(1))
    data_shuffled = data_tensor[:, perm, :]
    pathways_shuffled = pathways_tensor[:, perm, :]
    
    # --- 3. החלוקה הקריטית (85% לפיתוח) ---
    num_bacteria = data_shuffled.shape[1]
    idx_test_start = int(0.85 * num_bacteria) 
    
    # זה ה-Dev Set שלנו (כולל Train + Val לשעבר)
    dev_data = data_shuffled[:, :idx_test_start, :]
    dev_pathways = pathways_shuffled[:, :idx_test_start, :]
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_pearson_scores = []

    run = wandb.init(project="BacteriaMetric_MultiMetric_Opt", name=f"trial_{trial.number}", reinit=True, config=trial.params)

    print(f"Starting Trial {trial.number}...")

    # --- 4. לולאת ה-Cross Validation ---
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.arange(dev_data.shape[1]))):
        
        # יצירת Train/Val ל-Fold הנוכחי
        fold_train_data = dev_data[:, train_idx, :]
        fold_val_data = dev_data[:, val_idx, :]
        fold_val_pathways = dev_pathways[:, val_idx, :].mean(dim=0).numpy() 

        # הוספת הלא-מתוייגים לאימון
        fold_train_tensor = torch.cat((fold_train_data, unannotated_tensor), dim=1)
        
        # יצירת Loaders
        train_loader = create_dataloaders(fold_train_tensor, batch_size=batch_size)
        val_loader = create_dataloaders(fold_val_data, batch_size=batch_size)
        
        # מודל חדש (GPU 3)
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        model = SplitVAE(gene_dim=data.shape[-1], embedding_dim=embedding_dim).to(device)

        try:
            # הרצת האימון
            pearson = train_one_fold(
                model, train_loader, val_loader, fold_val_pathways, device, num_epochs, 
                learning_rate, fold_idx, trial.number, config.LAMBDA_WEIGHT, weight_decay, trial
            )
            fold_pearson_scores.append(pearson)
            print(f"  Fold {fold_idx}: Best Pearson (Max Metric) = {pearson:.4f}")
        
        except optuna.exceptions.TrialPruned:
            run.finish()
            raise
        except Exception as e:
            print(f"  Fold {fold_idx} Error: {e}")
            run.finish()
            raise optuna.exceptions.TrialPruned() # חיתוך במקרה של שגיאה טכנית

    run.finish()
    
    # ציון סופי: ממוצע הפירסון על פני כל ה-Folds
    avg_pearson = np.mean(fold_pearson_scores)
    return avg_pearson

if __name__ == "__main__":
    # מקסימיזציה של פירסון
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    
    # הרצת 35 ניסויים
    study.optimize(objective, n_trials=20)

    print("\n--- Finished ---")
    print(f"Best Pearson: {study.best_value}")
    print("Best Params:", study.best_trial.params)

    with open("best_params_multi_metric.txt", "w") as f:
        f.write(f"Best Pearson: {study.best_value}\n")
        f.write(str(study.best_trial.params))