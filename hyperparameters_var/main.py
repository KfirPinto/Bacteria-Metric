import os
import sys
import optuna
import torch
import wandb
import numpy as np
from sklearn.model_selection import KFold
from torch import optim

# --- הגדרות נתיבים (כדי שהייבוא יעבוד גם מתוך התיקייה) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import config
# ייבוא פונקציות עזר מהקבצים הקיימים בתיקייה (לא צריך לשנות אותם)
from hyperparameters_var.data_utils import load_data_tensor, load_metadata, normalize_tensor
from hyperparameters_var.training.model import SplitVAE
from hyperparameters_var.training.dataset import create_dataloaders
from hyperparameters_var.training.train import custom_loss

# --- קבועים ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_FOLDS = 5
NUM_EPOCHS = 100 

def train_one_fold(model, train_loader, val_loader, learning_rate, lambda_weight, weight_decay, trial_idx, fold_idx):
    """
    פונקציה זו מחליפה את train_model הישנה.
    היא רצה על פולד ספציפי ומחזירה את ה-Validation Loss הכי טוב.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_val_total_loss = float('inf')
    
    # פירוק משקולות ה-Loss
    w_recon = lambda_weight[0] if lambda_weight else 1.0
    w_bact = lambda_weight[1] if lambda_weight else 1.0
    w_sample = lambda_weight[2] if lambda_weight else 1.0

    for epoch in range(NUM_EPOCHS):
        # --- Train Loop ---
        model.train()
        for batch in train_loader:
            batch_tensor = batch[0].to(DEVICE).squeeze(0)
            optimizer.zero_grad()
            
            encoded, decoded, mu, logvar = model(batch_tensor)
            
            # שימוש ב-custom_loss מהקובץ הקיים (training/train.py)
            recon, bact, sample, wd, kl = custom_loss(
                batch_tensor, decoded, encoded, mu, logvar, model=model, weight_decay=0
            )
            
            # חישוב ה-Loss המשוקלל
            loss = (recon * w_recon) + (bact * w_bact) + (sample * w_sample) + kl + wd
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # --- Validation Loop ---
        model.eval()
        val_total_loss = 0.0
        val_recon_loss = 0.0
        val_bact_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch_tensor = batch[0].to(DEVICE).squeeze(0)
                encoded, decoded, mu, logvar = model(batch_tensor)
                
                recon, bact, sample, wd, kl = custom_loss(
                    batch_tensor, decoded, encoded, mu, logvar, model=model, weight_decay=0
                )
                
                total_loss = (recon * w_recon) + (bact * w_bact) + (sample * w_sample) + kl + wd
                
                val_total_loss += total_loss.item()
                val_recon_loss += recon.item()
                val_bact_loss += bact.item()
                num_batches += 1
        
        # ממוצעים
        avg_val_total = val_total_loss / num_batches
        avg_val_recon = val_recon_loss / num_batches
        avg_val_bact = val_bact_loss / num_batches

        # עדכון התוצאה הכי טובה
        if avg_val_total < best_val_total_loss:
            best_val_total_loss = avg_val_total
            
        # דיווח ל-WandB
        if wandb.run is not None:
            wandb.log({
                "epoch": epoch,
                f"trial_{trial_idx}_fold_{fold_idx}_val_total": avg_val_total,
                f"trial_{trial_idx}_fold_{fold_idx}_val_bact": avg_val_bact,
                f"trial_{trial_idx}_fold_{fold_idx}_val_recon": avg_val_recon
            })

    return best_val_total_loss

def objective(trial, dev_data, unannotated_tensor, gene_dim):
    # --- 1. הגדרת מרחב החיפוש (Hyperparameters) ---
    embedding_dim = trial.suggest_categorical("embedding_dim", [24, 32, 48, 64])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 1e-4)

    # שימוש בקונפיגורציה הקיימת שלך למשקולות (או שתגדיר כאן רשימה חדשה)
    lambda_weights = config.LAMBDA_WEIGHT 

    # אתחול WandB לניסוי הנוכחי
    run = wandb.init(
        project="BacteriaMetric_HyperOpt", 
        name=f"trial_{trial.number}", 
        group="optuna_cv",
        reinit=True, 
        config=trial.params
    )

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_losses = []
    
    n_samples = dev_data.shape[1] # הנחה: מימד 1 הוא דגימות
    
    try:
        # --- 5-Fold Cross Validation ---
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.arange(n_samples))):
            print(f"Trial {trial.number} | Fold {fold_idx + 1}/{K_FOLDS}")
            
            # הכנת הדאטה
            fold_train_data = dev_data[:, train_idx, :]
            fold_val_data = dev_data[:, val_idx, :]
            
            # הוספת Unannotated רק לאימון
            fold_train_tensor = torch.cat((fold_train_data, unannotated_tensor), dim=1)
            
            # יצירת DataLoaders
            train_loader = create_dataloaders(fold_train_tensor, batch_size=batch_size)
            val_loader = create_dataloaders(fold_val_data, batch_size=batch_size)
            
            # אתחול מודל חדש ונקי
            model = SplitVAE(gene_dim=gene_dim, embedding_dim=embedding_dim).to(DEVICE)
            
            # הרצת אימון על הפולד
            best_fold_loss = train_one_fold(
                model, train_loader, val_loader, 
                learning_rate, lambda_weights, weight_decay, 
                trial.number, fold_idx
            )
            
            fold_losses.append(best_fold_loss)
            
            # דיווח לאופטונה (Pruning) - אם הפולד הראשון גרוע מאוד, נעצור את הניסוי
            trial.report(np.mean(fold_losses), step=fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    except optuna.exceptions.TrialPruned:
        run.finish()
        raise
    except Exception as e:
        print(f"Error in trial {trial.number}: {e}")
        run.finish()
        return float('inf')

    run.finish()
    # המטרה: למזער את ממוצע ה-Total Loss
    return np.mean(fold_losses)

if __name__ == "__main__":
    # --- טעינת דאטה ראשית (פעם אחת בלבד) ---
    print("Loading Data...")
    # וודא שהנתיבים בקובץ config תקינים
    data = load_data_tensor(config.GENE_FAMILIES_TENSOR_PATH)
    unannotated_data = load_data_tensor(config.GENE_FAMILIES_COMPLEMENTRARY_TENSOR_PATH)
    
    # נרמול
    data_norm = normalize_tensor(data)
    unannotated_norm = normalize_tensor(unannotated_data)
    
    data_tensor = torch.tensor(data_norm, dtype=torch.float32)
    unannotated_tensor = torch.tensor(unannotated_norm, dtype=torch.float32)

    # Shuffle וחלוקה ל-Dev (85%) ו-Test (15%)
    torch.manual_seed(42)
    perm = torch.randperm(data_tensor.size(1))
    data_shuffled = data_tensor[:, perm, :]
    
    idx_test_start = int(0.85 * data_shuffled.shape[1]) 
    dev_data = data_shuffled[:, :idx_test_start, :] # רק על זה עושים אופטימיזציה!
    
    print(f"Data Loaded. Dev set size: {dev_data.shape[1]}")

    # --- Optuna Setup ---
    storage_name = "sqlite:///hyperparameters_var/db.sqlite3"
    
    study = optuna.create_study(
        study_name="bacteria_optimization", # חובה לתת שם כשעובדים עם DB
        storage=storage_name,               # מפנה לקובץ המשותף
        load_if_exists=True,                # טוען את המחקר אם הוא כבר קיים
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner()
    )
    
    # ★ הזרקת ה-Baseline שלך כנקודת התחלה ★
    print("Injecting Baseline Parameters...")
    study.enqueue_trial({
        "embedding_dim": 32,
        "learning_rate": 0.001,
        "batch_size": 64,
        "weight_decay": 0.0 
    })
    
    # הרצת המחקר
    print("Starting Optimization...")
    study.optimize(lambda trial: objective(trial, dev_data, unannotated_tensor, gene_dim=data.shape[-1]), n_trials=25)

    print("\n--- Optimization Finished ---")
    print(f"Best Total Val Loss: {study.best_value}")
    print("Best Params:", study.best_trial.params)