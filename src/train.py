"""
Training and Evaluation Pipeline for Tamil Nadu Crop Recommendation Models:
1. Decision Tree
2. Random Forest
3. XGBoost
4. FT-Transformer (Proposed Tabular Deep Learning Model)
"""

import os
import sys
import time
import json
import joblib

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.ft_transformer import FTTransformer

def evaluate_predictions(y_true, y_pred, model_name: str, split_name: str = "Test"):
    acc = float(accuracy_score(y_true, y_pred))
    p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    p_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    r_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n[{model_name} - {split_name} Set]")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  F1-Score (Macro):  {f1_macro:.4f}")
    print(f"  F1-Score (Weight): {f1_weighted:.4f}")
    print(f"  Precision (Macro): {p_macro:.4f}")
    print(f"  Recall (Macro):    {r_macro:.4f}")

    return {
        "model": model_name,
        "split": split_name,
        "accuracy": acc,
        "precision_macro": p_macro,
        "precision_weighted": p_weighted,
        "recall_macro": r_macro,
        "recall_weighted": r_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm.tolist()
    }

def train_and_evaluate_all(bundle_path: str, models_dir: str, epochs_ft: int = 15):
    print("Loading preprocessed data bundle...")
    bundle = joblib.load(bundle_path)
    tree_data = bundle['tree']
    trans_data = bundle['transformer']
    classes = bundle['target_classes']
    n_classes = len(classes)

    metrics_summary = {}
    cms_dict = {}

    # ==========================================
    # 1. Decision Tree
    # ==========================================
    print("\n" + "="*50)
    print("1. Training Decision Tree Classifier...")
    t0 = time.time()
    dt = DecisionTreeClassifier(max_depth=22, min_samples_split=5, random_state=42)
    dt.fit(tree_data['X_train'], tree_data['y_train'])
    dt_time = time.time() - t0
    print(f"Decision Tree trained in {dt_time:.2f}s")
    
    y_val_pred_dt = dt.predict(tree_data['X_val'])
    y_test_pred_dt = dt.predict(tree_data['X_test'])
    dt_test_metrics = evaluate_predictions(tree_data['y_test'], y_test_pred_dt, "Decision Tree", "Test")
    dt_test_metrics["train_time_sec"] = dt_time
    metrics_summary["Decision Tree"] = dt_test_metrics
    cms_dict["Decision Tree"] = dt_test_metrics["confusion_matrix"]
    joblib.dump(dt, os.path.join(models_dir, "decision_tree.joblib"))

    # ==========================================
    # 2. Random Forest
    # ==========================================
    print("\n" + "="*50)
    print("2. Training Random Forest Classifier...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_split=4, random_state=42, n_jobs=-1)
    rf.fit(tree_data['X_train'], tree_data['y_train'])
    rf_time = time.time() - t0
    print(f"Random Forest trained in {rf_time:.2f}s")
    
    y_val_pred_rf = rf.predict(tree_data['X_val'])
    y_test_pred_rf = rf.predict(tree_data['X_test'])
    rf_test_metrics = evaluate_predictions(tree_data['y_test'], y_test_pred_rf, "Random Forest", "Test")
    rf_test_metrics["train_time_sec"] = rf_time
    metrics_summary["Random Forest"] = rf_test_metrics
    cms_dict["Random Forest"] = rf_test_metrics["confusion_matrix"]
    joblib.dump(rf, os.path.join(models_dir, "random_forest.joblib"))

    # ==========================================
    # 3. XGBoost
    # ==========================================
    print("\n" + "="*50)
    print("3. Training XGBoost Classifier...")
    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    xgb.fit(tree_data['X_train'], tree_data['y_train'])
    xgb_time = time.time() - t0
    print(f"XGBoost trained in {xgb_time:.2f}s")
    
    y_val_pred_xgb = xgb.predict(tree_data['X_val'])
    y_test_pred_xgb = xgb.predict(tree_data['X_test'])
    xgb_test_metrics = evaluate_predictions(tree_data['y_test'], y_test_pred_xgb, "XGBoost", "Test")
    xgb_test_metrics["train_time_sec"] = xgb_time
    metrics_summary["XGBoost"] = xgb_test_metrics
    cms_dict["XGBoost"] = xgb_test_metrics["confusion_matrix"]
    joblib.dump(xgb, os.path.join(models_dir, "xgboost.joblib"))

    # ==========================================
    # 4. FT-Transformer (Proposed)
    # ==========================================
    print("\n" + "="*50)
    print("4. Training FT-Transformer Classifier (PyTorch)...")
    device = torch.device("cpu")
    
    train_dataset = TensorDataset(
        torch.tensor(trans_data['X_train_num'], dtype=torch.float32),
        torch.tensor(trans_data['X_train_cat'], dtype=torch.long),
        torch.tensor(trans_data['y_train'], dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(trans_data['X_val_num'], dtype=torch.float32),
        torch.tensor(trans_data['X_val_cat'], dtype=torch.long),
        torch.tensor(trans_data['y_val'], dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(trans_data['X_test_num'], dtype=torch.float32),
        torch.tensor(trans_data['X_test_cat'], dtype=torch.long),
        torch.tensor(trans_data['y_test'], dtype=torch.long)
    )

    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    ft_model = FTTransformer(
        n_num_features=trans_data['X_train_num'].shape[1],
        cat_cardinalities=trans_data['cat_dims'],
        n_classes=n_classes,
        d_token=48,
        n_layers=2,
        n_heads=4,
        ffn_factor=2.0,
        dropout=0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(ft_model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_ft)

    t0 = time.time()
    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(1, epochs_ft + 1):
        ft_model.train()
        total_loss, correct, total = 0.0, 0, 0
        for b_num, b_cat, b_y in train_loader:
            b_num, b_cat, b_y = b_num.to(device), b_cat.to(device), b_y.to(device)
            optimizer.zero_grad()
            logits = ft_model(b_num, b_cat)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(b_y)
            preds = logits.argmax(dim=-1)
            correct += (preds == b_y).sum().item()
            total += len(b_y)

        scheduler.step()
        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        ft_model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for b_num, b_cat, b_y in val_loader:
                b_num, b_cat, b_y = b_num.to(device), b_cat.to(device), b_y.to(device)
                logits = ft_model(b_num, b_cat)
                loss = criterion(logits, b_y)
                val_loss += loss.item() * len(b_y)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == b_y).sum().item()
                val_total += len(b_y)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch:02d}/{epochs_ft:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = ft_model.state_dict().copy()

    ft_time = time.time() - t0
    print(f"FT-Transformer trained in {ft_time:.2f}s (Best Val Loss: {best_val_loss:.4f})")

    # Load best model
    ft_model.load_state_dict(best_model_state)
    torch.save(best_model_state, os.path.join(models_dir, "ft_transformer.pt"))

    # Save model config
    ft_config = {
        "n_num_features": trans_data['X_train_num'].shape[1],
        "cat_cardinalities": trans_data['cat_dims'],
        "n_classes": n_classes,
        "d_token": 48,
        "n_layers": 2,
        "n_heads": 4,
        "ffn_factor": 2.0,
        "dropout": 0.1
    }
    with open(os.path.join(models_dir, "ft_transformer_config.json"), "w") as f:
        json.dump(ft_config, f, indent=2)

    # Test evaluation
    ft_model.eval()
    test_preds = []
    with torch.no_grad():
        for b_num, b_cat, _ in test_loader:
            b_num, b_cat = b_num.to(device), b_cat.to(device)
            logits = ft_model(b_num, b_cat)
            test_preds.extend(logits.argmax(dim=-1).cpu().numpy())

    test_preds = np.array(test_preds)
    ft_test_metrics = evaluate_predictions(trans_data['y_test'], test_preds, "FT-Transformer", "Test")
    ft_test_metrics["train_time_sec"] = ft_time
    metrics_summary["FT-Transformer"] = ft_test_metrics
    cms_dict["FT-Transformer"] = ft_test_metrics["confusion_matrix"]

    # Save all metrics
    with open(os.path.join(models_dir, "metrics_comparison.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    joblib.dump(cms_dict, os.path.join(models_dir, "confusion_matrices.joblib"))
    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY (Test Set):")
    print(f"{'Model':<18} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<12} | {'Time (s)':<8}")
    print("-" * 65)
    for m_name, m_data in metrics_summary.items():
        print(f"{m_name:<18} | {m_data['accuracy']:<10.4f} | {m_data['f1_macro']:<10.4f} | {m_data['f1_weighted']:<12.4f} | {m_data['train_time_sec']:<8.2f}")

    return metrics_summary

if __name__ == "__main__":
    bundle = r"C:\Users\Admin\.gemini\antigravity\scratch\tn_crop_recommendation_xai\models\processed_data_bundle.joblib"
    m_dir = r"C:\Users\Admin\.gemini\antigravity\scratch\tn_crop_recommendation_xai\models"
    train_and_evaluate_all(bundle, m_dir, epochs_ft=15)
