"""
Train CatBoost Classifier on Tamil Nadu Crop Recommendation dataset
and update metrics_comparison.json and confusion_matrices.joblib.
"""

import os
import sys
import time
import json
import joblib
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catboost import CatBoostClassifier
from src.train import evaluate_predictions

def train_catboost(bundle_path: str, models_dir: str):
    print("Loading data bundle for CatBoost training...")
    bundle = joblib.load(bundle_path)
    tree_data = bundle['tree']
    
    print("\nTraining CatBoost Classifier...")
    t0 = time.time()
    cb = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        loss_function='MultiClass',
        random_seed=42,
        thread_count=-1,
        verbose=50
    )
    cb.fit(tree_data['X_train'], tree_data['y_train'], eval_set=(tree_data['X_val'], tree_data['y_val']))
    cb_time = time.time() - t0
    print(f"CatBoost trained in {cb_time:.2f}s")

    y_test_pred_cb = cb.predict(tree_data['X_test']).flatten()
    cb_test_metrics = evaluate_predictions(tree_data['y_test'], y_test_pred_cb, "CatBoost", "Test")
    cb_test_metrics["train_time_sec"] = cb_time

    # Save model
    joblib.dump(cb, os.path.join(models_dir, "catboost.joblib"))
    print("Saved catboost.joblib")

    # Update metrics_comparison.json
    metrics_path = os.path.join(models_dir, "metrics_comparison.json")
    with open(metrics_path, "r") as f:
        metrics_summary = json.load(f)
    metrics_summary["CatBoost"] = cb_test_metrics
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print("Updated metrics_comparison.json")

    # Update confusion matrices
    cms_path = os.path.join(models_dir, "confusion_matrices.joblib")
    cms_dict = joblib.load(cms_path)
    cms_dict["CatBoost"] = cb_test_metrics["confusion_matrix"]
    joblib.dump(cms_dict, cms_path)
    print("Updated confusion_matrices.joblib")

    return cb_test_metrics

if __name__ == "__main__":
    bundle_p = r"C:\Users\Admin\.gemini\antigravity\scratch\tn_crop_recommendation_xai\models\processed_data_bundle.joblib"
    m_dir = r"C:\Users\Admin\.gemini\antigravity\scratch\tn_crop_recommendation_xai\models"
    train_catboost(bundle_p, m_dir)
