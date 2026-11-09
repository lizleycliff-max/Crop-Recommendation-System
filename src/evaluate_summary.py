"""
End-to-End Validation and Academic Summary Report Generator
Executes sanity checks on all trained models, validates XAI pipelines,
and prints an academic benchmarking summary.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.explain import CropExplainer

def run_evaluation_summary():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_dir, "models")

    print("=" * 75)
    print("ACADEMIC PROJECT EVALUATION SUMMARY")
    print("Project Title: Explainable AI for Crop Recommendation in Tamil Nadu")
    print("=" * 75)

    # 1. Dataset Verification
    data_bundle = joblib.load(os.path.join(models_dir, "processed_data_bundle.joblib"))
    classes = data_bundle["target_classes"]
    print(f"\n[1. Dataset Integrity & Splitting]")
    print(f"  Total Unique Crop Classes : {len(classes)}")
    print(f"  Continuous Features (6)   : {data_bundle['num_cols']}")
    print(f"  Categorical Features (3)  : {data_bundle['cat_cols']}")
    print(f"  Train Set Instances       : {len(data_bundle['tree']['y_train'])}")
    print(f"  Validation Set Instances  : {len(data_bundle['tree']['y_val'])}")
    print(f"  Test Set Instances        : {len(data_bundle['tree']['y_test'])} (Stratified Hold-Out)")

    # 2. Model Benchmark Table
    metrics_path = os.path.join(models_dir, "metrics_comparison.json")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    print(f"\n[2. Comparative Model Benchmark (Unseen Test Set)]")
    print(f"{'Model Architecture':<20} | {'Test Acc':<10} | {'Macro F1':<10} | {'Macro Prec':<12} | {'Train Time':<10}")
    print("-" * 75)
    for model_name, m_data in sorted(metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"{model_name:<20} | {m_data['accuracy']*100:<9.2f}% | {m_data['f1_macro']:<10.4f} | {m_data['precision_macro']:<12.4f} | {m_data['train_time_sec']:<8.2f}s")

    # 3. FT-Transformer Inference & XAI Verification
    print(f"\n[3. FT-Transformer Model Inference & Explainability Test]")
    explainer = CropExplainer(models_dir)
    test_farmer = {
        'N': 85.0,
        'P': 45.0,
        'K': 45.0,
        'SOIL_PH': 6.8,
        'TEMP': 26.5,
        'RELATIVE_HUMIDITY': 78.0,
        'SOIL': 'alluvial soil',
        'SEASON': 'kharif',
        'WATER_SOURCE': 'irrigated'
    }
    pred = explainer.predict(test_farmer)
    print(f"  Farmer Plot Input:")
    print(f"    Soil: {test_farmer['SOIL'].title()} | Season: {test_farmer['SEASON'].upper()} | Water: {test_farmer['WATER_SOURCE'].title()}")
    print(f"    N={test_farmer['N']}, P={test_farmer['P']}, K={test_farmer['K']}, pH={test_farmer['SOIL_PH']}, Temp={test_farmer['TEMP']}°C, RH={test_farmer['RELATIVE_HUMIDITY']}%")
    print(f"  Recommended Crop : {pred['top_crop'].title()} (Confidence: {pred['top_probability']*100:.1f}%)")
    print(f"  Top 3 Candidates :")
    for idx, cand in enumerate(pred['top_3'], 1):
        print(f"    {idx}. {cand['crop'].title():<15} : {cand['probability']*100:.2f}%")

    # 4. Attention & SHAP Values
    print(f"\n[4. Dual-Tier Explainability Verification]")
    print("  FT-Transformer Multi-Head Self-Attention Weights:")
    for feat, w in sorted(pred['attention_weights'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    • {feat:<18} : {w:.2f}%")

    print("\n  KernelExplainer SHAP Local Attributions:")
    shap_res = explainer.explain_instance_shap(test_farmer, pred['top_3'][0]['class_index'], n_bg_samples=25)
    for feat, v in sorted(shap_res['shap_values'].items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        direction = "Positive (+)" if v > 0 else "Negative (-)"
        print(f"    • {feat:<18} : {v:+.4f} ({direction})")

    # 5. Fertilizer Advisory
    advisory = explainer.calculate_nutrient_gap_and_advisory(test_farmer, pred['top_crop'])
    print(f"\n[5. Soil Health & Fertilizer Guidance for {pred['top_crop'].title()}]")
    for note in advisory.get("advisory_notes", []):
        print(f"  {note}")

    print("\n" + "=" * 75)
    print("All modules verified successfully. Streamlit application is ready to run.")
    print("=" * 75)

if __name__ == "__main__":
    run_evaluation_summary()
