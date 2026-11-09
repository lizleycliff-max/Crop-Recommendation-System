# Explainable AI for Crop Recommendation in Tamil Nadu

An academic, end-to-end decision-support system and comparative machine learning study for precision crop selection across Tamil Nadu agro-climatic zones, powered by the **Feature Tokenizer Transformer (FT-Transformer)** and model explainability using **SHAP (SHapley Additive exPlanations)** and multi-head self-attention.

---

## 1. Project Overview & Motivation

Agricultural productivity in Tamil Nadu depends critically on selecting cultivars that align with localized soil texture, macro-nutrient availability (Nitrogen, Phosphorus, Potassium), pH balance, seasonal photoperiods, and irrigation regimes. 

While conventional machine learning approaches often treat recommendation as a black box or inadvertently introduce target leakage (such as training on crop maturity duration or crop physiological water requirements), this project implements:
1. **Strict Data Leakage Elimination**: Ensuring only parameters verifiable prior to planting are utilized.
2. **Tabular Deep Learning**: Implementation of the **FT-Transformer** architecture (Gorishniy et al., NeurIPS 2021) combining numerical linear tokenizers and categorical embedding tables with multi-head self-attention.
3. **Comparative Empirical Benchmarking**: Side-by-side evaluation against Decision Tree, Random Forest, and XGBoost on a stratified 15% hold-out test set (8,550 samples).
4. **Dual-Tier Explainable AI (XAI)**: Intrinsic self-attention token attribution from the Transformer `[CLS]` token and game-theoretic marginal contribution analysis via SHAP KernelExplainer.
5. **Authentic Decision-Support Interface**: A clean, professional Streamlit application tailored for agricultural extension officers and farmers.

---

## 2. Dataset Audit & Ground Truth

* **Data Source**: Mendeley Data - *Crop recommendation dataset* ([Mendeley Link](https://data.mendeley.com/datasets/vynxnppr7j/1))
* **Total Instances**: 57,000 field records
* **Total Raw Columns**: 23 columns
* **Missing Values**: 0 (Complete dataset)
* **Duplicate Rows**: 0
* **Target Variable**: `CROPS` (57 distinct agricultural crops, perfectly balanced with 1,000 instances per crop)

### Data Leakage Audit & Feature Filtering

| Raw Column | Action | Agronomic & Machine Learning Rationale |
| :--- | :--- | :--- |
| `CROPDURATION` & `CROPDURATION_MAX` | **Excluded** | **Target Leakage**: Crop duration (e.g. 120 days for rice vs 300 for sugarcane) is an inherent biological trait of the crop itself, not a field condition available to a farmer before crop choice. |
| `WATERREQUIRED` & `WATERREQUIRED_MAX` | **Excluded** | **Target Leakage**: Represents the crop's physiological evapotranspiration demand, not the farmer's water availability. |
| `N_MAX`, `P_MAX`, `K_MAX`, `MAX_TEMP`, `SOIL_PH_HIGH`, `RELATIVE_HUMIDITY_MAX` | **Excluded** | **Deterministic Signature Leakage**: Empirical testing confirmed standard deviation within each crop is **0.00**. These columns are static handbook lookup constants; using them allows trivial 100% memorization. |
| `TYPE_OF_CROP` | **Excluded** | **Taxonomy Leakage**: Broad biological group (cereals, pulses) has 0 intra-crop variance and directly leaks the target category. |
| `HARVESTED` | **Excluded** | **Chronological Consequence**: Harvest month is determined by sowing date plus crop growth cycle. |
| `district` | **Not Present** | The Mendeley dataset does not contain administrative district labels. Soil texture and agro-climate serve as the geographical adaptation factors. |

### Final Selected Input Features (9 Core Features)

1. **`SOIL`** (*Categorical, 29 types*): Local soil texture (Alluvial, Loamy, Clay, Red, Black Cotton, Laterite, Sandy Loam, etc.). Standardized and cleaned of typographical discrepancies.
2. **`SEASON`** (*Categorical, 3 classes*): `kharif`, `rabi`, `zaid`.
3. **`WATER_SOURCE`** (*Categorical, 2 classes*): `irrigated`, `rainfed`.
4. **`N`** (*Continuous*): Available Soil Nitrogen (kg/ha).
5. **`P`** (*Continuous*): Available Soil Phosphorus (kg/ha).
6. **`K`** (*Continuous*): Available Soil Potassium (kg/ha).
7. **`SOIL_PH`** (*Continuous*): Measured soil pH level (3.5 - 9.0).
8. **`TEMP`** (*Continuous*): Ambient temperature during growth cycle (°C).
9. **`RELATIVE_HUMIDITY`** (*Continuous*): Ambient relative humidity (%).

---

## 3. Empirical Model Benchmarking Results

All models were evaluated on the exact same **stratified 15% hold-out test set** (8,550 records across 57 crops; 150 instances per crop).

| Model Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1 | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **98.76%** | **0.9883** | **0.9876** | **0.9875** | **0.9875** | 2.98 s |
| **XGBoost** | **98.61%** | **0.9863** | **0.9861** | **0.9861** | **0.9861** | 120.45 s |
| **CatBoost** | **98.54%** | **0.9860** | **0.9854** | **0.9854** | **0.9854** | 76.72 s |
| **Decision Tree** | **98.29%** | **0.9840** | **0.9829** | **0.9827** | **0.9827** | 0.33 s |
| **FT-Transformer (Proposed)** | **98.13%** | **0.9817** | **0.9813** | **0.9813** | **0.9813** | 233.87 s |

### Academic Integrity & Discussion
* Consistent with literature on tabular benchmarks (Gorishniy et al., 2021; Grinsztajn et al., 2022), tree-based ensembles (Random Forest at 98.76%, XGBoost at 98.61%, and CatBoost at 98.54%) achieved marginally higher raw classification accuracy on this clean tabular dataset.
* FT-Transformer achieves an exceptional **98.13% test accuracy** while offering major architectural advantages:
  1. Simultaneous end-to-end token embedding of continuous and categorical variables without manual one-hot dimensionality explosion.
  2. Native multi-head self-attention mechanisms that reveal feature-to-feature token interactions.
  3. Direct generation of probability distributions and calibrated Top-3 candidate sets for decision support.

---

## 4. Proposed Architecture: FT-Transformer

The FT-Transformer architecture is implemented in PyTorch (`src/ft_transformer.py`):
```
Input Layer
  ├── 6 Continuous Features (N, P, K, pH, Temp, RH)
  │     └── Numerical Tokenizer: e_j = x_j · W_j + b_j  (d=48)
  ├── 3 Categorical Features (Soil, Season, Water Source)
  │     └── Embedding Layers: e_k = Embedding(x_k, d=48)
  └── Learnable [CLS] Token (1 × 48)
            │
Sequence of 10 Tokens (Length=10, Dimension=48)
            │
Transformer Encoder Backbone (2 Layers, 4 Attention Heads)
  ├── Multi-Head Self-Attention (Attention matrices extracted for XAI)
  ├── Layer Normalization & Residual Add
  └── Feed-Forward Network with GELU Activations
            │
[CLS] Token Output Representation (Dimension=48)
            │
LayerNorm + Linear Classification Head
            │
Softmax Layer → 57 Crop Class Probabilities
```

---

## 5. Explainable AI (XAI) Framework

The system provides three complementary layers of interpretability:
1. **Global Feature Importance (SHAP)**: Computed across the test population to quantify the macro-level drivers in Tamil Nadu agriculture.
   - Empirical ranking: `SOIL` (0.314) > `RELATIVE_HUMIDITY` (0.218) > `SEASON` (0.155) > `K` (0.087) > `N` (0.082) > `TEMP` (0.054) > `P` (0.040) > `SOIL_PH` (0.039) > `WATER_SOURCE` (0.033).
2. **Local Feature Attribution (SHAP)**: Real-time calculation of marginal contributions for an individual farmer's plot, indicating which parameters favor or disfavor the recommended crop.
3. **Intrinsic Multi-Head Self-Attention Attribution**: Direct extraction of the attention weights linking the `[CLS]` token to each individual feature token in the FT-Transformer encoder.
4. **Automated Agronomic Advisory Generation**: Translates mathematical attributions into actionable agricultural language.

---

## 6. Streamlit Decision-Support Dashboard

The dashboard provides an authentic, clean, agricultural analytics interface:
* **Section 1: Executive Overview** - Dataset verification, leakage elimination principles, and 4-model performance matrix.
* **Section 2: Explore Agricultural Data** - Interactive nutrient histograms, macronutrient correlation heatmaps, seasonal temperature/humidity boxplots, and soil texture prevalence.
* **Section 3: Crop Recommendation** - Farmer input form with real-time FT-Transformer inference, primary crop card, Top-3 candidate comparison, and multi-head attention distribution.
* **Section 4: Why This Crop? (XAI)** - Local SHAP contribution charts, global SHAP importance benchmarks, and human-readable plain language advisory notes.
* **Section 5: What-If Scenario Analysis** - Interactive simulation enabling users to adjust NPK, pH, or irrigation and observe real-time probability shifts.
* **Section 6: Model Benchmarking & Architecture** - Confusion matrix diagnostics across all 57 crops and architectural schema.

---

## 7. Directory Structure

```
tn_crop_recommendation_xai/
├── app.py                     # Streamlit Decision-Support Application
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive documentation
├── data/
│   ├── crop_recommendation_dataset.csv   # Raw Mendeley dataset (57k rows)
│   └── inspect_data.py                  # Dataset audit script
├── models/
│   ├── cleaned_dataset.csv               # Leakage-free cleaned dataset
│   ├── preprocessor.joblib              # Fitted scaler & categorical encodings
│   ├── processed_data_bundle.joblib     # Stratified split matrices
│   ├── decision_tree.joblib             # Trained Decision Tree
│   ├── random_forest.joblib             # Trained Random Forest
│   ├── xgboost.joblib                   # Trained XGBoost
│   ├── ft_transformer.pt                # PyTorch FT-Transformer weights
│   ├── ft_transformer_config.json       # FT-Transformer architectural hyperparameters
│   ├── metrics_comparison.json          # Test set benchmarking metrics
│   ├── confusion_matrices.joblib        # Confusion matrices for all 4 models
│   ├── global_shap_importance.joblib    # Precomputed global SHAP values
│   └── eda_artifacts/                   # Static publication-quality EDA figures
└── src/
    ├── __init__.py
    ├── preprocess.py                    # Cleaning, encoding, and dataset splitting
    ├── ft_transformer.py                # PyTorch FT-Transformer implementation
    ├── train.py                         # Training pipeline for all 4 models
    ├── explain.py                       # Inference, SHAP, and attention attribution
    └── eda.py                           # EDA chart generator
```

---

## 8. Installation & Execution Guide

### Prerequisites
* Python 3.10, 3.11, or 3.12
* Windows, Linux, or macOS

### 1. Clone & Set Up Environment
```bash
git clone <repository_url>
cd tn_crop_recommendation_xai

# Create virtual environment (optional but recommended)
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run Pre-Trained Application (Instant Launch)
All model checkpoints, encoders, and SHAP background distributions are pre-saved in the `models/` folder. Launch the Streamlit dashboard immediately:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Full End-to-End Retraining (Optional)
To verify complete reproducibility from the raw CSV:
```bash
# Step 1: Preprocess and generate stratified splits
python src/preprocess.py

# Step 2: Train all 4 models & evaluate on test set
python src/train.py

# Step 3: Run XAI verification & precompute global SHAP
python src/explain.py

# Step 4: Generate EDA figures
python src/eda.py
```

---

## 9. Citation & Academic References

1. Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). *Revisiting Deep Learning Models for Tabular Data*. Advances in Neural Information Processing Systems (NeurIPS 2021).
2. Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems (NeurIPS 2017).
3. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. ACM SIGKDD.
4. Mendeley Data Dataset: *Crop recommendation dataset*, doi: 10.17632/vynxnppr7j.1.
