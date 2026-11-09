"""
Data Preprocessing Module for Tamil Nadu Crop Recommendation
Handles data cleaning, normalization, leakage removal, and train/val/test splitting.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

NUMERICAL_COLS = ['N', 'P', 'K', 'SOIL_PH', 'TEMP', 'RELATIVE_HUMIDITY']
CATEGORICAL_COLS = ['SOIL', 'SEASON', 'WATER_SOURCE']
TARGET_COL = 'CROPS'

def clean_soil_type(val: str) -> str:
    """Standardize soil type names, strip trailing spaces, remove non-breaking spaces."""
    if not isinstance(val, str):
        return "loamy soil"
    cleaned = val.replace('\xa0', ' ').strip().lower()
    # Normalize minor typo found in Mendeley dataset
    if cleaned == 'light soi':
        cleaned = 'light soil'
    return cleaned

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Tamil Nadu crop dataset."""
    df_clean = df.copy()
    df_clean['SOIL'] = df_clean['SOIL'].apply(clean_soil_type)
    df_clean['SEASON'] = df_clean['SEASON'].astype(str).str.strip().str.lower()
    df_clean['WATER_SOURCE'] = df_clean['WATER_SOURCE'].astype(str).str.strip().str.lower()
    if 'CROPS' in df_clean.columns:
        df_clean['CROPS'] = df_clean['CROPS'].astype(str).str.strip()
    return df_clean

class PreprocessingPipeline:
    def __init__(self):
        self.num_cols = NUMERICAL_COLS
        self.cat_cols = CATEGORICAL_COLS
        self.target_col = TARGET_COL
        self.scaler = StandardScaler()
        self.target_encoder = LabelEncoder()
        self.cat_maps = {}  # mapping: col -> {category_str: int_id}
        self.cat_inv_maps = {}
        self.cat_dims = []  # cardinality of each cat feature
        self.one_hot_encoder = None
        self.ohe_feature_names = None

    def fit(self, df: pd.DataFrame):
        df_clean = clean_dataset(df)
        
        # Fit target encoder
        self.target_encoder.fit(df_clean[self.target_col])
        
        # Fit numeric scaler
        self.scaler.fit(df_clean[self.num_cols])
        
        # Fit categorical ordinals (for FT-Transformer embeddings)
        self.cat_dims = []
        for col in self.cat_cols:
            cats = sorted(df_clean[col].unique().tolist())
            mapping = {cat: idx for idx, cat in enumerate(cats)}
            self.cat_maps[col] = mapping
            self.cat_inv_maps[col] = {idx: cat for cat, idx in mapping.items()}
            self.cat_dims.append(len(cats))
            
        # Fit OneHotEncoder (for Tree models)
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.one_hot_encoder.fit(df_clean[self.cat_cols])
        ohe_cols = self.one_hot_encoder.get_feature_names_out(self.cat_cols).tolist()
        self.ohe_feature_names = self.num_cols + ohe_cols

        return self

    def transform_trees(self, df: pd.DataFrame):
        """Prepare feature matrix for Decision Tree, Random Forest, XGBoost."""
        df_clean = clean_dataset(df)
        X_num = self.scaler.transform(df_clean[self.num_cols])
        X_cat_ohe = self.one_hot_encoder.transform(df_clean[self.cat_cols])
        X = np.hstack([X_num, X_cat_ohe])
        
        y = None
        if self.target_col in df_clean.columns:
            y = self.target_encoder.transform(df_clean[self.target_col])
        return X, y

    def transform_transformer(self, df: pd.DataFrame):
        """Prepare feature matrices for FT-Transformer: continuous and categorical indices."""
        df_clean = clean_dataset(df)
        X_num = self.scaler.transform(df_clean[self.num_cols]).astype(np.float32)
        
        # Categorical ordinal indices
        X_cats = []
        for col in self.cat_cols:
            mapping = self.cat_maps[col]
            # default to index 0 if unknown category encountered
            mapped = df_clean[col].map(lambda x: mapping.get(x, 0)).values
            X_cats.append(mapped[:, None])
        X_cat = np.hstack(X_cats).astype(np.int64)
        
        y = None
        if self.target_col in df_clean.columns:
            y = self.target_encoder.transform(df_clean[self.target_col]).astype(np.int64)
            
        return X_num, X_cat, y

    def transform_single_farmer_input(self, input_dict: dict):
        """Transform a single farmer input dictionary into feature arrays."""
        df_single = pd.DataFrame([input_dict])
        X_tree, _ = self.transform_trees(df_single)
        X_num, X_cat, _ = self.transform_transformer(df_single)
        return {
            'df': df_single,
            'X_tree': X_tree,
            'X_num': X_num,
            'X_cat': X_cat
        }

def prepare_splits(raw_csv_path: str, output_dir: str):
    """Load raw dataset, clean, split, fit preprocessor, and save artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(raw_csv_path)
    df_clean = clean_dataset(df)
    
    # Save clean dataset
    clean_csv_path = os.path.join(output_dir, "cleaned_dataset.csv")
    df_clean.to_csv(clean_csv_path, index=False)
    print(f"Cleaned dataset saved to {clean_csv_path}")

    # Stratified split: 70% train, 15% validation, 15% test
    train_df, temp_df = train_test_split(
        df_clean, test_size=0.30, random_state=42, stratify=df_clean[TARGET_COL]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df[TARGET_COL]
    )

    print(f"Dataset split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Fit preprocessor strictly on train_df
    preprocessor = PreprocessingPipeline()
    preprocessor.fit(train_df)

    # Save preprocessor
    prep_path = os.path.join(output_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, prep_path)
    print(f"Preprocessor saved to {prep_path}")

    # Transform all splits
    # 1. Tree matrices
    X_train_tree, y_train = preprocessor.transform_trees(train_df)
    X_val_tree, y_val = preprocessor.transform_trees(val_df)
    X_test_tree, y_test = preprocessor.transform_trees(test_df)

    # 2. Transformer matrices
    X_train_num, X_train_cat, _ = preprocessor.transform_transformer(train_df)
    X_val_num, X_val_cat, _ = preprocessor.transform_transformer(val_df)
    X_test_num, X_test_cat, _ = preprocessor.transform_transformer(test_df)

    # Save processed matrices
    data_bundle = {
        'tree': {
            'X_train': X_train_tree, 'y_train': y_train,
            'X_val': X_val_tree, 'y_val': y_val,
            'X_test': X_test_tree, 'y_test': y_test,
            'feature_names': preprocessor.ohe_feature_names
        },
        'transformer': {
            'X_train_num': X_train_num, 'X_train_cat': X_train_cat, 'y_train': y_train,
            'X_val_num': X_val_num, 'X_val_cat': X_val_cat, 'y_val': y_val,
            'X_test_num': X_test_num, 'X_test_cat': X_test_cat, 'y_test': y_test,
            'cat_dims': preprocessor.cat_dims
        },
        'target_classes': preprocessor.target_encoder.classes_.tolist(),
        'cat_maps': preprocessor.cat_maps,
        'num_cols': NUMERICAL_COLS,
        'cat_cols': CATEGORICAL_COLS
    }
    bundle_path = os.path.join(output_dir, "processed_data_bundle.joblib")
    joblib.dump(data_bundle, bundle_path)
    print(f"Processed data bundle saved to {bundle_path}")

    return preprocessor, data_bundle

if __name__ == "__main__":
    csv_file = r"C:\Users\Admin\.gemini\antigravity\scratch\tn_crop_recommendation_xai\data\crop_recommendation_dataset.csv"
    out_dir = r"C:\Users\Admin\.gemini\antigravity\scratch\tn_crop_recommendation_xai\models"
    prepare_splits(csv_file, out_dir)
