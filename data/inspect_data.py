import pandas as pd
import numpy as np

csv_path = r"C:\Users\Admin\.gemini\antigravity\scratch\tn_crop_recommendation_xai\data\crop_recommendation_dataset.csv"

# Load data
df = pd.read_csv(csv_path)

print("="*60)
print("DATASET SHAPE:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n" + "="*60)
print("COLUMNS & DATA TYPES:")
print(df.dtypes)

print("\n" + "="*60)
print("HEAD (First 5 rows):")
print(df.head())

print("\n" + "="*60)
print("MISSING VALUES:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values!")

print("\n" + "="*60)
print("DUPLICATE ROWS:")
print(f"Duplicate count: {df.duplicated().sum()}")

print("\n" + "="*60)
print("CATEGORICAL COLUMNS & UNIQUE VALUES:")
for col in df.select_dtypes(include=['object', 'category']).columns:
    unique_vals = df[col].unique()
    print(f"\n--- {col} (n_unique = {len(unique_vals)}) ---")
    if len(unique_vals) <= 50:
        print(df[col].value_counts())
    else:
        print(f"Sample values (total {len(unique_vals)}): {unique_vals[:15]}")

print("\n" + "="*60)
print("NUMERICAL COLUMNS SUMMARY:")
print(df.describe().T)
