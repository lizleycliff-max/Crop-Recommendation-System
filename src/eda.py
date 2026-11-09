"""
Exploratory Data Analysis (EDA) module for Tamil Nadu Crop Recommendation.
Generates publication-quality charts and summary statistics.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import clean_dataset

def run_eda(csv_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df_clean = clean_dataset(df)

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    palette = sns.color_palette("mako")

    print(f"Total records: {len(df_clean)}, Crops: {df_clean['CROPS'].nunique()}")

    # 1. Top 15 most frequent soil types
    plt.figure(figsize=(10, 6))
    soil_counts = df_clean['SOIL'].value_counts().head(12)
    sns.barplot(x=soil_counts.values, y=soil_counts.index, color="#2E7D32")
    plt.title("Distribution of Major Soil Types in Tamil Nadu Dataset", fontsize=13, weight="bold", pad=12)
    plt.xlabel("Sample Count", fontsize=11)
    plt.ylabel("Soil Classification", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "soil_distribution.png"), dpi=200)
    plt.close()

    # 2. Season and Water Source distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    season_counts = df_clean['SEASON'].value_counts()
    axes[0].bar(season_counts.index.str.capitalize(), season_counts.values, color=["#1E88E5", "#FFB300", "#43A047"], width=0.5)
    axes[0].set_title("Cropping Season Breakdown", fontsize=12, weight="bold")
    axes[0].set_ylabel("Count")

    water_counts = df_clean['WATER_SOURCE'].value_counts()
    axes[1].pie(water_counts.values, labels=water_counts.index.str.capitalize(), autopct='%1.1f%%', colors=["#0288D1", "#7CB342"], startangle=90)
    axes[1].set_title("Irrigation / Water Regime Ratio", fontsize=12, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "season_water_distribution.png"), dpi=200)
    plt.close()

    # 3. Correlation Heatmap of Environmental & Soil Features
    num_cols = ['N', 'P', 'K', 'SOIL_PH', 'TEMP', 'RELATIVE_HUMIDITY']
    corr = df_clean[num_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", cbar=True, square=True, linewidths=0.5)
    plt.title("Pearson Correlation: Soil Nutrients & Climate Variables", fontsize=12, weight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_correlation.png"), dpi=200)
    plt.close()

    # 4. Nutrient Distributions (N, P, K)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sns.histplot(df_clean['N'], kde=True, ax=axes[0], color="#2E7D32")
    axes[0].set_title("Nitrogen (N) Distribution (kg/ha)", weight="bold")
    axes[0].set_xlabel("N (kg/ha)")

    sns.histplot(df_clean['P'], kde=True, ax=axes[1], color="#E65100")
    axes[1].set_title("Phosphorus (P) Distribution (kg/ha)", weight="bold")
    axes[1].set_xlabel("P (kg/ha)")

    sns.histplot(df_clean['K'], kde=True, ax=axes[2], color="#0D47A1")
    axes[2].set_title("Potassium (K) Distribution (kg/ha)", weight="bold")
    axes[2].set_xlabel("K (kg/ha)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "npk_distributions.png"), dpi=200)
    plt.close()

    # 5. Temperature and Humidity by Season
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(x="SEASON", y="TEMP", data=df_clean, ax=axes[0], palette="Set2")
    axes[0].set_title("Ambient Temperature (°C) by Cropping Season", weight="bold")
    axes[0].set_xlabel("Season")
    axes[0].set_ylabel("Temperature (°C)")

    sns.boxplot(x="SEASON", y="RELATIVE_HUMIDITY", data=df_clean, ax=axes[1], palette="Set2")
    axes[1].set_title("Relative Humidity (%) by Cropping Season", weight="bold")
    axes[1].set_xlabel("Season")
    axes[1].set_ylabel("Humidity (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "climate_by_season.png"), dpi=200)
    plt.close()

    # Save summary stats to CSV
    summary_stats = df_clean[num_cols].describe().T
    summary_stats.to_csv(os.path.join(output_dir, "numerical_summary.csv"))
    print(f"EDA charts and summary saved to {output_dir}")

if __name__ == "__main__":
    raw_data = r"C:\Users\Admin\.gemini\antigravity\scratch\tn_crop_recommendation_xai\data\crop_recommendation_dataset.csv"
    out_charts = r"C:\Users\Admin\.gemini\antigravity\scratch\tn_crop_recommendation_xai\models\eda_artifacts"
    run_eda(raw_data, out_charts)
