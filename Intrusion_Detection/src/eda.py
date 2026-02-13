import os
import matplotlib

# Use a non-interactive backend so scripts do not block waiting for GUI windows
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from src.config import RANDOM_SEED

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Tuning knobs to keep EDA quick on large datasets
MAX_PLOT_COLS = 3
HIST_SAMPLE_SIZE = 50000
CORR_SAMPLE_SIZE = 100000


def analyze_target_distribution(df, target_column):
    print("\n===== TARGET DISTRIBUTION =====")
    value_counts = df[target_column].value_counts(dropna=False)
    print(value_counts)
    print("\nPercentage Distribution:")
    print(df[target_column].value_counts(normalize=True, dropna=False) * 100)

    plt.figure()
    sns.countplot(x=target_column, data=df)
    plt.title("Binary Target Distribution")
    plt.tight_layout()
    plot_path = os.path.join(FIGURES_DIR, f"{target_column}_distribution.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved target distribution plot to {plot_path}")


def analyze_feature_types(df):
    print("\n===== FEATURE TYPE ANALYSIS =====")
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = df.select_dtypes(exclude=["object"]).columns.tolist()

    print(f"Categorical Columns: {categorical_cols}")
    print(f"Number of Numerical Columns: {len(numerical_cols)}")

    return categorical_cols, numerical_cols


def plot_numeric_distributions(df, numeric_columns):
    print("\n===== NUMERIC DISTRIBUTIONS =====")
    if not numeric_columns:
        print("No numeric columns to plot.")
        return

    sample_size = min(len(df), HIST_SAMPLE_SIZE)
    sampled_df = df[numeric_columns].sample(sample_size, random_state=RANDOM_SEED)

    for col in numeric_columns[:MAX_PLOT_COLS]:
        plt.figure()
        sns.histplot(sampled_df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plot_path = os.path.join(FIGURES_DIR, f"dist_{col}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved distribution for {col} to {plot_path}")


def correlation_analysis(df):
    print("\n===== CORRELATION ANALYSIS =====")
    numeric_df = df.select_dtypes(exclude=["object"])

    if numeric_df.empty:
        print("No numeric columns available for correlation analysis.")
        return

    # Sample to keep the correlation matrix computation reasonable on large datasets
    sample_size = min(len(numeric_df), CORR_SAMPLE_SIZE)
    numeric_df = numeric_df.sample(sample_size, random_state=RANDOM_SEED)

    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plot_path = os.path.join(FIGURES_DIR, "correlation_matrix.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved correlation heatmap to {plot_path}")
