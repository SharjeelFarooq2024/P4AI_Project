import numpy as np


def check_basic_info(df):
    print("===== BASIC DATASET INFO =====")
    print(f"Total samples: {df.shape[0]}")
    print(f"Total features: {df.shape[1]}")
    print("\nColumn Names:")
    print(df.columns.tolist())
    print("\nData Types:")
    print(df.dtypes)
    print("\nMemory Usage (MB):")
    print(round(df.memory_usage(deep=True).sum() / (1024**2), 2))


def check_duplicates(df):
    duplicates = df.duplicated().sum()
    print("\n===== DUPLICATE CHECK =====")
    print(f"Number of duplicate rows: {duplicates}")
    return duplicates


def check_missing_values(df):
    print("\n===== MISSING VALUES CHECK =====")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(missing if not missing.empty else "No missing values found.")


def check_infinite_values(df):
    print("\n===== INFINITE VALUES CHECK =====")
    numeric_df = df.select_dtypes(include=[np.number])
    inf_count = np.isinf(numeric_df).sum().sum()
    print(f"Number of infinite values: {inf_count}")
