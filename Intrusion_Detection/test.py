from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

FEATURES_CSV = Path("data/NUSW-NB15_features.csv")
DATA_PATH = Path("data")
OUTPUT_PATH = Path("processed_data_full/attacks_normal_half.csv")

def load_features_names():
    df = pd.read_csv(FEATURES_CSV, encoding="latin1")
    return df["Name"].str.strip().str.lower().tolist()

def read_file(path: Path, names: list[str]) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"Warning: {path} not found; skipping")
        return None
    try:
        df = pd.read_csv(path, header=0, names= names, encoding="latin1", encoding_errors = "replace", low_memory=False)
        if df.shape[1] != len(names):
            raise ValueError(f"column mismatch: read {df.shape[1]} vs expected {len(names)}")
    except Exception as exc:
        print(f"Error reading {path} with catalog names: {exc}; retrying with file header")
        try:
            df = pd.read_csv(path, header = 0, encoding= "latin1", encoding_errors = "replace", low_memory= False)
        except Exception as exc2:
            print(f"Error reading {path} without catalog names {exc2}; skipping")
            return None
    return df

def load_dataset():
    names = load_features_names()
    parts = [
        DATA_PATH / "UNSW-NB15_1.csv",
        DATA_PATH / "UNSW-NB15_2.csv",
        DATA_PATH / "UNSW-NB15_3.csv",
        DATA_PATH / "UNSW-NB15_4.csv",
    ]
    dfs = []
    for path in parts:
        df = read_file(path, names)
        if df is not None:
            dfs.append(df)
        
        if not dfs:
            raise FileNotFoundError("No UNSW-NB15 part files found under data/")
        combined = pd.concat(dfs, ignore_index=True)
        combined.columns = combined.columns.str.strip().str.lower()
    return combined

def build_attack_normal_subset(df: pd.DataFrame) -> pd.DataFrame:
    attack_df = df[df["label"] == 1]
    normal_df = df[df["label"] == 0]
    
    attack_count = len(attack_df)
    target_normals = max(1, ((attack_count//8) - 5000))
    normal_sample = normal_df.sample(n=target_normals, random_state=42)
    if len(normal_df) <= target_normals:
        sampled_normals = normal_df
        print(f"Normals available {len(normal_df):,} < target {target_normals:,}; taking all normals.")
    else:
        sampled_normals = normal_df.sample(n=target_normals, random_state=42)
        print(f"Sampled {len(sampled_normals):,} normals out of {len(normal_df):,} to match 1/8 of attacks.")
    
    subset = pd.concat([attack_df, sampled_normals], axis= 0)
    subset = subset.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return subset

def main():
    df = load_dataset()
    subset = build_attack_normal_subset(df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved attack-normal half subset to {OUTPUT_PATH} with {len(subset):,} rows.")
    
    
if __name__ == "__main__":
    main()