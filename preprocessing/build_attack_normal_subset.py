from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

FEATURES_CSV = Path("data/NUSW-NB15_features.csv")
DATA_PATH = Path("data")
OUTPUT_PATH = Path("processed_data_full/attacks_normal_half.csv")


def load_feature_names() -> list[str]:
    catalog = pd.read_csv(FEATURES_CSV, encoding="latin1")
    return catalog["Name"].str.strip().str.lower().tolist()


def read_with_fallback(path: Path, names: list[str]) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"Warning: {path} not found; skipping")
        return None
    try:
        df = pd.read_csv(
            path,
            header=0,
            names=names,
            encoding="latin1",
            encoding_errors="replace",
            low_memory=False,
        )
        if df.shape[1] != len(names):
            raise ValueError(f"column mismatch: read {df.shape[1]} vs expected {len(names)}")
    except Exception as exc:
        print(f"Error reading {path} with catalog names: {exc}; retrying with file header")
        try:
            df = pd.read_csv(
                path,
                header=0,
                encoding="latin1",
                encoding_errors="replace",
                low_memory=False,
            )
        except Exception as exc2:
            print(f"Error reading {path} without catalog names: {exc2}; skipping")
            return None
    return df


def load_full_dataset(data_dir: Path) -> pd.DataFrame:
    names = load_feature_names()
    parts = [
        data_dir / "UNSW_NB15_training-set.csv",
        data_dir / "UNSW_NB15_testing-set.csv",
        data_dir / "UNSW-NB15_1.csv",
        data_dir / "UNSW-NB15_2.csv",
        data_dir / "UNSW-NB15_3.csv",
        data_dir / "UNSW-NB15_4.csv",
    ]
    dfs = []
    for path in parts:
        df = read_with_fallback(path, names)
        if df is not None:
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No UNSW-NB15 data files were found.")
    combined = pd.concat(dfs, ignore_index=True)
    combined.columns = combined.columns.str.strip().str.lower()
    return combined


def build_attack_normal_half(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    if "label" not in df.columns:
        raise ValueError("label column missing")
    labels = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df = df.assign(label=labels)

    attacks = df[df["label"] == 1]
    normals = df[df["label"] == 0]

    attack_count = len(attacks)
    target_normals = max(1, attack_count // 8)

    if len(normals) <= target_normals:
        sampled_normals = normals
        print(f"Normals available {len(normals):,} < target {target_normals:,}; taking all normals.")
    else:
        sampled_normals = normals.sample(n=target_normals, random_state=seed)
        print(f"Sampled {len(sampled_normals):,} normals out of {len(normals):,} to match 1/8 of attacks.")

    subset = pd.concat([attacks, sampled_normals], axis=0)
    subset = subset.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return subset


def main():
    df = load_full_dataset(DATA_PATH)
    print(f"Loaded full dataset: {len(df):,} rows, {df.shape[1]} columns")

    subset = build_attack_normal_half(df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(OUTPUT_PATH, index=False)

    attack_count = (subset["label"] == 1).sum()
    normal_count = (subset["label"] == 0).sum()
    print(
        f"Saved {OUTPUT_PATH} with {len(subset):,} rows. Attacks: {attack_count:,}, Normals: {normal_count:,}."
    )


if __name__ == "__main__":
    main()
