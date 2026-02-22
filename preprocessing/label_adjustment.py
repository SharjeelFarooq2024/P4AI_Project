from pathlib import Path
import pandas as pd

DATA_PATH = Path("processed_data_full")
OUTPUT_PATH = Path("processed_data_mc")

MIN_ATTACK_COUNT = 9_000

FEATURES_CSV = Path("data/NUSW-NB15_features.csv")

# def load_feature_names() -> list[str]:
#     # Feature catalog has a non-UTF8 apostrophe; use latin1 to avoid decode errors
#     catalog = pd.read_csv(FEATURES_CSV, encoding="latin1")
#     return catalog["Name"].str.strip().str.lower().tolist()

def load_combined(data_path: Path) -> pd.DataFrame:
    # names = load_feature_names()
    files = [
        DATA_PATH / "attacks_normal_half.csv",
    ]
    dfs = []
    for name in files:
        file_path = Path(name)
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping")
            continue
        try:
            # First try with explicit names (catalog). If column count mismatches, retry with file header.
            df = pd.read_csv(
                file_path,
                header=0, # names=names,
                encoding="latin1",
                encoding_errors="replace",
                low_memory=False,
            )
            # if df.shape[1] != len(names):
            #     raise ValueError(f"column mismatch: read {df.shape[1]} vs expected {len(names)}")
        except Exception as exc:
            print(f"Error reading {file_path} with catalog names: {exc}; retrying with file header")
            try:
                df = pd.read_csv(
                    file_path,
                    header=0,
                    encoding="latin1",
                    encoding_errors="replace",
                    low_memory=False,
                )
            except Exception as exc2:
                print(f"Error reading {file_path} without catalog names: {exc2}; skipping")
                continue
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No UNSW-NB15 part files found under data/")
    return pd.concat(dfs, ignore_index=True)

def main():
    df = load_combined(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()
    
    if "label" not in df.columns:
        raise ValueError("label column missing after load")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    
    attack_cat = df.get("attack_cat", pd.Series(["Normal"] * len(df)))
    attack_cat = (
        attack_cat.fillna("Normal")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.title()
        .replace({"-": "Normal", "Normal": "Normal"})
    )
    attack_counts = attack_cat[attack_cat != "Normal"].value_counts()
    rare_attacks = attack_counts[attack_counts < MIN_ATTACK_COUNT].index.tolist()
    if rare_attacks:
        print(f"Grouping rare attack categories (<{MIN_ATTACK_COUNT} rows) into Others: {rare_attacks}")
    attack_cat_grouped = attack_cat.where(~attack_cat.isin(rare_attacks), "Others")

    df["attack_cat"] = attack_cat_grouped
    df["target_mc"] = attack_cat_grouped
    df.loc[df["label"] == 0, "target_mc"] = "Normal"
    
    df = df.drop_duplicates().reset_index(drop=True)
    
    df.to_csv(OUTPUT_PATH / "combined_raw_split.csv", index=False)
    print(f"Combined dataset saved to {OUTPUT_PATH / 'combined_raw_split.csv'} with {len(df):,} rows and {df.shape[1]} columns")
    
if __name__ == "__main__":
    main()

