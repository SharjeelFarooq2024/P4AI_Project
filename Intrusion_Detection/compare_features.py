"""Compare captured_features.csv against UNSW-NB15_1.csv and print deltas."""
import pandas as pd
from pathlib import Path
from src.data_loader import load_feature_names


def main() -> None:
    base = Path(__file__).parent  # Intrusion_Detection folder
    captured_path = base / "captured_features.csv"
    unsw_path = base / "data" / "raw" / "UNSW-NB15_1.csv"

    if not captured_path.exists():
        raise FileNotFoundError(f"Missing {captured_path}. Run run_capture.py first.")

    captured = pd.read_csv(captured_path)
    unsw = pd.read_csv(unsw_path, header=None, low_memory=False)
    unsw.columns = load_feature_names()

    # Column coverage
    captured_cols = set(captured.columns)
    unsw_cols = set(unsw.columns)
    missing = sorted(unsw_cols - captured_cols)
    extra = sorted(captured_cols - unsw_cols)
    print("Columns missing vs UNSW:", missing)
    print("Columns extra vs UNSW:", extra)

    # Key numeric comparisons
    key_cols = ["sbytes", "dbytes", "sttl", "dttl", "Spkts", "Dpkts"]
    print("\nKey column non-zero fraction and means (captured vs UNSW):")
    for col in key_cols:
        if col not in captured:
            continue
        nz = (captured[col] != 0).mean()
        mean_c = captured[col].mean()
        mean_u = unsw[col].mean() if col in unsw else float("nan")
        print(f"{col:6s} nz={nz:5.3f} | mean cap={mean_c:12.3f} | mean unsw={mean_u:12.3f}")

    # Proto/service distribution (top 5)
    if "proto" in captured:
        print("\nCaptured proto counts (top 5):")
        print(captured["proto"].value_counts().head())
    if "service" in captured:
        print("\nCaptured service counts (top 5):")
        print(captured["service"].value_counts().head())

    # Sample rows for sanity
    print("\nCaptured sample rows:")
    print(captured.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
