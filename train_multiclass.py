"""Multi-class training for UNSW-NB15 with leakage-safe split and SMOTE.

Defaults:
- Split: train 40%, val 20%, test 40% (stratified on target_mc)
- Target: target_mc (Normal vs attack categories)
- Preprocessing: drop noisy cols, cap outliers, encode proto/service/state, robust scale
- Balancing: SMOTE on train only after encoding/scaling
- Model: RandomForestClassifier

Supports optional --sample-rows for quick dry runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from preprocessing.training_utils import (
    balance_train,
    default_mc_config,
    evaluate_split,
    fit_preprocess_mc,
    load_mc_dataset,
    save_artifacts,
    split_mc,
    train_rf,
)

DATA_DIR = Path("data")
OUTPUT_DIR = Path("processed_data_mc")
MODEL_DIR = Path("models/saved")

def main():
    parser = argparse.ArgumentParser(description="Train multi-class UNSW-NB15 model with SMOTE")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--sample-rows", type=int, default=None, help="Optional row limit for dry runs")
    args = parser.parse_args()

    X, y = load_mc_dataset(args.data_dir, args.sample_rows)
    config = default_mc_config(args.data_dir, OUTPUT_DIR)

    X_train, X_val, X_test, y_train, y_val, y_test = split_mc(X, y)
    pipe, X_train_p, y_train_p, X_val_p, X_test_p = fit_preprocess_mc(X_train, y_train, X_val, X_test, config)

    X_train_bal, y_train_bal = balance_train(X_train_p, y_train_p)
    model = train_rf(X_train_bal, y_train_bal)

    val_metrics = evaluate_split(model, X_val_p, y_val, "val")
    test_metrics = evaluate_split(model, X_test_p, y_test, "test")

    results = {
        "val": val_metrics,
        "test": test_metrics,
        "train_size": int(len(X_train_bal)),
        "val_size": int(len(X_val_p)),
        "test_size": int(len(X_test_p)),
    }
    save_artifacts(pipe, model, results, OUTPUT_DIR, MODEL_DIR)
    print("Artifacts saved to processed_data_mc and models/saved")


if __name__ == "__main__":
    main()
