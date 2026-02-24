"""Run multiclass inference on captured_features.csv using existing pipeline/model.

Expects:
- Preprocessing pipeline: processed_data_mc/combined_preprocessing_pipeline.joblib
- Model bundle: saved_models/combined_rf.joblib (dict with keys: model, target_classes, pipeline_columns, ...)
- Captured data CSV: captured_features.csv (change via --input)

Usage (from Intrusion_Detection):
    python run_captured_inference.py --input captured_features.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).parent
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_artifacts(pipeline_path: Path, model_path: Path):
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Missing pipeline artifact at {pipeline_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact at {model_path}")

    pipeline = joblib.load(pipeline_path)
    bundle = joblib.load(model_path)
    model = bundle.get("model", bundle)
    classes = bundle.get("target_classes")
    return pipeline, model, classes


def predict(df: pd.DataFrame, pipeline, model, classes):
    # Normalize column names to match training (lowercase, stripped)
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    # Best-effort alignment before transform: ensure expected scaled cols exist
    if getattr(pipeline, "scaler", None):
        expected_cols = set(getattr(pipeline.scaler, "scaled_columns_", []))
        expected_cols.update(getattr(pipeline, "wide_range_cols_", []) or [])
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0

    X_proc = pipeline.transform(df)

    # Align to model's feature names if available
    if hasattr(model, "feature_names_in_"):
        feat_names = list(model.feature_names_in_)
        # If transform returned ndarray, wrap to DataFrame
        if not isinstance(X_proc, pd.DataFrame):
            X_proc = pd.DataFrame(X_proc, columns=feat_names[: X_proc.shape[1]])
        # Add missing columns as zeros
        for col in feat_names:
            if col not in X_proc.columns:
                X_proc[col] = 0
        # Drop extras and order exactly
        X_proc = X_proc[feat_names]

    preds = model.predict(X_proc)
    proba = model.predict_proba(X_proc) if hasattr(model, "predict_proba") else None

    if classes is not None:
        preds_labels = [classes[int(i)] for i in preds]
    else:
        preds_labels = preds
    return preds, preds_labels, proba


def main():
    parser = argparse.ArgumentParser(description="Run inference on captured features with existing multiclass model")
    parser.add_argument("--input", type=Path, default=ROOT / "captured_features.csv", help="Path to captured_features.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "captured_predictions.csv", help="Where to save predictions")
    parser.add_argument("--pipeline", type=Path, default=ROOT / "processed_data_mc/combined_preprocessing_pipeline.joblib", help="Path to preprocessing pipeline joblib")
    parser.add_argument("--model", type=Path, default=ROOT / "saved_models/combined_rf.joblib", help="Path to trained model bundle joblib")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    print(f"Loaded captured data: {df.shape}")

    pipeline, model, classes = load_artifacts(args.pipeline, args.model)
    preds, preds_labels, proba = predict(df, pipeline, model, classes)

    out_df = pd.DataFrame({
        "prediction_code": preds,
        "prediction_label": preds_labels,
    })
    if proba is not None:
        proba_df = pd.DataFrame(proba, columns=[f"proba_{i}" for i in range(proba.shape[1])])
        out_df = pd.concat([out_df.reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")
    if classes is not None:
        print("Class mapping:")
        for idx, name in enumerate(classes):
            print(f"  {idx}: {name}")


if __name__ == "__main__":
    main()
