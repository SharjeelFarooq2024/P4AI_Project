"""Error analysis for multiclass model on processed test split.

Loads:
- processed_data_mc/test_eval_processed.csv (expects 'target' column of class indices)
- saved_models/combined_rf.joblib (expects dict with 'model' and 'target_classes')

Outputs:
- Prints per-class confusion, metrics, low-confidence cases, and feature-wise patterns.
- Writes a summary report to analysis_output/error_analysis_report.txt
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).parent
TEST_PATH = ROOT / "processed_data_mc/test_eval_processed.csv"
MODEL_PATH = ROOT / "saved_models/combined_rf.joblib"
REPORT_PATH = ROOT / "analysis_output/error_analysis_report.txt"


def load_data():
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test split not found at {TEST_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model bundle not found at {MODEL_PATH}")

    df = pd.read_csv(TEST_PATH)
    if "target" not in df.columns:
        raise ValueError("Expected 'target' column in test data")

    bundle = joblib.load(MODEL_PATH)
    model = bundle.get("model", bundle)
    classes = bundle.get("target_classes")
    return df, model, classes


def align_features(X: pd.DataFrame, model) -> pd.DataFrame:
    """Align columns to model.feature_names_in_, filling missing with zero and dropping extras."""
    if not hasattr(model, "feature_names_in_"):
        return X
    feat_names = list(model.feature_names_in_)
    X_aligned = X.copy()
    for col in feat_names:
        if col not in X_aligned.columns:
            X_aligned[col] = 0
    X_aligned = X_aligned[feat_names]
    return X_aligned


def per_class_confusion(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    return df_cm


def misclassification_table(y_true, y_pred, classes):
    mask = y_true != y_pred
    if not mask.any():
        return pd.DataFrame()
    true_labels = [classes[int(i)] for i in y_true[mask]]
    pred_labels = [classes[int(i)] for i in y_pred[mask]]
    ct = pd.crosstab(pd.Series(true_labels, name="true"), pd.Series(pred_labels, name="pred"))
    return ct


def feature_importance(model, top_k: int = 8):
    if not hasattr(model, "feature_importances_"):
        return []
    importances = model.feature_importances_
    names = getattr(model, "feature_names_in_", [f"f{i}" for i in range(len(importances))])
    pairs = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
    return pairs[:top_k]


def compare_feature_distribution(X, y_true, y_pred, top_features):
    mask_err = y_true != y_pred
    mask_ok = ~mask_err
    rows = []
    for feat in top_features:
        if feat not in X.columns:
            continue
        c_med = X.loc[mask_ok, feat].median()
        e_med = X.loc[mask_err, feat].median()
        rows.append({
            "feature": feat,
            "median_correct": float(c_med),
            "median_error": float(e_med),
        })
    return rows


def low_confidence_cases(proba, threshold: float = 0.6):
    maxp = proba.max(axis=1)
    mask = maxp < threshold
    return mask, maxp


def main():
    df, model, classes = load_data()
    y_true = df["target"].to_numpy()
    X = df.drop(columns=["target"], errors="ignore")
    X = align_features(X, model)

    proba = model.predict_proba(X)
    y_pred = proba.argmax(axis=1)

    report = classification_report(y_true, y_pred, labels=range(len(classes)), target_names=classes, output_dict=True, zero_division=0)
    cm_df = per_class_confusion(y_true, y_pred, classes)
    miscls = misclassification_table(y_true, y_pred, classes)

    top_feats = [f for f, _ in feature_importance(model, top_k=8)]
    feat_drift = compare_feature_distribution(X, y_true, y_pred, top_feats)

    low_mask, maxp = low_confidence_cases(proba, threshold=0.6)

    # Summaries
    summary = {
        "dataset": {
            "rows": int(len(df)),
            "features": int(X.shape[1]),
        },
        "per_class_metrics": report,
        "confusion_matrix": cm_df.to_dict(),
        "misclassifications": miscls.to_dict(),
        "top_features": top_feats,
        "feature_medians_correct_vs_error": feat_drift,
        "low_confidence": {
            "threshold": 0.6,
            "count": int(low_mask.sum()),
            "fraction": float(low_mask.mean()),
            "by_true_class": pd.Series(y_true[low_mask]).value_counts().to_dict(),
            "by_pred_class": pd.Series(proba[low_mask].argmax(axis=1)).value_counts().to_dict(),
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(summary, indent=2))

    # Also print concise console view
    print("Rows:", len(df), "Features:", X.shape[1])
    print("\nPer-class metrics (precision/recall/f1):")
    for cls in classes:
        m = report[cls]
        print(f"  {cls}: p={m['precision']:.3f}, r={m['recall']:.3f}, f1={m['f1-score']:.3f}, support={m['support']}")

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm_df)

    if not miscls.empty:
        print("\nTop misclassifications:")
        print(miscls)

    if feat_drift:
        print("\nMedian comparison (correct vs error) on top features:")
        for row in feat_drift:
            print(f"  {row['feature']}: correct={row['median_correct']:.3g}, error={row['median_error']:.3g}")

    print(f"\nLow-confidence (<0.6) count: {low_mask.sum()} ({low_mask.mean():.3f})")
    print("Report written to", REPORT_PATH)


if __name__ == "__main__":
    main()
