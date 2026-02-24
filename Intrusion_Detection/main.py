import json
import time
import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure local src/ is importable when running from repo root.
ROOT = Path(__file__).parent
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyzer import analyze_dataset
from src.preprocessing import PreprocessingConfig, PreprocessingPipeline, DataLoader
from src.models.trainer import ModelTrainer


DATA_DIR = Path("data")
OUTPUT_DIR = Path("processed_data_full")
MODEL_DIR = Path("saved_models")
ANALYZER_SAMPLE_SIZE = 150_000


def _json_default(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="list")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    return str(obj)


def _save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def load_full_dataset(data_dir: Path) -> Tuple[pd.DataFrame, pd.Series]:
    loader = DataLoader(data_dir)
    loader.load_feature_catalog()

    print("\n=== Loading combined raw files (1-4) ===")
    df = loader.load_combined_data()
    df.columns = df.columns.str.strip().str.lower()

    if "label" not in df.columns:
        raise ValueError("label column missing after load")

    y = df["label"].astype(int)
    X = df.drop(columns=["label", "attack_cat"], errors="ignore")

    print(f"Full dataset: {len(df):,} rows, {X.shape[1]} features after dropping targets")
    print("Class distribution:", y.value_counts(normalize=True).round(4).to_dict())
    return X, y


def run_preprocessing_step(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    config: PreprocessingConfig,
):
    pipeline = PreprocessingPipeline(config)
    print("\n=== Fitting preprocessing pipeline ===")
    t0 = time.time()
    X_train_proc, y_train_proc = pipeline.fit_transform(
        X_train, y_train, apply_imbalance_handling=False
    )
    X_test_proc = pipeline.transform(X_test)
    duration = time.time() - t0
    print(f"Preprocessing done in {duration:.2f}s")
    return pipeline, X_train_proc, y_train_proc, X_test_proc, duration


def save_preprocessing_artifacts(
    pipeline: PreprocessingPipeline,
    X_train_proc: pd.DataFrame,
    y_train_proc: pd.Series,
    X_test_proc: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    train_out = X_train_proc.copy()
    train_out["label"] = y_train_proc.values
    test_out = X_test_proc.copy()
    test_out["label"] = y_test.values

    train_path = output_dir / "train_processed_full.csv"
    test_path = output_dir / "test_processed_full.csv"
    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)
    print(f"Saved processed train to {train_path}")
    print(f"Saved processed test to {test_path}")

    pipeline_path = output_dir / "preprocessing_pipeline_full.joblib"
    joblib.dump(pipeline, pipeline_path)
    print(f"Saved pipeline to {pipeline_path}")

    report_text = pipeline.generate_report()
    (output_dir / "preprocessing_report_full.txt").write_text(report_text)

    stats_serializable = {
        k: sorted(v) if isinstance(v, set) else v for k, v in pipeline.pipeline_stats_.items()
    }
    _save_json(output_dir / "pipeline_stats_full.json", stats_serializable)

    enc_report = pipeline.encoder.get_encoding_report()
    if not enc_report.empty:
        enc_report.to_csv(output_dir / "encoding_report_full.csv", index=False)


def run_training_step(
    X_train_proc: pd.DataFrame,
    y_train_proc: pd.Series,
    X_test_proc: pd.DataFrame,
    y_test: pd.Series,
    model_dir: Path,
):
    trainer = ModelTrainer(verbose=True)
    print("\n=Training model")
    t1 = time.time()
    trainer.train(X_train_proc, y_train_proc)
    train_time = time.time() - t1

    print("\n Evaluating model ")
    metrics = trainer.evaluate(X_test_proc, y_test)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "random_forest_full_pipeline.joblib"
    trainer.save_model(model_path)

    if trainer.feature_importance_ is not None:
        top_path = OUTPUT_DIR / "feature_importance_top50.csv"
        trainer.get_top_features(50).to_csv(top_path, index=False)

    return trainer, metrics, train_time, model_path


def main():
    print("UNSW-NB15 FULL DATASET PIPELINE")

    print("\nRunning analyzer to derive preprocessing parameters")
    analysis = analyze_dataset(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        sample_size=ANALYZER_SAMPLE_SIZE,
        missing_threshold=0.20,
        correlation_threshold=0.90,
        mi_threshold=0.0,
        high_cardinality_threshold=120,
        wide_range_threshold=1_000.0,
        preset_drop=["unnamed: 37", "unnamed: 38", "unnamed: 47"],
    )

    config = PreprocessingConfig(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        imbalance_strategy="none",
        scaling_strategy="robust",
        encoding_strategy="hybrid",
        columns_to_drop=sorted(set(analysis.columns_to_drop + ["srcip", "dstip"])),
        categorical_columns=["proto", "service", "state"],
        binary_columns=sorted(set(analysis.binary_columns + ["is_sm_ips_ports"])),
        target_encoding_columns=["proto"],
        onehot_encoding_columns=["service", "state"],
        skip_scaling_columns=sorted(set(analysis.skip_scaling_columns + ["label", "is_sm_ips_ports"])),
        correlation_threshold=0.90,
        mi_threshold=0.005,
        wide_range_threshold=analysis.thresholds["wide_range_threshold"],
    )

    X, y = load_full_dataset(DATA_DIR)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=config.random_state,
        shuffle=True,
    )

    print(f"Train: {len(X_train):,} rows; Test: {len(X_test):,} rows")

    pipeline, X_train_proc, y_train_proc, X_test_proc, preprocess_time = run_preprocessing_step(
        X_train, X_test, y_train, config
    )

    save_preprocessing_artifacts(
        pipeline, X_train_proc, y_train_proc, X_test_proc, y_test, OUTPUT_DIR
    )

    trainer, metrics, train_time, model_path = run_training_step(
        X_train_proc, y_train_proc, X_test_proc, y_test, MODEL_DIR
    )

    results = {
        "train_samples": int(len(X_train_proc)),
        "test_samples": int(len(X_test_proc)),
        "preprocess_time_sec": preprocess_time,
        "train_time_sec": train_time,
        "analyzer_summary": analysis.stats_summary,
        "metrics": {k: v for k, v in metrics.items() if k not in {"confusion_matrix", "classification_report"}},
        "confusion_matrix": metrics.get("confusion_matrix"),
        "classification_report": metrics.get("classification_report"),
        "model_path": str(model_path),
    }

    _save_json(OUTPUT_DIR / "full_dataset_results.json", results)
    print(f"Model saved to: {model_path}")
    print(f"Results saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
