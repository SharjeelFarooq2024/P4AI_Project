from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

from preprocessing.config import PreprocessingConfig
from preprocessing.pipeline import PreprocessingPipeline
from models.trainer import ModelTrainer


def main():
    data_path = Path("processed_data_mc/combined_raw_split.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    # Use the multiclass/binary target already built as target_mc; drop label/attack_cat
    if "target_mc" not in df.columns:
        raise ValueError("target_mc column missing in dataset")

    drop_cols = ["label", "attack_cat", "target_mc"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y_raw = df["target_mc"].fillna("Normal")

    # Factorize target_mc to numeric labels for the model, keep class map
    y, class_labels = pd.factorize(y_raw)
    y = pd.Series(y, name="target_mc")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    config = PreprocessingConfig(
        data_dir=Path("data"),
        output_dir=Path("processed_data_mc")
    )
    # Softer imbalance handling: SMOTE only for non-majority classes with smaller k
    config.imbalance_strategy = "smote"
    config.smote_sampling_strategy = "not majority"
    config.smote_k_neighbors = 3
    pipeline = PreprocessingPipeline(config)

    # Fit only on training split to avoid leakage; test is transformed with the fitted pipeline
    X_train_prep, y_train_prep = pipeline.fit_transform(
        X_train,
        y_train,
        apply_imbalance_handling=True,
    )
    X_test_prep = pipeline.transform(X_test)

    trainer = ModelTrainer(model_type="random_forest", random_state=42)
    trainer.train(X_train_prep, y_train_prep)
    metrics = trainer.evaluate(X_test_prep, y_test)

    # Echo key performance to console
    print("\nEvaluation summary (test set):")
    for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        if k in metrics:
            print(f"  {k}: {metrics[k]:.4f}")

    out_dir = Path("processed_data_mc")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist processed splits to inspect/avoid reprocessing
    pd.concat([X_train_prep, pd.Series(y_train_prep, name="target")], axis=1).to_csv(
        out_dir / "train_eval_processed.csv", index=False
    )
    pd.concat([X_test_prep, pd.Series(y_test, name="target")], axis=1).to_csv(
        out_dir / "test_eval_processed.csv", index=False
    )

    with (out_dir / "combined_eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    model_path = Path("models/saved/combined_rf.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": trainer.model,
        "feature_importance": trainer.feature_importance_,
        "training_stats": trainer.training_stats_,
        "pipeline_columns": list(X_train_prep.columns),
        "target_classes": list(class_labels),
    }, model_path)

    pipeline_path = out_dir / "combined_preprocessing_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)

    print("Saved metrics to", out_dir / "combined_eval_metrics.json")
    print("Saved model to", model_path)
    print("Saved preprocessing pipeline to", pipeline_path)


if __name__ == "__main__":
    main()
