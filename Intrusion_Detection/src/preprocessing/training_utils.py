"""Utility helpers for multi-class UNSW-NB15 training.

This module centralizes dataset loading, splitting, preprocessing, balancing,
model training, evaluation, and artifact saving so training scripts can stay small.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE

from preprocessing import PreprocessingConfig, PreprocessingPipeline
from preprocessing.data_loader import DataLoader


MIN_ATTACK_COUNT = 9_000


def default_mc_config(data_dir: Path, output_dir: Path) -> PreprocessingConfig:
    return PreprocessingConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        target_column="target_mc",
        attack_category_column="attack_cat",
        imbalance_strategy="smote",
        encoding_strategy="hybrid",
        scaling_strategy="robust",
        columns_to_drop=sorted({"unnamed: 37", "unnamed: 38", "unnamed: 47", "srcip", "dstip"}),
        categorical_columns=["proto", "service", "state"],
        binary_columns=["is_sm_ips_ports", "is_ftp_login"],
        target_encoding_columns=["proto"],
        onehot_encoding_columns=["service", "state"],
        skip_scaling_columns=["is_ftp_login", "is_sm_ips_ports", "label", "target_mc"],
        correlation_threshold=0.90,
        mi_threshold=0.005,
        wide_range_threshold=1_000.0,
    )


def load_mc_dataset(data_dir: Path, sample_rows: int | None) -> Tuple[pd.DataFrame, pd.Series]:
    loader = DataLoader(data_dir)
    loader.load_feature_catalog()
    df = loader.load_combined_data(nrows=sample_rows)
    df.columns = df.columns.str.strip().str.lower()
    if "label" not in df.columns:
        raise ValueError("label column missing")

    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df["attack_cat"] = df.get("attack_cat", pd.Series(["Normal"] * len(df))).fillna("Normal")
    df["attack_cat"] = df["attack_cat"].replace({"-": "Normal", "normal": "Normal"})
    attack_counts = df.loc[df["attack_cat"] != "Normal", "attack_cat"].value_counts()
    rare_attacks = attack_counts[attack_counts < MIN_ATTACK_COUNT].index.tolist()
    if rare_attacks:
        print(f"Grouping rare attack categories (<{MIN_ATTACK_COUNT} rows) into Others: {rare_attacks}")
    df["attack_cat"] = df["attack_cat"].where(~df["attack_cat"].isin(rare_attacks), "Others")

    df["target_mc"] = df["attack_cat"]
    df.loc[df["label"] == 0, "target_mc"] = "Normal"

    y = df["target_mc"].astype(str)
    X = df.drop(columns=["target_mc", "attack_cat", "label"], errors="ignore")

    X, y = shuffle(X, y, random_state=42)
    dup_mask = X.duplicated()
    if dup_mask.any():
        X = X.loc[~dup_mask].reset_index(drop=True)
        y = y.loc[~dup_mask].reset_index(drop=True)
        print(f"Dropped {dup_mask.sum():,} duplicate rows; new size {len(X):,}")

    return X, y


def split_mc(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    vc = y.value_counts()
    rare = vc[vc < 3].index.tolist()
    if rare:
        print(f"Merging rare classes (<3 samples) into RARE: {rare}")
    y_strat = y.where(~y.isin(rare), "RARE")
    if y_strat.value_counts().min() < 2:
        keep_mask = y_strat != "RARE"
        removed = int((~keep_mask).sum())
        X = X.loc[keep_mask].reset_index(drop=True)
        y = y.loc[keep_mask].reset_index(drop=True)
        y_strat = y_strat.loc[keep_mask].reset_index(drop=True)
        print(f"Dropped {removed} ultra-rare rows (<2) to enable stratification")
    strat_base = y_strat if y_strat.value_counts().min() >= 2 else None
    X_train, X_tmp, y_train, y_tmp, y_strat_train, y_strat_tmp = train_test_split(
        X,
        y,
        y_strat,
        test_size=0.6,
        stratify=strat_base,
        random_state=seed,
        shuffle=True,
    )

    strat_tmp = y_strat_tmp if y_strat_tmp.value_counts().min() >= 2 else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=2 / 3,
        stratify=strat_tmp,
        random_state=seed,
        shuffle=True,
    )
    print(
        f"Splits: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}; "
        f"classes={y.nunique()}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_preprocess_mc(X_train, y_train, X_val, X_test, config: PreprocessingConfig):
    pipe = PreprocessingPipeline(config)
    label_codes = y_train.astype("category").cat.codes
    pipe.fit(X_train, label_codes)
    X_train_proc = pipe.transform(X_train)
    X_val_proc = pipe.transform(X_val)
    X_test_proc = pipe.transform(X_test)
    return pipe, X_train_proc, y_train, X_val_proc, X_test_proc


def balance_train(X_train_proc, y_train_proc):
    counts = y_train_proc.value_counts()
    min_count = counts.min()
    if min_count < 2:
        print(f"Skipping SMOTE: min class count {min_count} < 2 -> using original train set")
        return X_train_proc, y_train_proc

    smote = SMOTE(random_state=42, k_neighbors=min(5, min_count - 1))
    Xb, yb = smote.fit_resample(X_train_proc, y_train_proc)
    print(f"SMOTE: {len(X_train_proc):,}->{len(Xb):,} rows")
    return Xb, yb


def train_rf(X_train, y_train) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_split(model, X, y, split: str):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    report = classification_report(y, preds, output_dict=True)
    cm = confusion_matrix(y, preds).tolist()
    print(f"{split} accuracy: {acc:.4f}")
    return {"accuracy": acc, "report": report, "confusion_matrix": cm}


def save_artifacts(pipe, model, results, output_dir: Path, model_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, output_dir / "preprocessing_pipeline_mc.joblib")
    joblib.dump(model, model_dir / "random_forest_mc.joblib")
    with (output_dir / "mc_results.json").open("w") as f:
        json.dump(results, f, indent=2)
