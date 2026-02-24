"""
Lightweight analyzer stub for UNSW-NB15.
Provides the fields expected by main.py without heavy computation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.preprocessing.data_loader import DataLoader


@dataclass
class AnalysisResult:
    columns_to_drop: List[str]
    binary_columns: List[str]
    skip_scaling_columns: List[str]
    thresholds: Dict[str, float]
    stats_summary: Dict[str, object]


def analyze_dataset(
    data_dir: Path,
    output_dir: Path,
    sample_size: int = 150_000,
    missing_threshold: float = 0.2,
    correlation_threshold: float = 0.9,
    mi_threshold: float = 0.0,
    high_cardinality_threshold: int = 120,
    wide_range_threshold: float = 1_000.0,
    preset_drop: List[str] | None = None,
) -> AnalysisResult:
    # Load a sample of the combined raw splits
    loader = DataLoader(data_dir)
    df = loader.load_combined_data(nrows=sample_size)

    # Basic cleaning
    df.columns = df.columns.str.strip().str.lower()

    # Identify obvious drops
    preset_drop = preset_drop or []
    columns_to_drop = set(col.lower() for col in preset_drop)

    # Heuristic binary columns
    binary_columns: List[str] = []
    for col in df.columns:
        uniq = df[col].dropna().unique()
        if len(uniq) <= 2:
            binary_columns.append(col)

    # Skip scaling for label-like fields
    skip_scaling_columns = [c for c in df.columns if c in {"label", "is_sm_ips_ports", "is_ftp_login"}]

    # Stats summary for transparency
    stats_summary = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "missing_pct": df.isna().mean().round(4).to_dict(),
    }

    return AnalysisResult(
        columns_to_drop=sorted(columns_to_drop),
        binary_columns=sorted(set(binary_columns)),
        skip_scaling_columns=sorted(set(skip_scaling_columns)),
        thresholds={"wide_range_threshold": wide_range_threshold},
        stats_summary=stats_summary,
    )
