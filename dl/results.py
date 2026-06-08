"""Extended results I/O for DL pipeline — detailed per-class metrics + confusion matrices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def save_extended_results_csv(
    rows: list[dict[str, object]],
    path: Path,
    classes: list[str] | None = None,
) -> None:
    """Save extended results rows to CSV with all per-class and CM columns."""
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_extended_results(
    csv_path: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
    list[str],
    dict[str, dict[str, dict[str, float]]],
]:
    """Load extended DL results CSV.

    Returns:
        Tuple of (results_df, confusion_matrices, class_names, per_class_reports).
    """
    df = pd.read_csv(csv_path)

    # Infer class names from per-class metric columns
    class_names = []
    for col in df.columns:
        if col.endswith("_precision") and not col.startswith("CM_"):
            cls_name = col.replace("_precision", "")
            if cls_name not in class_names:
                class_names.append(cls_name)

    # Parse confusion matrices from JSON column
    confusion_matrices: dict[str, np.ndarray] = {}
    cm_col = "Confusion_Matrix"
    if cm_col in df.columns:
        for _, row in df.iterrows():
            model_name = row["model"]
            if pd.notna(row.get(cm_col)):
                try:
                    cm_list = json.loads(row[cm_col])
                    confusion_matrices[model_name] = np.array(cm_list)
                except (json.JSONDecodeError, TypeError):
                    pass

    # Parse per-class reports
    per_class_reports: dict[str, dict[str, dict[str, float]]] = {}
    for _, row in df.iterrows():
        model_name = row["model"]
        report: dict[str, dict[str, float]] = {}
        for cls_name in class_names:
            p_col = f"{cls_name}_precision"
            r_col = f"{cls_name}_recall"
            f1_col = f"{cls_name}_f1"
            if p_col in row and pd.notna(row.get(p_col)):
                report[cls_name] = {
                    "precision": float(row[p_col]) if pd.notna(row.get(p_col)) else 0.0,
                    "recall": float(row[r_col]) if pd.notna(row.get(r_col)) else 0.0,
                    "f1-score": float(row[f1_col]) if pd.notna(row.get(f1_col)) else 0.0,
                }
        if report:
            per_class_reports[model_name] = report

    return df, confusion_matrices, class_names, per_class_reports
