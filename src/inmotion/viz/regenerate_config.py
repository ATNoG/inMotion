"""Typed options for this pipeline.

Derived from the argparse definition it replaced; field names are unchanged so
the pipeline body did not need editing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunConfig:
    """Command-line options for this pipeline."""

    results_dir: str = None
    csv: str = None
    detailed_csv: str = None
    plots_dir: str = None
    top_n: int = 15
    format: str = "pdf"
    eda: bool = False
    data: str = "data/processed/dataset.csv"
    seed: int = 42
    skip_classification_plots: bool = False
