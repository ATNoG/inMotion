"""Typed options for this pipeline.

Derived from the argparse definition it replaced; field names are unchanged so
the pipeline body did not need editing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunConfig:
    """Command-line options for this pipeline."""

    data: str = "data/processed/dataset.csv"
    quick: bool = False
    no_ensembles: bool = False
    optimize: bool = False
    n_trials: int = 100
    cv_folds: int = 5
    seed: int = 42
    n_jobs: int = -1
    skip_eda: bool = False
    storage: str = "sqlite:///optuna_studies.db"
    dashboard: bool = False
    no_seed_suffix: bool = False
    suffix_after_seed: str = ""
