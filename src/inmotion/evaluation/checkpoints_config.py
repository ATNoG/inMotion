"""Typed options for this pipeline.

Derived from the argparse definition it replaced; field names are unchanged so
the pipeline body did not need editing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from inmotion.data.paths import processed


@dataclass
class RunConfig:
    """Command-line options for this pipeline."""

    seed: int = 42
    data: Path = field(default_factory=lambda: processed("dataset.csv"))
    checkpoints_dir: Path = field(default_factory=lambda: Path("models/dl"))
    output: Path = None
    batch_size: int = 64
    no_wandb: bool = False
    models_dir: Path = None
    optuna_db: str = os.environ.get("INMOTION_OPTUNA_STORAGE", "sqlite:///optuna_dl.db")
