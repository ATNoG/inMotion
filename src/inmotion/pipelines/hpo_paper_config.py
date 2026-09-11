"""Typed options for this pipeline.

Derived from the argparse definition it replaced; field names are unchanged so
the pipeline body did not need editing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from inmotion.data.paths import processed


@dataclass
class RunConfig:
    """Command-line options for this pipeline."""

    data: Path = field(default_factory=lambda: processed("dataset.csv"))
    seed: int = 42
    epochs: int = 150
    patience: int = 20
    # None means 'every model in PAPER_HPO_CONFIGS'; set from --models.
    models: list[str] | None = None
    models_dir: Path = field(default_factory=lambda: Path("models/dl"))
    results_dir: Path = field(default_factory=lambda: Path("results/dl"))
