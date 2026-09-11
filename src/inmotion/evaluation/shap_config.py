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
    model: str = "ts_jepa"
    checkpoint: Path = field(default_factory=lambda: Path("backup/ts_jepa_ft_seed42.pt"))
    seed: int = 42
    samples: int = 50
    nsamples: int = 150
    output_dir: Path = field(default_factory=lambda: Path("results"))
