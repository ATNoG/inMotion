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
    out: Path = None
    report_dir: Path = field(default_factory=lambda: Path("results"))
    within_floor: float = 0.15
    lodo_floor: float = 0.05
    keep_shifted: bool = False
    drop_ambiguous: bool = False
    ambiguity_thresh: float = 0.85
