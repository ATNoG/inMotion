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
    steps: int = 100
    no_tabpfn: bool = False
    no_classical: bool = False
    members: str = "all"
    select: bool = False
    size_frontier: bool = False
