"""Typed options for the DL permutation feature-importance pipeline.

Derived from the argparse definition it replaced; the field names are unchanged,
so the pipeline body did not need editing. The CLI builds this from typed
options (`inmotion evaluate feature-importance`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from inmotion.data.paths import processed

# Repository root, for the artifact directories this pipeline writes to.
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class RunConfig:
    """Command-line options for the feature-importance pipeline."""

    data: Path | None = None
    seed: int = 42
    epochs: int = 100
    patience: int = 20
    output_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "docs" / "icaisf" / "paper" / "images"
    )
    checkpoints_dir: Path = field(default_factory=lambda: REPO_ROOT / "models" / "dl")
    n_repeats: int | None = None

    def resolved_data(self) -> Path:
        """The dataset to use, falling back to the processed default."""
        return self.data if self.data is not None else processed("dataset.csv")
