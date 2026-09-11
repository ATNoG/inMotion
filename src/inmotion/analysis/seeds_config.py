"""Typed options for this pipeline.

Derived from the argparse definition it replaced; field names are unchanged so
the pipeline body did not need editing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunConfig:
    """Command-line options for this pipeline."""

    seeds: str = "3,5,42"
    output_dir: str = "seeds_results_combined"
