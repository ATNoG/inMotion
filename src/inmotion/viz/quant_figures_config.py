"""Typed options for the quantization-figure pipeline.

Kept beside the pipeline like every other `regenerate_*_config.py`, so the CLI
holds no figure logic and the pipeline holds no argument parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunConfig:
    """Command-line options for the quantization figures."""

    data_dir: str = "results/quant"
    """Where the statistics CSVs and the device benchmark JSONs live."""
    perf_dir: str = "results/perf"
    """Where the before/after latency CSVs live."""
    out_dir: str = "results/quant/figures"
    """Figure output directory (PDF + PNG, plus the materialised data/)."""
    dpi: int = 600
    formats: tuple[str, ...] = field(default_factory=lambda: ("pdf", "png"))
    seed: int = 42
    """Recorded in the materialised data tables for traceability."""
