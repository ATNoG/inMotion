"""Typed options for the bus-network figures.

Kept beside the pipeline like every other ``*_config.py`` in this repository, so
the CLI holds no figure logic and the pipeline holds no argument parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlotConfig:
    """Command-line options for the OD-matrix figures."""

    processed_dir: str = "data/bus/processed"
    """Where ``stop_times.csv`` and ``od_matrix*.csv`` live."""

    out_dir: str = "plots/bus"
    """Figure output directory."""

    service_group: str = "weekday"
    """Which service to compare: ``weekday`` or ``weekend``."""

    minimum_stops: int = 6
    """Routes with fewer stops than this are not compared."""

    label_stops: bool = True
    """Print every stop name beside its ladder. Off leaves only numbered ticks."""

    dpi: int = 200
    formats: tuple[str, ...] = field(default_factory=lambda: ("png",))
