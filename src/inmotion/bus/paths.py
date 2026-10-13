"""Canonical locations for the bus-network artifacts and derived tables.

The three tiers mirror `inmotion.data.paths`, with the bus network kept apart
from the WiFi captures so neither can be mistaken for the other:

``raw/``
    Whatever the operator sites served, byte for byte: BusWay line pages and
    AveiroBus line PDFs, plus ``manifest.json``. A cache, not a deliverable.
``interim/``
    One CSV per operator, straight out of its parser, before the two networks
    are reconciled.
``processed/``
    The merged stop/trip/stop-time tables and the OD matrices derived from them.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

BUS_DIR = REPO_ROOT / "data" / "bus"
RAW_DIR = BUS_DIR / "raw"
INTERIM_DIR = BUS_DIR / "interim"
PROCESSED_DIR = BUS_DIR / "processed"

MANIFEST = RAW_DIR / "manifest.json"


def raw_dir(operator: str) -> Path:
    """Raw artifact directory of one operator, created on demand."""
    path = RAW_DIR / operator
    path.mkdir(parents=True, exist_ok=True)
    return path


def interim(name: str) -> Path:
    """Path to an interim table, by bare filename."""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    return INTERIM_DIR / name


def processed(name: str) -> Path:
    """Path to a processed table, by bare filename."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return PROCESSED_DIR / name
