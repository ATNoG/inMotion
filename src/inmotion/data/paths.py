"""Canonical locations for data files, and resolution of dataset references.

The project has three data tiers, matching how a dataset is actually produced:

``raw/``
    Captures straight off the access point (Wavecom ``.txt`` dumps). Never
    edited; the input to `inmotion data export`.
``interim/``
    One CSV per collection session, as written by ``export``. Input to
    `inmotion data merge`.
``processed/``
    Merged and augmented datasets that training and evaluation consume.

Before this module the tiers all lived in the repository root or in
``wavecom_files/``, and code referred to datasets by bare filename
(``dataset.csv``) from whatever the current directory happened to be.

:func:`resolve` keeps those bare references working by searching the tiers in
order, so an old command line such as ``--data dataset.csv`` still finds the
file, while new code uses explicit paths.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Tier search order for a bare filename or relative path.
SEARCH_ORDER = (PROCESSED_DIR, INTERIM_DIR, RAW_DIR, REPO_ROOT)


def resolve(reference: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve a dataset reference to a path.

    Accepts an absolute path, a path relative to the repository root, a path
    relative to the current directory, or a bare filename. Bare filenames are
    searched across the data tiers in :data:`SEARCH_ORDER`.

    With ``must_exist=True`` an unresolved reference raises, which is what
    training entry points should use so a typo fails immediately rather than
    producing an empty dataset.
    """
    path = Path(reference)
    if path.is_absolute():
        if path.exists() or not must_exist:
            return path
        raise FileNotFoundError(f"dataset not found: {path}")

    # Absolute candidates only, so callers are unaffected by the current
    # directory (training scripts are launched from several places).
    candidates = [
        (REPO_ROOT / path).resolve(),
        path.resolve(),
        *((tier / path).resolve() for tier in SEARCH_ORDER),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    if must_exist:
        searched = ", ".join(str(c) for c in dict.fromkeys(candidates))
        raise FileNotFoundError(f"{reference!r} not found. Searched: {searched}")
    # Unresolved: assume the caller wants a new file in the processed tier.
    return (PROCESSED_DIR / path.name).resolve()


def processed(name: str) -> Path:
    """Path to a processed dataset, by bare filename."""
    return PROCESSED_DIR / name


def raw(name: str) -> Path:
    """Path to a raw capture, by bare filename."""
    return RAW_DIR / name


def interim(name: str) -> Path:
    """Path to an interim per-collection CSV, by bare filename."""
    return INTERIM_DIR / name


def list_processed() -> list[Path]:
    """Every file in the processed tier, sorted."""
    if not PROCESSED_DIR.is_dir():
        return []
    return sorted(p for p in PROCESSED_DIR.iterdir() if p.suffix == ".csv")


def list_raw() -> list[Path]:
    """Every raw capture, sorted."""
    if not RAW_DIR.is_dir():
        return []
    return sorted(RAW_DIR.rglob("*.txt"))
