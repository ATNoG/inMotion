"""Stop identity: one canonical key per stop per operator, and the stop dictionary."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd


def canonical(name: str) -> str:
    """A stable stop key: lowercase, unaccented, alphanumerics only.

    Ordinal indicators are dropped before decomposition rather than folded into
    a letter, because the bitmap OCR reads ``Qtª`` as both ``Qta`` and ``Qt?``:
    dropping it puts every reading of the same stop on one key.
    """
    stripped = unicodedata.normalize("NFKD", name.replace("\u00aa", "").replace("\u00ba", ""))
    ascii_only = "".join(c for c in stripped if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", ascii_only.lower())).strip()


def stop_id(operator: str, stop: str) -> str:
    """The canonical id of one stop of one operator."""
    return f"{operator}:{canonical(stop).replace(' ', '-')}"


def stops_frame(stop_times: pd.DataFrame) -> pd.DataFrame:
    """The stop dictionary: one row per operator-specific stop."""
    served = stop_times[stop_times["served"]]
    grouped = (
        served.groupby(["operator", "stop_id", "stop"], as_index=False)
        .agg(
            trips=("trip_id", "nunique"),
            lines=("line", lambda values: " ".join(sorted(set(values)))),
        )
        .sort_values(["operator", "stop"])
    )
    return grouped[["stop_id", "operator", "stop", "trips", "lines"]]
