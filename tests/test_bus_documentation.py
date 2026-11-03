"""The results document states numbers; this holds them to the tables.

`AI_RESULT.md` is the record of what the bus build produced, and nothing
regenerates it. A rebuild that is not written down therefore leaves a document
that reads as authoritative and is not — which is how the counts once came to
describe a build two parser revisions old, with the aggregate still quoting the
pre-fix banding.

The numbers below are read *out of the document* rather than repeated here, so
the record and the tables can only move together: a rebuild that changes them
fails this, and the fix is to state the new numbers in the document.

Skips when the tables are absent, since `data/bus/processed` is not tracked.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest
from conftest import REPO_ROOT

from inmotion.bus import od, paths

DOCUMENT = REPO_ROOT / "AI_RESULT.md"

requires_tables = pytest.mark.skipif(
    not (paths.PROCESSED_DIR / "od_matrix.csv").exists(),
    reason="bus tables not built (data/bus/processed is not tracked)",
)
requires_document = pytest.mark.skipif(
    not DOCUMENT.exists(),
    reason="AI_RESULT.md is not tracked, so a clone has no record to hold to the tables",
)


def _stated(pattern: str, text: str) -> str:
    """The number the document states, refusing to pass when it stopped stating it."""
    match = re.search(pattern, text)
    assert match is not None, f"AI_RESULT.md no longer states anything matching {pattern!r}"
    return match.group(1).replace(",", "")


def _document() -> str:
    return DOCUMENT.read_text()


@requires_tables
@requires_document
def test_the_document_states_the_row_counts_of_the_tables() -> None:
    text = _document()
    stated = {
        "stop_times.csv": _stated(r"\| `stop_times\.csv` \| ([\d,]+) \|", text),
        "trips.csv": _stated(r"\| `trips\.csv` \| ([\d,]+) \|", text),
        "stops.csv": _stated(r"\| `stops\.csv` \| ([\d,]+) \|", text),
        "od_pairs.csv": _stated(r"\| `od_pairs\.csv` \| ([\d,]+) \|", text),
        "od_matrix.csv": _stated(r"\| `od_matrix\.csv` \| ([\d,]+) \|", text),
        od.TIMELINE_FILE: _stated(rf"\| `{od.TIMELINE_FILE}` \| ([\d,]+) \|", text),
    }

    actual = {
        name: len(pd.read_csv(paths.PROCESSED_DIR / name, low_memory=False)) for name in stated
    }
    assert {name: int(value) for name, value in stated.items()} == actual, (
        "the results document and the tables disagree on the row counts"
    )


@requires_tables
@requires_document
def test_the_document_states_the_totals_the_build_reported() -> None:
    text = _document()
    pairs = _stated(r"od pairs\s+([\d,]+)\s+od cells", text)
    cells = _stated(r"od cells\s+([\d,]+)", text)
    timeline_cells = _stated(r"timeline cells ([\d,]+)", text)
    repairs = _stated(r"ocr repairs\s+([\d,]+)", text)

    assert int(pairs) == len(pd.read_csv(paths.processed("od_pairs.csv"), low_memory=False))
    assert int(cells) == len(pd.read_csv(paths.processed("od_matrix.csv"), low_memory=False))
    assert int(timeline_cells) == len(
        pd.read_csv(paths.processed(od.TIMELINE_FILE), low_memory=False)
    )

    import json

    build = json.loads((paths.PROCESSED_DIR / "build.json").read_text())
    assert int(repairs) == build["ocr_repairs"]
    assert int(timeline_cells) == build["timeline_cells"]


@requires_tables
@requires_document
def test_the_document_states_the_shape_of_each_service_day() -> None:
    """The per-service table, and the day it says the slices add back up to."""
    text = _document()
    rows = re.findall(
        r"\| (BusWay|AveiroBus) \| (weekday|weekend) \| ([\d,]+) "
        r"\| ([\d:]{5}-[\d:]{5}) \| ([\d,]+) \|",
        text,
    )
    assert len(rows) == 4, "the dynamic-view table in AI_RESULT.md lost a row or changed shape"

    timeline = pd.read_csv(paths.processed(od.TIMELINE_FILE), low_memory=False)
    names = {"BusWay": "busway", "AveiroBus": "aveirobus"}
    for stated_operator, group, departures, label, peak in rows:
        sub = timeline[
            (timeline["operator"] == names[stated_operator]) & (timeline["service_group"] == group)
        ]
        busiest = sub.groupby("interval")["trips"].sum().nlargest(1)
        assert int(departures.replace(",", "")) == int(sub["trips"].sum())
        assert label == str(busiest.index[0])
        assert int(peak.replace(",", "")) == int(busiest.iloc[0])

    matrix = pd.read_csv(paths.processed("od_matrix.csv"), low_memory=False)
    both = _stated(r"([\d,]+) direct departures either way", text)
    assert int(both) == int(timeline["trips"].sum()) == int(matrix["trips"].sum())
