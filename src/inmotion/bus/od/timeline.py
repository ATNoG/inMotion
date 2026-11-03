"""OD matrices by time interval: the same relation, sliced by departure.

A static matrix aggregates a whole service day into one number per stop pair. A
*dynamic* OD matrix keeps the same rows and columns and adds when the movement
happened: ``M_ij(t)``, the direct trips from ``i`` to ``j`` that left ``i``
inside interval ``t``.

The interval is stamped on the departure from the row's own origin, which is the
instant the departure bands are keyed on. Two consequences worth naming: a trip
is never split between slices — a 23:50 departure that arrives after midnight
belongs to ``23:00-24:00`` in full — and a rider who boards mid-trip is counted
in the slice they left in, not the one the vehicle started in.

Portuguese timetables keep printing past 24:00 for the last trips of the day, so
the grid keeps counting past it: ``24:00-25:00`` is a real interval, not an
error.

``window`` is how long a slice lasts and ``step`` how far apart the slices start.
The defaults partition the service day into hours, which is what makes the slices
add up to the static matrix; a ``step`` shorter than the window overlaps them, so
a trip is counted by every slice containing its departure and the slices must not
be summed, while a longer one samples the day and leaves the gaps between slices
unread.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from inmotion.bus import paths
from inmotion.bus.od.bands import format_minutes, service_group
from inmotion.bus.od.matrix import wide_matrix
from inmotion.bus.od.pairs import per_trip_pairs

WINDOW = 60
STEP = 60

TIMELINE_FILE = "od_timeline.csv"


def window_starts(minutes: int, *, window: int = WINDOW, step: int = STEP) -> list[int]:
    """The start of every slice that contains ``minutes``, latest first.

    Slices align to the service day (00:00, 00:00 + ``step``, ...) and one that
    would start before midnight is not offered, because the service day begins
    there: the previous day's last trips are printed past 24:00 instead. A
    negative reading is clamped to the start of the day — the data is wrong at
    that point and :mod:`inmotion.bus.od.validate` says so, but slicing must not
    drop the row in silence.
    """
    if window <= 0 or step <= 0:
        raise ValueError(f"window and step are lengths in minutes: {window}, {step}")
    last = max(minutes, 0) // step * step
    starts = []
    for index in range(-(-window // step)):
        start = last - index * step
        if start < 0:
            break
        starts.append(start)
    return starts


def interval_label(start: int, *, window: int = WINDOW) -> str:
    """A slice as ``07:00-08:00``, keeping hours past 24."""
    return f"{format_minutes(start)}-{format_minutes(start + window)}"


def od_timeline(
    stop_times: pd.DataFrame, *, window: int = WINDOW, step: int = STEP
) -> pd.DataFrame:
    """The dynamic OD matrix: direct trips per stop pair, per time interval.

    One row per operator, service group, interval and stop pair, carrying the
    same measures as the static matrix so a cell can be read against the day it
    was cut out of. ``trips`` counts the departures in that interval, as the
    static matrix does, which is what lets a partition of the day add back up to
    it: :func:`inmotion.bus.od.validate.validate_timeline` checks exactly that.
    """
    per_trip = per_trip_pairs(stop_times)
    if per_trip.empty:
        return per_trip
    per_trip["service_group"] = per_trip["service"].map(service_group)
    slices = [
        (index, start)
        for index, minutes in enumerate(per_trip["departure_minutes"])
        for start in window_starts(int(minutes), window=window, step=step)
    ]
    sliced = per_trip.take([index for index, _ in slices]).reset_index(drop=True)
    sliced["interval_start"] = [start for _, start in slices]
    timeline = (
        sliced.groupby(
            [
                "operator",
                "service_group",
                "interval_start",
                "origin_id",
                "origin",
                "destination_id",
                "destination",
            ],
            as_index=False,
        )
        .agg(
            trips=("trip_id", "size"),
            lines=("line", lambda values: " ".join(sorted(set(values)))),
            first_minutes=("departure_minutes", "min"),
            last_minutes=("departure_minutes", "max"),
            min_run_min=("run_min", "min"),
            median_run_min=("run_min", "median"),
            max_run_min=("run_min", "max"),
        )
        .sort_values(["operator", "service_group", "interval_start", "origin_id", "destination_id"])
    )
    timeline["interval"] = [
        interval_label(start, window=window) for start in timeline["interval_start"]
    ]
    timeline["first_departure"] = timeline["first_minutes"].map(format_minutes)
    timeline["last_departure"] = timeline["last_minutes"].map(format_minutes)
    return timeline


def interval_matrices(
    timeline: pd.DataFrame, *, operator: str, service_group: str
) -> Iterator[tuple[int, pd.DataFrame]]:
    """Every interval of one operator and service group as a dense grid.

    The axes are that operator's stops for the whole service group, not only the
    ones moving in this interval, so every grid of the series has the same shape
    and a pair with no direct trip in an interval is a zero rather than a missing
    row. Yields ``(interval start in minutes, grid)``, oldest first.
    """
    mine = timeline[
        (timeline["operator"] == operator) & (timeline["service_group"] == service_group)
    ]
    if mine.empty:
        return
    origins = mine[["operator", "origin_id", "origin"]].drop_duplicates()
    index = pd.MultiIndex.from_frame(origins.sort_values(["operator", "origin_id"]))
    columns = sorted(mine["destination_id"].unique())
    for start, group in mine.groupby("interval_start", sort=True):
        grid = wide_matrix(group.drop(columns=["interval_start", "interval"]))
        aligned = grid.set_index(["operator", "origin_id", "origin"]).reindex(
            index=index, columns=columns, fill_value=0
        )
        yield int(start), aligned.reset_index()


def read_stop_times(directory: str | Path = paths.PROCESSED_DIR) -> pd.DataFrame:
    """The tidy timetable the slices are cut from.

    ``line`` is read as text: AveiroBus numbers its lines and also calls two of
    them ``azul`` and ``verde``, which pandas would otherwise split in two.
    """
    path = Path(directory) / "stop_times.csv"
    if not path.exists():
        raise RuntimeError(f"no {path}: run `inmotion bus build` first")
    return pd.read_csv(path, dtype={"line": str})


def write_timeline(
    timeline: pd.DataFrame, out_dir: str | Path, *, wide: bool = False
) -> list[Path]:
    """Write the long timeline, and with ``wide`` a grid per interval.

    The long table is ``od_timeline.csv``. Each grid is
    ``od_timeline_wide_<operator>_<service>_<HHMM>.csv``, named after the minute
    its interval starts, because a directory of them is read as a series.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    long_path = directory / TIMELINE_FILE
    timeline.to_csv(long_path, index=False)
    written = [long_path]
    if not wide or timeline.empty:
        return written
    keys = (
        timeline[["operator", "service_group"]]
        .drop_duplicates()
        .sort_values(["operator", "service_group"])
    )
    for operator, group in keys.itertuples(index=False, name=None):
        for start, grid in interval_matrices(timeline, operator=operator, service_group=group):
            clock = f"{start // 60:02d}{start % 60:02d}"
            path = directory / f"od_timeline_wide_{operator}_{group}_{clock}.csv"
            grid.to_csv(path, index=False)
            written.append(path)
    return written
