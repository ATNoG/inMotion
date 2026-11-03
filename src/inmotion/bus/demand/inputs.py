"""The supply tables a scenario is generated from, and the user's stop attributes.

Only ``stop_times.csv`` and ``stops.csv`` are read: the first already carries every
fact the model keys on, so ``trips.csv`` would be a second copy that could
disagree with it. The optional attributes file is a trust boundary, checked before
anything uses it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from inmotion.bus import paths

#: The columns the generator reads out of the tidy timetable.
STOP_TIME_COLUMNS = (
    "trip_id",
    "operator",
    "line",
    "direction",
    "service",
    "service_group",
    "stop_seq",
    "stop_id",
    "stop",
    "minutes",
    "served",
)

#: The columns the optional per-stop table may carry. ``stop_id`` is the key;
#: ``weight`` overrides the attraction a stop draws, ``category`` is free text
#: carried through to the outputs, and ``lat``/``lon`` override the coordinates.
STOP_ATTRIBUTE_COLUMNS = ("stop_id", "weight", "category", "lat", "lon")


@dataclass(frozen=True)
class Network:
    """The supply side of a generated day: where the buses go, and when."""

    stop_times: pl.DataFrame
    stops: pl.DataFrame

    def served(self) -> pl.DataFrame:
        """The calls a trip actually makes, with the clock as whole minutes.

        A dash in the timetable means the trip does not call at that stop at
        all, which is a different fact from calling there at an unknown time:
        those rows are dropped, not filled in.
        """
        return (
            self.stop_times.filter(pl.col("served"))
            .with_columns(pl.col("minutes").cast(pl.Int64))
            .select(STOP_TIME_COLUMNS)
        )

    def dictionary(self) -> pl.DataFrame:
        """One row per stop id, with the spellings of the same stop reconciled.

        ``stops.csv`` holds a row per operator, id *and* name, and OCR reads the same stop
        two ways often enough that an id appears twice. Lines are the union of both
        readings, the name is the alphabetically first, so the label is stable.

        Joining the raw dictionary instead multiplies anything it is joined to by the
        number of spellings — that is how a pair once appeared twice in one day and lost
        riders in the apportionment.
        """
        return (
            self.stops.group_by("stop_id", "operator")
            .agg(
                pl.col("stop").min().alias("stop"),
                pl.col("trips").max().alias("trips"),
                pl.col("lines")
                .str.split(" ")
                .list.explode()
                .unique()
                .sort()
                .str.join(" ")
                .alias("lines"),
            )
            .sort("stop_id")
        )

    def names(self) -> pl.DataFrame:
        """The stop dictionary's own spelling, one row per stop id."""
        return self.dictionary().select("stop_id", "stop")

    def trips(self) -> pl.DataFrame:
        """One row per scheduled trip, with where and when it starts."""
        served = self.served()
        return (
            served.group_by("trip_id")
            .agg(
                pl.col("operator").first(),
                pl.col("line").first(),
                pl.col("direction").first(),
                pl.col("service").first(),
                pl.col("service_group").first(),
                pl.col("minutes").min().alias("departure_minutes"),
                pl.len().alias("stops_served"),
            )
            .sort("trip_id")
        )


def read_network(directory: str | Path = paths.PROCESSED_DIR) -> Network:
    """Read the processed tables `inmotion bus build` writes.

    ``line`` is read as text: AveiroBus numbers its lines and also calls two of
    them ``azul`` and ``verde``, which would otherwise be split into two types.
    """
    root = Path(directory)
    missing = [name for name in ("stop_times.csv", "stops.csv") if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"no {', '.join(missing)} in {root}: run `inmotion bus build` first")
    stop_times = pl.read_csv(
        root / "stop_times.csv",
        schema_overrides={"line": pl.String},
        columns=list(STOP_TIME_COLUMNS),
    )
    # ``lines`` is text: a stop served by a single line is written as ``85``,
    # which pandas-free inference would read as a number and break the split
    # that counts them.
    stops = pl.read_csv(root / "stops.csv", schema_overrides={"lines": pl.String})
    if stops.height == 0:
        raise RuntimeError(f"{root / 'stops.csv'} holds no stops")
    return Network(stop_times=stop_times, stops=stops)


def read_stop_attributes(path: str | Path | None) -> pl.DataFrame:
    """The optional per-stop table, or an empty one with the full schema.

    Coordinates given here win over anything resolved from OpenStreetMap: the
    operator's own knowledge of where a stop is beats a name match.
    """
    if path is None:
        return pl.DataFrame(
            schema={
                "stop_id": pl.String,
                "weight": pl.Float64,
                "category": pl.String,
                "lat": pl.Float64,
                "lon": pl.Float64,
            }
        )
    frame = pl.read_csv(Path(path), schema_overrides={"stop_id": pl.String})
    missing = [name for name in ("stop_id",) if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} needs a stop_id column")
    unknown = [name for name in frame.columns if name not in STOP_ATTRIBUTE_COLUMNS]
    if unknown:
        raise ValueError(
            f"{path} has columns this generator does not read: {', '.join(sorted(unknown))}"
        )
    for name in ("weight", "lat", "lon"):
        if name in frame.columns:
            frame = frame.with_columns(pl.col(name).cast(pl.Float64))
    duplicates = frame.filter(pl.col("stop_id").is_duplicated())["stop_id"].unique().to_list()
    if duplicates:
        raise ValueError(f"{path} names the same stop twice: {', '.join(sorted(duplicates)[:5])}")
    if "weight" in frame.columns:
        bad = frame.filter(pl.col("weight").is_not_null() & (pl.col("weight") <= 0))
        if bad.height:
            raise ValueError(f"{path}: weight must be positive, e.g. {bad['stop_id'][0]!r}")
    for axis, limit in (("lat", 90.0), ("lon", 180.0)):
        if axis in frame.columns:
            bad = frame.filter(
                pl.col(axis).is_not_null() & ((pl.col(axis).abs() > limit) | pl.col(axis).is_nan())
            )
            if bad.height:
                raise ValueError(f"{path}: {axis} out of range at {bad['stop_id'][0]!r}")
    return frame


def check(network: Network, attributes: pl.DataFrame) -> list[str]:
    """What is worth reporting about the inputs, before a scenario is generated.

    These are warnings, not refusals: a stop with no attribute row is normal,
    and a stop the attributes name but the timetable never serves is worth
    saying out loud rather than failing the run over.
    """
    problems: list[str] = []
    known = set(network.stops["stop_id"].to_list())
    unknown = set(network.served()["stop_id"].unique().to_list()) - known
    if unknown:
        problems.append(
            f"{len(unknown)} stops in stop_times are missing from stops.csv "
            f"(e.g. {sorted(unknown)[0]})"
        )
    if attributes.height:
        named = set(attributes["stop_id"].to_list())
        orphans = named - known
        if orphans:
            problems.append(
                f"{len(orphans)} stops in the attributes file are not in the network "
                f"(e.g. {sorted(orphans)[0]})"
            )
        unattributed = len(known - named)
        if unattributed == len(known):
            problems.append("the attributes file names none of the network's stops")
    unserved = network.stop_times.filter(~pl.col("served")).height
    if unserved:
        problems.append(f"{unserved} timetable cells are dashes and carry no clock")
    return problems
