"""Every ordered stop pair a trip serves directly: the domain demand lives on.

The relation ``inmotion.bus.od.pairs`` derives, in polars, with a parity test
holding the two together — a second implementation is only safe if it cannot
become a second answer. Two calls naming one stop at the same minute are one call,
as the operators print them.
"""

from __future__ import annotations

import polars as pl

from inmotion.bus.od.bands import band, service_group

#: The pair relation's columns, in the order ``od.per_trip_pairs`` emits them,
#: so the two can be compared row for row.
PAIR_COLUMNS = (
    "operator",
    "line",
    "direction",
    "service",
    "trip_id",
    "origin_id",
    "origin",
    "destination_id",
    "destination",
    "departure_minutes",
    "run_min",
)

_TRIP_KEYS = ("operator", "line", "direction", "service", "trip_id")


def dedupe_calls(served: pl.DataFrame) -> pl.DataFrame:
    """Drop a call that repeats the previous stop at the same minute.

    ``served`` must already be sorted: the rule is about consecutive calls. The
    comparison is null for each trip's first call and a null predicate is not true, so
    it is filled before negating — without that, ``~`` drops every first call.
    """
    repeats = (pl.col("stop_id") == pl.col("stop_id").shift(1).over("trip_id")) & (
        pl.col("minutes") == pl.col("minutes").shift(1).over("trip_id")
    )
    return served.filter(~repeats.fill_null(False))


def per_trip_pairs(served: pl.DataFrame) -> pl.DataFrame:
    """Every ordered stop pair one trip serves, one row per trip and pair.

    ``departure_minutes`` is the clock at the pair's own origin — when *this* ride
    leaves, the instant the bands and the intervals are keyed on. Dashes are dropped
    here, as the reference implementation does: a row with no clock would become a
    pair with no journey time.
    """
    calls = served.filter(pl.col("served")) if "served" in served.columns else served
    calls = dedupe_calls(calls.sort(["trip_id", "stop_seq"])).select(
        *[pl.col(name) for name in (*_TRIP_KEYS, "stop_seq", "stop_id", "stop", "minutes")]
    )
    destinations = calls.select(
        *[pl.col(name) for name in _TRIP_KEYS],
        pl.col("stop_seq").alias("stop_seq_destination"),
        pl.col("stop_id").alias("destination_id"),
        pl.col("stop").alias("destination"),
        pl.col("minutes").alias("destination_minutes"),
    )
    return (
        calls.join(destinations, on=list(_TRIP_KEYS), how="inner")
        .filter(pl.col("stop_seq") < pl.col("stop_seq_destination"))
        .sort(["trip_id", "stop_seq", "stop_seq_destination"])
        .select(
            *[pl.col(name) for name in _TRIP_KEYS],
            pl.col("stop_id").alias("origin_id"),
            pl.col("stop").alias("origin"),
            pl.col("destination_id"),
            pl.col("destination"),
            pl.col("minutes").alias("departure_minutes"),
            (pl.col("destination_minutes") - pl.col("minutes")).alias("run_min"),
        )
    )


def stop_dictionary(pairs: pl.DataFrame, names: pl.DataFrame) -> pl.DataFrame:
    """One name per stop id: the dictionary's, plus a fallback for its gaps.

    A few ids in the timetable are missing from ``stops.csv``, and those would keep
    whichever spelling the page used — enough to make one pair appear twice in a day.
    The fallback is the alphabetically first spelling the relation carries.
    """
    both = pl.concat(
        [
            pairs.select(pl.col("origin_id").alias("stop_id"), pl.col("origin").alias("stop")),
            pairs.select(
                pl.col("destination_id").alias("stop_id"), pl.col("destination").alias("stop")
            ),
        ]
    )
    fallback = (
        both.filter(pl.col("stop").is_not_null())
        .unique()
        .group_by("stop_id")
        .agg(pl.col("stop").min())
    )
    return pl.concat([names.select("stop_id", "stop"), fallback]).unique(
        subset=["stop_id"], keep="first"
    )


def name_stops(pairs: pl.DataFrame, names: pl.DataFrame) -> pl.DataFrame:
    """Replace the pair relation's stop names with the dictionary's own."""
    names = stop_dictionary(pairs, names)
    return (
        pairs.drop("origin", "destination")
        .join(
            names.rename({"stop_id": "origin_id", "stop": "origin"}),
            on="origin_id",
            how="left",
        )
        .join(
            names.rename({"stop_id": "destination_id", "stop": "destination"}),
            on="destination_id",
            how="left",
        )
        .with_columns(
            pl.col("origin").fill_null(pl.col("origin_id")),
            pl.col("destination").fill_null(pl.col("destination_id")),
        )
        .select(*PAIR_COLUMNS)
    )


def with_time_keys(pairs: pl.DataFrame) -> pl.DataFrame:
    """Stamp each pair with the service group and band of its own departure.

    Banding before any aggregation is the whole point: a (line, direction,
    service) group can span the entire service day, so a band derived later from
    the group's earliest departure would file every trip of that group under one
    morning band.
    """
    return pairs.with_columns(
        pl.col("service")
        .map_elements(service_group, return_dtype=pl.String)
        .alias("service_group"),
        pl.col("departure_minutes")
        .map_elements(band, return_dtype=pl.String)
        .alias("departure_band"),
    )
