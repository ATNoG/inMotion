"""The passenger OD matrices: the supply tables' keys, one measure apart.

``od_matrix.csv`` counts the departures a rider could take; these count the riders
who took them. The static matrix and its hourly slices are built together, and
``validate`` holds the second to adding back up to the first.
"""

from __future__ import annotations

import polars as pl

from inmotion.bus.od.bands import format_minutes
from inmotion.bus.od.timeline import interval_label, window_starts

#: The keys a passenger matrix is aggregated on, in the supply matrix's order.
MATRIX_KEYS = (
    "operator",
    "service_group",
    "departure_band",
    "origin_id",
    "origin",
    "destination_id",
    "destination",
)

WINDOW = 60
STEP = 60


def passenger_matrix(od_flow: pl.DataFrame) -> pl.DataFrame:
    """Boardings per stop pair, by operator, service day and departure band.

    ``trips`` counts the departures that carried at least one of those riders, so
    a cell that is busy with passengers on a thin service is distinguishable from
    one that is busy because the service is frequent.
    """
    if not od_flow.height:
        return pl.DataFrame(
            schema={
                **dict.fromkeys(MATRIX_KEYS, pl.String),
                "passengers": pl.Int64,
                "trips": pl.Int64,
            }
        )
    return (
        od_flow.group_by(*MATRIX_KEYS)
        .agg(
            pl.col("passengers").sum(),
            pl.col("trip_id").n_unique().alias("trips"),
            pl.col("departure_minutes").min().alias("first_minutes"),
            pl.col("departure_minutes").max().alias("last_minutes"),
        )
        .with_columns(
            pl.col("first_minutes")
            .map_elements(format_minutes, return_dtype=pl.String)
            .alias("first_departure"),
            pl.col("last_minutes")
            .map_elements(format_minutes, return_dtype=pl.String)
            .alias("last_departure"),
        )
        .select(
            *MATRIX_KEYS,
            "passengers",
            "trips",
            "first_minutes",
            "last_minutes",
            "first_departure",
            "last_departure",
        )
        .sort([*MATRIX_KEYS[:3], "origin_id", "destination_id"])
    )


def passenger_timeline(
    od_flow: pl.DataFrame, *, window: int = WINDOW, step: int = STEP
) -> pl.DataFrame:
    """The same boardings per stop pair, sliced by when the ride left.

    A boarding is filed under the interval its own origin's departure falls in,
    which is the instant the bands are keyed on too: a trip is never split
    between slices, and a rider who boards mid-route belongs to the slice they
    left in.
    """
    if not od_flow.height:
        return pl.DataFrame(
            schema={
                "operator": pl.String,
                "service_group": pl.String,
                "interval_start": pl.Int64,
                "interval": pl.String,
                "origin_id": pl.String,
                "origin": pl.String,
                "destination_id": pl.String,
                "destination": pl.String,
                "passengers": pl.Int64,
                "trips": pl.Int64,
            }
        )
    sliced = (
        od_flow.with_columns(
            pl.col("departure_minutes")
            .map_elements(
                lambda minutes: window_starts(int(minutes), window=window, step=step),
                return_dtype=pl.List(pl.Int64),
            )
            .alias("interval_start")
        )
        .explode("interval_start")
        .filter(pl.col("interval_start").is_not_null())
    )
    return (
        sliced.group_by(
            "operator",
            "service_group",
            "interval_start",
            "origin_id",
            "origin",
            "destination_id",
            "destination",
        )
        .agg(pl.col("passengers").sum(), pl.col("trip_id").n_unique().alias("trips"))
        .with_columns(
            pl.col("interval_start")
            .map_elements(
                lambda start: interval_label(int(start), window=window), return_dtype=pl.String
            )
            .alias("interval")
        )
        .select(
            "operator",
            "service_group",
            "interval_start",
            "interval",
            "origin_id",
            "origin",
            "destination_id",
            "destination",
            "passengers",
            "trips",
        )
        .sort("operator", "service_group", "interval_start", "origin_id", "destination_id")
    )


def wide(matrix: pl.DataFrame, axis: pl.DataFrame, *, measure: str = "passengers") -> pl.DataFrame:
    """Pivot a long matrix into origins by destinations, absent cells as zero.

    ``axis`` is the full list of stops of the operator — taken from the supply
    side rather than from the matrix — so a stop that only ever receives riders
    still gets a row, and the grid lines up column for column with the supply
    grid it is meant to be read against.
    """
    if not matrix.height or not axis.height:
        return pl.DataFrame()
    keys = ["operator", "origin_id", "origin"]
    grid = matrix.pivot(on="destination_id", index=keys, values=measure, aggregate_function="sum")
    complete = (
        axis.select(*keys).unique().join(grid, on=keys, how="left").sort("operator", "origin_id")
    )
    destinations = [name for name in complete.columns if name not in keys]
    return complete.with_columns([pl.col(name).fill_null(0) for name in destinations])
