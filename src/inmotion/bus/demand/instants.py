"""The flows on the 10-second grid the WiFi captures already use.

``AA`` stayed inside the bus, ``BB`` stayed at the stop, ``BA`` boarded, ``AB``
alighted. Rows exist around a stop visit — ``window`` seconds either side — because
that is all a capture at that stop can see.
"""

from __future__ import annotations

import polars as pl

from inmotion.bus.demand.config import ScenarioConfig

INSTANT_SCHEMA = {
    "trip_id": pl.String,
    "operator": pl.String,
    "line": pl.String,
    "service_group": pl.String,
    "stop_id": pl.String,
    "stop": pl.String,
    "visit_minutes": pl.Int64,
    "visit_time": pl.String,
    "offset_s": pl.Int64,
    "instant_s": pl.Int64,
    "instant": pl.String,
    "inside_bus": pl.Int64,
    "at_stop": pl.Int64,
    "boarded": pl.Int64,
    "alighted": pl.Int64,
}


def clock(seconds: int) -> str:
    """``HH:MM:SS`` on a continuous scale, keeping hours past 24."""
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def instants(trips: pl.DataFrame, config: ScenarioConfig) -> pl.DataFrame:
    """One row per stop visit and instant, in the label vocabulary of the data.

    The visiting instant itself carries the two transitions as well as the
    counts on either side of them: at that instant a passenger who boards is
    ``BA``, one who alights is ``AB``, and the counts are the two states the
    capture can distinguish.
    """
    if not trips.height:
        return pl.DataFrame(schema=INSTANT_SCHEMA)
    offsets = list(
        range(-config.instants_window_s, config.instants_window_s + 1, config.instants_step_s)
    )
    grid = (
        trips.select(
            "trip_id",
            "operator",
            "line",
            "service_group",
            "stop_id",
            "stop",
            "minutes",
            "time",
            "onboard_before",
            "onboard_after",
            "waiting_at_arrival",
            "boarded",
            "alighted",
        )
        .join(pl.DataFrame({"offset_s": offsets}, schema={"offset_s": pl.Int64}), how="cross")
        .with_columns((pl.col("minutes") * 60 + pl.col("offset_s")).alias("instant_s"))
    )
    return (
        grid.with_columns(
            pl.when(pl.col("offset_s") < 0)
            .then(0)
            .when(pl.col("offset_s") == 0)
            .then(pl.col("onboard_before"))
            .otherwise(pl.col("onboard_after"))
            .alias("inside_bus"),
            pl.when(pl.col("offset_s") < 0)
            .then(pl.col("waiting_at_arrival"))
            .otherwise(pl.col("waiting_at_arrival") - pl.col("boarded"))
            .alias("at_stop"),
            pl.when(pl.col("offset_s") == 0).then(pl.col("boarded")).otherwise(0).alias("boarded"),
            pl.when(pl.col("offset_s") == 0)
            .then(pl.col("alighted"))
            .otherwise(0)
            .alias("alighted"),
            pl.col("minutes").alias("visit_minutes"),
            pl.col("time").alias("visit_time"),
        )
        .with_columns(
            pl.col("instant_s").map_elements(clock, return_dtype=pl.String).alias("instant"),
        )
        .select(*INSTANT_SCHEMA)
        .sort("trip_id", "visit_minutes", "offset_s")
    )
