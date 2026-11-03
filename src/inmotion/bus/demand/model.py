"""Where the riders want to go: a gravity model over the pairs the network runs.

``A_i = w_i * (1 + lines_i)^alpha`` — what a stop draws.
``D_ij = exp(-beta * t_ij)`` — how sharply a longer ride is avoided.
``T_ij ∝ A_i * A_j * D_ij``, normalised to the day's target and *drawn* per rider:
rounding the expected value of every cell would empty most of the matrix, while
drawing makes it sparse with a long tail, as real OD data is.

The dials are conventional, not calibrated: every number reproduces from
``scenario.json``, and none of it is a forecast of Aveiro's real ridership.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from inmotion.bus.demand import allocate, inputs
from inmotion.bus.demand.config import ScenarioConfig

#: The keys a demand row is identified by, and the keys the assignment splits on.
DEMAND_KEYS = ("operator", "service_group", "origin_id", "destination_id")


@dataclass(frozen=True)
class Demand:
    """The generated demand for one day, and anything worth saying about it."""

    pairs: pl.DataFrame
    warnings: list[str] = field(default_factory=list)

    def total(self) -> int:
        """The boardings the day's demand adds up to."""
        return int(self.pairs["demand"].sum()) if self.pairs.height else 0


def attraction(stops: pl.DataFrame, attributes: pl.DataFrame, alpha: float) -> pl.DataFrame:
    """How much each stop draws, from its own service level and any override.

    ``lines`` is the space-separated list ``bus build`` writes, so a stop served
    by nine lines draws more than one served by two. The ``+1`` keeps a stop
    that no line serves (possible in the stop dictionary, impossible in the
    timetable) from collapsing to zero, which would erase it from the matrix.
    """
    line_count = (
        pl.when(pl.col("lines").str.strip_chars().str.len_chars() == 0)
        .then(0)
        .otherwise(pl.col("lines").str.split(" ").list.len())
        .cast(pl.Float64)
    )
    # The dictionary is deduplicated first: an id with two spellings would
    # otherwise multiply every pair it is joined to.
    frame = (
        stops.select("stop_id", line_count.alias("lines_served"))
        .group_by("stop_id")
        .agg(pl.col("lines_served").max())
    )
    if attributes.height:
        overrides = attributes.select(
            "stop_id",
            pl.col("weight").cast(pl.Float64).alias("weight"),
        )
        frame = frame.join(overrides, on="stop_id", how="left")
    else:
        frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("weight"))
    return frame.with_columns(
        (pl.col("weight").fill_null(1.0) * (1.0 + pl.col("lines_served")).pow(alpha)).alias(
            "attraction"
        )
    ).select("stop_id", "attraction")


def service_targets(pairs: pl.DataFrame, config: ScenarioConfig) -> pl.DataFrame:
    """The boardings each operator and service day has to add up to.

    Without an explicit total the day scales with the service it runs, so a thin
    Sunday is a thin Sunday. With one, that many boardings are split across every
    generated operator and service group by departures — otherwise two operators
    would quietly produce twice the passengers asked for.
    """
    departures = (
        pairs.group_by("operator", "service_group")
        .agg(pl.col("trip_id").n_unique().alias("departures"))
        .sort("operator", "service_group")
    )
    if config.daily_boardings is None:
        return departures.with_columns(
            (pl.col("departures") * config.boardings_per_departure)
            .round()
            .cast(pl.Int64)
            .alias("target")
        )
    scaled = departures.with_columns(
        pl.lit(1, dtype=pl.Int64).alias("_all"),
        pl.lit(config.daily_boardings, dtype=pl.Int64).alias("total"),
    )
    split = allocate.apportion(
        scaled,
        weights="departures",
        total="total",
        keys=["_all"],
        seed=config.seed,
        salt="service-split",
        name="target",
    )
    return split.select("operator", "service_group", "departures", "target")


def pair_demand(
    stamped_pairs: pl.DataFrame,
    stops: pl.DataFrame,
    attributes: pl.DataFrame,
    config: ScenarioConfig,
    targets: pl.DataFrame | None = None,
) -> Demand:
    """The day's demand per stop pair, on the pairs the network actually runs.

    Only pairs with a direct ride get demand: the supply tables describe what
    can be ridden without changing buses, and a demand model that invented
    journeys no timetable provides would produce an OD matrix no operator could
    serve.
    """
    if not stamped_pairs.height:
        return Demand(pairs=stamped_pairs, warnings=["no stop pairs"])
    targets = service_targets(stamped_pairs, config) if targets is None else targets
    served = (
        stamped_pairs.group_by(*DEMAND_KEYS, "origin", "destination")
        .agg(
            pl.col("run_min").min().alias("min_run_min"),
            pl.col("departure_minutes").min().alias("first_minutes"),
            pl.col("trip_id").n_unique().alias("trips"),
        )
        .sort(*DEMAND_KEYS)
    )
    draw = attraction(stops, attributes, config.alpha)
    weights = (
        served.join(
            draw.rename({"stop_id": "origin_id", "attraction": "origin_draw"}),
            on="origin_id",
            how="left",
        )
        .join(
            draw.rename({"stop_id": "destination_id", "attraction": "destination_draw"}),
            on="destination_id",
            how="left",
        )
        .with_columns(
            (
                pl.col("origin_draw").fill_null(1.0)
                * pl.col("destination_draw").fill_null(1.0)
                # A handful of pairs in the scraped timetables run backwards —
                # the operator prints the last trips of the day without rolling
                # them past midnight. Left alone, a negative ride would *raise*
                # the attraction of an impossible journey; clipped, it is the
                # shortest ride there is, and the count is reported.
                * (-config.beta * pl.col("min_run_min").clip(lower_bound=0)).exp()
            ).alias("weight")
        )
    )
    scale = weights.join(
        targets.select("operator", "service_group", "target"),
        on=["operator", "service_group"],
        how="left",
    )
    scaled = pl.concat(
        [
            group.with_columns(
                pl.Series(
                    "demand",
                    allocate.sampled(
                        group["weight"].to_numpy(),
                        int(group["target"][0]),
                        seed=config.seed,
                        salt=f"demand:{operator}:{service_group}",
                    ),
                    dtype=pl.Int64,
                )
            ).drop("target")
            for (operator, service_group), group in scale.group_by(
                "operator", "service_group", maintain_order=True
            )
        ]
    )
    warnings = []
    for row in (
        scaled.group_by("operator", "service_group")
        .agg(pl.col("demand").sum())
        .iter_rows(named=True)
    ):
        if row["demand"] == 0:
            warnings.append(
                f"{row['operator']}/{row['service_group']}: the day's demand rounds to zero"
            )
    return Demand(pairs=scaled, warnings=warnings)


def generate(
    stamped_pairs: pl.DataFrame,
    network: inputs.Network,
    attributes: pl.DataFrame,
    config: ScenarioConfig,
    *,
    stops: pl.DataFrame | None = None,
) -> Demand:
    """The demand of one scenario: the pairs, scaled to the day's target."""
    return pair_demand(
        stamped_pairs,
        network.dictionary() if stops is None else stops,
        attributes,
        config,
    )
