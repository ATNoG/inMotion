"""Which bus carries whom: demand spread over the trips serving each pair.

Each serving trip is weighted by its departure band, so demand follows the
commutes; each trip then gets one of the fleet's sizes, by the scenario's fleet
policy, and that size is what later limits how many of those riders fit.

The result is an intent — this many riders want this trip, from this stop, to
that one. Whether they get on is the queue's decision, in ``simulate``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from inmotion.bus.demand import allocate
from inmotion.bus.demand.config import BusSize, ScenarioConfig
from inmotion.bus.demand.model import DEMAND_KEYS

#: The columns of an intent row: one (trip, origin, destination) and how many
#: riders mean to board there.
INTENT_COLUMNS = (
    "trip_id",
    "operator",
    "line",
    "direction",
    "service_group",
    "origin_id",
    "origin",
    "destination_id",
    "destination",
    "departure_minutes",
    "departure_band",
    "riders",
)

#: The same columns, typed, so an empty assignment still has the shape the
#: queue expects rather than a frame of strings.
INTENT_SCHEMA = {
    "trip_id": pl.String,
    "operator": pl.String,
    "line": pl.String,
    "direction": pl.String,
    "service_group": pl.String,
    "origin_id": pl.String,
    "origin": pl.String,
    "destination_id": pl.String,
    "destination": pl.String,
    "departure_minutes": pl.Int64,
    "departure_band": pl.String,
    "riders": pl.Int64,
}

#: The fleet as it reaches the trips: one size and capacity per scheduled trip,
#: including the trips nobody wants to board — the flow tables are a view of the
#: timetable, so a trip with no riders still has a bus.
FLEET_SCHEMA = {"trip_id": pl.String, "bus_size": pl.String, "capacity": pl.Int64}


def ordered_fleet(fleet: tuple[BusSize, ...]) -> list[BusSize]:
    """The fleet smallest first, with ties settled by name so a run is stable."""
    return sorted(fleet, key=lambda size: (size.capacity, size.name))


@dataclass(frozen=True)
class Assignment:
    """The demand, per trip, and the fleet that has to carry it."""

    intents: pl.DataFrame
    fleet: pl.DataFrame
    warnings: list[str]

    def summary(self) -> str:
        """Fleet and load, one line, for the report."""
        if not self.fleet.height:
            return "nothing assigned"
        fleet = (
            self.fleet.group_by("bus_size", "capacity")
            .agg(pl.col("trip_id").n_unique().alias("trips"))
            .sort("capacity")
        )
        parts = [
            f"{row['bus_size']} ({row['capacity']} seats) x {row['trips']} trips"
            for row in fleet.iter_rows(named=True)
        ]
        return ", ".join(parts)


def band_preference(stamped: pl.DataFrame, config: ScenarioConfig) -> pl.Series:
    """How strongly each departure attracts riders, from its own band.

    A band the profile does not mention falls back to the profile's own average
    rather than to zero: a trip at a clock the bands do not cover must still be
    boardable, or the timetable would have services that no model can use.
    """
    weights = config.band_weights
    average = sum(weights.values()) / len(weights) if weights else 1.0
    return stamped["departure_band"].replace_strict(weights, default=average).cast(pl.Float64)


def fleet_sizes(
    trips: pl.DataFrame,
    riders: pl.DataFrame,
    fleet: tuple[BusSize, ...],
    policy: str,
    seed: int,
) -> pl.DataFrame:
    """One bus size per trip, by the scenario's fleet policy.

    ``cycle`` walks the sizes in departure order; ``demand`` gives the biggest
    vehicles to the busiest lines, cycling the fleet's multiset down the ranking;
    ``random`` draws per trip, weighted by how many of each the fleet holds.
    """
    if not fleet:
        raise ValueError("a fleet needs at least one bus size")
    sizes = ordered_fleet(fleet)
    ordered = trips.sort("service_group", "line", "direction", "departure_minutes", "trip_id")
    if policy == "cycle":
        vehicles = [size for size in sizes for _ in range(max(size.count, 1))]
        assigned = [vehicles[index % len(vehicles)] for index in range(ordered.height)]
        return ordered.select("trip_id").with_columns(
            pl.Series("bus_size", [size.name for size in assigned]),
            pl.Series("capacity", [size.capacity for size in assigned], dtype=pl.Int64),
        )
    if policy == "demand":
        vehicles = sorted(
            (size for size in sizes for _ in range(max(size.count, 1))),
            key=lambda size: (-size.capacity, size.name),
        )
        # Keyed on operator as well as line: AveiroBus's line 1 and BusWay's
        # line 1 are different services that happen to share a number.
        ranked = (
            riders.group_by("operator", "line")
            .agg(pl.col("riders").sum())
            .sort(["riders", "operator", "line"], descending=[True, False, False])
        )
        lines = ranked.height
        by_line = ranked.with_columns(
            pl.Series("bus_size", [vehicles[index % len(vehicles)].name for index in range(lines)]),
            pl.Series(
                "capacity",
                [vehicles[index % len(vehicles)].capacity for index in range(lines)],
                dtype=pl.Int64,
            ),
        )
        smallest = sizes[0]
        return (
            ordered.select("trip_id", "operator", "line")
            .join(by_line, on=["operator", "line"], how="left")
            .with_columns(
                pl.col("bus_size").fill_null(smallest.name),
                pl.col("capacity").fill_null(smallest.capacity),
            )
            .select("trip_id", "bus_size", "capacity")
        )
    if policy != "random":
        raise ValueError(f"unknown fleet policy {policy!r}")
    counts = np.array([max(size.count, 1) for size in sizes], dtype=float)
    generator = np.random.default_rng(abs(seed) + 1)
    drawn = generator.choice(len(sizes), size=ordered.height, p=counts / counts.sum())
    return ordered.select("trip_id").with_columns(
        pl.Series("bus_size", [sizes[index].name for index in drawn]),
        pl.Series("capacity", [sizes[index].capacity for index in drawn], dtype=pl.Int64),
    )


def trip_table(stamped_pairs: pl.DataFrame) -> pl.DataFrame:
    """One row per scheduled trip: which line it runs, and when it starts.

    ``departure_minutes`` is the minimum over the trip's pairs, which is its
    first served call. The pair relation carries the clock of each pair's own
    origin, so a trip appears once per pair and anything that counts trips must
    collapse it first or count it once per pair.
    """
    return (
        stamped_pairs.group_by("trip_id")
        .agg(
            pl.col("operator").first(),
            pl.col("line").first(),
            pl.col("direction").first(),
            pl.col("service_group").first(),
            pl.col("departure_minutes").min().alias("departure_minutes"),
        )
        .sort("trip_id")
    )


def assign(demand: pl.DataFrame, stamped_pairs: pl.DataFrame, config: ScenarioConfig) -> Assignment:
    """Spread each pair's demand over the trips serving it, then size the fleet."""
    if not demand.height or not stamped_pairs.height:
        return Assignment(
            intents=pl.DataFrame(schema=INTENT_SCHEMA),
            fleet=pl.DataFrame(schema=FLEET_SCHEMA),
            warnings=["no demand to assign"],
        )
    serving = stamped_pairs.join(
        demand.select(*DEMAND_KEYS, "demand"), on=list(DEMAND_KEYS), how="inner"
    )
    if not serving.height:
        return Assignment(
            intents=pl.DataFrame(schema=INTENT_SCHEMA),
            fleet=pl.DataFrame(schema=FLEET_SCHEMA),
            warnings=["no trip serves a pair that has demand"],
        )
    serving = serving.with_columns(band_preference(serving, config).alias("band_weight"))
    split = allocate.apportion(
        serving,
        weights="band_weight",
        total="demand",
        keys=list(DEMAND_KEYS),
        seed=config.seed,
        salt="assignment",
        name="riders",
    )
    riders = split.filter(pl.col("riders") > 0)
    fleet = fleet_sizes(
        trip_table(stamped_pairs),
        riders.select("operator", "line", "riders"),
        config.fleet,
        config.fleet_policy,
        config.seed,
    )
    intents = riders.select(*INTENT_COLUMNS).sort("trip_id", "origin_id", "destination_id")
    warnings = []
    if intents.height and intents["riders"].sum() != demand["demand"].sum():
        warnings.append(
            f"assignment moved {intents['riders'].sum():,} of the day's "
            f"{demand['demand'].sum():,} boardings"
        )
    return Assignment(intents=intents, fleet=fleet, warnings=warnings)
