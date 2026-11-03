"""The queue at each stop, and the bus that either takes the riders or leaves them.

``arrivals`` riders turn up for a departure; ``boarded`` (entry) is how many fit;
``alighted`` (exit) is who got off here; ``waiting`` (stay in the stop) is who is
still standing there, and a later bus for their destination can still take them.
What is left when the day's last bus has gone is *stranded*, reported not dropped.

Two orderings are load-bearing: events follow the order buses arrive at each stop,
not the order they left their first one, and each service group is a separate day,
so a weekday queue never boards a weekend bus.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

import polars as pl

from inmotion.bus.demand import allocate
from inmotion.bus.demand import pairs as pair_relation
from inmotion.bus.demand.config import ScenarioConfig
from inmotion.bus.od.bands import band, format_minutes
from inmotion.bus.od.timeline import window_starts

#: One row per trip and stop: what the vehicle did, and what the stop did.
TRIP_FLOW_SCHEMA = {
    "trip_id": pl.String,
    "operator": pl.String,
    "line": pl.String,
    "direction": pl.String,
    "service_group": pl.String,
    "stop_seq": pl.Int64,
    "stop_id": pl.String,
    "stop": pl.String,
    "minutes": pl.Int64,
    "time": pl.String,
    "departure_band": pl.String,
    "bus_size": pl.String,
    "capacity": pl.Int64,
    "arrivals": pl.Int64,
    "boarded": pl.Int64,
    "alighted": pl.Int64,
    "onboard_arrival": pl.Int64,
    "onboard_before": pl.Int64,
    "onboard_after": pl.Int64,
    "load_factor": pl.Float64,
    "waiting_at_arrival": pl.Int64,
    "denied": pl.Int64,
}

#: One row per stop and interval: the queue, which is what "stay in the bus stop"
#: means once the day is cut into slices.
STOP_FLOW_SCHEMA = {
    "stop_id": pl.String,
    "operator": pl.String,
    "stop": pl.String,
    "service_group": pl.String,
    "interval_start": pl.Int64,
    "interval": pl.String,
    "arrivals": pl.Int64,
    "boarded": pl.Int64,
    "denied": pl.Int64,
    "waiting_start": pl.Int64,
    "waiting_end": pl.Int64,
}

#: One row per trip, boarding stop and destination: the ride actually taken, and
#: the only table the passenger OD matrices are built from.
OD_FLOW_SCHEMA = {
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
    "run_min": pl.Int64,
    "arrival_minutes": pl.Int64,
    "passengers": pl.Int64,
}

WINDOW = 60


@dataclass(frozen=True)
class Flows:
    """What one generated day moved: per trip and stop, per stop and hour."""

    trips: pl.DataFrame
    stops: pl.DataFrame
    od: pl.DataFrame
    stranded: int = 0
    warnings: list[str] = field(default_factory=list)


def simulate(
    served: pl.DataFrame,
    intents: pl.DataFrame,
    fleet: pl.DataFrame,
    config: ScenarioConfig,
    *,
    names: pl.DataFrame | None = None,
    window: int = WINDOW,
) -> Flows:
    """Run every generated service day's buses over that day's riders."""
    groups = served["service_group"].unique(maintain_order=True).to_list()
    if not groups:
        return Flows(
            trips=pl.DataFrame(schema=TRIP_FLOW_SCHEMA),
            stops=pl.DataFrame(schema=STOP_FLOW_SCHEMA),
            od=pl.DataFrame(schema=OD_FLOW_SCHEMA),
            warnings=["the timetable has no served calls"],
        )
    parts = [
        _simulate_day(
            served.filter(pl.col("service_group") == group),
            intents.filter(pl.col("service_group") == group) if intents.height else intents,
            fleet,
            config,
            names=names,
            window=window,
            salt=group,
        )
        for group in groups
    ]
    flows = Flows(
        trips=pl.concat([part.trips for part in parts]),
        stops=pl.concat([part.stops for part in parts]),
        od=pl.concat([part.od for part in parts]),
        stranded=sum(part.stranded for part in parts),
        warnings=[warning for part in parts for warning in part.warnings],
    )
    if flows.stranded:
        flows.warnings.append(
            f"{flows.stranded:,} riders were still waiting when their day's last buses left"
        )
    return flows


def _simulate_day(
    served: pl.DataFrame,
    intents: pl.DataFrame,
    fleet: pl.DataFrame,
    config: ScenarioConfig,
    *,
    names: pl.DataFrame | None,
    window: int,
    salt: str,
) -> Flows:
    """One service day: the bus events in the order they reach each stop."""
    calls = pair_relation.dedupe_calls(served.sort(["trip_id", "stop_seq"]))
    ordered = (
        calls.group_by("trip_id", maintain_order=True)
        .agg(pl.col("stop_id"), pl.col("stop_seq"))
        .sort("trip_id")
    )
    stops_of: dict[str, list[str]] = {}
    position_of: dict[tuple[str, int], int] = {}
    for row in ordered.iter_rows(named=True):
        trip = row["trip_id"]
        stops_of[trip] = row["stop_id"]
        for position, sequence in enumerate(row["stop_seq"]):
            position_of[(trip, int(sequence))] = position
    # Each position's remaining calls, as stop id -> the clock it is reached at.
    # The clock is what lets an exit be placed in time the way an entry is: a
    # rider who boards at 07:50 and rides for 20 minutes alights at 08:10, in
    # another interval and possibly after midnight. A stop that a circular route
    # reaches twice keeps the first of the two, which is the one this leg uses.
    clocks: dict[str, list[int | None]] = {}
    sequence_of: dict[str, list[int]] = {}
    for trip, group in calls.group_by("trip_id", maintain_order=True):
        key = trip[0] if isinstance(trip, tuple) else trip
        clocks[key] = [None if value is None else int(value) for value in group["minutes"]]
        sequence_of[key] = [int(value) for value in group["stop_seq"]]
    beyond: dict[tuple[str, int], dict[str, int]] = {}
    for trip, stops in stops_of.items():
        for position in range(len(stops)):
            later: dict[str, int] = {}
            for ahead in range(position + 1, len(stops)):
                stop = stops[ahead]
                if stop in later:
                    continue
                clock = clocks[trip][ahead] if clocks.get(trip) else None
                if clock is not None:
                    later[stop] = clock
            beyond[(trip, position)] = later

    # One spelling per stop id: the dictionary's when the caller has it, and
    # otherwise the first the timetable prints.
    if names is not None and names.height:
        name_of = dict(names.select("stop_id", "stop").iter_rows())
    else:
        name_of = dict(
            calls.select("stop_id", "stop").unique(subset=["stop_id"], keep="first").iter_rows()
        )
    capacity = {row["trip_id"]: int(row["capacity"]) for row in fleet.iter_rows(named=True)}
    size = {row["trip_id"]: row["bus_size"] for row in fleet.iter_rows(named=True)}
    fallback = config.fleet[0] if config.fleet else None
    for trip in stops_of:
        if fallback is not None:
            capacity.setdefault(trip, fallback.capacity)
            size.setdefault(trip, fallback.name)

    pending: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for row in intents.iter_rows(named=True):
        pending[(row["trip_id"], row["origin_id"])].append(
            (row["destination_id"], int(row["riders"]))
        )

    queue: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    onboard: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    load: dict[str, int] = defaultdict(int)
    # The tie-break inside a full bus has to be reproducible from the seed and
    # independent of row order, which is what one seeded stream over a
    # deterministic event order gives.
    rng = random.Random(f"{config.seed}:{salt}")

    trip_rows: list[tuple] = []
    od_rows: list[tuple] = []
    stop_events: dict[str, list[tuple]] = defaultdict(list)

    events = calls.select(
        "trip_id",
        "stop_id",
        "stop",
        "stop_seq",
        "minutes",
        "operator",
        "line",
        "direction",
        "service_group",
    ).sort(["minutes", "trip_id", "stop_seq"])
    for event in events.iter_rows(named=True):
        trip = event["trip_id"]
        stop = event["stop_id"]
        minutes = int(event["minutes"])
        position = position_of[(trip, int(event["stop_seq"]))]
        onboard_arrival = load[trip]
        alighted = onboard[trip].pop(stop, 0)
        load[trip] -= alighted

        arrivals = 0
        for destination, riders in pending.pop((trip, stop), ()):
            queue[stop][destination] += riders
            arrivals += riders
        waiting = sum(queue[stop].values())

        served_beyond = beyond[(trip, position)]
        candidates = {
            destination: count
            for destination, count in queue[stop].items()
            if count > 0 and destination in served_beyond
        }
        free = capacity[trip] - load[trip]
        boarded = 0
        if candidates and free > 0:
            destinations = sorted(candidates)
            shares = allocate.largest_remainder(
                [candidates[destination] for destination in destinations],
                min(free, sum(candidates.values())),
                tie_break=[rng.random() for _ in destinations],
            )
            for destination, riders in zip(destinations, shares, strict=True):
                if not riders:
                    continue
                queue[stop][destination] -= riders
                onboard[trip][destination] += riders
                load[trip] += riders
                boarded += riders
                arrival = served_beyond.get(destination, minutes)
                od_rows.append(
                    (
                        trip,
                        event["operator"],
                        event["line"],
                        event["direction"],
                        event["service_group"],
                        stop,
                        name_of.get(stop, event["stop"]),
                        destination,
                        name_of.get(destination, destination),
                        minutes,
                        band(minutes),
                        arrival - minutes,
                        arrival,
                        riders,
                    )
                )
        onboard_before = onboard_arrival - alighted
        onboard_after = onboard_before + boarded
        trip_rows.append(
            (
                trip,
                event["operator"],
                event["line"],
                event["direction"],
                event["service_group"],
                int(event["stop_seq"]),
                stop,
                name_of.get(stop, event["stop"]),
                minutes,
                format_minutes(minutes),
                band(minutes),
                size[trip],
                capacity[trip],
                arrivals,
                boarded,
                alighted,
                onboard_arrival,
                onboard_before,
                onboard_after,
                onboard_after / capacity[trip] if capacity[trip] else 0.0,
                waiting,
                sum(candidates.values()) - boarded,
            )
        )
        interval_start = window_starts(minutes, window=window, step=window)[0]
        stop_events[stop].append(
            (
                interval_start,
                arrivals,
                boarded,
                sum(candidates.values()) - boarded,
                name_of.get(stop, event["stop"]),
                event["operator"],
                event["service_group"],
            )
        )

    stranded = sum(sum(bydestination.values()) for bydestination in queue.values())
    return Flows(
        trips=pl.DataFrame(trip_rows, schema=TRIP_FLOW_SCHEMA, orient="row"),
        stops=_stop_flow(stop_events, window=window),
        od=pl.DataFrame(od_rows, schema=OD_FLOW_SCHEMA, orient="row"),
        stranded=stranded,
    )


def _stop_flow(stop_events: dict[str, list[tuple]], *, window: int) -> pl.DataFrame:
    """The queue per stop and interval, with the empty intervals carried through.

    An interval in which no bus came still has to appear: the riders standing at
    the stop did not go anywhere, and a table that skipped the hour would show a
    queue appearing out of nothing in the next one.
    """
    rows: list[tuple] = []
    for stop, events in sorted(stop_events.items()):
        by_interval: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
        for interval_start, arrivals, boarded, denied, *_ in events:
            cell = by_interval[interval_start]
            cell[0] += arrivals
            cell[1] += boarded
            cell[2] += denied
        first, last = min(by_interval), max(by_interval)
        _, _, _, _, name, operator, service_group = events[0]
        waiting = 0
        for start in range(first, last + window, window):
            arrivals, boarded, denied = by_interval.get(start, [0, 0, 0])
            end = waiting + arrivals - boarded
            rows.append(
                (
                    stop,
                    operator,
                    name,
                    service_group,
                    start,
                    f"{format_minutes(start)}-{format_minutes(start + window)}",
                    arrivals,
                    boarded,
                    denied,
                    waiting,
                    end,
                )
            )
            waiting = end
    return pl.DataFrame(rows, schema=STOP_FLOW_SCHEMA, orient="row")
