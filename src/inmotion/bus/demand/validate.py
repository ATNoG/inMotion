"""The invariants a generated day has to hold, and the supply it must trace to.

A synthetic day can be wrong in ways that look plausible — a bus over capacity, a
queue boarding riders who were never there, a matrix that does not add up. Each is
checked here, including that every trip, call and clock came from the scrape, and a
failure is reported rather than written.
"""

from __future__ import annotations

import polars as pl

from inmotion.bus.demand import inputs
from inmotion.bus.demand.model import DEMAND_KEYS
from inmotion.bus.demand.simulate import Flows


def _report(problems: list[str], rows: pl.DataFrame, message: str, example: str) -> None:
    """Add a problem, naming one offending row so it can be found again."""
    if not rows.height:
        return
    first = rows.row(0, named=True)
    problems.append(f"{rows.height} {message} ({example.format(**first)})")


def check_flows(flows: Flows) -> list[str]:
    """Every identity a trip and a stop have to satisfy, on the flow tables."""
    problems: list[str] = []
    trips = flows.trips
    if not trips.height:
        return ["no flows were generated"]

    _report(
        problems,
        trips.filter(pl.col("onboard_after") != pl.col("onboard_before") + pl.col("boarded")),
        "stop visits where the load does not follow from the boardings",
        "{trip_id} at {stop}: {onboard_before} + {boarded} != {onboard_after}",
    )
    _report(
        problems,
        trips.filter(pl.col("onboard_before") != pl.col("onboard_arrival") - pl.col("alighted")),
        "stop visits where the alightings do not follow from the load",
        "{trip_id} at {stop}: {onboard_arrival} - {alighted} != {onboard_before}",
    )
    _report(
        problems,
        trips.filter(pl.col("onboard_after") > pl.col("capacity")),
        "stop visits over the vehicle's capacity",
        "{trip_id} at {stop}: {onboard_after} of {capacity} seats",
    )
    _report(
        problems,
        trips.filter(
            (pl.col("boarded") < 0)
            | (pl.col("alighted") < 0)
            | (pl.col("arrivals") < 0)
            | (pl.col("denied") < 0)
            | (pl.col("onboard_arrival") < 0)
        ),
        "stop visits with a negative count",
        "{trip_id} at {stop}",
    )
    _report(
        problems,
        trips.filter(
            (pl.col("denied") + pl.col("boarded"))
            > pl.col("waiting_at_arrival").fill_null(0) + pl.col("arrivals")
        ),
        "stop visits that boarded or refused more riders than were present",
        "{trip_id} at {stop}: {boarded} boarded and {denied} refused of {waiting_at_arrival}",
    )

    stops = flows.stops
    _report(
        problems,
        stops.filter(
            pl.col("waiting_end")
            != pl.col("waiting_start") + pl.col("arrivals") - pl.col("boarded")
        ),
        "stop intervals where the queue does not conserve riders",
        "{stop_id} {interval}: {waiting_start} + {arrivals} - {boarded} != {waiting_end}",
    )
    _report(
        problems,
        stops.filter(pl.col("waiting_end") < 0),
        "stop intervals with a negative queue",
        "{stop_id} {interval}",
    )
    return problems


def check_conservation(
    flows: Flows, demand: pl.DataFrame, matrix: pl.DataFrame | None = None
) -> list[str]:
    """The day's totals: what was generated, what boarded, what was left."""
    problems: list[str] = []
    boarded = int(flows.trips["boarded"].sum()) if flows.trips.height else 0
    od_total = int(flows.od["passengers"].sum()) if flows.od.height else 0
    if boarded != od_total:
        problems.append(
            f"the flow tables board {boarded:,} riders but the OD relation records {od_total:,}"
        )
    if demand.height:
        requested = int(demand["demand"].sum())
        served = boarded + flows.stranded
        if requested != served:
            problems.append(
                f"the day requested {requested:,} boardings but {served:,} were accounted for "
                f"({boarded:,} boarded, {flows.stranded:,} stranded)"
            )
    if matrix is not None and matrix.height and int(matrix["passengers"].sum()) != boarded:
        problems.append(
            f"the passenger matrix holds {int(matrix['passengers'].sum()):,} riders "
            f"against {boarded:,} in the flows"
        )
    return problems


def check_demand(demand: pl.DataFrame) -> list[str]:
    """The demand matrix itself: whole riders, on pairs that exist, non-negative."""
    problems: list[str] = []
    if not demand.height:
        return ["the demand model produced no pairs"]
    _report(
        problems,
        demand.filter(pl.col("demand") < 0),
        "stop pairs with a negative demand",
        "{origin_id} to {destination_id}",
    )
    _report(
        problems,
        demand.filter(pl.col("weight").is_null() | (pl.col("weight") < 0)),
        "stop pairs with no attraction weight",
        "{origin_id} to {destination_id}",
    )
    duplicate_keys = demand.group_by(*DEMAND_KEYS).len().filter(pl.col("len") > 1)
    _report(
        problems,
        duplicate_keys,
        "stop pairs that appear twice in one day",
        "{origin_id} to {destination_id}",
    )
    return problems


def check_timeline(matrix: pl.DataFrame, timeline: pl.DataFrame) -> list[str]:
    """The slices of a day have to hold the boardings of the day they came from.

    Summed over the slices, every cell must equal the same cell of the static
    matrix. It is the only check that catches a boarding dropped or counted
    twice by the slicing. Only meaningful for a partitioning scheme
    (``step == window``): overlapping slices are meant to count a ride in each.
    """
    if matrix.height == 0 or timeline.height == 0:
        return []
    keys = ["operator", "service_group", "origin_id", "destination_id"]
    day = matrix.group_by(keys).agg(pl.col("passengers").sum())
    sliced = timeline.group_by(keys).agg(pl.col("passengers").sum())
    drift = day.join(sliced, on=keys, how="full", suffix="_slices").filter(
        pl.col("passengers").fill_null(0) != pl.col("passengers_slices").fill_null(0)
    )
    if not drift.height:
        return []
    first = drift.row(0, named=True)
    return [
        f"{drift.height} od cells do not add up over the intervals "
        f"(e.g. {first['origin_id']}->{first['destination_id']}: "
        f"{first['passengers']} over the day, {first['passengers_slices']} over the slices)"
    ]


def check(report_inputs: dict[str, pl.DataFrame], flows: Flows) -> list[str]:
    """Every invariant, in one call, for a run to attach to its report."""
    problems = check_demand(report_inputs.get("demand", pl.DataFrame()))
    problems.extend(check_flows(flows))
    problems.extend(
        check_conservation(
            flows, report_inputs.get("demand", pl.DataFrame()), report_inputs.get("matrix")
        )
    )
    problems.extend(
        check_timeline(
            report_inputs.get("matrix", pl.DataFrame()),
            report_inputs.get("timeline", pl.DataFrame()),
        )
    )
    return problems


def check_provenance(
    flows: Flows, demand: pl.DataFrame, network: inputs.Network
) -> tuple[list[str], dict[str, int]]:
    """Every vehicle movement must trace back to the scraped timetable.

    The demand is invented; the supply is not. Easy to claim and worth proving, since
    a future change could quietly synthesise a route, a stop or a timing. Returns the
    problems and the counts checked, so a report can state what was verified.
    """
    served = network.served()
    known_stops = set(network.stops["stop_id"].to_list())
    supply_trips = set(served["trip_id"].unique().to_list())
    problems: list[str] = []

    generated = flows.trips.select("trip_id").unique()["trip_id"].to_list()
    invented = sorted(set(generated) - supply_trips)
    if invented:
        problems.append(
            f"{len(invented)} trips in the flows are not in the scraped timetable "
            f"(e.g. {invented[0]})"
        )

    # a stop visit exists in the supply, with the same clock
    visits = flows.trips.select("trip_id", "stop_id", "minutes").unique()
    known_visits = served.select("trip_id", "stop_id", "minutes").unique()
    unmatched = visits.join(known_visits, on=["trip_id", "stop_id", "minutes"], how="anti")
    if unmatched.height:
        first = unmatched.row(0, named=True)
        problems.append(
            f"{unmatched.height} stop visits in the flows are not calls in the timetable "
            f"(e.g. {first['trip_id']} at {first['stop_id']} at {first['minutes']} min)"
        )

    # every stop named anywhere is one the stop dictionary knows
    used = set(flows.trips["stop_id"].unique().to_list())
    used |= set(flows.od["origin_id"].unique().to_list())
    used |= set(flows.od["destination_id"].unique().to_list())
    if demand.height:
        used |= set(demand["origin_id"].unique().to_list())
        used |= set(demand["destination_id"].unique().to_list())
    unknown = sorted(used - known_stops)
    if unknown:
        problems.append(
            f"{len(unknown)} stops in the output are not in stops.csv (e.g. {unknown[0]})"
        )

    # a trip's line, direction and service are the operator's, not ours
    identity = served.select("trip_id", "line", "direction", "service_group").unique()
    ambiguous = identity.group_by("trip_id").len().filter(pl.col("len") > 1)
    if ambiguous.height:
        problems.append(
            f"{ambiguous.height} trips are listed under two lines or directions in the supply"
        )
    checked = (
        flows.trips.select("trip_id", "line", "direction", "service_group")
        .unique()
        .join(identity, on="trip_id", how="left", suffix="_supply")
    )
    mismatched = checked.filter(
        (pl.col("line") != pl.col("line_supply"))
        | (pl.col("direction") != pl.col("direction_supply"))
        | (pl.col("service_group") != pl.col("service_group_supply"))
    )
    if mismatched.height:
        problems.append(
            f"{mismatched.height} trips carry a line or direction the supply does not give them"
        )

    # an OD row is two calls of the same trip, in the order they happen
    calls = served.select("trip_id", "stop_id", "stop_seq").unique()
    origins = (
        flows.od.select("trip_id", "origin_id", "destination_id")
        .unique()
        .join(
            calls.rename({"stop_id": "origin_id", "stop_seq": "origin_seq"}),
            on=["trip_id", "origin_id"],
            how="left",
        )
    )
    legs = origins.join(
        calls.rename({"stop_id": "destination_id", "stop_seq": "destination_seq"}),
        on=["trip_id", "destination_id"],
        how="left",
    )
    unknown_calls = legs.filter(
        pl.col("origin_seq").is_null() | pl.col("destination_seq").is_null()
    )
    if unknown_calls.height:
        problems.append(
            f"{unknown_calls.height} OD rows ride between stops the trip does not call at"
        )
    # A circular trip calls at the same stop twice, so one OD row can join to
    # several occurrences of each end. What has to hold is that *some* occurrence
    # of the origin comes before *some* occurrence of the destination — checking
    # every combination would call a legitimate loop ride backwards.
    ordered = legs.group_by("trip_id", "origin_id", "destination_id").agg(
        pl.col("origin_seq").min().alias("first_origin"),
        pl.col("destination_seq").max().alias("last_destination"),
    )
    backwards = ordered.filter(pl.col("last_destination") <= pl.col("first_origin"))
    if backwards.height:
        problems.append(f"{backwards.height} OD rows ride backwards along their trip")
    counts = {
        "supply_trips": len(supply_trips),
        "supply_stops": len(known_stops),
        "generated_trips": len(generated),
        "stop_visits": visits.height,
        "od_rows": flows.od.height,
    }
    return problems, counts
