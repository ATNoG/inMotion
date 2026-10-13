"""The OD relation: every ordered stop pair a rider can ride *directly*.

Pairs are enumerated per trip, so a row is a direct ride rather than a
connection, and the band is stamped on each trip before the group is aggregated.
Banding afterwards, from the group's earliest departure, would file a whole day
of service under one morning band.
"""

from __future__ import annotations

import pandas as pd

from inmotion.bus.od.bands import band, format_minutes, service_group


def od_pairs(stop_times: pd.DataFrame) -> pd.DataFrame:
    """Every reachable ordered stop pair, aggregated over the trips serving it.

    Enumerated per trip, so each pair is a *direct* ride on that trip rather
    than a connection: one row per line, direction, service and pair.
    """
    served = stop_times[stop_times["served"]].sort_values(["trip_id", "stop_seq"])
    rows = []
    for keys, group in served.groupby(
        ["operator", "line", "direction", "service", "trip_id"], sort=False
    ):
        operator, line, direction, service, trip_id = keys
        calls = list(group[["stop_id", "stop", "minutes"]].itertuples(index=False, name=None))
        for index, origin in enumerate(calls):
            for destination in calls[index + 1 :]:
                rows.append(
                    (
                        operator,
                        line,
                        direction,
                        service,
                        trip_id,
                        origin[0],
                        origin[1],
                        destination[0],
                        destination[1],
                        int(origin[2]),
                        int(destination[2]) - int(origin[2]),
                    )
                )
    per_trip = pd.DataFrame(
        rows,
        columns=[
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
        ],
    )
    if per_trip.empty:
        return per_trip
    per_trip["service_group"] = per_trip["service"].map(service_group)
    # Band every trip by its own departure. A (line, direction, service) group can
    # span the whole service day, so deriving the band later from first_minutes
    # would file every trip of that group under its earliest departure.
    per_trip["departure_band"] = per_trip["departure_minutes"].map(band)
    aggregated = (
        per_trip.groupby(
            [
                "operator",
                "service_group",
                "departure_band",
                "line",
                "direction",
                "service",
                "origin_id",
                "origin",
                "destination_id",
                "destination",
            ],
            as_index=False,
        )
        .agg(
            trips=("trip_id", "nunique"),
            first_minutes=("departure_minutes", "min"),
            last_minutes=("departure_minutes", "max"),
            min_run_min=("run_min", "min"),
            median_run_min=("run_min", "median"),
            max_run_min=("run_min", "max"),
        )
        .sort_values(["operator", "line", "direction", "service", "origin_id", "destination_id"])
    )
    aggregated["first_departure"] = aggregated["first_minutes"].map(format_minutes)
    aggregated["last_departure"] = aggregated["last_minutes"].map(format_minutes)
    return aggregated
