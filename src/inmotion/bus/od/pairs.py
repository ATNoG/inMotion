"""The OD relation: every ordered stop pair a rider can ride *directly*.

Pairs are enumerated per trip, so a row is a direct ride rather than a
connection, and the band is stamped on each trip before the group is aggregated.
Banding afterwards, from the group's earliest departure, would file a whole day
of service under one morning band.

`per_trip_pairs` is that enumeration; `od_pairs` is one projection of it, and
`inmotion.bus.od.timeline` is the other. Both stamp their own time key on the
same rows, which is why they agree cell by cell.
"""

from __future__ import annotations

import pandas as pd

from inmotion.bus.od.bands import band, format_minutes, service_group

PAIR_COLUMNS = [
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
]


def per_trip_pairs(stop_times: pd.DataFrame) -> pd.DataFrame:
    """Every ordered stop pair one trip serves, one row per trip and pair.

    ``departure_minutes`` is the time at the pair's own origin, not at the trip's
    first stop: it is when *this* ride leaves, which is the instant both the
    departure bands and the time intervals are keyed on.

    Two consecutive calls that name the same stop at the same minute are one call,
    not two. That is how the operators print it: BusWay's line 107 page lists
    ``Rio Tinto Sul`` twice at 07:41, and a bus cannot be at one stop twice in the
    same minute. The tidy table keeps both rows, because it records what the page
    says; the relation must not, or every pair through that stop is counted twice.
    """
    served = stop_times[stop_times["served"]].sort_values(["trip_id", "stop_seq"])
    rows = []
    for keys, group in served.groupby(
        ["operator", "line", "direction", "service", "trip_id"], sort=False
    ):
        operator, line, direction, service, trip_id = keys
        calls = list(group[["stop_id", "stop", "minutes"]].itertuples(index=False, name=None))
        calls = [call for index, call in enumerate(calls) if index == 0 or call != calls[index - 1]]
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
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def od_pairs(stop_times: pd.DataFrame) -> pd.DataFrame:
    """Every reachable ordered stop pair, aggregated over the trips serving it.

    Enumerated per trip, so each pair is a *direct* ride on that trip rather
    than a connection: one row per line, direction, service, departure band and
    pair. ``trips`` counts the departures serving the pair — a circular route
    that passes the same two stops twice counts twice, because a rider can take
    it twice — which is also the only unit a time slicing of this table can add
    back up to.
    """
    per_trip = per_trip_pairs(stop_times)
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
            trips=("trip_id", "size"),
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
