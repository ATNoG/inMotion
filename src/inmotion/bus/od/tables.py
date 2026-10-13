"""The tidy tables: one row per trip, and one row per trip and stop.

These are the input every other module in this package reads, so they are kept
free of aggregation: a value here is what a single trip did, never a sum over
trips.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from inmotion.bus.od.bands import service_group
from inmotion.bus.od.stops import stop_id
from inmotion.bus.records import Trip


def stop_times_frame(trips: Iterable[Trip]) -> pd.DataFrame:
    """The full tidy timetable: one row per trip and stop."""
    rows = []
    for trip in trips:
        for call in trip.calls:
            rows.append(
                {
                    "trip_id": trip.trip_id,
                    "operator": trip.operator,
                    "line": trip.line,
                    "direction": trip.direction,
                    "service": trip.service,
                    "service_group": service_group(trip.service),
                    "stop_seq": call.stop_seq,
                    "stop_id": stop_id(trip.operator, call.stop),
                    "stop": call.stop,
                    "time": call.time,
                    "minutes": call.minutes,
                    "served": call.minutes is not None,
                    "markers": call.markers,
                }
            )
    return pd.DataFrame(rows)


def trips_frame(trips: Iterable[Trip]) -> pd.DataFrame:
    """One row per trip, with its endpoints and its span."""
    rows = []
    for trip in trips:
        served = trip.served
        rows.append(
            {
                "trip_id": trip.trip_id,
                "operator": trip.operator,
                "line": trip.line,
                "direction": trip.direction,
                "service": trip.service,
                "service_group": service_group(trip.service),
                "trip_index": trip.trip_index,
                "season": trip.season,
                "school_only": trip.school_only,
                "colour": trip.colour,
                "block": trip.block,
                "stops_served": len(served),
                "origin": served[0].stop if served else "",
                "destination": served[-1].stop if served else "",
                "departure": served[0].time if served else "",
                "arrival": served[-1].time if served else "",
                "departure_minutes": served[0].minutes if served else None,
                "duration_min": (served[-1].minutes - served[0].minutes if served else None),
                "source": trip.source,
                "notes": trip.notes,
            }
        )
    return pd.DataFrame(rows)
