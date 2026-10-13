"""Turn parsed trips into the tidy tables and the origin-destination matrices.

The operators publish supply, not demand: nobody records how many people made a
trip. What can be built from a timetable is therefore a *service* OD matrix —
for every ordered pair of stops, how many direct trips connect them, how long
that takes and when the first and last one leaves. That is the quantity an
accessibility or headway study needs, and it is what these tables hold.

Outputs, all in ``data/bus/processed``:

``stops.csv``
    one row per stop, per operator, with the canonical id used everywhere else
``trips.csv``
    one row per trip: endpoints, span, and where it came from
``stop_times.csv``
    the full tidy timetable — one row per trip and stop
``od_pairs.csv``
    the OD relation broken down by line and direction, so a pair can be traced
    back to the trips that provide it
``od_matrix.csv``
    the matrix itself, aggregated over lines: operator, service, departure band
    and stop pair against direct trips and travel time
``od_matrix_wide_<service>.csv``
    the same matrix as a dense origins-by-destinations grid, for a quick look

Stop identity is per operator and derived from the OCR'd name, so it survives
the artefacts a bitmap timetable produces: punctuation, accents and the
occasional ``?`` for an ordinal all normalise to the same key.

Submodules:

``bands``     departure bands and the operators' service labels
``stops``     stop identity, and the stop dictionary built from it
``tables``    the tidy per-trip and per-stop tables
``pairs``     the OD relation, one row per line, direction and stop pair
``matrix``    the matrices: aggregated over lines, long and wide
``validate``  the invariants a trustworthy matrix has to satisfy
"""

from __future__ import annotations

from inmotion.bus.od.bands import BANDS, WEEKEND, band, format_minutes, service_group
from inmotion.bus.od.matrix import od_matrix, wide_matrix
from inmotion.bus.od.pairs import od_pairs
from inmotion.bus.od.stops import canonical, stop_id, stops_frame
from inmotion.bus.od.tables import stop_times_frame, trips_frame
from inmotion.bus.od.validate import validate

__all__ = [
    "BANDS",
    "WEEKEND",
    "band",
    "canonical",
    "format_minutes",
    "od_matrix",
    "od_pairs",
    "service_group",
    "stop_id",
    "stop_times_frame",
    "stops_frame",
    "trips_frame",
    "validate",
    "wide_matrix",
]
