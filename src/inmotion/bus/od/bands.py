"""Departure bands and service labels, as the two operators write them.

Every classification the OD tables depend on lives here, because a band applied
at the wrong point silently rewrites the matrix: a (line, direction, service)
group can span the whole service day, so banding such a group by its earliest
departure files every one of its trips under that first band.
"""

from __future__ import annotations

# Departure bands, in minutes past midnight. Portuguese timetables run past
# 24:00 for the last trips of the day, so the last band is open-ended.
BANDS: tuple[tuple[str, int, int], ...] = (
    ("early", 0, 7 * 60),
    ("am_peak", 7 * 60, 9 * 60 + 30),
    ("midday", 9 * 60 + 30, 16 * 60),
    ("pm_peak", 16 * 60, 20 * 60),
    ("evening", 20 * 60, 24 * 60),
    ("night", 24 * 60, 48 * 60),
)

WEEKEND = {"saturday", "sunday", "weekend", "holiday"}


def band(minutes: int | None) -> str:
    """The departure band a trip's first stop belongs to."""
    if minutes is None:
        return "unknown"
    for name, low, high in BANDS:
        if low <= minutes < high:
            return name
    return "night"


def format_minutes(minutes: int | None) -> str:
    """Minutes past midnight back to ``HH:MM``, keeping hours past 24."""
    if minutes is None:
        return ""
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def service_group(service: str) -> str:
    """Collapse the operators' own service labels to weekday or weekend."""
    return "weekend" if service in WEEKEND else "weekday"
