"""The record shapes both operator parsers produce.

Keeping one definition for both networks is what lets `inmotion.bus.od` build
the OD matrices without caring which site a trip came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Call:
    """One trip calling at one stop.

    ``time`` is empty when the trip does not call there at all — BusWay prints a
    dash, and the same thing is reconstructed from the grid for AveiroBus.
    """

    stop_seq: int
    stop: str
    time: str
    minutes: int | None
    markers: str = ""


@dataclass
class Trip:
    """One scheduled trip, with every stop it calls at in order."""

    trip_id: str
    operator: str
    line: str
    direction: str
    service: str
    season: str
    trip_index: int
    source: str
    notes: str = ""
    block: int = 0
    school_only: bool = False
    colour: str = ""
    calls: list[Call] = field(default_factory=list)

    @property
    def served(self) -> list[Call]:
        """The calls the trip actually makes, in order."""
        return [call for call in self.calls if call.minutes is not None]


def roll_past_midnight(trip: Trip) -> int:
    """Put a trip's times on one continuous scale across midnight.

    Both operators write the last trips of the day as ``23:58`` then ``00:02``,
    which read literally runs backwards and makes every stop-to-stop journey
    time between the two negative. A time more than twelve hours earlier than
    the one before it is therefore the next day, and the trip's remaining calls
    are shifted with it — the same way the operators' own early-morning services
    are printed. Returns how many midnight crossings were found.
    """
    offset = 0
    previous: int | None = None
    shifted = 0
    for call in trip.calls:
        if call.minutes is None:
            continue
        minutes = call.minutes + offset
        if previous is not None and minutes < previous - 12 * 60:
            offset += 24 * 60
            minutes += 24 * 60
            shifted += 1
        call.minutes = minutes
        call.time = f"{minutes // 60:02d}:{minutes % 60:02d}"
        previous = minutes
    return shifted


def timed(calls: list[Call]) -> list[Call]:
    """Calls with a time, in stop order."""
    return [call for call in calls if call.minutes is not None]
