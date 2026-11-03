"""Generate passenger flows over the scraped Aveiro timetables, and the OD matrices.

The operators publish supply, never demand: a timetable says where the buses go
and when, and nothing about who was on them. This package invents the demand a
timetable could be carrying — reproducibly, from a scenario file, in the
vocabulary the rest of the project already uses.

Per scenario, under ``data/bus/scenarios/<label>``: ``flows.csv`` (trip and stop:
boarded, alighted, load, left behind), ``stop_flow.csv`` (the queue per stop and
hour), ``od_flow.csv``, ``instants.csv`` (the AA/BB/BA/AB grid at 10 s),
``od_demand.csv``, ``od_passengers.csv`` with its wide and hourly variants,
``scenario.json`` and ``report.json`` — plus three pages: the map, the
space-time cube and the route replay. A batch adds ``index.csv``.

Submodules: ``config``, ``inputs``, ``pairs``, ``model`` (the gravity demand),
``assignment`` (demand to trips, and the fleet), ``simulate`` (the queue),
``instants``, ``matrices``, ``validate``, ``coords`` (OpenStreetMap), ``map`` and
``pages`` (the viewers), ``build`` (the run), ``allocate`` (apportionment).
"""

from __future__ import annotations

from inmotion.bus.demand.assignment import Assignment, assign
from inmotion.bus.demand.build import Report, run, run_space
from inmotion.bus.demand.config import (
    FLEET_MIXES,
    FLEET_POLICIES,
    BusSize,
    ScenarioConfig,
    ScenarioSpace,
    sweep_configs,
)
from inmotion.bus.demand.inputs import Network, read_network, read_stop_attributes
from inmotion.bus.demand.instants import instants
from inmotion.bus.demand.matrices import (
    passenger_matrix,
    passenger_timeline,
    wide,
)
from inmotion.bus.demand.model import Demand, generate
from inmotion.bus.demand.pairs import per_trip_pairs, with_time_keys
from inmotion.bus.demand.simulate import Flows, simulate
from inmotion.bus.demand.validate import check

__all__ = [
    "FLEET_MIXES",
    "FLEET_POLICIES",
    "Assignment",
    "BusSize",
    "Demand",
    "Flows",
    "Network",
    "Report",
    "ScenarioConfig",
    "ScenarioSpace",
    "assign",
    "check",
    "generate",
    "instants",
    "passenger_matrix",
    "passenger_timeline",
    "per_trip_pairs",
    "read_network",
    "read_stop_attributes",
    "run",
    "run_space",
    "simulate",
    "sweep_configs",
    "wide",
    "with_time_keys",
]
