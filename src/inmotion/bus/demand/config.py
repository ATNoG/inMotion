"""One scenario's parameters, and the space a batch samples from.

Every value that changes a number in the output lives here, which makes a run
reproducible from its own ``scenario.json`` rather than from the command line that
produced it. :class:`ScenarioSpace` turns ranges and choices into N scenarios with
distinct seeds.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Any

#: ``random`` draws a size per trip from the fleet mix, ``cycle`` walks them in
#: departure order, ``demand`` gives the biggest vehicles to the busiest lines.
FLEET_POLICIES: tuple[str, ...] = ("random", "cycle", "demand")

#: Basemaps a page can draw under the flows. CARTO needs a key (``CARTO_KEY_ENV``,
#: never stored in a scenario); OSM's tiles need an HTTP referer, so they only work
#: for a page served over HTTP; ``network`` is the timetable's own geometry.
BASEMAPS: tuple[str, ...] = ("carto-dark", "carto-light", "network", "osm", "none")

#: The configuration value that means "whatever this machine can actually draw":
#: CARTO when its key is in the environment, the network's own legs otherwise.
AUTO_BASEMAP = "auto"

#: Where the CARTO key comes from. Not a scenario parameter on purpose: it is a
#: secret, and the pages and records this package writes are meant to be shared.
CARTO_KEY_ENV = "CARTO_API_KEY"

#: What a scenario uses when it is not told otherwise. ``network`` rather than
#: ``auto`` because a configuration object should mean what it says; the command
#: line is where ``auto`` belongs.
DEFAULT_BASEMAP = "network"

#: The share of a day's riders per departure band (see ``inmotion.bus.od.bands``).
DEFAULT_BAND_WEIGHTS: dict[str, float] = {
    "early": 0.04,
    "am_peak": 0.28,
    "midday": 0.18,
    "pm_peak": 0.32,
    "evening": 0.16,
    "night": 0.02,
}


def resolve_basemap(name: str, environment: Mapping[str, str] | None = None) -> str:
    """Turn :data:`AUTO_BASEMAP` into the best basemap this machine can draw.

    A keyed basemap in a scenario nobody can render is worse than no basemap: the
    page would come out with a grid of "API KEY REQUIRED" tiles and no hint why.
    So the choice is made once, here, against the environment, and the name that
    is written into ``scenario.json`` is the one that was actually drawn.
    """
    if name != AUTO_BASEMAP:
        return name
    variables = os.environ if environment is None else environment
    return "carto-dark" if variables.get(CARTO_KEY_ENV) else "network"


@dataclass(frozen=True)
class BusSize:
    """One size of vehicle in the fleet, and how many of them there are."""

    name: str
    capacity: int
    count: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a bus size needs a name")
        if self.capacity <= 0:
            raise ValueError(f"{self.name}: capacity must be positive, got {self.capacity}")
        if self.count < 0:
            raise ValueError(f"{self.name}: count cannot be negative, got {self.count}")

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable copy, for ``scenario.json``."""
        return {"name": self.name, "capacity": self.capacity, "count": self.count}

    @classmethod
    def parse(cls, text: str) -> list[BusSize]:
        """Parse ``standard:40:8,articulated:70:3`` into sizes, smallest first.

        Ordering by capacity is deliberate: every policy that walks the fleet
        then walks it from the smallest vehicle to the largest, so the result
        does not depend on the order the sizes were typed in.
        """
        sizes = []
        for item in text.split(","):
            chunk = item.strip()
            if not chunk:
                continue
            parts = chunk.split(":")
            if len(parts) != 3:
                raise ValueError(f"{chunk!r} is not a bus size; expected name:capacity:count")
            name, capacity, count = parts
            try:
                sizes.append(cls(name.strip(), int(capacity), int(count)))
            except ValueError as error:
                if "invalid literal" in str(error):
                    raise ValueError(
                        f"{chunk!r} is not a bus size; capacity and count must be integers"
                    ) from error
                raise
        if not sizes:
            raise ValueError("a fleet needs at least one bus size")
        return sorted(sizes, key=lambda size: (size.capacity, size.name))


#: The fleet a scenario falls back to: mostly standard buses, with minibuses for
#: the thin lines and articulated buses for the busiest, which is roughly the
#: shape of a small regional operator.
DEFAULT_FLEET: tuple[BusSize, ...] = (
    BusSize("minibus", 20, 2),
    BusSize("standard", 40, 8),
    BusSize("articulated", 70, 3),
)

#: Fleet mixes a scenario batch can be sampled from. The counts are vehicles,
#: not trips: the number is what the size policy distributes.
FLEET_MIXES: tuple[tuple[BusSize, ...], ...] = (
    DEFAULT_FLEET,
    (BusSize("minibus", 20, 4), BusSize("standard", 40, 10)),
    (BusSize("standard", 40, 12), BusSize("articulated", 70, 2)),
)


@dataclass(frozen=True)
class ScenarioConfig:
    """Everything one generated scenario depends on."""

    seed: int = 42
    service_groups: tuple[str, ...] = ("weekday",)
    fleet: tuple[BusSize, ...] = (BusSize("standard", 40, 1),)
    fleet_policy: str = "random"
    #: Riders per scheduled departure, when no daily total is given: a magnitude,
    #: not a measurement. 12 puts the scraped network at ~17,000 riders a day.
    boardings_per_departure: float = 12.0
    #: Total boardings for the day, split over the network by departures. Replaces
    #: ``boardings_per_departure`` when set.
    daily_boardings: int | None = None
    #: Attraction exponent: how much a stop's own service level draws riders.
    alpha: float = 1.0
    #: Journey-time decay, per minute: how sharply a longer ride is avoided.
    beta: float = 0.06
    band_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_BAND_WEIGHTS))
    #: The ground-truth grid ``instants.csv`` writes: step, and reach either side.
    instants_step_s: int = 10
    instants_window_s: int = 30
    #: What the map draws underneath the flows: see :data:`BASEMAPS`.
    basemap: str = DEFAULT_BASEMAP
    #: Stops on the space-time cube's lattice (the busiest ones).
    cube_stops: int = 60
    label: str = ""

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if not self.service_groups:
            raise ValueError("a scenario needs at least one service group")
        unknown = set(self.service_groups) - {"weekday", "weekend"}
        if unknown:
            raise ValueError(f"unknown service group(s): {', '.join(sorted(unknown))}")
        if not self.fleet:
            raise ValueError("a scenario needs at least one bus size")
        if sum(size.count for size in self.fleet) <= 0:
            raise ValueError("the fleet holds no buses")
        if self.fleet_policy not in FLEET_POLICIES:
            raise ValueError(
                f"unknown fleet policy {self.fleet_policy!r}: "
                f"expected one of {', '.join(FLEET_POLICIES)}"
            )
        if self.boardings_per_departure <= 0 and self.daily_boardings is None:
            raise ValueError("boardings_per_departure must be positive when no daily total is set")
        if self.daily_boardings is not None and self.daily_boardings <= 0:
            raise ValueError(f"daily_boardings must be positive, got {self.daily_boardings}")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha and beta are non-negative dials")
        if any(weight < 0 for weight in self.band_weights.values()):
            raise ValueError("band weights are shares and cannot be negative")
        if sum(self.band_weights.values()) <= 0:
            raise ValueError("band weights sum to zero, so no hour attracts any rider")
        if self.basemap not in BASEMAPS:
            raise ValueError(
                f"unknown basemap {self.basemap!r}: expected one of {', '.join(BASEMAPS)}"
            )
        if self.cube_stops < 2:
            raise ValueError(f"a cube needs at least two stops, got {self.cube_stops}")
        if self.instants_step_s <= 0 or self.instants_window_s < 0:
            raise ValueError(
                f"instants need a positive step and a non-negative window, "
                f"got {self.instants_step_s} and {self.instants_window_s}"
            )

    # -- identity ----------------------------------------------------------

    def resolved_label(self) -> str:
        """The scenario's directory name: what it is, in one readable string."""
        if self.label:
            return self.label
        scale = (
            f"daily{self.daily_boardings}"
            if self.daily_boardings is not None
            else f"bpd{self.boardings_per_departure:g}"
        )
        return f"s{self.seed:04d}-{self.fleet_policy}-{scale}-{'-'.join(self.service_groups)}"

    def _payload(self) -> dict[str, Any]:
        """The parameters alone: no label, no fingerprint, so hashing terminates."""
        return {
            "seed": self.seed,
            "service_groups": list(self.service_groups),
            "fleet": [size.to_dict() for size in self.fleet],
            "fleet_policy": self.fleet_policy,
            "boardings_per_departure": self.boardings_per_departure,
            "daily_boardings": self.daily_boardings,
            "alpha": self.alpha,
            "beta": self.beta,
            "band_weights": dict(sorted(self.band_weights.items())),
            "instants_step_s": self.instants_step_s,
            "instants_window_s": self.instants_window_s,
            "basemap": self.basemap,
            "cube_stops": self.cube_stops,
        }

    def fingerprint(self) -> str:
        """A short digest of the parameters, so two scenarios can be told apart."""
        payload = json.dumps(self._payload(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        """The configuration as ``scenario.json`` holds it."""
        return {
            "label": self.resolved_label(),
            "fingerprint": self.fingerprint(),
            **self._payload(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScenarioConfig:
        """Rebuild a configuration from ``scenario.json``."""
        return cls(
            seed=int(payload["seed"]),
            service_groups=tuple(payload["service_groups"]),
            fleet=tuple(
                BusSize(size["name"], int(size["capacity"]), int(size["count"]))
                for size in payload["fleet"]
            ),
            fleet_policy=str(payload["fleet_policy"]),
            boardings_per_departure=float(payload["boardings_per_departure"]),
            daily_boardings=(
                None if payload.get("daily_boardings") is None else int(payload["daily_boardings"])
            ),
            alpha=float(payload["alpha"]),
            beta=float(payload["beta"]),
            band_weights={key: float(value) for key, value in payload["band_weights"].items()},
            instants_step_s=int(payload["instants_step_s"]),
            instants_window_s=int(payload["instants_window_s"]),
            basemap=str(payload.get("basemap", DEFAULT_BASEMAP)),
            cube_stops=int(payload.get("cube_stops", 60)),
            label=str(payload.get("label", "")),
        )


def with_peak_share(base: dict[str, float], share: float) -> dict[str, float]:
    """Rescale a band profile so the two peaks hold ``share`` of the day.

    Sampling a scenario by redrawing every band weight independently would
    produce profiles that are inconsistent with themselves — shares that no
    longer sum to one, or a midday larger than both peaks together. Rescaling
    keeps the shape and moves one number, which is the axis a sweep wants.
    """
    if not 0 < share < 1:
        raise ValueError(f"peak share must be a share of the day, got {share}")
    peaks = [name for name in base if name in ("am_peak", "pm_peak")]
    if not peaks:
        raise ValueError("the band profile has no peak bands to rescale")
    peak_total = sum(base[name] for name in peaks)
    rest_total = sum(value for name, value in base.items() if name not in peaks)
    if peak_total <= 0 or rest_total <= 0:
        raise ValueError("the band profile needs both peak and off-peak weight")
    scaled = {name: base[name] / peak_total * share for name in peaks}
    scaled.update(
        {
            name: value / rest_total * (1 - share)
            for name, value in base.items()
            if name not in peaks
        }
    )
    return scaled


@dataclass(frozen=True)
class ScenarioSpace:
    """The ranges a batch of scenarios is sampled from."""

    scenarios: int = 1
    seed: int = 42
    boardings_per_departure: tuple[float, float] = (4.0, 30.0)
    alpha: tuple[float, float] = (0.5, 1.5)
    beta: tuple[float, float] = (0.02, 0.12)
    peak_share: tuple[float, float] = (0.45, 0.75)
    fleets: tuple[tuple[BusSize, ...], ...] = FLEET_MIXES
    fleet_policies: tuple[str, ...] = FLEET_POLICIES
    service_groups: tuple[str, ...] = ("weekday", "weekend")

    def sample(self) -> list[ScenarioConfig]:
        """Draw ``scenarios`` configurations, each with its own seed."""
        if self.scenarios < 1:
            raise ValueError(f"a batch holds at least one scenario, got {self.scenarios}")
        fleets = self.fleets or (DEFAULT_FLEET,)
        policies = self.fleet_policies or FLEET_POLICIES
        rng = random.Random(self.seed)
        drawn: list[ScenarioConfig] = []
        seen: set[str] = set()
        for index in range(self.scenarios):
            config = ScenarioConfig(
                seed=self.seed + index,
                service_groups=self.service_groups,
                fleet=fleets[index % len(fleets)],
                fleet_policy=policies[index % len(policies)],
                boardings_per_departure=rng.uniform(*self.boardings_per_departure),
                alpha=rng.uniform(*self.alpha),
                beta=rng.uniform(*self.beta),
                band_weights=with_peak_share(DEFAULT_BAND_WEIGHTS, rng.uniform(*self.peak_share)),
            )
            label = config.resolved_label()
            if label in seen:
                label = f"{label}-{index:02d}"
            seen.add(label)
            drawn.append(replace(config, label=label))
        return drawn


def sweep_configs(path: str | Path, base: ScenarioConfig) -> list[ScenarioConfig]:
    """Every combination of the values a sweep file lists.

    The file is a JSON object of parameter names to lists of values, e.g.
    ``{"fleet_policy": ["random", "demand"], "beta": [0.02, 0.1]}``. Values are
    crossed, not zipped: a sweep is a grid, and each cell becomes a scenario
    with its own seed, so no two scenarios in a batch share one.
    """
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{path} must hold an object of parameter names to value lists")
    names = sorted(payload)
    for name in names:
        values = payload[name]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name}: a sweep parameter needs a non-empty list of values")
        if not hasattr(base, name):
            raise ValueError(f"{name} is not a scenario parameter")
    configs = []
    for index, combination in enumerate(product(*(payload[name] for name in names))):
        overrides: dict[str, Any] = dict(zip(names, combination, strict=True))
        if "fleet" in overrides:
            overrides["fleet"] = tuple(
                size if isinstance(size, BusSize) else BusSize(**size)
                for size in overrides["fleet"]
            )
        if "band_weights" in overrides:
            overrides["band_weights"] = {
                str(key): float(value) for key, value in overrides["band_weights"].items()
            }
        configs.append(replace(base, seed=base.seed + index, label="", **overrides))
    labels: list[str] = []
    for index, config in enumerate(configs):
        label = config.resolved_label()
        if label in labels:
            label = f"{label}-{index:02d}"
        labels.append(label)
    return [replace(config, label=label) for config, label in zip(configs, labels, strict=True)]
