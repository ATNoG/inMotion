"""Generate one scenario end to end, or a batch of them, and write them out.

Read the supply, lay the demand over the pairs the network runs, spread it across
the trips, run the buses over the queues, cut the result into the matrices, check
every invariant — including that the supply is the scraped one — then write. Each
scenario lands in its own directory with the configuration that produced it, and a
batch adds ``index.csv`` for comparing them.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from inmotion.bus import paths
from inmotion.bus.demand import (
    assignment,
    inputs,
    matrices,
    model,
)
from inmotion.bus.demand import (
    coords as coordinates,
)
from inmotion.bus.demand import (
    map as map_module,
)
from inmotion.bus.demand import (
    pairs as pair_relation,
)
from inmotion.bus.demand.config import ScenarioConfig, ScenarioSpace

# Imported from their own modules rather than through the package's re-exports:
# `simulate`, `instants` and `validate` are functions there, so the package
# attributes of those names are the functions and not the modules — a
# distinction that only shows up as an AttributeError when the import order
# inside `__init__` happens to change.
from inmotion.bus.demand.instants import instants
from inmotion.bus.demand.simulate import simulate
from inmotion.bus.demand.validate import check, check_provenance

#: The supply files a scenario was generated from, digested into its record.
DIGEST_INPUTS = ("stop_times.csv", "stops.csv")

INDEX_FILE = "index.csv"

INDEX_COLUMNS = (
    "label",
    "seed",
    "fleet_policy",
    "boardings_per_departure",
    "daily_boardings",
    "alpha",
    "beta",
    "service_groups",
    "trips",
    "stop_visits",
    "demand",
    "boardings",
    "stranded",
    "denied",
    "peak_load_factor",
    "matrix_cells",
    "timeline_cells",
    "problems",
    "warnings",
)


@dataclass
class Report:
    """What a generated scenario produced, and what it had to work around."""

    label: str = ""
    seed: int = 0
    fingerprint: str = ""
    fleet_policy: str = ""
    boardings_per_departure: float = 0.0
    daily_boardings: int | None = None
    alpha: float = 0.0
    beta: float = 0.0
    service_groups: str = ""
    trips: int = 0
    stop_visits: int = 0
    pairs: int = 0
    demand: int = 0
    boardings: int = 0
    alightings: int = 0
    denied: int = 0
    stranded: int = 0
    peak_load_factor: float = 0.0
    matrix_cells: int = 0
    timeline_cells: int = 0
    instant_rows: int = 0
    cube_points: int = 0
    replay_routes: int = 0
    #: What the run verified about where its supply came from: every trip, stop
    #: and clock is the scraped timetable's, while the demand is generated.
    provenance: dict[str, int] = field(default_factory=dict)
    fleet: str = ""
    coords_resolved: int = 0
    coords_total: int = 0
    seconds: float = 0.0
    files: list[str] = field(default_factory=list)
    #: Invariants that failed. A run with problems is not a usable scenario, and
    #: the command that produced it says so with its exit code.
    problems: list[str] = field(default_factory=list)
    #: Things worth knowing that did not break anything: a missing cache, a stop
    #: with no coordinates, riders left waiting at the end of the day.
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """The scenario in numbers, for a terminal or a report."""
        lines = [
            f"scenario       {self.label}",
            f"seed           {self.seed}",
            f"fleet          {self.fleet or 'none'}",
            f"trips          {self.trips:,}",
            f"stop visits    {self.stop_visits:,}",
            f"stop pairs     {self.pairs:,}",
            f"demand         {self.demand:,}",
            f"boarded        {self.boardings:,}",
            f"alighted       {self.alightings:,}",
            f"left behind    {self.denied:,}",
            f"never boarded  {self.stranded:,}",
            f"peak load      {self.peak_load_factor:.2f}",
            f"matrix cells   {self.matrix_cells:,}",
            f"timeline cells {self.timeline_cells:,}",
            f"instants       {self.instant_rows:,}",
            f"cube rides     {self.cube_points:,}",
            f"replay routes  {self.replay_routes:,}",
        ]
        if self.coords_total:
            lines.append(f"coordinates    {self.coords_resolved:,}/{self.coords_total:,}")
        lines.append(f"problems       {len(self.problems)}")
        lines.append(f"warnings       {len(self.warnings)}")
        if self.provenance:
            lines.append(
                f"supply         {self.provenance['supply_trips']:,} trips / "
                f"{self.provenance['supply_stops']:,} stops, all from the scrape"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """The report as ``report.json`` holds it."""
        return asdict(self)

    def index_row(self) -> dict:
        """The columns ``index.csv`` carries, so a batch can be compared."""
        row = {name: getattr(self, name) for name in INDEX_COLUMNS}
        row["problems"] = len(self.problems)
        row["warnings"] = len(self.warnings)
        return row


def digest(directory: Path, names: tuple[str, ...] = DIGEST_INPUTS) -> dict[str, str]:
    """The sha256 of each input file, so a scenario names the tables it read."""
    digests = {}
    for name in names:
        path = directory / name
        if path.exists():
            digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def stop_coordinates(directory: Path, attributes: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Coordinates per stop: the geocoded table, with attributes as overrides."""
    warnings: list[str] = []
    path = directory / coordinates.COORDS_FILE
    if path.exists():
        frame = pl.read_csv(path, schema_overrides={"stop_id": pl.String})
    else:
        frame = pl.DataFrame(
            schema={"stop_id": pl.String, "lat": pl.Float64, "lon": pl.Float64, "source": pl.String}
        )
        warnings.append(
            f"no {path.name}; run `inmotion bus geocode` for a map "
            f"(the flows do not need coordinates)"
        )
    overrides = (
        attributes.filter(pl.col("lat").is_not_null() & pl.col("lon").is_not_null()).select(
            "stop_id", "lat", "lon"
        )
        if attributes.height and "lat" in attributes.columns
        else pl.DataFrame(schema={"stop_id": pl.String, "lat": pl.Float64, "lon": pl.Float64})
    )
    if overrides.height:
        frame = (
            frame.drop("lat", "lon")
            .join(overrides, on="stop_id", how="full", coalesce=True)
            .with_columns(pl.lit("manual").alias("source"))
        )
    if "source" not in frame.columns:
        frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias("source"))
    return frame, warnings


def run(
    config: ScenarioConfig,
    *,
    source_dir: str | Path = paths.PROCESSED_DIR,
    out_dir: str | Path | None = None,
    network: inputs.Network | None = None,
    attributes: pl.DataFrame | None = None,
    with_instants: bool = True,
    with_map: bool = True,
    with_cube: bool = True,
    with_replay: bool = True,
) -> Report:
    """Generate one scenario and write its tables. Returns what it produced."""
    started = time.monotonic()
    root = Path(source_dir)
    network = inputs.read_network(root) if network is None else network
    attributes = inputs.read_stop_attributes(None) if attributes is None else attributes
    directory = Path(out_dir) if out_dir is not None else paths.scenario(config.resolved_label())
    directory.mkdir(parents=True, exist_ok=True)

    report = Report(
        label=config.resolved_label(),
        seed=config.seed,
        fingerprint=config.fingerprint(),
        fleet_policy=config.fleet_policy,
        boardings_per_departure=config.boardings_per_departure,
        daily_boardings=config.daily_boardings,
        alpha=config.alpha,
        beta=config.beta,
        service_groups=" ".join(config.service_groups),
        warnings=inputs.check(network, attributes),
    )

    coords, coord_warnings = stop_coordinates(root, attributes)
    if with_map and coords.height and map_module.placed_count(coords):
        # Before a single table is written: a run whose map cannot be drawn says
        # so now, rather than leaving a scenario behind that looks finished.
        map_module.require_key(config.basemap)

    served = network.served()
    dictionary = network.dictionary()
    names = dictionary.select("stop_id", "stop")
    stamped = pair_relation.with_time_keys(
        pair_relation.name_stops(pair_relation.per_trip_pairs(served), names)
    )
    demand = model.generate(stamped, network, attributes, config, stops=dictionary)
    report.warnings.extend(demand.warnings)
    assigned = assignment.assign(demand.pairs, stamped, config)
    report.warnings.extend(assigned.warnings)
    flows = simulate(served, assigned.intents, assigned.fleet, config, names=names)
    report.warnings.extend(flows.warnings)

    matrix = matrices.passenger_matrix(flows.od)
    timeline = matrices.passenger_timeline(flows.od)
    table = instants(flows.trips, config) if with_instants else pl.DataFrame()

    report.problems = check({"demand": demand.pairs, "matrix": matrix, "timeline": timeline}, flows)
    provenance_problems, provenance_counts = check_provenance(flows, demand.pairs, network)
    report.problems.extend(provenance_problems)
    report.provenance = provenance_counts
    if not provenance_problems:
        report.warnings.append(
            f"supply verified: {provenance_counts['generated_trips']:,} trips, "
            f"{provenance_counts['stop_visits']:,} stop visits and "
            f"{provenance_counts['od_rows']:,} OD rows all trace to the scraped timetable "
            f"(the demand itself is generated)"
        )
    report.trips = network.trips().height
    report.stop_visits = flows.trips.height
    report.pairs = demand.pairs.height
    report.demand = demand.total()
    report.boardings = int(flows.trips["boarded"].sum()) if flows.trips.height else 0
    report.alightings = int(flows.trips["alighted"].sum()) if flows.trips.height else 0
    report.denied = int(flows.trips["denied"].sum()) if flows.trips.height else 0
    report.stranded = flows.stranded
    report.peak_load_factor = (
        float(flows.trips["load_factor"].max() or 0.0) if flows.trips.height else 0.0
    )
    report.matrix_cells = matrix.height
    report.timeline_cells = timeline.height
    report.instant_rows = table.height
    report.fleet = assigned.summary()

    written = [
        _write_json(directory / "scenario.json", config.to_dict() | {"inputs": digest(root)}),
        _write(directory / "flows.csv", flows.trips),
        _write(directory / "stop_flow.csv", flows.stops),
        _write(directory / "od_flow.csv", flows.od),
        _write(directory / "od_demand.csv", demand.pairs),
        _write(directory / "od_passengers.csv", matrix),
        _write(directory / "od_passengers_timeline.csv", timeline),
    ]
    if table.height:
        written.append(_write(directory / "instants.csv", table))
    written.extend(_write_wide(directory, matrix, stamped))

    report.warnings.extend(coord_warnings)
    if coords.height:
        known = set(coords.filter(pl.col("lat").is_not_null())["stop_id"].to_list())
        report.coords_total = dictionary.height
        report.coords_resolved = len(known & set(dictionary["stop_id"].to_list()))
        if with_map and report.coords_resolved:
            written.extend(
                map_module.write(
                    directory,
                    flows=flows.trips,
                    od=flows.od,
                    stops=dictionary,
                    coords=coords,
                    matrix=matrix,
                    config=config,
                    report=report,
                )
            )
        if with_cube and report.coords_resolved:
            cube = map_module.write_cube(
                directory,
                od=flows.od,
                flows=flows.trips,
                stops=dictionary,
                coords=coords,
                config=config,
                report=report,
                stop_total=dictionary.height,
            )
            if cube.exists():
                written.append(cube)
                report.cube_points = len(
                    map_module.cube_payload(
                        flows.od, flows.trips, dictionary, coords, stops_limit=config.cube_stops
                    )["points"]
                )
        if with_replay and report.coords_resolved:
            pages_written = map_module.write_replay(
                directory,
                flows=flows.trips,
                stops=dictionary,
                coords=coords,
                config=config,
                report=report,
            )
            if pages_written:
                written.extend(pages_written)
                report.replay_routes = len(
                    map_module.replay_payload(flows.trips, dictionary, coords)["routes"]
                )
    report.files = [path.name for path in written]
    report.seconds = round(time.monotonic() - started, 2)
    _write_json(directory / "report.json", report.to_dict())
    return report


def _write(path: Path, frame: pl.DataFrame) -> Path:
    """Write a table, with a header even when it is empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path)
    return path


def _write_json(path: Path, payload: dict) -> Path:
    """Write a record with the timestamp of when it was generated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stamped = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"), **payload}
    path.write_text(json.dumps(stamped, indent=2, ensure_ascii=False) + "\n")
    return path


def _write_wide(directory: Path, matrix: pl.DataFrame, stamped: pl.DataFrame) -> list[Path]:
    """One dense grid per operator and service day, for a quick look."""
    written: list[Path] = []
    if not matrix.height:
        return written
    for row in (
        matrix.select("operator", "service_group")
        .unique()
        .sort("operator", "service_group")
        .iter_rows(named=True)
    ):
        operator, group = row["operator"], row["service_group"]
        subset = matrix.filter(
            (pl.col("operator") == operator) & (pl.col("service_group") == group)
        )
        axis = pair_relation_grid_axis(stamped, operator)
        grid = matrices.wide(subset, axis)
        if grid.height:
            written.append(_write(directory / f"od_passengers_wide_{operator}_{group}.csv", grid))
    return written


def pair_relation_grid_axis(stamped: pl.DataFrame, operator: str) -> pl.DataFrame:
    """Every stop of one operator, as the axis a dense grid is laid out on."""
    mine = stamped.filter(pl.col("operator") == operator)
    return pl.concat(
        [
            mine.select("operator", pl.col("origin_id"), pl.col("origin")),
            mine.select(
                "operator",
                pl.col("destination_id").alias("origin_id"),
                pl.col("destination").alias("origin"),
            ),
        ]
    ).unique()


def run_space(
    space: ScenarioSpace | Sequence[ScenarioConfig],
    *,
    source_dir: str | Path = paths.PROCESSED_DIR,
    out_root: str | Path | None = None,
    with_instants: bool = True,
    with_map: bool = True,
    with_cube: bool = True,
    with_replay: bool = True,
    progress: Callable[[Report], None] | None = None,
) -> list[Report]:
    """Generate a batch of scenarios, sharing the supply tables between them.

    Takes either a space to sample or the scenarios themselves: a sweep file has
    already decided which cells to run, and sampling them a second time would
    silently ignore what the file asked for.
    """
    configs = space.sample() if isinstance(space, ScenarioSpace) else list(space)
    network = inputs.read_network(source_dir)
    attributes = inputs.read_stop_attributes(None)
    root = Path(out_root) if out_root is not None else paths.SCENARIO_DIR
    reports = []
    for config in configs:
        report = run(
            config,
            source_dir=source_dir,
            out_dir=root / config.resolved_label(),
            network=network,
            attributes=attributes,
            with_instants=with_instants,
            with_map=with_map,
            with_cube=with_cube,
            with_replay=with_replay,
        )
        reports.append(report)
        if progress is not None:
            progress(report)
    _write(
        root / INDEX_FILE,
        pl.DataFrame([report.index_row() for report in reports], schema=index_schema()),
    )
    return reports


def index_schema() -> dict:
    """The types of ``index.csv``, so an empty batch still writes a header."""
    return {
        "label": pl.String,
        "seed": pl.Int64,
        "fleet_policy": pl.String,
        "boardings_per_departure": pl.Float64,
        "daily_boardings": pl.Int64,
        "alpha": pl.Float64,
        "beta": pl.Float64,
        "service_groups": pl.String,
        "trips": pl.Int64,
        "stop_visits": pl.Int64,
        "demand": pl.Int64,
        "boardings": pl.Int64,
        "stranded": pl.Int64,
        "denied": pl.Int64,
        "peak_load_factor": pl.Float64,
        "matrix_cells": pl.Int64,
        "timeline_cells": pl.Int64,
        "problems": pl.Int64,
        "warnings": pl.Int64,
    }
