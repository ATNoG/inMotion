"""`inmotion bus` — scrape the Aveiro operators and build the OD matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, help="Scrape Aveiro bus networks and build OD matrices.")
NAME = "bus"
HELP = "Scrape Aveiro bus networks and build OD matrices."


@app.command("fetch", help="Download the raw BusWay pages and AveiroBus PDFs.")
def fetch(
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Re-download even when cached.")
    ] = False,
) -> None:
    """Populate ``data/bus/raw`` and the checksum manifest."""
    from inmotion.bus import sources

    with sources.Fetcher(force=refresh) as fetcher:
        busway_pages = sources.busway_inventory(fetcher)
        aveirobus_pdfs = sources.aveirobus_inventory(fetcher)
    typer.secho(
        f"busway pages {len(busway_pages)}, aveirobus pdfs {len(aveirobus_pdfs)}",
        fg=typer.colors.GREEN,
    )


@app.command("timeline", help="Slice the OD matrices by departure interval.")
def timeline(
    window: Annotated[int, typer.Option("--window", help="Slice length in minutes.")] = 60,
    step: Annotated[int, typer.Option("--step", help="Minutes between slice starts.")] = 60,
    wide: Annotated[
        bool,
        typer.Option("--wide", help="Also write a dense grid per operator, service and slice."),
    ] = False,
    out_dir: Annotated[Path, typer.Option("--out", help="Where to write the tables.")] = Path(
        "data/bus/processed"
    ),
) -> None:
    """Read the processed stop times and write one matrix per time interval.

    The slices are cut from `data/bus/processed/stop_times.csv`, so a different
    window costs nothing — no scrape, no OCR. `--out` is for an experiment that
    should not replace the hourly tables `bus build` writes.
    """
    from inmotion.bus.od import timeline as slices

    if window <= 0 or step <= 0:
        typer.secho(
            "--window and --step are lengths in minutes and must be positive.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        table = slices.od_timeline(slices.read_stop_times(), window=window, step=step)
    except RuntimeError as error:
        typer.secho(f"timeline refused: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    for path in slices.write_timeline(table, out_dir, wide=wide):
        typer.echo(f"wrote {path}")
    if not table.empty:
        typer.secho(
            f"{len(table):,} timeline cells over {table['interval'].nunique()} slices "
            f"of {window} min every {step}",
            fg=typer.colors.GREEN,
        )


@app.command("plot", help="Draw the two most similar routes of each operator.")
def plot(
    service_group: Annotated[
        str, typer.Option("--service", help="Which service to compare: weekday or weekend.")
    ] = "weekday",
    out_dir: Annotated[Path, typer.Option("--out", help="Figure output directory.")] = Path(
        "plots/bus"
    ),
    no_labels: Annotated[
        bool, typer.Option("--no-labels", help="Leave out the stop names on the ladders.")
    ] = False,
) -> None:
    """Render one figure per operator: both routes' OD grids and route ladders."""
    from inmotion.bus.viz import run
    from inmotion.bus.viz_config import PlotConfig

    code = run(
        PlotConfig(
            service_group=service_group,
            out_dir=str(out_dir),
            label_stops=not no_labels,
        )
    )
    if code:
        raise typer.Exit(code=code)


@app.command("build", help="Parse both networks and write the tables and OD matrices.")
def build(
    refresh: Annotated[bool, typer.Option("--refresh", help="Re-download before parsing.")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Only report the summary.")] = False,
) -> None:
    """Fetch if needed, parse, then write `data/bus/{interim,processed}`."""
    from inmotion.bus import build as builder

    try:
        code = builder.run(refresh=refresh, verbose=not quiet)
    except RuntimeError as error:
        # A missing OCR prerequisite is an environment problem, not a crash:
        # say what to install instead of printing a traceback.
        typer.secho(f"build refused: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error
    if code:
        raise typer.Exit(code=code)
    typer.secho("wrote data/bus/processed", fg=typer.colors.GREEN)


@app.command("geocode", help="Place the stops on the map, from OpenStreetMap.")
def geocode(
    bbox: Annotated[
        str,
        typer.Option(
            "--bbox",
            help="Area to index, as south,west,north,east (default: the Aveiro district).",
        ),
    ] = "40.30,-8.95,41.10,-8.10",
    stops: Annotated[
        Path | None,
        typer.Option("--stops", help="Per-stop table with lat/lon, which wins over any match."),
    ] = None,
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Re-fetch the OpenStreetMap index.")
    ] = False,
    web: Annotated[
        bool,
        typer.Option(
            "--web",
            help="Also ask Nominatim about the stops the local index could not place "
            "(one request a second, so this takes a while).",
        ),
    ] = False,
) -> None:
    """Resolve stop coordinates once, so a scenario run stays offline.

    Writes `stop_coords.csv`, `stop_clusters.csv` and `stops_unresolved.csv` into
    `data/bus/processed`. A stop that cannot be placed is left blank and listed,
    never guessed: a phantom coordinate is worse than a missing one.
    """
    from inmotion.bus.demand import coords as geocoder

    try:
        south, west, north, east = (float(part) for part in bbox.split(","))
    except ValueError:
        typer.secho(
            "--bbox takes four numbers: south,west,north,east", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=2) from None

    try:
        resolution = geocoder.geocode(
            attributes=stops, bbox=(south, west, north, east), refresh=refresh, web=web
        )
    except RuntimeError as error:
        typer.secho(f"geocode refused: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error
    typer.secho(resolution.summary(), fg=typer.colors.GREEN)
    for warning in resolution.warnings:
        typer.secho(f"  ! {warning}", fg=typer.colors.YELLOW)
    typer.echo("wrote data/bus/processed/stop_coords.csv")


def _pin(
    configs: list, *, fleet, fleet_policy, alpha, beta, boardings_per_departure, daily_boardings
) -> list:
    """Hold the dials the caller stated, and let the batch sample the rest.

    Without this, an explicit ``--alpha`` in a batch would be silently replaced
    by a sampled one, which is the kind of quiet override that makes a sweep
    impossible to reproduce from the command that ran it.
    """
    from dataclasses import replace

    overrides = {}
    if fleet:
        overrides["fleet"] = tuple(fleet)
    for name, value in (
        ("fleet_policy", fleet_policy),
        ("alpha", alpha),
        ("beta", beta),
        ("boardings_per_departure", boardings_per_departure),
        ("daily_boardings", daily_boardings),
    ):
        if value is not None:
            overrides[name] = value
    if not overrides:
        return configs
    return [replace(config, label="", **overrides) for config in configs]


@app.command("demand", help="Generate synthetic passenger flows and their OD matrices.")
def demand(
    scenarios: Annotated[
        int, typer.Option("--scenarios", "-n", help="How many scenarios to sample and generate.")
    ] = 1,
    seed: Annotated[
        int, typer.Option("--seed", help="Seed of the batch; scenario i uses seed+i.")
    ] = 42,
    fleet: Annotated[
        str | None,
        typer.Option("--fleet", help="Bus sizes as name:capacity:count, comma separated."),
    ] = None,
    fleet_policy: Annotated[
        str | None,
        typer.Option("--fleet-policy", help="random, cycle or demand."),
    ] = None,
    service: Annotated[
        str,
        typer.Option("--service", help="Which days to generate: weekday, weekend or both."),
    ] = "both",
    boardings_per_departure: Annotated[
        float | None,
        typer.Option(
            "--boardings-per-departure",
            help="Demand dial, when no daily total is set. Sampled when a batch is requested.",
        ),
    ] = None,
    daily_boardings: Annotated[
        int | None,
        typer.Option("--daily-boardings", help="Total riders for the day, split over the network."),
    ] = None,
    alpha: Annotated[
        float | None,
        typer.Option(
            "--alpha", help="Attraction exponent of a stop's service. Sampled in a batch."
        ),
    ] = None,
    beta: Annotated[
        float | None,
        typer.Option("--beta", help="Journey-time decay, per minute. Sampled in a batch."),
    ] = None,
    step: Annotated[
        int, typer.Option("--step", help="Seconds between ground-truth instants.")
    ] = 10,
    basemap: Annotated[
        str,
        typer.Option(
            "--basemap",
            help="What the map draws under the flows: auto (CARTO when CARTO_API_KEY is "
            "set, the timetable's own legs otherwise), carto-dark, carto-light, network, "
            "osm (needs the page served over HTTP) or none.",
        ),
    ] = "auto",
    sweep: Annotated[
        Path | None,
        typer.Option(
            "--sweep", help="JSON grid of parameters; every combination becomes a scenario."
        ),
    ] = None,
    no_instants: Annotated[
        bool, typer.Option("--no-instants", help="Skip the per-instant ground-truth table.")
    ] = False,
    no_map: Annotated[
        bool, typer.Option("--no-map", help="Skip the map and the GeoJSON files.")
    ] = False,
    no_cube: Annotated[
        bool,
        typer.Option("--no-cube", help="Skip the origin-by-destination-by-time cube page."),
    ] = False,
    cube_stops: Annotated[
        int,
        typer.Option("--cube-stops", help="How many of the busiest stops the cube holds."),
    ] = 60,
    no_replay: Annotated[
        bool,
        typer.Option("--no-replay", help="Skip the single-route day replay page."),
    ] = False,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Where to write: the scenario itself for a single run, "
            "or the directory holding one per scenario for a batch.",
        ),
    ] = None,
) -> None:
    """Generate flows for the processed network and write them under `data/bus/scenarios`."""
    from inmotion.bus.demand import build as generator
    from inmotion.bus.demand import config as scenario_config

    try:
        sizes = (
            scenario_config.BusSize.parse(fleet) if fleet else list(scenario_config.DEFAULT_FLEET)
        )
        groups = ("weekday", "weekend") if service == "both" else (service,)
    except ValueError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error
    if service not in ("weekday", "weekend", "both"):
        typer.secho("--service takes weekday, weekend or both", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    defaults = scenario_config.ScenarioConfig()
    try:
        basemap = scenario_config.resolve_basemap(basemap)
        base = scenario_config.ScenarioConfig(
            seed=seed,
            service_groups=groups,
            fleet=tuple(sizes),
            fleet_policy=fleet_policy or defaults.fleet_policy,
            boardings_per_departure=(
                defaults.boardings_per_departure
                if boardings_per_departure is None
                else boardings_per_departure
            ),
            daily_boardings=daily_boardings,
            alpha=defaults.alpha if alpha is None else alpha,
            beta=defaults.beta if beta is None else beta,
            instants_step_s=step,
            basemap=basemap,
            cube_stops=cube_stops,
        )
        if sweep is not None:
            configs = scenario_config.sweep_configs(sweep, base)
        elif scenarios > 1:
            configs = _pin(
                scenario_config.ScenarioSpace(
                    scenarios=scenarios, seed=seed, service_groups=groups
                ).sample(),
                fleet=tuple(sizes) if fleet else None,
                fleet_policy=fleet_policy,
                alpha=alpha,
                beta=beta,
                boardings_per_departure=boardings_per_departure,
                daily_boardings=daily_boardings,
            )
        else:
            configs = [base]
    except ValueError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error

    def progress(report: generator.Report) -> None:
        typer.echo(f"{report.label}: {report.boardings:,} boardings, {report.stranded:,} stranded")

    try:
        if len(configs) == 1:
            report = generator.run(
                configs[0],
                out_dir=out_dir,
                with_instants=not no_instants,
                with_map=not no_map,
                with_cube=not no_cube,
                with_replay=not no_replay,
            )
            typer.echo(report.summary())
        else:
            reports = generator.run_space(
                configs,
                out_root=out_dir,
                with_instants=not no_instants,
                with_map=not no_map,
                with_cube=not no_cube,
                with_replay=not no_replay,
                progress=progress,
            )
            report = reports[-1]
            typer.echo(f"wrote {len(reports)} scenarios and {generator.INDEX_FILE}")
    except RuntimeError as error:
        typer.secho(f"demand refused: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    for problem in report.problems:
        typer.secho(f"  x {problem}", fg=typer.colors.RED, err=True)
    for warning in report.warnings[:10]:
        typer.secho(f"  ! {warning}", fg=typer.colors.YELLOW)
    if len(report.warnings) > 10:
        typer.secho(
            f"  ! ... and {len(report.warnings) - 10} more in report.json",
            fg=typer.colors.YELLOW,
        )
    if report.problems:
        # A scenario whose invariants failed is not a scenario: say so with the
        # exit code rather than leaving a table that looks authoritative.
        typer.secho(f"{len(report.problems)} invariants failed", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.secho("wrote data/bus/scenarios", fg=typer.colors.GREEN)
