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
