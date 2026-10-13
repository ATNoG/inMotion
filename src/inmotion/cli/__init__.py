"""The `inmotion` command-line interface.

One package, one module per command group:

``model``     inspect, verify, predict and adopt checkpoints
``datasets``  dataset versions, provenance and checksums
``data``      build and clean datasets (export, merge, clean, augment)
``bus``       scrape the Aveiro bus networks and build the OD matrices
``train``     train models (classification, dl, exotic, hpo-paper)
``evaluate``  evaluate and combine models (ensemble, checkpoints, shap, importance)
``analyze``   analyse results (seeds, interference)
``plots``     regenerate figures from saved results
``doctor``    report on the environment and the artifact store

Each command builds a typed ``RunConfig`` and calls the pipeline's ``run()``.
The CLI holds no pipeline logic, and the pipelines hold no argument parsing, so
either side can change without touching the other.
"""

from __future__ import annotations

from typing import Annotated

import typer

from inmotion import __version__
from inmotion.cli import (
    analyze,
    bus,
    data,
    datasets,
    doctor,
    evaluate,
    model,
    plots,
    quantize,
    train,
)

app = typer.Typer(
    name="inmotion",
    help="WiFi RSSI mobility classification: data, models, training and evaluation.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)

for sub in (model, datasets, data, bus, quantize, train, evaluate, analyze, plots):
    app.add_typer(sub.app, name=sub.NAME, help=sub.HELP)

app.command("doctor", help="Report on the environment and the artifact store.")(doctor.doctor)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"inmotion {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """[bold]inMotion[/] — WiFi RSSI mobility classification."""


__all__ = ["app", "main"]
