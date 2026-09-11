"""`inmotion analyze` — analyse results across seeds and routes."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from inmotion.cli._common import run_pipeline

app = typer.Typer(no_args_is_help=True, help="Analyse results.")
NAME = "analyze"
HELP = "Analyse results."


@app.command("seeds", help="Combine and plot results from multi-seed runs.")
def seeds(
    results_dir: Annotated[
        Path, typer.Option("--results-dir", help="Directory of per-seed result CSVs.")
    ] = Path("results"),
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Where to write the combined analysis.")
    ] = Path("combined"),
    seeds_csv: Annotated[
        str, typer.Option("--seeds", help="Comma-separated seeds to combine.")
    ] = "3,5,42",
) -> None:
    """Compute cross-seed statistics and ranking stability."""
    from inmotion.analysis.seeds import RunConfig, run

    run_pipeline(run, RunConfig(results_dir=results_dir, output_dir=output_dir, seeds=seeds_csv))


@app.command("interference", help="Cross-route interference from the concurrent_noise_path column.")
def interference(
    data: Annotated[
        Path, typer.Option("--data", help="Dataset CSV with concurrent_noise_path.")
    ] = Path("data/processed/dataset.csv"),
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Figure output directory.")
    ] = Path("plots/interference"),
) -> None:
    """Quantify how a concurrent device's route degrades classification."""
    from inmotion.analysis.interference import RunConfig, run

    run_pipeline(run, RunConfig(data=data, output_dir=output_dir))
