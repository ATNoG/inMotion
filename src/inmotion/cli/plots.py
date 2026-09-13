"""`inmotion plots` — regenerate figures from saved results."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from inmotion.cli._common import run_pipeline

app = typer.Typer(no_args_is_help=True, help="Regenerate figures from saved results.")
NAME = "plots"
HELP = "Regenerate figures from saved results."


@app.command("classical", help="Rebuild the classical-ML figures from saved CSVs.")
def classical(
    csv: Annotated[
        Path | None, typer.Option("--csv", help="A single classification_results.csv.")
    ] = None,
    results_dir: Annotated[
        Path | None, typer.Option("--results-dir", help="A directory of result CSVs.")
    ] = None,
    plots_dir: Annotated[Path, typer.Option("--plots-dir", help="Figure output directory.")] = Path(
        "plots"
    ),
    eda: Annotated[bool, typer.Option("--eda", help="Also regenerate the EDA figures.")] = False,
    data: Annotated[
        Path | None, typer.Option("--data", help="Dataset CSV, needed with --eda.")
    ] = None,
) -> None:
    """Rebuild the figures without retraining."""
    from inmotion.viz.regenerate import RunConfig, run

    run_pipeline(
        run,
        RunConfig(
            csv=csv,
            results_dir=results_dir,
            plots_dir=plots_dir,
            eda=eda,
            data=data,
        ),
    )


@app.command("dl", help="Rebuild the DL figures from saved CSVs.")
def dl(
    results_csv: Annotated[
        Path, typer.Option("--results-csv", help="Extended DL results CSV.")
    ] = Path("results/dl/dl_detailed_seed42.csv"),
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Figure output directory.")
    ] = Path("plots/dl"),
    eda: Annotated[bool, typer.Option("--eda", help="Also regenerate the EDA figures.")] = False,
    data: Annotated[
        Path | None, typer.Option("--data", help="Dataset CSV, needed with --eda.")
    ] = None,
) -> None:
    """Rebuild the DL figures without retraining."""
    from inmotion.viz.regenerate_dl import RunConfig, run

    run_pipeline(
        run,
        RunConfig(
            results_csv=results_csv,
            output_dir=output_dir,
            eda=eda,
            data=data,
        ),
    )


@app.command("quantization", help="Rebuild the quantization figures from the statistics CSVs.")
def quantization(
    data_dir: Annotated[
        Path, typer.Option("--data-dir", help="Statistics CSVs and device benchmark JSONs.")
    ] = Path("results/quant"),
    perf_dir: Annotated[Path, typer.Option("--perf-dir", help="Before/after latency CSVs.")] = Path(
        "results/perf"
    ),
    out_dir: Annotated[Path, typer.Option("--out-dir", help="Figure output directory.")] = Path(
        "results/quant/figures"
    ),
    dpi: Annotated[int, typer.Option("--dpi", help="Raster DPI (PDF is vector).")] = 600,
) -> None:
    """Rebuild the six quantization figures without re-measuring anything."""
    from inmotion.viz.quant_figures import RunConfig, run

    run_pipeline(
        run,
        RunConfig(
            data_dir=str(data_dir),
            perf_dir=str(perf_dir),
            out_dir=str(out_dir),
            dpi=dpi,
        ),
    )
