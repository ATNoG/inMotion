"""`inmotion data` — build and clean datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from inmotion.cli._common import existing_file, run_pipeline

app = typer.Typer(no_args_is_help=True, help="Build and clean datasets.")
NAME = "data"
HELP = "Build and clean datasets."


@app.command("export", help="Turn a raw Wavecom capture into a per-route CSV.")
def export(
    input_file: Annotated[Path, typer.Option("--input", "-i", help="Raw Wavecom capture (.txt).")],
    output_file: Annotated[Path, typer.Option("--output", "-o", help="Destination CSV.")],
    group: Annotated[
        list[str] | None, typer.Option("--group", help="label:mac1,mac2,... (repeatable).")
    ] = None,
    group_label: Annotated[
        list[str] | None,
        typer.Option("--group-label", help="group_name:final_label (repeatable)."),
    ] = None,
) -> None:
    """Pivot 10-second captures into one row per route."""
    from inmotion.data.export import export_to_csv

    existing_file(input_file, "input capture")
    export_to_csv(str(input_file), str(output_file), groups=group, group_labels=group_label)
    typer.secho(f"wrote {output_file}", fg=typer.colors.GREEN)


@app.command("merge", help="Merge per-route CSVs into one dataset.")
def merge(
    input_dir: Annotated[Path, typer.Option("--input", "-i", help="Directory of per-route CSVs.")],
    output_file: Annotated[Path, typer.Option("--output", "-o", help="Merged dataset CSV.")],
    pattern: Annotated[str, typer.Option("--pattern", help="Glob for input files.")] = "*.csv",
) -> None:
    """Concatenate the interim CSVs and label concurrent-noise rows."""
    from inmotion.data.merge import merge_csv_files

    if not input_dir.is_dir():
        typer.secho(f"input directory not found: {input_dir}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    merge_csv_files(str(input_dir), str(output_file), pattern=pattern)
    typer.secho(f"wrote {output_file}", fg=typer.colors.GREEN)


@app.command("clean", help="Audit a dataset per device and drop rows that poison training.")
def clean(
    data: Annotated[Path, typer.Option("--data", help="Dataset CSV to audit.")],
    output: Annotated[
        Path | None, typer.Option("--output", help="Write the cleaned CSV here.")
    ] = None,
    keep_shifted: Annotated[
        bool, typer.Option("--keep-shifted", help="Keep device-shifted but self-consistent rows.")
    ] = False,
    drop_ambiguous: Annotated[
        bool, typer.Option("--drop-ambiguous", help="Also drop samples the model rejects.")
    ] = False,
    ambiguity_thresh: Annotated[
        float,
        typer.Option(
            "--ambiguity-thresh",
            help="Confidence above which a wrong prediction is dropped.",
        ),
    ] = 0.9,
) -> None:
    """Quantify which devices or samples carry inconsistent labels."""
    from inmotion.data.clean import RunConfig, run

    existing_file(data, "dataset")
    run_pipeline(
        run,
        RunConfig(
            data=data,
            output=output,
            keep_shifted=keep_shifted,
            drop_ambiguous=drop_ambiguous,
            ambiguity_thresh=ambiguity_thresh,
        ),
    )


@app.command("augment", help="Augment a dataset with noise injection and intra-class mixup.")
def augment(
    data: Annotated[Path, typer.Option("--data", help="Dataset CSV to augment.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Destination CSV.")],
) -> None:
    """Add class-conditioned noise and mixup rows, flagged `synthetic`."""
    from inmotion.data.augment import augment

    existing_file(data, "dataset")
    augment(str(data), str(output))
    typer.secho(f"wrote {output}", fg=typer.colors.GREEN)
