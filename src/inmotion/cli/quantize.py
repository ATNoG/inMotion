"""`inmotion quantize` — quantize models and measure what it costs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from inmotion.quant.spec import FLOAT_PRECISIONS, PRESET_SPECS

app = typer.Typer(no_args_is_help=True, help="Quantize models and measure the cost.")
NAME = "quantize"
HELP = "Quantize models and measure the cost."

#: Models the quantization sweep evaluates: the deployable family plus the
#: SIGReg CNN, which :data:`inmotion.quant.evaluate.BASELINE_MCC` also records.
DEPLOY_MODELS = ("lejepa", "t_jepa", "ts_jepa", "cf_jepa", "sigreg")
ALL_FORMATS = (*FLOAT_PRECISIONS, *sorted(PRESET_SPECS))


def _parse_models(value: str | None) -> list[str]:
    if not value:
        return list(DEPLOY_MODELS)
    return [m.strip() for m in value.split(",") if m.strip()]


def _parse_formats(value: str | None) -> tuple[str, ...]:
    if not value:
        # One definition, in the module that owns the baseline. Imported here
        # rather than at module scope so `inmotion --help` stays fast.
        from inmotion.quant.evaluate import DEFAULT_PRECISIONS

        return tuple(DEFAULT_PRECISIONS)
    chosen = tuple(f.strip() for f in value.split(",") if f.strip())
    unknown = [f for f in chosen if f not in ALL_FORMATS]
    if unknown:
        raise typer.BadParameter(
            f"unknown format(s) {unknown}; choose from {', '.join(ALL_FORMATS)}"
        )
    return chosen


@app.command("formats", help="List the available quantization schemes.")
def formats() -> None:
    """Show each scheme, its bit width and its storage granularity."""
    from inmotion.quant.model import CAST_BYTES, CAST_DTYPES

    typer.echo(f"{'format':8s} {'bits':>4s}  granularity / detail")
    for name in FLOAT_PRECISIONS:
        detail = "cast-only (dtype round-trip)"
        typer.echo(f"{name:8s} {CAST_BYTES[name] * 8:>4d}  {detail}")
    for name, spec in sorted(PRESET_SPECS.items(), key=lambda kv: -kv[1].bits):
        gran = spec.granularity.value
        if spec.granularity.value == "per_group":
            gran = f"per_group(group={spec.group_size})"
        typer.echo(
            f"{name:8s} {spec.bits:>4d}  {gran}, {'symmetric' if spec.symmetric else 'asymmetric'}"
        )
    typer.echo("")
    typer.echo(f"cast dtypes: {', '.join(sorted(CAST_DTYPES))}")


@app.command("show", help="Quantize one model and report size and error.")
def show(
    model: Annotated[str, typer.Argument(help="Registered model name.")],
    format: Annotated[str, typer.Option("--format", "-f", help="Scheme, e.g. q4.")] = "int8",
) -> None:
    """Report the size, compression and weight error for one scheme."""
    from inmotion.models import registry as reg
    from inmotion.quant.model import apply, apply_cast
    from inmotion.quant.spec import resolve as resolve_spec

    if format not in ALL_FORMATS:
        typer.secho(
            f"unknown format {format!r}; choose from {', '.join(ALL_FORMATS)}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    base = reg.load(reg.by_name(model), strict=True)
    fp32_params = sum(p.numel() for p in base.parameters())

    if format in FLOAT_PRECISIONS:
        _, report = apply_cast(base, format)
    else:
        _, report = apply(base, resolve_spec(format))

    typer.echo(f"model        : {model} ({fp32_params:,} parameters)")
    typer.echo(f"format       : {format}  ({report.bits} bits)")
    typer.echo(f"quantized    : {report.quantized_bytes / 1e6:.3f} MB")
    typer.echo(f"unquantized  : {report.skipped_param_bytes / 1e6:.3f} MB (norms, biases, tokens)")
    typer.echo(f"buffers      : {report.buffer_bytes / 1e6:.3f} MB")
    typer.echo(f"total        : {report.total_bytes / 1e6:.3f} MB")
    typer.echo(f"compression  : {report.compression:.2f}x on quantized weights")
    typer.echo(f"max |w err|  : {report.max_error:.6f}")
    typer.echo(f"tensors      : {len(report.records)} quantized, {len(report.skipped)} skipped")


@app.command("export", help="Write a portable artifact for a non-Python runtime.")
def export_cmd(
    model: Annotated[str, typer.Argument(help="Registered model name.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Destination directory.")],
    format: Annotated[
        str, typer.Option("--format", "-f", help="Scheme, or fp32 for none.")
    ] = "fp32",
    golden: Annotated[bool, typer.Option("--golden", help="Also write golden vectors.")] = True,
    tol: Annotated[
        float, typer.Option("--tol", help="Parity tolerance recorded in golden.json.")
    ] = 2e-4,
) -> None:
    """Export an artifact plus the golden vectors a Rust runtime must reproduce."""
    from inmotion.quant.export import export, write_golden

    if format not in ALL_FORMATS:
        typer.secho(
            f"unknown format {format!r}; choose from {', '.join(ALL_FORMATS)}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    quant = None if format == "fp32" else format
    report = export(model, out, quant=quant)
    typer.secho(f"wrote {out}", fg=typer.colors.GREEN)
    for key, value in report.summary().items():
        typer.echo(f"  {key:18s} {value}")

    if golden:
        path = write_golden(model, out, quant=quant, dataset=None)
        import json

        payload = json.loads(path.read_text())
        payload["tol"] = tol
        path.write_text(json.dumps(payload, indent=2) + "\n")
        typer.secho(f"wrote {path}  ({len(payload['cases'])} cases)", fg=typer.colors.GREEN)
        typer.echo("")
        typer.echo("verify with:")
        typer.echo(
            "  cd rust && cargo run --release -p inmotion-cli -- check "
            f"--artifact {out} --golden {out}"
        )


@app.command("eval", help="Measure accuracy under quantization on dataset.csv.")
def evaluate(
    models: Annotated[
        str | None,
        typer.Option("--models", help="Comma-separated; default: all five world models."),
    ] = None,
    formats: Annotated[
        str | None, typer.Option("--formats", help="Comma-separated; default: DEFAULT_PRECISIONS.")
    ] = None,
    data: Annotated[
        Path | None, typer.Option("--data", help="Dataset CSV; default: dataset.csv.")
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Split seed.")] = 42,
    out: Annotated[Path, typer.Option("--out", help="Where to write the results CSV.")] = Path(
        "results/quant/sweep_dataset.csv"
    ),
) -> None:
    """Sweep models x formats and write one row per measurement.

    The unquantized run must reproduce the baseline MCC recorded in
    ``inmotion.quant.evaluate.BASELINE_MCC`` for dataset.csv; if it does not,
    the harness is wrong and the command fails rather than publishing numbers
    nobody can trace. The published table is a different dataset and is not
    consulted.
    """
    from inmotion.quant.evaluate import check_baseline, sweep, write_csv

    try:
        chosen_formats = _parse_formats(formats)
    except typer.BadParameter as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    rows = sweep(_parse_models(models), chosen_formats, dataset=data, seed=seed)
    if not rows:
        typer.secho("no measurements produced", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    write_csv(rows, out)
    problems = check_baseline(rows)
    if problems:
        typer.secho(
            "\nBASELINE MISMATCH — these numbers are not trustworthy:",
            fg=typer.colors.RED,
            err=True,
        )
        for problem in problems:
            typer.secho(f"  {problem}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.secho(
        "\nbaseline reproduced: fp32 matches BASELINE_MCC on dataset.csv",
        fg=typer.colors.GREEN,
    )


@app.command("stats", help="Write the quantization statistics tables the figures read.")
def stats(
    models: Annotated[
        str | None,
        typer.Option("--models", help="Comma-separated; default: all five world models."),
    ] = None,
    formats: Annotated[
        str | None, typer.Option("--formats", help="Comma-separated; default: DEFAULT_PRECISIONS.")
    ] = None,
    data: Annotated[
        Path | None, typer.Option("--data", help="Dataset CSV; default: dataset.csv.")
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Split seed.")] = 42,
    sweep: Annotated[
        Path,
        typer.Option("--sweep", help="Accuracy/size CSV from `quantize eval`."),
    ] = Path("results/quant/sweep_dataset.csv"),
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Where the tables and raw capture go.")
    ] = Path("results/quant"),
    latency_iters: Annotated[
        int,
        typer.Option("--latency-iters", help="Batch-1 torch reference timings; 0 disables."),
    ] = 200,
    latency_warmup: Annotated[
        int, typer.Option("--latency-warmup", help="Warmup iterations per measurement.")
    ] = 20,
) -> None:
    """Produce size, accuracy, coverage, latency and per-layer tables.

    Inputs are artifacts, never transcribed numbers: the sweep CSV from
    ``quantize eval``, a raw per-tensor capture this command writes, and every
    device benchmark JSON already in ``--out-dir``. The fp32 rows are re-checked
    against ``BASELINE_MCC`` before anything is written, so a table that
    disagrees with the enforced baseline fails instead of publishing.
    """
    from inmotion.viz.quant_stats import (
        MANIFEST_JSON,
        RAW_JSON,
        build_tables,
        collect,
        device_json_paths,
        latency_rows,
        measure_reference_latency,
        write_json,
    )

    try:
        chosen_formats = _parse_formats(formats)
    except typer.BadParameter as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    chosen_models = _parse_models(models)
    typer.echo("capturing per-tensor quantization reports")
    payload = collect(chosen_models, chosen_formats, dataset=data, seed=seed)
    write_json(payload, RAW_JSON)

    artifacts = device_json_paths(out_dir)
    typer.echo(f"ingesting {len(artifacts)} device artifact(s): {[p.name for p in artifacts]}")
    latency = latency_rows(artifacts)

    if latency_iters > 0:
        # One row per model, at fp32: on the dequantizing torch path the
        # precision does not change what is executed, so timing every precision
        # would only manufacture a latency curve out of noise.
        typer.echo("measuring torch reference latency on this machine (fp32)")
        for model_name in chosen_models:
            latency.append(
                measure_reference_latency(
                    model_name,
                    "fp32",
                    dataset=data,
                    seed=seed,
                    iterations=latency_iters,
                    warmup=latency_warmup,
                )
            )

    written = build_tables(out_dir, sweep=sweep, raw=RAW_JSON, latency=latency)
    typer.secho(
        f"\nwrote {len(written)} tables + {MANIFEST_JSON.name}",
        fg=typer.colors.GREEN,
    )
