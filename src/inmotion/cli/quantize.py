"""`inmotion quantize` — quantize models and measure what it costs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from inmotion.quant.spec import FLOAT_PRECISIONS, PRESET_SPECS

app = typer.Typer(no_args_is_help=True, help="Quantize models and measure the cost.")
NAME = "quantize"
HELP = "Quantize models and measure the cost."

#: Models that can be deployed; the JEPA family is the target.
DEPLOY_MODELS = ("lejepa", "t_jepa", "ts_jepa", "cf_jepa")
ALL_FORMATS = (*FLOAT_PRECISIONS, *sorted(PRESET_SPECS))


def _parse_models(value: str | None) -> list[str]:
    if not value:
        return list(DEPLOY_MODELS)
    return [m.strip() for m in value.split(",") if m.strip()]


def _parse_formats(value: str | None) -> tuple[str, ...]:
    if not value:
        return ("fp32", "bf16", "fp16", "int8", "q8", "q4", "q2")
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
    from inmotion.quant.model import CAST_DTYPES

    typer.echo(f"{'format':8s} {'bits':>4s}  granularity / detail")
    for name in FLOAT_PRECISIONS:
        detail = "cast-only (dtype round-trip)"
        typer.echo(f"{name:8s} {4 if name == 'fp32' else 2:>4d}  {detail}")
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


@app.command("eval", help="Measure accuracy under quantization on the paper's split.")
def evaluate(
    models: Annotated[
        str | None, typer.Option("--models", help="Comma-separated; default: the JEPA family.")
    ] = None,
    formats: Annotated[
        str | None, typer.Option("--formats", help="Comma-separated; default: all.")
    ] = None,
    data: Annotated[
        Path | None, typer.Option("--data", help="Dataset CSV; default: dataset.icaisf.csv.")
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Split seed.")] = 42,
    out: Annotated[Path, typer.Option("--out", help="Where to write the results CSV.")] = Path(
        "results/quant/sweep.csv"
    ),
) -> None:
    """Sweep models x formats and write one row per measurement.

    The unquantized run must reproduce the MCC recorded in MASTER_RESULTS.md; if
    it does not, the harness is wrong and the command fails rather than
    publishing numbers nobody can trace.
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
    typer.secho("\nbaseline reproduced: fp32 matches MASTER_RESULTS", fg=typer.colors.GREEN)
