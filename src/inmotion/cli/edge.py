"""`inmotion edge` — the edge-quantization campaign, phase by phase.

Every command is a thin wrapper: the work lives in ``inmotion.edge.*`` and the
imports happen inside the command body, so ``inmotion --help`` stays fast and a
missing optional backend cannot break the whole CLI.

The one-command entry point is ``campaign``; the rest exist to rerun a single
phase without redoing the run (and are what the campaign calls internally).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, help="Edge-quantization campaign on the wavecom routers.")
NAME = "edge"
HELP = "Edge-quantization campaign on the wavecom routers."

_DEVICE_OPTION = Annotated[
    str, typer.Option("--device", help="Device key: wave-router, wave-router-2 or local.")
]
_PROFILE_OPTION = Annotated[str, typer.Option("--profile", help="smoke, quick or full.")]
_ARTIFACTS_OPTION = Annotated[
    Path, typer.Option("--artifacts-dir", help="Directory of <model>-<precision> artifacts.")
]
_TABLES_OPTION = Annotated[Path, typer.Option("--tables-dir", help="Directory for canonical CSVs.")]
_RAW_OPTION = Annotated[Path, typer.Option("--raw-dir", help="Directory for raw device output.")]
_BINARY_OPTION = Annotated[
    Path,
    typer.Option("--binary", help="Static musl runtime binary to deploy."),
]
_WINDOWS_OPTION = Annotated[
    Path, typer.Option("--windows-dir", help="Directory of per-model windows/labels/classes.")
]


def _split(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


@app.command("preflight", help="Report what each device is and has.")
def preflight(
    devices: Annotated[
        str, typer.Option("--devices", help="Comma-separated device keys (default: both routers).")
    ] = "wave-router,wave-router-2",
) -> None:
    """Probe kernel, CPU, memory, tools and the deployed binary."""
    import json

    from inmotion.edge import transport

    for key in _split(devices):
        device = transport.open_device(key)
        facts = device.preflight()
        typer.echo(f"{key}:")
        typer.echo(json.dumps(facts, indent=2, sort_keys=True))
        close = getattr(device, "close", None)
        if callable(close):
            close()


@app.command("artifacts", help="Export the deployable artifact matrix.")
def artifacts(
    artifacts_dir: Annotated[
        Path, typer.Option("--artifacts-dir", help="Output directory for artifacts.")
    ] = Path("results/edge/artifacts"),
    models: Annotated[
        str | None, typer.Option("--models", help="Comma-separated models (default: contract set).")
    ] = None,
    precisions: Annotated[
        str | None,
        typer.Option("--precisions", help="Comma-separated precisions (default: deployable set)."),
    ] = None,
) -> None:
    """Write ``<model>-<precision>/`` plus ``artifacts.csv``."""
    from inmotion.edge import accuracy, contracts

    accuracy.export_artifacts(
        artifacts_dir,
        models=_split(models) or contracts.MODELS,
        precisions=_split(precisions) or contracts.PRECISIONS_DEPLOYABLE,
    )
    typer.secho(f"artifacts -> {artifacts_dir}", fg=typer.colors.GREEN)


@app.command("accuracy", help="Score every artifact from its own weights.")
def accuracy_command(
    artifacts_dir: _ARTIFACTS_OPTION = Path("results/edge/artifacts"),
    tables_dir: _TABLES_OPTION = Path("results/edge/tables"),
) -> None:
    """Write accuracy.csv, per_class.csv and confusion.csv."""
    from inmotion.edge import accuracy

    paths = accuracy.evaluate_artifacts(artifacts_dir, tables_dir)
    typer.secho(
        f"accuracy -> {', '.join(str(path) for path in paths.values())}", fg=typer.colors.GREEN
    )


@app.command("windows", help="Export the replayed test windows and labels.")
def windows(
    windows_dir: _WINDOWS_OPTION = Path("results/edge/windows"),
    models: Annotated[
        str | None, typer.Option("--models", help="Comma-separated models (default: contract set).")
    ] = None,
) -> None:
    """Write per-model windows.txt, labels.txt and classes.json."""
    from inmotion.edge import accuracy, contracts

    paths = accuracy.export_windows(windows_dir, models=_split(models) or contracts.MODELS)
    typer.secho(f"windows -> {len(paths)} model(s) under {windows_dir}", fg=typer.colors.GREEN)


@app.command("bench", help="Run the load/latency matrix on a device.")
def bench_command(
    device: _DEVICE_OPTION = "wave-router",
    profile: _PROFILE_OPTION = "quick",
    artifacts_dir: _ARTIFACTS_OPTION = Path("results/edge/artifacts"),
    tables_dir: _TABLES_OPTION = Path("results/edge/tables"),
    raw_dir: _RAW_OPTION = Path("results/edge/raw"),
    binary: _BINARY_OPTION = Path("rust/target/x86_64-unknown-linux-musl/release/inmotion-infer"),
) -> None:
    """Write load.csv, bench.csv and cores.csv for one device."""
    from inmotion.edge import bench, config, transport

    chosen = config.profile(profile)
    with _open(transport, device) as node:
        path = bench.run(node, artifacts_dir, tables_dir, raw_dir, chosen, binary=binary)
    typer.secho(f"bench -> {path}", fg=typer.colors.GREEN)


@app.command("replay", help="Replay the test windows on a device and compare.")
def replay_command(
    device: _DEVICE_OPTION = "wave-router",
    profile: _PROFILE_OPTION = "quick",
    artifacts_dir: _ARTIFACTS_OPTION = Path("results/edge/artifacts"),
    windows_dir: _WINDOWS_OPTION = Path("results/edge/windows"),
    tables_dir: _TABLES_OPTION = Path("results/edge/tables"),
    raw_dir: _RAW_OPTION = Path("results/edge/raw"),
    binary: _BINARY_OPTION = Path("rust/target/x86_64-unknown-linux-musl/release/inmotion-infer"),
) -> None:
    """Write replay.csv after ``check --golden`` and a full stream."""
    from inmotion.edge import config, replay, transport

    chosen = config.profile(profile)
    with _open(transport, device) as node:
        path = replay.run(
            node, artifacts_dir, windows_dir, tables_dir, raw_dir, chosen, binary=binary
        )
    typer.secho(f"replay -> {path}", fg=typer.colors.GREEN)


@app.command("power", help="Estimate or measure energy per inference.")
def power_command(
    device: _DEVICE_OPTION = "wave-router",
    tables_dir: _TABLES_OPTION = Path("results/edge/tables"),
    meter_file: Annotated[
        Path | None, typer.Option("--meter-file", help="Torque meter reading file.")
    ] = None,
    meter_cmd: Annotated[
        str | None, typer.Option("--meter-cmd", help="Command that prints watts or joules.")
    ] = None,
) -> None:
    """Write power.csv from bench.csv."""
    from inmotion.edge import power, transport

    with _open(transport, device) as node:
        path = power.run(node, tables_dir, meter_file=meter_file, meter_cmd=meter_cmd)
    typer.secho(f"power -> {path}", fg=typer.colors.GREEN)


@app.command("stress", help="Run the ABBA interference protocol on a device.")
def stress_command(
    device: _DEVICE_OPTION = "wave-router",
    profile: _PROFILE_OPTION = "quick",
    artifacts_dir: _ARTIFACTS_OPTION = Path("results/edge/artifacts"),
    tables_dir: _TABLES_OPTION = Path("results/edge/tables"),
    raw_dir: _RAW_OPTION = Path("results/edge/raw"),
    binary: _BINARY_OPTION = Path("rust/target/x86_64-unknown-linux-musl/release/inmotion-infer"),
) -> None:
    """Write stress_raw.csv and stress_probes.csv."""
    from inmotion.edge import config, stress, transport

    chosen = config.profile(profile)
    with _open(transport, device) as node:
        path = stress.run(node, artifacts_dir, tables_dir, raw_dir, chosen, binary=binary)
    typer.secho(f"stress -> {path}", fg=typer.colors.GREEN)


@app.command("analyze", help="Compute the interference statistics.")
def analyze_command(
    tables_dir: _TABLES_OPTION = Path("results/edge/tables"),
    raw_dir: Annotated[
        Path | None,
        typer.Option("--raw-dir", help="Raw directory (for drop counters and inference windows)."),
    ] = None,
) -> None:
    """Write impact.csv and inference_impact.csv."""
    from inmotion.edge import analysis

    path = analysis.run(tables_dir, raw_dir)
    typer.secho(f"analyze -> {path}", fg=typer.colors.GREEN)


@app.command("figures", help="Build the eight figures from the tables.")
def figures_command(
    tables_dir: _TABLES_OPTION = Path("results/edge/tables"),
    out_dir: Annotated[Path, typer.Option("--out-dir", help="Figure output directory.")] = Path(
        "results/edge/figures"
    ),
    formats: Annotated[
        str, typer.Option("--formats", help="Comma-separated formats: pdf,png.")
    ] = "pdf,png",
    dpi: Annotated[int, typer.Option("--dpi", help="Raster resolution.")] = 300,
) -> None:
    """Write the figure stems in ``contracts.FIGURE_STEMS``."""
    from inmotion.edge import figures

    paths = figures.build(tables_dir, out_dir, formats=_split(formats) or ("pdf", "png"), dpi=dpi)
    typer.secho(f"figures -> {len(paths)} file(s) under {out_dir}", fg=typer.colors.GREEN)


@app.command("report", help="Summarise the tables into a markdown report.")
def report_command(
    tables_dir: _TABLES_OPTION = Path("results/edge/tables"),
    out: Annotated[Path, typer.Option("--out", help="Report path.")] = Path(
        "results/edge/report.md"
    ),
) -> None:
    """Write the run summary (the Lead owns the final REPORT.md)."""
    from inmotion.edge import campaign

    path = campaign.build_report(tables_dir, out)
    typer.secho(f"report -> {path}", fg=typer.colors.GREEN)


@app.command("campaign", help="Run the full campaign and write a resumable run directory.")
def campaign_command(
    profile: _PROFILE_OPTION = "quick",
    devices: Annotated[
        str,
        typer.Option(
            "--devices", help="Comma-separated device keys (wave-router,wave-router-2,local)."
        ),
    ] = "wave-router",
    run_dir: Annotated[
        Path | None, typer.Option("--run-dir", help="Explicit run directory.")
    ] = None,
    results_root: Annotated[
        Path, typer.Option("--results-root", help="Root for runs/artifacts/windows.")
    ] = Path("results/edge"),
    artifacts_dir: Annotated[
        Path | None, typer.Option("--artifacts-dir", help="Override the artifact directory.")
    ] = None,
    windows_dir: Annotated[
        Path | None, typer.Option("--windows-dir", help="Override the windows directory.")
    ] = None,
    phases: Annotated[
        str | None,
        typer.Option("--phases", help="Comma-separated phase subset (default: all)."),
    ] = None,
    meter_file: Annotated[
        Path | None, typer.Option("--meter-file", help="Meter reading file.")
    ] = None,
    meter_cmd: Annotated[str | None, typer.Option("--meter-cmd", help="Meter command.")] = None,
    resume: Annotated[bool, typer.Option("--resume", help="Continue an interrupted run.")] = False,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing run.")] = False,
) -> None:
    """Run preflight -> artifacts -> ... -> manifest in one directory."""
    import sys

    from inmotion.edge import campaign

    keys = _split(devices)
    if not keys:
        typer.secho("campaign: --devices is empty", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    try:
        built = campaign.run(
            profile,
            keys,
            run_dir=run_dir,
            resume=resume,
            force=force,
            results_root=results_root,
            artifacts_dir=artifacts_dir,
            windows_dir=windows_dir,
            meter_file=meter_file,
            meter_cmd=meter_cmd,
            phases=_split(phases) or None,
            log=lambda message: print(message, file=sys.stderr, flush=True),
        )
    except (RuntimeError, FileExistsError, FileNotFoundError) as exc:
        typer.secho(f"campaign: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"run dir: {built.run_dir}")
    typer.echo(f"report:  {built.run_dir / 'report.md'}")
    typer.echo(f"manifest: {built.manifest_path}")


class _open:
    """Open a transport and close it afterwards, for the single-device commands."""

    def __init__(self, transport_module: object, key: str) -> None:
        self._transport = transport_module
        self._key = key
        self._device = None

    def __enter__(self):
        self._device = self._transport.open_device(self._key)
        return self._device

    def __exit__(self, *_exc: object) -> None:
        close = getattr(self._device, "close", None)
        if callable(close):
            close()
