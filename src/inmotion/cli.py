"""inmotion command-line interface.

One entry point for the whole project. `model` and `datasets` are implemented
natively; the training and evaluation pipelines are currently invoked as
subprocesses because their internals still parse ``sys.argv`` directly. That
delegation is temporary and tracked in ``docs/reports/MIGRATION.md``:

    inmotion run dl -- --seed 42          # legacy argparse passthrough

The important distinction is that `model` does not delegate: loading, verifying
and running a trained checkpoint is a first-class operation with no prior
knowledge of the training scripts required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from . import __version__
from .data.preprocessing import CONVENTION_WHOLE_CSV, FeaturePipeline
from .models import registry as reg
from .models.adopt import infer
from .models.checkpoint import save as save_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]

app = typer.Typer(
    name="inmotion",
    help="WiFi RSSI mobility classification: data, models, training and evaluation.",
    no_args_is_help=True,
    add_completion=False,
)
model_app = typer.Typer(help="Inspect, verify and run trained models.", no_args_is_help=True)
datasets_app = typer.Typer(help="Dataset versions and provenance.", no_args_is_help=True)
data_app = typer.Typer(help="Build and clean datasets.", no_args_is_help=True)
run_app = typer.Typer(help="Run a training or evaluation pipeline.", no_args_is_help=True)
app.add_typer(model_app, name="model")
app.add_typer(datasets_app, name="datasets")
app.add_typer(data_app, name="data")
app.add_typer(run_app, name="run")

# The CSV a model's scaling statistics are reconstructed from, by feature set.
# `ds-augmented-v3` is what the world models were trained on.
DEFAULT_REFERENCE = {
    True: "dataset_augmented3.csv",
    False: "dataset.csv",
}


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"inmotion {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True)
    ] = False,
) -> None:
    """inMotion — WiFi RSSI mobility classification."""


# ═══════════════════════════════════════════════════════════════════════════
# model
# ═══════════════════════════════════════════════════════════════════════════


@model_app.command("list")
def model_list(
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """List every registered model."""
    rows = []
    for name in reg.spec_names():
        spec = reg.by_name(name)
        rows.append(
            {
                "name": name,
                "factory": spec.factory,
                "params": spec.params,
                "rich": spec.rich,
                "present": spec.exists(),
                "path": str(spec.path),
                "note": spec.note,
            }
        )

    if as_json:
        typer.echo(json.dumps(rows, indent=2))
        return

    missing = [r for r in rows if not r["present"]]
    typer.echo(f"{'name':22s} {'factory':20s} {'params':>12s}  {'feat':>4s}  present")
    for r in rows:
        mark = "yes" if r["present"] else "NO"
        typer.echo(
            f"{r['name']:22s} {r['factory']:20s} {r['params']:>12,}  "
            f"{'rich' if r['rich'] else 'raw':>4s}  {mark}"
        )
    if missing:
        typer.echo(
            f"\n{len(missing)} checkpoint(s) not on disk. Model binaries are "
            "gitignored; see docs/reports/ARTIFACTS.md.",
            err=True,
        )


@model_app.command("info")
def model_info(name: Annotated[str, typer.Argument(help="Registered model name.")]) -> None:
    """Show the architecture and provenance of one model."""
    spec = reg.by_name(name)
    typer.echo(f"name      : {spec.name}")
    typer.echo(f"factory   : {spec.factory}")
    typer.echo(f"params    : {spec.params:,}")
    typer.echo(f"features  : {'18 (rich)' if spec.rich else '4 (raw)'}")
    typer.echo(f"prefix    : {spec.prefix or '-'}")
    typer.echo(f"path      : {spec.abs_path}")
    typer.echo(f"exists    : {spec.exists()}")
    typer.echo(f"arch      : {json.dumps(spec.arch, sort_keys=True)}")
    if spec.published_m is not None:
        actual_m = spec.params / 1e6
        flag = "matches" if abs(actual_m - spec.published_m) < 0.002 else "DIFFERS from"
        typer.echo(f"published : {spec.published_m}M ({flag} MASTER_RESULTS.md)")
    if spec.name in reg.PROVENANCE_GAPS:
        typer.echo("")
        typer.secho("provenance gap:", fg=typer.colors.YELLOW)
        typer.echo(f"  {reg.PROVENANCE_GAPS[spec.name]}")


@model_app.command("verify")
def model_verify(
    name: Annotated[Optional[str], typer.Argument(help="Verify one model.")] = None,
) -> None:
    """Strict-load every model and check it against its recorded parameter count."""
    names = [name] if name else reg.spec_names()
    failures: list[str] = []
    for model_name in names:
        spec = reg.by_name(model_name)
        if not spec.exists():
            failures.append(f"{model_name}: checkpoint missing at {spec.path}")
            typer.secho(f"  MISSING  {model_name}", fg=typer.colors.RED)
            continue
        try:
            loaded = reg.load(spec, strict=True)
            counts = sum(p.numel() for p in loaded.parameters())
            if spec.params and counts != spec.params:
                raise ValueError(f"{counts:,} params != recorded {spec.params:,}")
            typer.secho(f"  OK       {model_name}", fg=typer.colors.GREEN)
        except Exception as exc:  # noqa: BLE001 - reported per model
            failures.append(f"{model_name}: {type(exc).__name__}: {exc}")
            typer.secho(f"  FAILED   {model_name}: {exc}", fg=typer.colors.RED)

    if failures:
        typer.echo("")
        typer.secho(f"{len(failures)} model(s) failed verification.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.secho(f"\nAll {len(names)} model(s) verified.", fg=typer.colors.GREEN)


@model_app.command("predict")
def model_predict(
    name: Annotated[str, typer.Argument(help="Registered model name.")],
    input_csv: Annotated[Path, typer.Option("--input", "-i", help="CSV of RSSI rows.")],
    reference: Annotated[
        Optional[Path], typer.Option("--reference", help="CSV used to fit scaling statistics.")
    ] = None,
    output: Annotated[Optional[Path], typer.Option("--out", "-o")] = None,
) -> None:
    """Run a trained model over a CSV of RSSI windows.

    The model's preprocessing statistics were never persisted by the original
    training code, so scaling is reconstructed from ``--reference`` (the same CSV
    the model trained on) unless a saved pipeline exists next to the weights.
    """
    import numpy as np
    import pandas as pd
    import torch

    spec = reg.by_name(name)
    model = reg.load(spec, strict=True)

    from .data.paths import resolve

    ref_path = reference or resolve(DEFAULT_REFERENCE[spec.rich])
    if not ref_path.exists():
        typer.secho(
            f"reference CSV not found: {ref_path}\n"
            "Pass --reference pointing at the CSV this model was trained on; "
            "without it, predictions will not be reproducible.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    pipeline = FeaturePipeline.fit(
        ref_path, rich=spec.rich, seq_len=10, convention=CONVENTION_WHOLE_CSV
    )
    typer.secho(
        f"note: scaling statistics reconstructed from {ref_path.name} "
        "(they were not saved with the checkpoint).",
        fg=typer.colors.YELLOW,
    )

    df = pd.read_csv(input_csv)
    X = pipeline.transform(df)

    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        probs = torch.softmax(logits, dim=-1).numpy()

    classes = pipeline.classes or [str(c) for c in range(probs.shape[1])]
    labels = [classes[int(i)] for i in np.argmax(probs, axis=1)]
    result = pd.DataFrame({"predicted_class": labels})
    result["confidence"] = probs.max(axis=1).round(6)
    for idx, label in enumerate(classes):
        result[f"p_{label}"] = probs[:, idx].round(6)

    if output:
        result.to_csv(output, index=False)
        typer.secho(f"wrote {len(result)} predictions to {output}", fg=typer.colors.GREEN)
    else:
        typer.echo(result.to_string(index=False))


@model_app.command("adopt")
def model_adopt(
    checkpoint: Annotated[Path, typer.Argument(help="Path to a .pt checkpoint.")],
    name: Annotated[
        Optional[str], typer.Option("--name", help="Name for the adopted model.")
    ] = None,
    factory: Annotated[
        Optional[str], typer.Option("--factory", help="Pin the architecture when ambiguous.")
    ] = None,
    destination: Annotated[
        Optional[Path],
        typer.Option("--write", help="Write a sidecar-tagged copy to this directory."),
    ] = None,
) -> None:
    """Recover a checkpoint's architecture and write a sidecar.

    Use --factory to resolve structural ambiguity such as TS-JEPA versus CF-JEPA
    at the same width, which are indistinguishable from weights alone.
    """
    try:
        sidecar = infer(checkpoint, name=name, factory=factory)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho(f"adopted {checkpoint}", fg=typer.colors.GREEN)
    typer.echo(f"  factory : {sidecar.factory}")
    typer.echo(f"  params  : {sidecar.params:,}")
    typer.echo(f"  arch    : {json.dumps(sidecar.arch, sort_keys=True)}")
    typer.echo(f"  prefix  : {sidecar.prefix or '-'}")
    ambiguous = sidecar.provenance.get("ambiguous_with")
    if ambiguous:
        typer.secho(
            f"  ambiguous with: {', '.join(ambiguous)} — pass --factory to choose.",
            fg=typer.colors.YELLOW,
        )

    if destination:
        model = (
            reg.load(reg.by_name(sidecar.name), strict=True)
            if sidecar.name in reg.REGISTRY
            else None
        )
        if model is None:
            from .models.factories import build as build_model

            model = build_model(sidecar.factory, sidecar.arch)
            import torch

            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            from .models.wrappers import normalise_state

            model.load_state_dict(normalise_state(state, sidecar.prefix), strict=True)
        save_checkpoint(model, destination, sidecar)
        typer.secho(f"  wrote sidecar-tagged copy to {destination}", fg=typer.colors.GREEN)


@model_app.command("inventory")
def model_inventory(
    directory: Annotated[Path, typer.Argument(help="Directory to scan for .pt files.")],
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Adopt every checkpoint under a directory and report what is recoverable."""
    paths = sorted(p for p in Path(directory).rglob("*.pt"))
    if not paths:
        typer.secho(f"no .pt files under {directory}", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    rows: list[dict] = []
    for path in paths:
        try:
            sidecar = infer(path)
            rows.append(
                {
                    "checkpoint": str(path.relative_to(REPO_ROOT) if path.is_absolute() else path),
                    "factory": sidecar.factory,
                    "params": sidecar.params,
                    "status": "ok",
                    "ambiguous_with": sidecar.provenance.get("ambiguous_with", []),
                }
            )
        except Exception as exc:  # noqa: BLE001 - reported per checkpoint
            rows.append(
                {
                    "checkpoint": str(path),
                    "status": "unrecoverable",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if as_json:
        typer.echo(json.dumps(rows, indent=2))
        return

    ok = sum(r["status"] == "ok" for r in rows)
    for row in rows:
        if row["status"] == "ok":
            flag = (
                f"  (ambiguous: {', '.join(row['ambiguous_with'])})"
                if row["ambiguous_with"]
                else ""
            )
            typer.echo(
                f"  OK  {row['factory']:20s} {row['params']:>12,}  {row['checkpoint']}{flag}"
            )
        else:
            typer.secho(
                f"  ??  {row['checkpoint']}: {row.get('error', '')}", fg=typer.colors.YELLOW
            )
    typer.echo(f"\n{ok}/{len(rows)} checkpoint(s) recovered.")


# ═══════════════════════════════════════════════════════════════════════════
# data
# ═══════════════════════════════════════════════════════════════════════════


@data_app.command("export")
def data_export(
    input_file: Annotated[Path, typer.Option("--input", "-i", help="Wavecom capture file.")],
    output_file: Annotated[Path, typer.Option("--output", "-o", help="Destination CSV.")],
    group: Annotated[
        list[str], typer.Option("--group", help="label:mac1,mac2,... (repeatable).")
    ] = [],
    group_label: Annotated[
        list[str], typer.Option("--group-label", help="group_name:final_label (repeatable).")
    ] = [],
) -> None:
    """Turn a raw Wavecom capture into a per-route CSV."""
    from .data.export import export_to_csv

    if not input_file.exists():
        typer.secho(f"input not found: {input_file}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    export_to_csv(
        str(input_file),
        str(output_file),
        group=group or None,
        group_label=group_label or None,
    )
    typer.secho(f"wrote {output_file}", fg=typer.colors.GREEN)


@data_app.command("merge")
def data_merge(
    input_dir: Annotated[Path, typer.Option("--input", "-i", help="Directory of per-route CSVs.")],
    output_file: Annotated[Path, typer.Option("--output", "-o", help="Merged dataset CSV.")],
    pattern: Annotated[str, typer.Option("--pattern", help="Glob for input files.")] = "*.csv",
) -> None:
    """Merge per-route CSVs into one dataset."""
    from .data.merge import merge_csv_files

    if not input_dir.is_dir():
        typer.secho(f"input directory not found: {input_dir}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    merge_csv_files(str(input_dir), str(output_file), pattern=pattern)
    typer.secho(f"wrote {output_file}", fg=typer.colors.GREEN)


def _delegate(module: str, args: list[str]) -> None:
    """Run a pipeline module in process, preserving its argparse interface."""
    import importlib

    from .pipelines.registry import _argv

    full = f"inmotion.{module}"
    entry = getattr(importlib.import_module(full), "main", None)
    if entry is None:
        typer.secho(f"{full} has no main()", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    with _argv([full, *args]):
        result = entry()
    raise typer.Exit(code=0 if result is None else int(result))


@data_app.command(
    "clean", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def data_clean(ctx: typer.Context) -> None:
    """Audit and clean a dataset (delegates to clean_dataset.py)."""
    _delegate("data.clean", list(ctx.args))


@data_app.command(
    "augment", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def data_augment(ctx: typer.Context) -> None:
    """Augment a dataset (delegates to augment_data.py)."""
    _delegate("data.augment", list(ctx.args))


# ═══════════════════════════════════════════════════════════════════════════
# run — training / evaluation pipelines
# ═══════════════════════════════════════════════════════════════════════════


@run_app.command("list")
def run_list() -> None:
    """List the available pipelines."""
    from .pipelines.registry import PIPELINES

    width = max(len(p.name) for p in PIPELINES)
    for pipeline in PIPELINES:
        typer.echo(f"  {pipeline.name:<{width}}  {pipeline.summary}")
    typer.echo("\nRun one with: inmotion run pipeline <name> -- <args>")


@run_app.command(
    "pipeline",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def run_pipeline(
    ctx: typer.Context,
    pipeline: Annotated[str, typer.Argument(help="Pipeline name; see `inmotion run list`.")],
) -> None:
    """Run one pipeline, passing remaining arguments straight through.

    Example:  inmotion run pipeline exotic -- --model ts_jepa --seed 42
    """
    from .pipelines.registry import run

    try:
        code = run(pipeline, list(ctx.args))
    except KeyError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    if code:
        raise typer.Exit(code=code)


@app.command("doctor")
def doctor() -> None:
    """Report on the environment and the artifact store."""
    import platform

    typer.echo(f"inmotion  : {__version__}")
    typer.echo(f"python    : {platform.python_version()}")
    typer.echo(f"repo root : {REPO_ROOT}")

    try:
        import torch

        typer.echo(f"torch     : {torch.__version__}  cuda={torch.cuda.is_available()}")
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"torch     : unavailable ({exc})", fg=typer.colors.YELLOW)

    present = [n for n in reg.spec_names() if reg.by_name(n).exists()]
    absent = [n for n in reg.spec_names() if not reg.by_name(n).exists()]
    typer.echo(f"models    : {len(present)}/{len(reg.spec_names())} checkpoints on disk")
    if absent:
        typer.secho(f"  missing : {', '.join(absent)}", fg=typer.colors.YELLOW)
        typer.echo("  model binaries are gitignored; see docs/reports/ARTIFACTS.md")

    from .datasets import load_manifest

    manifest = load_manifest()
    if manifest:
        typer.echo(f"datasets  : {len(manifest['datasets'])} version(s) recorded")
    else:
        typer.secho("datasets  : no manifest found", fg=typer.colors.YELLOW)


# ═══════════════════════════════════════════════════════════════════════════
# datasets
# ═══════════════════════════════════════════════════════════════════════════


@datasets_app.command("list")
def datasets_list(
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List recorded dataset versions, newest lineage last."""
    from .datasets import load_manifest

    manifest = load_manifest()
    if manifest is None:
        typer.secho(
            "no manifest; run `inmotion datasets manifest --write`",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    if as_json:
        typer.echo(json.dumps(manifest, indent=2))
        return

    canonical = manifest["canonical"]
    typer.echo(f"{'id':18s} {'file':26s} {'rows':>7s} {'dev':>4s} {'status':11s} {'labels':16s}")
    for entry in manifest["datasets"]:
        mark = "*" if entry["id"] == canonical else " "
        typer.echo(
            f"{mark}{entry['id']:17s} {entry['file']:26s} {entry['rows']:>7,} "
            f"{entry['devices']:>4} {entry['status']:11s} {entry['label_audit']:16s}"
        )
    typer.echo(f"\n* canonical training set: {canonical}")
    typer.echo("  regenerate measured fields: inmotion datasets manifest --write")


@datasets_app.command("show")
def datasets_show(
    identifier: Annotated[str, typer.Argument(help="Dataset id or file name.")],
) -> None:
    """Show lineage and label provenance for one dataset version."""
    from .datasets import resolve

    version = resolve(identifier)
    typer.echo(f"id           : {version.id}")
    typer.echo(f"file         : {version.file}")
    typer.echo(f"rows         : {version.rows:,}")
    typer.echo(f"columns      : {', '.join(version.columns)}")
    typer.echo(f"classes       : {', '.join(version.classes)}")
    typer.echo(f"class counts : {json.dumps(version.class_counts)}")
    typer.echo(f"devices      : {version.devices}")
    typer.echo(f"synthetic col: {version.synthetic}")
    typer.echo(f"size         : {version.size_bytes:,} bytes")
    typer.echo(f"sha256       : {version.sha256}")
    typer.echo(f"first commit : {version.first_commit} ({version.first_commit_date})")
    typer.echo(f"role         : {version.role}")
    typer.echo(f"status       : {version.status}")
    typer.echo(f"lineage      : {json.dumps(version.lineage)}")

    typer.echo(f"label audit  : {version.label_audit}")
    if version.label_audit == "known_incorrect":
        typer.secho(f"  {version.validity_note}", fg=typer.colors.RED)
    elif version.validity_note:
        typer.echo(f"  {version.validity_note}")


@datasets_app.command("verify")
def datasets_verify() -> None:
    """Re-check every dataset against its recorded checksum and row count."""
    from .datasets import verify as verify_datasets

    problems = verify_datasets()
    if not problems:
        from .datasets import load_manifest

        count = len(load_manifest()["datasets"])
        typer.secho(f"All {count} dataset version(s) match the manifest.", fg=typer.colors.GREEN)
        return

    for identifier, problem in problems:
        typer.secho(f"  {identifier}: {problem}", fg=typer.colors.RED)
    typer.secho(f"\n{len(problems)} problem(s).", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


@datasets_app.command("manifest")
def datasets_manifest(
    write: Annotated[bool, typer.Option("--write", help="Rewrite data/DATASETS.json.")] = False,
) -> None:
    """Regenerate the manifest's measured fields, or print it."""
    from .datasets import build_manifest, write_manifest

    if write:
        path = write_manifest()
        typer.secho(f"wrote {path}", fg=typer.colors.GREEN)
        return
    typer.echo(json.dumps(build_manifest(), indent=2))


@datasets_app.command("resolve")
def datasets_resolve(
    identifier: Annotated[str, typer.Argument(help="Dataset id or file name.")],
) -> None:
    """Print the path of a dataset version, warning when it is not trustworthy."""
    from .datasets import canonical, resolve

    version = resolve(identifier)
    if version.label_audit == "known_incorrect":
        typer.secho(
            f"warning: {version.id} is marked as having incorrect class labels.",
            fg=typer.colors.RED,
            err=True,
        )
    if version.id != canonical().id and version.status != "current":
        typer.secho(
            f"note: {version.id} is '{version.status}'; "
            f"the canonical training set is {canonical().id}.",
            fg=typer.colors.YELLOW,
            err=True,
        )
    typer.echo(str(version.path))
