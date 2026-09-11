"""`inmotion model` - inspect, verify, predict and adopt checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from inmotion.data.paths import REPO_ROOT, resolve
from inmotion.data.preprocessing import CONVENTION_WHOLE_CSV, FeaturePipeline
from inmotion.models import registry as reg
from inmotion.models.adopt import infer
from inmotion.models.checkpoint import save as save_checkpoint

# The CSV a model's scaling statistics are reconstructed from, by feature
# set. ds-augmented-v3 is the set the world models were trained on.
DEFAULT_REFERENCE = {True: "dataset_augmented3.csv", False: "dataset.csv"}

app = typer.Typer(no_args_is_help=True, help="Inspect, verify and run trained models.")
NAME = "model"
HELP = "Inspect, verify and run trained models."


@app.command("list")
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


@app.command("info")
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
    if spec.note:
        typer.echo(f"note      : {spec.note}")
    if spec.published_m is not None:
        actual_m = spec.params / 1e6
        flag = "matches" if abs(actual_m - spec.published_m) < 0.002 else "DIFFERS from"
        typer.echo(f"published : {spec.published_m}M ({flag} MASTER_RESULTS.md)")
    if spec.name in reg.PROVENANCE_GAPS:
        typer.echo("")
        typer.secho("provenance gap:", fg=typer.colors.YELLOW)
        typer.echo(f"  {reg.PROVENANCE_GAPS[spec.name]}")


@app.command("verify")
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


@app.command("predict")
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


@app.command("adopt")
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


@app.command("inventory")
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
