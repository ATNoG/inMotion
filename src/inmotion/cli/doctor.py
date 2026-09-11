"""`inmotion doctor` - report on the environment and the artifact store."""

from __future__ import annotations

import platform

import typer

from inmotion import __version__
from inmotion.data.paths import REPO_ROOT
from inmotion.models import registry as reg


def doctor() -> None:
    """Report on the environment and the artifact store."""

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

    from inmotion.datasets import load_manifest

    manifest = load_manifest()
    if manifest:
        typer.echo(f"datasets  : {len(manifest['datasets'])} version(s) recorded")
    else:
        typer.secho("datasets  : no manifest found", fg=typer.colors.YELLOW)
