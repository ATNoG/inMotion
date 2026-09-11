"""`inmotion datasets` - dataset versions, provenance and checksums."""

from __future__ import annotations

import json
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, help="Dataset versions and provenance.")
NAME = "datasets"
HELP = "Dataset versions and provenance."


@app.command("list")
def datasets_list(
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List recorded dataset versions, newest lineage last."""
    from inmotion.datasets import load_manifest

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


@app.command("show")
def datasets_show(
    identifier: Annotated[str, typer.Argument(help="Dataset id or file name.")],
) -> None:
    """Show lineage and label provenance for one dataset version."""
    from inmotion.datasets import resolve

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


@app.command("verify")
def datasets_verify() -> None:
    """Re-check every dataset against its recorded checksum and row count."""
    from inmotion.datasets import verify as verify_datasets

    problems = verify_datasets()
    if not problems:
        from inmotion.datasets import load_manifest

        count = len(load_manifest()["datasets"])
        typer.secho(f"All {count} dataset version(s) match the manifest.", fg=typer.colors.GREEN)
        return

    for identifier, problem in problems:
        typer.secho(f"  {identifier}: {problem}", fg=typer.colors.RED)
    typer.secho(f"\n{len(problems)} problem(s).", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


@app.command("manifest")
def datasets_manifest(
    write: Annotated[bool, typer.Option("--write", help="Rewrite data/DATASETS.json.")] = False,
) -> None:
    """Regenerate the manifest's measured fields, or print it."""
    from inmotion.datasets import build_manifest, write_manifest

    if write:
        path = write_manifest()
        typer.secho(f"wrote {path}", fg=typer.colors.GREEN)
        return
    typer.echo(json.dumps(build_manifest(), indent=2))


@app.command("resolve")
def datasets_resolve(
    identifier: Annotated[str, typer.Argument(help="Dataset id or file name.")],
) -> None:
    """Print the path of a dataset version, warning when it is not trustworthy."""
    from inmotion.datasets import canonical, resolve

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
