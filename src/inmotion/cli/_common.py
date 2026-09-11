"""Shared helpers for the CLI command modules."""

from __future__ import annotations

from pathlib import Path

import typer


def run_pipeline(run, config) -> None:
    """Call a pipeline's ``run(config)`` and translate its exit code.

    Pipelines return an int (0 for success). Anything non-zero is surfaced as a
    typer exit so a shell pipeline and CI can detect failure.
    """
    code = run(config)
    if code:
        raise typer.Exit(code=int(code))


def existing_file(path: Path, label: str = "file") -> Path:
    """Fail early, with a clear message, when an input path does not exist."""
    if not path.exists():
        typer.secho(f"{label} not found: {path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    return path


def parse_csv_list(value: str | None) -> list[str]:
    """Split a comma-separated option into a list, dropping empties."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]
