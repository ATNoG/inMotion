"""`python -m inmotion.pipelines.exotic` runs the world-model pipeline's CLI."""

from __future__ import annotations

import sys

from typer.main import get_command

from inmotion.cli.train import app

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "exotic", *sys.argv[1:]]
    get_command(app)()
