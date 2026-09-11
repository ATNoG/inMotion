"""`python -m inmotion.pipelines.dl` runs the DL pipeline's CLI.

Delegates to the typer command so `--help` and shell completion behave exactly
as they do through `inmotion train dl`.
"""

from __future__ import annotations

import sys

from typer.main import get_command

from inmotion.cli.train import app

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "dl", *sys.argv[1:]]
    get_command(app)()
