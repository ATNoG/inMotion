"""Shared pytest configuration for the inMotion test-suite.

The suite must be runnable from the repository root with `uv run pytest`.
It deliberately avoids importing heavy optional deps at collection time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# The mamba_ssm CUDA extension is hang-prone; force the pure-PyTorch fallback
# before any project import can pull it in.
os.environ.setdefault("MAMBA_SSM_AVAILABLE", "0")

REPO_ROOT = Path(__file__).resolve().parent.parent

# Root-level scripts (run_dl, run_exotic, mega_ensemble, ...) are imported by
# module name, so the repository root has to be importable.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

# Model checkpoints are large binaries kept out of git (see .gitignore). When a
# checkout has no artifact store, the model tests cannot verify anything, so they
# skip with an explicit reason rather than fail as if the models were broken.
ARTIFACTS_PRESENT = (
    (REPO_ROOT / "backup").is_dir()
    and (REPO_ROOT / "20-aug" / "models").is_dir()
)

requires_artifacts = __import__("pytest").mark.skipif(
    not ARTIFACTS_PRESENT,
    reason="model checkpoints not present (backup/ and 20-aug/models/ are gitignored)",
)
