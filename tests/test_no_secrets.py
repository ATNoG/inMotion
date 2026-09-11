"""Guard against credentials being committed again.

A live Weights & Biases key was previously hardcoded in ten job scripts. Even
after removing it the value remains in git history, so the only useful defence is
to stop a new one being added: this scans the tracked files and fails on the
shape of a W&B key or a populated assignment to an obvious secret name.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Real W&B keys look like `wandb_v1_<base64ish>`; the placeholder never matches.
WANDB_KEY = re.compile(r"wandb_v1_[A-Za-z0-9_\-]{20,}")

# `NAME=<something>` for names that should only ever come from the environment.
SECRET_ASSIGNMENT = re.compile(
    r"""(?ix)
    ^\s*(?:export\s+)?
    (?:WANDB_API_KEY|INMOTION_ROUTER_PASS|AWS_SECRET_ACCESS_KEY|GITHUB_TOKEN)
    \s*=\s*
    ['"]?(?!\s*$)(?!<)[^\s'"]{12,}['"]?\s*$
    """,
)

SKIP_SUFFIXES = {".lock", ".png", ".pdf", ".joblib", ".pt", ".npz"}
SKIP_PARTS = {".git", ".venv", "node_modules", ".omp", "20-aug", "backup", "plots", "results"}


def _scanned_files() -> list[Path]:
    """Every text file in the working tree, not just the committed ones.

    Scanning the working tree matters: a secret added to a not-yet-committed
    file would slip past a `git ls-files`-based check, which is exactly when a
    guard is most useful.
    """
    return [p for p in REPO_ROOT.rglob("*") if _scannable(p)]


def _scannable(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix in SKIP_SUFFIXES or set(path.parts) & SKIP_PARTS:
        return False
    if path.name == "env.example" or path.name.startswith("test_no_secrets"):
        return False
    return "__pycache__" not in path.parts


@pytest.mark.parametrize("path", _scanned_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_credentials_in_tracked_files(path: Path) -> None:
    try:
        text = path.read_text(errors="ignore")
    except (OSError, UnicodeDecodeError):
        pytest.skip("unreadable as text")

    for lineno, line in enumerate(text.splitlines(), start=1):
        assert not WANDB_KEY.search(line), (
            f"{path.relative_to(REPO_ROOT)}:{lineno} looks like a live W&B key"
        )
        assert not SECRET_ASSIGNMENT.match(line), (
            f"{path.relative_to(REPO_ROOT)}:{lineno} assigns a literal secret; "
            "read it from the environment instead"
        )
