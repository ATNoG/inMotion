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

# A CARTO basemap key looks like `cb1_<...>`. It belongs in `.env` only: the
# generated map pages carry it (that is how a client-side key works), so a
# scenario committed with its page would put it in the repository.
CARTO_KEY = re.compile(r"cb1_[A-Za-z0-9_\-]{12,}")

# `NAME=<something>` for names that should only ever come from the environment.
SECRET_ASSIGNMENT = re.compile(
    r"""(?ix)
    ^\s*(?:export\s+)?
    (?:WANDB_API_KEY|CARTO_API_KEY|INMOTION_ROUTER_PASS|AWS_SECRET_ACCESS_KEY|GITHUB_TOKEN)
    \s*=\s*
    ['"]?(?!\s*$)(?!<)[^\s'"]{12,}['"]?\s*$
    """,
)

SKIP_SUFFIXES = {".lock", ".png", ".pdf", ".joblib", ".pt", ".npz"}
# Directories that never contain source: tool caches, checkouts of the artefact
# stores, and generated output. Scanning them wastes time and makes the test
# count depend on unrelated churn.
SKIP_PARTS = {
    ".git",
    ".venv",
    # `data/bus/raw` is the scrape cache and `data/bus/scenarios` the generated
    # passenger-flow output: tens of megabytes, regenerable, and the generated
    # map page is *supposed* to carry a basemap key in its tile URL. The rule
    # they would otherwise break is stated where it belongs — env.example and
    # AI_RESULT.md both say a scenario built with a key should not be committed.
    "raw",
    "scenarios",
    "node_modules",
    ".omp",
    ".ruff_cache",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    "20-aug",
    "backup",
    "plots",
    "results",
    "combined",
    "catboost_info",
    "dist",
    ".astro",
}


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
    # `.env` is where a secret is supposed to live (it is gitignored, and
    # env.example documents every name in it), so scanning it would forbid the
    # one place the project asks for one.
    if path.name in {".env", "env.example"} or path.name.startswith("test_no_secrets"):
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
        assert not CARTO_KEY.search(line), (
            f"{path.relative_to(REPO_ROOT)}:{lineno} looks like a live CARTO key; "
            "it belongs in .env, and the pages that embed it should not be committed"
        )
        assert not SECRET_ASSIGNMENT.match(line), (
            f"{path.relative_to(REPO_ROOT)}:{lineno} assigns a literal secret; "
            "read it from the environment instead"
        )
