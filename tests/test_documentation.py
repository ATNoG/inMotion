"""Documentation must agree with the code.

The reference documents are generated from the source, so the only way they can
be wrong is by going stale. This test regenerates them in memory and compares, so
a change to the registry, the CLI or the dataset manifest that is not reflected
in the docs fails here rather than quietly misleading a reader.

It also checks the hand-written documents for the specific claims most likely to
rot: links to files that no longer exist, and references to the pre-move script
names.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import REPO_ROOT

DOCS = REPO_ROOT / "docs"
GENERATOR = REPO_ROOT / "scripts" / "docs" / "generate_reference.py"


def test_reference_docs_are_not_stale() -> None:
    """`generate_reference.py --check` must report no drift."""
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "reference docs are stale; run "
        "`uv run python scripts/docs/generate_reference.py`\n"
        f"{result.stdout}\n{result.stderr}"
    )


def _markdown_files() -> list[Path]:
    return sorted(DOCS.rglob("*.md"))


# Paths the documentation names *because they are missing*. The docs describe
# these as absent, so requiring them to exist would contradict the documents.
DOCUMENTED_MISSING = {
    # Named because it is absent; the paper table cites a stale path.
    "models/dl/augmented/DeepStackEnsemble_seed42.pt",
    # Output directories the CLI creates on first run.
    "models/exotic",
    "models/exotic/viz",
}


# A backticked path with a slash or a known extension, e.g. `docs/foo.md`,
# `src/inmotion/cli.py`, `backup/x.pt`.
_PATH_IN_BACKTICKS = re.compile(
    r"`((?:\.github|docs|src|tests|scripts|data|demo|dl|ml_classification|backup|models|20-aug)"
    r"/[A-Za-z0-9_./{}<>-]+)`"
)


@pytest.mark.parametrize("path", _markdown_files(), ids=lambda p: p.name)
def test_doc_references_exist(path: Path) -> None:
    """Backticked repository paths in the docs should point at real files.

    Paths containing glob/placeholder characters are skipped: they describe
    patterns rather than a single file.
    """
    text = path.read_text(errors="ignore")
    missing: list[str] = []
    for match in _PATH_IN_BACKTICKS.finditer(text):
        ref = match.group(1)
        if any(ch in ref for ch in "{}<>*"):
            continue
        if "..." in ref:  # abbreviated in the document, e.g. `a/.../b.pt`
            continue
        if ref.endswith("/"):
            continue
        if ref in DOCUMENTED_MISSING:
            continue
        if ref.startswith("backup/") and ref.endswith(".pt"):
            continue  # checkpoints are gitignored; docs cite them as examples
        if not (REPO_ROOT / ref).exists():
            missing.append(ref)
    assert not missing, (
        f"{path.relative_to(REPO_ROOT)} references missing paths: {sorted(set(missing))}"
    )


# The modules were moved into src/inmotion; docs must not instruct the reader to
# run the old root-level scripts.
# Command forms that were superseded. Current docs must not use them;
# `docs/reports/` records history and is exempt.
_STALE_INVOCATIONS = (
    "inmotion run pipeline",
    "python run_dl.py",
    "python run_exotic.py",
    "python run_ensemble.py",
    "python mega_ensemble.py",
    "python run_classification.py",
    "python train_hpo_paper.py",
    "python evaluate_checkpoints.py",
    "python compute_shap.py",
    "python feature_importance_dl.py",
    "python analyze_seeds.py",
    "python analyze_interference.py",
    "python regenerate_plots.py",
    "python regenerate_dl_plots.py",
    "python export.py",
    "python merge_csv.py",
    "python clean_dataset.py",
    "python augment_data.py",
)


@pytest.mark.parametrize(
    "path",
    [p for p in _markdown_files() if "reports" not in p.parts],
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_current_docs_do_not_use_pre_move_commands(path: Path) -> None:
    """Current documentation must show the CLI, not the pre-move scripts.

    `docs/reports/` is excluded: it records history, so older command forms there
    are intentional.
    """
    text = path.read_text(errors="ignore")
    found = [cmd for cmd in _STALE_INVOCATIONS if cmd in text]
    assert not found, (
        f"{path.relative_to(REPO_ROOT)} still shows pre-move commands {found}; "
        "use the `inmotion train|evaluate|analyze|plots` command instead"
    )


def test_readme_and_docs_index_exist() -> None:
    assert (REPO_ROOT / "README.md").is_file()
    assert (DOCS / "README.md").is_file()
    assert (DOCS / "architecture.md").is_file()
    assert (DOCS / "reproducibility.md").is_file()
    assert (DOCS / "audit" / "AUDIT_CHECKLIST.md").is_file()
    assert (DOCS / "audit" / "MODEL_CARDS.md").is_file()
