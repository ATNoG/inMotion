"""The `before`-build script must never write into the working tree by accident.

`results/perf/make_before_kernels.py` rewrites its target in place. It used to
resolve the checkout's own `rust/inmotion-core/src/ops.rs` whenever a caller let it
guess; it now requires an explicit target and refuses an in-repo destination without
`--in-repo`. That file's reviewed revision is uncommitted — so one accidental run
converts the tree to the pre-optimisation kernels, and a later `git checkout` then
restores `HEAD`, destroying the repair while leaving the tree clean. That happened
once; these tests are the lock, not the warning.

The script is deliberately not imported: the contract under test is how it behaves
when invoked, from a given working directory, with given arguments.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from conftest import REPO_ROOT
import shutil

SCRIPT = REPO_ROOT / "results" / "perf" / "make_before_kernels.py"
OPS = REPO_ROOT / "rust" / "inmotion-core" / "src" / "ops.rs"


def run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bare_invocation_refuses_and_leaves_the_checkout_alone() -> None:
    """From the repository root, no arguments: refuse, write nothing."""
    before = digest(OPS)
    result = run(cwd=REPO_ROOT)
    assert result.returncode != 0, "a bare invocation must not succeed"
    assert "refusing to guess" in (result.stdout + result.stderr)
    assert digest(OPS) == before, "a bare invocation touched the checkout's ops.rs"


def test_named_in_place_target_inside_the_checkout_is_refused() -> None:
    """Naming the checkout's own ops.rs still must not write it without --in-repo."""
    before = digest(OPS)
    result = run(str(OPS), cwd=REPO_ROOT)
    assert result.returncode != 0, "an in-place write into the checkout must be refused"
    assert "refusing to write" in (result.stdout + result.stderr)
    assert digest(OPS) == before, "the refused run modified ops.rs anyway"


def test_out_writes_elsewhere_and_leaves_the_checkout_alone(tmp_path: Path) -> None:
    """The documented safe route still works: read the checkout, write to a scratch path."""
    before = digest(OPS)
    target = tmp_path / "ops-before.rs"
    result = run("--out", str(target), cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr
    assert digest(OPS) == before, "--out modified the file it read"
    assert target.is_file() and target.stat().st_size > 0
    assert digest(target) != before, "--out produced an identity instead of a revert"


def test_scratch_tree_reverts_in_place(tmp_path: Path) -> None:
    """A copy outside the checkout is the legitimate in-place target."""
    shutil.copytree(REPO_ROOT / "rust", tmp_path / "rust", ignore=shutil.ignore_patterns("target"))
    copy_ops = tmp_path / "rust" / "inmotion-core" / "src" / "ops.rs"
    before = digest(copy_ops)
    result = run(str(copy_ops), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert digest(copy_ops) != before, "the scratch copy was left as an identity"
