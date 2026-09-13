#!/usr/bin/env python3
"""Rewrite the four optimised kernels in ``ops.rs`` back to their pre-optimisation bodies.

    python3 results/perf/make_before_kernels.py --out FILE [ops.rs] [--check] [--in-repo]
    python3 results/perf/make_before_kernels.py <ops.rs> [--out FILE] [--check] [--in-repo]

The write destination is never implicit. ``--out FILE`` writes the reverted source
to a different path and leaves whatever it read alone; without ``--out`` the
positional ``ops.rs`` is both the source and the destination, so it is required.
Nothing about the current directory can name the target for you: with neither
``--out`` nor a positional target the script exits non-zero rather than guessing,
because the guess used to be the checkout's own
``rust/inmotion-core/src/ops.rs`` and writing that in place destroys the working
tree. Dropping one argument from a documented command is therefore a refusal, not
an in-place write. Point it at a scratch copy made with
``cp -r rust /tmp/before-build/rust``:

    python3 results/perf/make_before_kernels.py \
        /tmp/before-build/rust/inmotion-core/src/ops.rs

``--out`` is the explicit route that reads the checkout's own ``ops.rs`` and writes
somewhere else, which is the way to produce the ``before`` source without touching
the tree:

    python3 results/perf/make_before_kernels.py --out /tmp/ops-before.rs

Writing inside the repository is refused even when the path is named, unless
``--in-repo`` says you really mean it; the refusal names the file it would have
overwritten.

``--check`` only reports whether the target still carries the optimised bodies
(exit 0 when it is already pre-optimisation, 1 when it is not) and writes nothing.

Here is the command pattern to avoid when regenerating:

    # WRONG: both sides become the current optimised source, so nothing changes
    python3 results/perf/make_before_kernels.py --baseline /tmp/base.rs /tmp/base.rs

The pre-optimisation bodies come from ``ops_before_optimisation.rs`` next to this
script, which is tracked, so building the ``before`` binary needs nothing outside
the repository. If a target already carries those bodies the replacements are
identities and the run says so rather than emitting a different file:

* ``impl PackedWeight`` .. ``}``     (the original (out, in) layout)
* ``pub fn linear_packed`` .. ``}``  (the original serial dot-product loop)
* ``pub fn self_attention`` .. ``}`` (the original triple loop)
* ``pub fn conv1d_same`` .. ``}``    (the original quadruple loop)

Every item is located by the line that opens it, so the replacement survives the
line shifts that other edits introduce, and the printed old and new line counts
are the evidence that a body really changed.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RELATIVE_TARGET = Path("rust") / "inmotion-core" / "src" / "ops.rs"


def checkout_root() -> Path | None:
    """The git checkout this script belongs to, or None when it is not in one.

    The guard protects *this* checkout, identified by the ``.git`` entry above
    ``results/perf/``. A self-contained copy of the whole repository is therefore
    free to revert its own tree, while the original stays protected even when the
    copy is invoked from it.
    """
    for base in (REPO, *REPO.parents):
        if (base / ".git").exists():
            return base
    return None


def redirected_script() -> Path | None:
    """The repository copy of this script when a relative invocation reached us.

    ``cd some-tree && python3 ../results/perf/make_before_kernels.py`` from outside
    a checkout resolves to the checkout this script lives in; this walks up from
    the caller's directory to that same copy and defers to it, so the two copies
    cannot disagree. A relative path that reaches an *in-repo copy* does so from
    ``<repo>/results/perf/`` as well, so it resolves to that copy directly and no
    redirect happens — there is no path that fails to converge.
    """
    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        candidate = (base / "results" / "perf" / "make_before_kernels.py").resolve()
        if candidate == Path(__file__).resolve():
            return None  # the invocation already reached this copy
        if candidate.is_file():
            return candidate
    return None

OPEN = {
    "impl PackedWeight": ("impl PackedWeight {", "}"),
    "pub fn linear_packed": ("pub fn linear_packed(", "}"),
    "pub fn self_attention": ("pub fn self_attention(", "}"),
    "pub fn conv1d_same": ("pub fn conv1d_same(", "}"),
}

USAGE = (
    "usage: make_before_kernels.py --out FILE [ops.rs] [--check] [--in-repo]\n"
    "       make_before_kernels.py <ops.rs> [--out FILE] [--check] [--in-repo]\n"
    "       the write destination is never implicit: --out writes elsewhere, a\n"
    "       positional ops.rs is written in place, and neither means no guess and\n"
    "       no write — a bare invocation exits non-zero"
)


def blocks(lines: list[str], open_text: str, close_text: str) -> list[tuple[int, int]]:
    """Return (start, end) line indices, end inclusive, for items opening with `open_text`."""
    out = []
    for i, line in enumerate(lines):
        if not line.rstrip("\n").startswith(open_text):
            continue
        for j in range(i, len(lines)):
            if lines[j].rstrip("\n") == close_text:
                out.append((i, j))
                break
        else:
            raise SystemExit(f"unterminated {open_text!r} at line {i + 1}")
    return out


def main() -> int:
    other = redirected_script()
    if other is not None:
        # A relative invocation reached a copy of this script; defer to the copy
        # the caller's directory actually points at.
        import subprocess

        return subprocess.call([sys.executable, str(other), *sys.argv[1:]])

    argv = sys.argv[1:]
    baseline = HERE / "ops_before_optimisation.rs"
    out: Path | None = None
    target: Path | None = None
    check = False
    in_repo_ok = False

    while argv:
        arg = argv.pop(0)
        if arg in ("-h", "--help"):
            print(USAGE)
            print(__doc__)
            return 0
        if arg == "--baseline":
            if not argv:
                raise SystemExit("--baseline needs a file")
            baseline = Path(argv.pop(0))
        elif arg == "--out":
            if not argv:
                raise SystemExit("--out needs a file")
            out = Path(argv.pop(0))
        elif arg == "--check":
            check = True
        elif arg == "--in-repo":
            in_repo_ok = True
        elif arg.startswith("-"):
            raise SystemExit(f"unknown option {arg}\n\n{USAGE}")
        elif target is None:
            target = Path(arg)
        else:
            raise SystemExit(f"unexpected extra argument {arg}\n\n{USAGE}")

    # There is no implicit target. The old default was the checkout's own
    # rust/inmotion-core/src/ops.rs whenever the caller stood in a repository, so
    # a bare invocation rewrote the working tree — the incident this guard exists
    # for. Reading that file is fine as long as the write goes elsewhere, which is
    # what --out says.
    if target is None and out is None:
        raise SystemExit(
            "no target given; refusing to guess one.\n"
            f"  --out FILE   read {RELATIVE_TARGET} here and write the reverted source to FILE\n"
            "  <ops.rs>     revert that file in place (a scratch copy, not the checkout)\n"
            "\n"
            "A bare invocation used to default to this checkout's own "
            f"{RELATIVE_TARGET}, which is unrecoverable in place.\n\n{USAGE}"
        )
    if target is None:
        caller_target = Path.cwd() / RELATIVE_TARGET
        if not caller_target.is_file():
            raise SystemExit(
                f"no ops.rs to revert: {caller_target} does not exist.\n"
                "With --out and no positional target the source is ./"
                f"{RELATIVE_TARGET} under the caller's directory; pass the ops.rs "
                "you mean as an argument, or run this from the repository root.\n\n"
                f"{USAGE}"
            )
        target = caller_target

    if not target.is_file():
        raise SystemExit(
            f"no ops.rs to revert: {target} does not exist.\n"
            "Pass the ops.rs you mean as an argument.\n\n"
            f"{USAGE}"
        )
    if not baseline.is_file():
        raise SystemExit(f"no baseline: {baseline} does not exist (it is tracked next to this script)")

    destination = out or target
    # The dangerous shape is an in-place write into the checkout's own source,
    # which is the incident this guard exists for. The check is on the DESTINATION,
    # not on which flag named it: `--out rust/inmotion-core/src/ops.rs` is still the
    # checkout's own kernels, and exempting --out from the refusal was the hole.
    #
    # This only sees a checkout that carries `.git`: a copy without it is not a
    # checkout but a scratch tree, which is what a clean-host `before` build uses,
    # so such a copy may still revert its own tree.
    root = checkout_root()
    writes_into_repo = root is not None and destination.resolve().is_relative_to(root.resolve())
    if writes_into_repo and not in_repo_ok:
        raise SystemExit(
            f"refusing to write {destination.resolve()} in place: it is inside the "
            f"tree this run would edit ({root if root is not None else Path.cwd()}), "
            "and reverting an ops.rs where it stands is what destroys a checkout.\n"
            "Write to a scratch path with --out, point a positional target at a "
            "scratch copy, or pass --in-repo if you really mean to overwrite the "
            "file named above.\n\n"
            f"{USAGE}"
        )

    old = baseline.read_text().splitlines(keepends=True)
    new = target.read_text().splitlines(keepends=True)
    changed = 0

    for key, (open_text, close_text) in OPEN.items():
        src = blocks(old, open_text, close_text)
        dst = blocks(new, open_text, close_text)
        if len(src) != 1 or len(dst) != 1:
            raise SystemExit(f"{key}: expected one item on each side, got {len(src)} and {len(dst)}")
        (s0, s1), (d0, d1) = src[0], dst[0]
        before = d1 - d0 + 1
        after = s1 - s0 + 1
        if new[d0 : d1 + 1] != old[s0 : s1 + 1]:
            changed += 1
        new[d0 : d1 + 1] = old[s0 : s1 + 1]
        note = "  (identity: already reverted)" if before == after else ""
        print(f"reverted {key}: {before} -> {after} lines{note}")

    if changed == 0:
        print("nothing changed: the target already carries the pre-optimisation bodies")

    if check:
        # Report only, so a caller can tell whether `before` is already in place
        # without risking the tree: exit 1 means the optimised bodies are still there.
        print("check only: nothing written")
        return 1 if changed else 0

    destination.write_text("".join(new))
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
