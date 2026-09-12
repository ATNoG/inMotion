#!/usr/bin/env bash
#
# Every quality gate, runnable locally with no GitHub Actions minutes.
#
# This replaces the CI workflow on purpose: the same checks, on demand, at no
# cost. Run the fast half constantly; `--full` adds the slow sweeps.
#
#   scripts/check.sh          lint, format, docs, unit tests, rust checks
#   scripts/check.sh --full   also the demo suite and the cross-language parity
#
set -euo pipefail

cd "$(dirname "$0")/.."

# Importing mamba_ssm can hang for minutes while it builds CUDA kernels.
export MAMBA_SSM_AVAILABLE=0

FULL=0
[ "${1:-}" = "--full" ] && FULL=1

fail=0
step() {
    local name="$1"; shift
    printf '\n=== %s ===\n' "$name"
    if "$@"; then
        printf '  ok\n'
    else
        printf '  FAILED\n'
        fail=1
    fi
}

printf 'inMotion checks%s\n' "$([ $FULL -eq 1 ] && echo ' (full)')"

step "python: lint"        uv run ruff check src tests scripts
step "python: formatting"  uv run ruff format --check src tests scripts
step "python: reference docs are current" \
     uv run python scripts/docs/generate_reference.py --check
step "python: credentials guard" uv run pytest tests/test_no_secrets.py -q
step "python: tests"       uv run pytest tests/ -q

if command -v cargo >/dev/null 2>&1; then
    # `cargo fmt` has no --manifest-path, so the Rust steps run from rust/.
    step "rust: formatting"  bash -c 'cd rust && cargo fmt --check'
    step "rust: lints"       bash -c 'cd rust && cargo clippy --all-targets -- -D warnings'
    step "rust: unit tests"  bash -c 'cd rust && cargo test'
else
    printf '\n=== rust ===\n  cargo not on PATH, skipped\n'
fi

if [ $FULL -eq 1 ]; then
    step "demo: tests"     bash -c 'cd demo && uv run --project . pytest -q'
    step "rust: parity with PyTorch" \
         uv run pytest tests/test_quant_export.py -k rust_port -q
fi

printf '\n'
if [ $fail -eq 0 ]; then
    printf 'all checks passed\n'
else
    printf 'SOME CHECKS FAILED\n'
fi
exit $fail
