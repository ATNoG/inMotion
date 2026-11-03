#!/usr/bin/env bash
# One-command entry point for the edge campaign.
#
#   scripts/edge/run_campaign.sh --profile quick --devices wave-router
#   scripts/edge/run_campaign.sh --profile full  --devices wave-router,wave-router-2
#
# Checks the environment before touching a device, pins SOURCE_DATE_EPOCH so a
# rerun is byte-identical, runs the CLI, then prints the run directory and the
# report path. The password is only ever read from $WAVECOM_PASSWORD.
set -euo pipefail

PROFILE="quick"
DEVICES="wave-router"
RUN_DIR=""
EXTRA=()

usage() {
    cat <<'EOF'
usage: scripts/edge/run_campaign.sh [options]

  --profile NAME       smoke | quick | full          (default: quick)
  --devices LIST       comma-separated device keys   (default: wave-router)
                       keys: wave-router, wave-router-2, local
  --run-dir PATH       explicit run directory         (default: results/edge/runs/<UTC>)
  --artifacts-dir PATH override the artifact directory
  --windows-dir PATH   override the windows directory
  --phases LIST        comma-separated phase subset
  --meter-file PATH    meter reading file for power.csv
  --meter-cmd CMD      meter command for power.csv
  --resume             continue an interrupted run
  --force              overwrite an existing run
  -h, --help           this text
EOF
}

while (($#)); do
    case "$1" in
        --profile|-p)    PROFILE="${2:?--profile needs a value}"; shift 2 ;;
        --devices|-d)    DEVICES="${2:?--devices needs a value}"; shift 2 ;;
        --run-dir)       RUN_DIR="${2:?--run-dir needs a value}"; shift 2 ;;
        --artifacts-dir) EXTRA+=(--artifacts-dir "${2:?--artifacts-dir needs a value}"); shift 2 ;;
        --windows-dir)   EXTRA+=(--windows-dir "${2:?--windows-dir needs a value}"); shift 2 ;;
        --phases)        EXTRA+=(--phases "${2:?--phases needs a value}"); shift 2 ;;
        --meter-file)    EXTRA+=(--meter-file "${2:?--meter-file needs a value}"); shift 2 ;;
        --meter-cmd)     EXTRA+=(--meter-cmd "${2:?--meter-cmd needs a value}"); shift 2 ;;
        --resume)        EXTRA+=(--resume); shift ;;
        --force)         EXTRA+=(--force); shift ;;
        -h|--help)       usage; exit 0 ;;
        *)               echo "error: unknown option '$1' (try --help)" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

fail() {
    echo "error: $*" >&2
    exit 1
}

# ── environment ──────────────────────────────────────────────────────────
command -v uv >/dev/null 2>&1 \
    || fail "uv is not installed; see https://docs.astral.sh/uv/ (or run: cargo --version)"

BINARY="rust/target/x86_64-unknown-linux-musl/release/inmotion-infer"
[[ -f "$BINARY" ]] \
    || fail "runtime binary missing at $BINARY; build it with: cargo build --release --target x86_64-unknown-linux-musl -p inmotion-cli"
command -v cargo >/dev/null 2>&1 \
    || echo "warning: cargo not found; using the prebuilt binary" >&2

needs_password=0
needs_device1=0
IFS=',' read -ra KEYS <<<"$DEVICES"
for raw_key in "${KEYS[@]}"; do
    key="${raw_key// /}"
    case "$key" in
        wave-router)   needs_device1=1 ;;
        wave-router-2) needs_password=1; needs_device1=1 ;;
        local)         ;;
        *) fail "unknown device '$key' (known: wave-router, wave-router-2, local)" ;;
    esac
done

if ((needs_password)); then
    [[ -n "${WAVECOM_PASSWORD:-}" ]] \
        || fail "WAVECOM_PASSWORD is unset: export it (value in wavecom-devices.md; never commit it) before a device-2 or both-device run"
fi
if ((needs_device1)); then
    ssh -o BatchMode=yes -o ConnectTimeout=8 wave-router true >/dev/null 2>&1 \
        || fail "device 1 (wave-router) is not reachable over SSH with key auth"
fi

if [[ -z "${SOURCE_DATE_EPOCH:-}" ]]; then
    SOURCE_DATE_EPOCH="$(git log -1 --format=%ct 2>/dev/null || date +%s)"
    export SOURCE_DATE_EPOCH
fi

# ── run ──────────────────────────────────────────────────────────────────
CMD=(uv run inmotion edge campaign --profile "$PROFILE" --devices "$DEVICES")
[[ -n "$RUN_DIR" ]] && CMD+=(--run-dir "$RUN_DIR")
if ((${#EXTRA[@]})); then
    CMD+=("${EXTRA[@]}")
fi

echo "campaign: profile=$PROFILE devices=$DEVICES source_date_epoch=$SOURCE_DATE_EPOCH"
echo "campaign: ${CMD[*]}"
"${CMD[@]}"

LATEST="results/edge/latest"
RESOLVED=""
if [[ -L "$LATEST" ]]; then
    RESOLVED="$(readlink -f "$LATEST")"
elif [[ -n "$RUN_DIR" ]]; then
    RESOLVED="$(cd "$RUN_DIR" && pwd)"
fi

if [[ -z "$RESOLVED" ]]; then
    echo "warning: could not resolve the run directory from $LATEST" >&2
    exit 0
fi

echo
echo "run dir: $RESOLVED"
if [[ -f "$RESOLVED/report.md" ]]; then
    echo "report:  $RESOLVED/report.md"
else
    echo "report:  (not built; the report phase was not requested)"
fi
