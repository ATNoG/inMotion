#!/bin/sh
# Reproduce results/perf/bench_{host,router}.csv.
#
#   sh run_bench.sh host     # on the development machine
#   sh run_bench.sh router   # on the wave-router (J3455)
#
# Both variants interleave `before` and `after` runs of the same artifact so that
# machine load drifts cancel instead of landing entirely on one side. Each cell
# is measured `REPS` times; summary.py keeps the best run per cell.
#
# `before` is the pre-optimisation kernel. It is rebuilt from this tree by
# reverting only the bodies of `ops::linear_packed`, `ops::self_attention` and
# `ops::conv1d_same` to their original loop shapes (see README.md, "The before
# build"); the call sites, the loader and the artifact format are untouched.
set -e
DEVICE=${1:?usage: run_bench.sh host|router}
REPS=${REPS:-3}
MODELS="lejepa t_jepa ts_jepa cf_jepa sigreg"
PRECS="fp32 int8 q4"

if [ "$DEVICE" = host ]; then
  ART=${ART:-/tmp/art}
  BEFORE=${BEFORE:-/tmp/perf/before-gnu}
  AFTER=${AFTER:-/tmp/perf/after-gnu}
  ITER=${ITER:-300}
  WARM=${WARM:-40}
  OUT=${OUT:-bench_host.csv}
  PIN="taskset -c 6"
  line() { # json -> csv row
    printf '%s\n' "$1" | python3 -c "
import json,sys
d=json.load(sys.stdin)
print('host,$2,$3,$4,$5,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f' % (d['min_ms'],d['median_ms'],d['mean_ms'],d['p95_ms'],d['max_ms'],d['throughput_per_s']))"
  }
else
  cd "${DEPLOY:-/mnt/extra/cargo/inmotion/perf}"
  ART=${ART:-art}
  BEFORE=${BEFORE:-./before-musl}
  AFTER=${AFTER:-./after-musl}
  ITER=${ITER:-200}
  WARM=${WARM:-30}
  OUT=${OUT:-bench_router.csv}
  PIN=""
  # The router is busybox: no python3, so parse the JSON with sed.
  line() {
    f() { printf '%s\n' "$1" | sed -n "s/.*\"$2\":\([-0-9.]*\).*/\1/p"; }
    echo "router,$3,$4,$5,$6,$(f min_ms),$(f median_ms),$(f mean_ms),$(f p95_ms),$(f max_ms),$(f throughput_per_s)"
  }
fi

echo "device,model,precision,build,run,min_ms,median_ms,mean_ms,p95_ms,max_ms,throughput_per_s" > "$OUT"
for model in $MODELS; do
  for prec in $PRECS; do
    run=1
    while [ "$run" -le "$REPS" ]; do
      for build in before after; do
        bin=$BEFORE
        [ "$build" = after ] && bin=$AFTER
        json=$($PIN "$bin" bench --artifact "$ART/$model-$prec" \
                 --iterations "$ITER" --warmup "$WARM" --json)
        line "$json" x "$model" "$prec" "$build" "$run" >> "$OUT"
      done
      run=$((run + 1))
    done
  done
done
