#!/bin/sh
# Measure the Rust runtime on the router.
#
# busybox notes: no python, so the bench JSON is parsed with sed; `sleep` here
# rejects fractional arguments, so the sampling interval uses usleep; getconf is
# absent, so USER_HZ is taken as 100 (the x86 Linux default). The wall window
# comes from the binary itself rather than being inferred from the throughput.
#
# Usage: router-measure.sh ARTIFACT_ROOT [ITER] [WARM]
ROOT=${1:?usage: router-measure.sh ARTIFACT_ROOT [ITER] [WARM]}
ITER=${2:-200}
WARM=${3:-30}
BIN=./inmotion-infer
HZ=100

field() { sed -n "s/.*\"$1\":\([-0-9.]*\).*/\1/p" "$2"; }

echo "model,precision,min_ms,median_ms,p95_ms,max_ms,throughput_per_s,wall_ms,total_ms,peak_rss_kb,cpu_ticks"

for dir in "$ROOT"/*/; do
  name=$(basename "$dir")
  # Producer suffixes are part of the PRECISION, not the model: the model name
  # is what the FLOPs table is joined on, and `${name%-*}` would turn
  # `lejepa_ft_seed42-torchao-int8` into a model called `lejepa_ft_seed42-torchao`.
  case "$name" in
    *-torchao-int8) model=${name%-torchao-int8}; prec=torchao-int8 ;;
    *) model=${name%-*}; prec=${name##*-} ;;
  esac
  json=/tmp/bench.json
  $BIN bench --artifact "$dir" --iterations "$ITER" --warmup "$WARM" --json > "$json" 2>/dev/null &
  pid=$!
  peak=0
  ticks=0
  while kill -0 "$pid" 2>/dev/null; do
    v=$(sed -n 's/^VmHWM:[[:space:]]*\([0-9]*\).*/\1/p' /proc/$pid/status 2>/dev/null)
    if [ -n "$v" ] && [ "$v" -gt "$peak" ]; then peak=$v; fi
    t=$(awk '{print $14+$15}' /proc/$pid/stat 2>/dev/null)
    if [ -n "$t" ]; then ticks=$t; fi
    usleep 50000
  done
  wait "$pid"
  echo "$model,$prec,$(field min_ms "$json"),$(field median_ms "$json"),$(field p95_ms "$json"),$(field max_ms "$json"),$(field throughput_per_s "$json"),$(field wall_ms "$json"),$(field total_ms "$json"),$peak,$ticks"
done
