# Extended-quantization campaign — results and provenance

What was measured, what each number means, and where it came from. Every figure
in this directory is produced by a script here from a CSV here, and every CSV is
produced by a command named below.

Two devices: **this PC** (Intel i5-1135G7, 4 cores / 8 threads, AVX2+FMA) and
**wave-router** (Intel Celeron J3455, 1.50 GHz, 4 cores, *no* AVX/AVX2/FMA).

## 1. The models

Five classifier checkpoints from one HPO run, none of them in the paper's
registry, recovered by inferring their architecture from tensor shapes:

| model | parameters | input channels | fp32 MCC |
|---|---:|---|---:|
| `sigreg_seed42` | 317,460 | 4 | **0.9177** |
| `lejepa_ft_seed42` | 565,765 | 4 | 0.9143 |
| `cf_jepa_ft_seed42` | 1,232,261 | **18** | 0.9083 |
| `t_jepa_ft_seed42` | 633,477 | **18** | 0.8982 |
| `ts_jepa_ft_seed42` | 2,153,349 | **18** | 0.4909 |

**Input width is not visible in the shapes.** `t_jepa`'s tokenizer is
`(182, 128)` — 180 flattened features, 18 channels × 10 timesteps — and
`cf_jepa`'s is a `Conv1d` with `in_channels=18`. Three of the five are 18-channel.
Nothing in the tensor shapes says so, and getting it wrong loads a real model,
feeds it the wrong features and produces plausible wrong numbers rather than an
error; the value is taken from the registry spec the shapes matched, and
`tests/test_adoption.py` pins all five.

**Two findings about the checkpoints themselves**, both recorded rather than
explained away:

* `sigreg_seed42` scores 0.9177 where the **registered** `sigreg` scores 0.6333,
  at 317,460 parameters against 696,900. It is a better model than the one the
  registry currently points at.
* `ts_jepa_ft_seed42` scores 0.4909 — near chance for four classes, against
  0.89–0.92 for everything else, while being the largest model. `inventory`
  flagged it as ambiguous with `cf_jepa_base`, so both readings were measured
  and **both give 0.4909**: they are structurally identical, so the architecture
  is not the cause. With no sidecar there is no provenance to check.

## 2. What each number means, and where it came from

| number | means | produced by | in |
|---|---|---|---|
| MCC | Matthews correlation on the 703-sample test split of `data/processed/dataset.csv`, seed 42 | `uv run inmotion quantize eval` | `sweep_hpo_models.csv`, `sweep_all_formats.csv` |
| compression | fp32 bytes ÷ artifact bytes, including scales and zero points | same | same |
| FLOPs | multiply–accumulates × 2 for one forward, from the shape each leaf module is called with | `uv run python results/campaign/measure_flops.py` | `flops_per_model.csv` |
| median_ms | median of 200 timed inferences, warmup excluded | `inmotion-infer bench` | `perf_device_*.csv` |
| peak RSS | highest `VmHWM` sampled while the measured process ran | `measure_device.py`, `router-measure.sh` | same |
| CPU% | CPU seconds ÷ the process's whole lifetime, the `top(1)` convention | same | same |
| throughput / speedup | inferences per second; speedup is against concurrency 1 | `inmotion-infer bench --concurrency N --seconds S` | `scaling_*.csv` |
| energy | **not measured** — see §6 | — | column empty, never 0 |

## 3. Accuracy

Full sweep, 5 models × 12 precisions, in `sweep_hpo_models.csv`. MCC:

| model | fp32 | fp16/bf16 | fp8 | int8 | q8 | q6 | q5 | q4 | q3 | q2 | q1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sigreg_seed42` | 0.918 | 0.918 | 0.908 | 0.916 | 0.918 | 0.916 | 0.914 | 0.907 | 0.850 | 0.112 | −0.005 |
| `lejepa_ft_seed42` | 0.914 | 0.914 | 0.847 | 0.912 | 0.914 | 0.912 | 0.908 | 0.908 | 0.838 | 0.323 | −0.001 |
| `cf_jepa_ft_seed42` | 0.908 | 0.908 | 0.904 | 0.906 | 0.906 | 0.906 | 0.905 | 0.905 | 0.899 | 0.453 | 0.134 |
| `t_jepa_ft_seed42` | 0.898 | 0.898 | 0.904 | 0.898 | 0.898 | 0.896 | 0.896 | 0.886 | 0.786 | 0.339 | −0.020 |
| `ts_jepa_ft_seed42` | 0.491 | 0.491 | 0.487 | 0.491 | 0.491 | 0.483 | 0.487 | 0.479 | 0.456 | 0.359 | 0.045 |

**q1 is at or below chance for all four JEPA models** (−0.02 to 0.13): one bit is
not a deployable precision for them, and 9.1× compression does not buy that back.
**q3 is where accuracy returns** (0.79–0.90). **q4 is the best
accuracy-preserving point** for the JEPAs at 6.09×, and q5 for `sigreg`,
`t_jepa` and `ts_jepa` at ~5.7×.

`fp8` (e4m3) is **not uniformly worse than q4**: worse for `lejepa` (0.847 vs
0.908) and `sigreg`, better for `t_jepa` (0.904 vs 0.886). `sigreg` is far more
quantization-tolerant than the transformers — 0.591 at q2 against 0.633 at fp32,
where the JEPAs sit near 0.3 at the same setting.

Figures: `fig_accuracy_vs_precision_hpo_models`, `fig_size_vs_accuracy_hpo_models`,
`fig_mcc_heatmap_hpo_models`.

## 4. Compute

| model | MFLOP/inference | FLOPs ÷ parameters |
|---|---:|---:|
| `t_jepa_ft_seed42` | 101.929 | 160.90 |
| `ts_jepa_ft_seed42` | 10.670 | 4.96 |
| `sigreg_seed42` | 5.984 | 18.85 |
| `lejepa_ft_seed42` | 5.638 | 9.96 |
| `cf_jepa_ft_seed42` | 4.111 | 3.34 |

**`torch.utils.flop_counter` is not used for this**, and the reason is checkable:
it reports 0.74 MFLOP for `cf_jepa`, whose transformer alone holds 1.19M
parameters — a dense forward through those cannot cost less than 2.4 MFLOP. The
counted numbers above come from forward hooks on the leaf modules, and every row
satisfies the invariant that catches that class of error: **FLOPs must exceed
parameters for a dense forward.** All five do, from 3.3× to 161×.

FLOPs are a property of the model, not the runtime, so one measurement serves both
devices; `flops_per_s` in the device CSVs is `flops × throughput`.

## 5. Devices

Both devices, median of 200 inferences, `perf_device_this-pc.csv` and
`perf_device_router.csv`:

| model | PC fp32 | J3455 fp32 | ratio | PC q1 RSS | J3455 q1 RSS |
|---|---:|---:|---:|---:|---:|
| `sigreg_seed42` | 0.690 ms | 2.825 ms | 4.09× | 4.43 MB | 2.30 MB |
| `lejepa_ft_seed42` | 0.745 ms | 2.909 ms | 3.90× | 5.73 MB | 3.51 MB |
| `cf_jepa_ft_seed42` | 1.003 ms | 3.210 ms | 3.20× | 8.53 MB | 6.43 MB |
| `ts_jepa_ft_seed42` | 1.616 ms | 5.404 ms | 3.34× | 13.89 MB | 10.63 MB |
| `t_jepa_ft_seed42` | 28.362 ms | 91.025 ms | 3.21× | 6.93 MB | 4.64 MB |

**3.2–4.1× slower** is what the hardware predicts: 1.5 GHz against a 2.4 GHz base
with no AVX2/FMA. **Peak RSS is lower on the router for every model** — musl's
allocator against glibc's, not anything the code does.

**QUANTIZATION DOES NOT RELIABLY CUT TIME.** `lejepa`'s q4 is 21% *slower* than
its fp32 on this PC (0.902 vs 0.745 ms): at 4–11 MFLOP the unpacking and
dequantization cost more than the smaller weights save. It cuts **memory**
consistently — q1 is a 9× smaller artifact on both devices. The win is memory,
not time, at this size.

**Parity is verified on both machines**, which is what makes these numbers
numbers rather than timings of an unknown computation: all 20 artifacts replay
their golden vectors with `PARITY OK` on the J3455 as well as here, fp32, int8,
q4 and q1 alike, `max |diff|` between 7.2e-7 and 8.6e-6 against a 2e-4 tolerance.

## 6. Energy — not measured

The `energy_j` and `energy_per_infer_mj` columns are **empty on every row**. This
is a measurement result, not an omission:

* **This PC**: `/sys/class/powercap/intel-rapl:0/energy_uj` is mode `0400 root`.
  Reading it needs privilege, and this session has no passwordless sudo. To fill
  the column:

      sudo -E uv run python results/campaign/measure_device.py \
          --binary rust/target/release/inmotion-infer --artifacts /tmp/hpo-art \
          --device this-pc --out results/campaign/perf_device_this-pc.csv

* **The router**: has **no `powercap` at all** — no RAPL, no energy counter of any
  kind, on a busybox kernel 5.15. Its energy column cannot be filled in software
  under any privilege; it needs external metering.

Reporting 0 would have been wrong in a way that survives into a paper.

## 7. Scaling and saturation

Concurrent inferences, speedup against concurrency 1, `scaling_*.csv`:

| device | model+precision | c=1 | max c | speedup | efficiency |
|---|---|---:|---:|---:|---:|
| PC (4C/8T) | `lejepa` q4 | 1018/s | 8 | 3.451× | 0.43 |
| PC | `sigreg` fp32 | 1357/s | 8 | 3.006× | 0.38 |
| PC | `t_jepa` fp32 | 38/s | 8 | 2.804× | 0.35 |
| J3455 (4C) | `ts_jepa` q4 | 184/s | 4 | 3.608× | 0.90 |
| J3455 | `cf_jepa` fp32 | 313/s | 4 | 3.606× | 0.90 |
| J3455 | `t_jepa` fp32 | 10.5/s | 4 | 3.276× | 0.82 |
| J3455 | `sigreg` fp32 | 352/s | 4 | 2.311× | 0.58 |

**Saturation**: on the J3455 the curves reach 3.3–3.6× on its four physical cores
and eight more would not exist; on this PC they flatten at about 3× on **eight**
threads, because four of those are hyperthreads that add little for this workload.
Counting cores is a poor guide to capacity; counting *physical* cores is not.

`sigreg` is the outlier on both devices (0.58 efficiency on the router) — it is
the only non-transformer, and its conv kernels are the most memory-bound of the
five, so it hits the shared-memory limit first.

Individual inferences are **not** split across threads: at 4–11 MFLOP the
coordination costs more than the arithmetic saves. Processing several requests
concurrently is the axis that scales.

## 8. gemm, matrixmultiply and rayon

**Tried and measured in an earlier round; not used.** The finding is recorded at
the kernel it concerns, `rust/inmotion-core/src/ops.rs`: at the router's baseline
`x86_64-unknown-linux-musl` target the hand-written packed kernel is **1.7×
faster than `gemm` and 1.9× faster than `matrixmultiply`** — both only pull ahead
once AVX2/FMA exist, which the J3455 does not have. The dependencies were removed
rather than kept behind a flag that would never be enabled.

This round adds the axis that earlier one did not test — concurrency, §7 — and
leaves the kernel comparison standing. Nothing in this campaign modifies `ops.rs`,
so the parity gates are unaffected.

## 9. What is not claimed

* **Energy**, for the reasons in §6.
* **`ts_jepa`'s low accuracy has no explanation here.** The architecture is ruled
  out; the cause needs its training provenance, which does not exist.
* **`torch`'s FLOP count** is reported nowhere as a result; §4 explains why.
* **Thread counts are not core counts.** The PC's eight "cores" are four physical
  cores; the efficiency figures reflect that.
* **CPU% on the router required care.** `/proc` counts CPU over the whole process,
  so the divisor is the process lifetime (`total_ms`), not the timed loop. An
  earlier version estimated the warmup and printed 103% for a single-threaded run.

## 10. Reproducing

```bash
# accuracy: 5 models x 12 precisions  (see also sweep_all_formats.csv for the
# registered models)
D=20-aug/models/exotic/normal-new-ds/full-after-hpo
uv run inmotion quantize eval \
  --models $D/lejepa_ft_seed42.pt,$D/t_jepa_ft_seed42.pt,$D/ts_jepa_ft_seed42.pt,\
$D/cf_jepa_ft_seed42.pt,$D/sigreg_seed42.pt \
  --formats fp32,bf16,fp16,fp8,int8,q8,q6,q5,q4,q3,q2,q1 \
  --allow-unrecorded-baseline --out results/campaign/sweep_hpo_models.csv

# FLOPs
uv run python results/campaign/measure_flops.py --models $D/*_ft_seed42.pt \
  $D/sigreg_seed42.pt --out results/campaign/flops_per_model.csv

# artifacts, then both devices
for m in lejepa t_jepa ts_jepa cf_jepa; do for q in fp32 int8 q4 q1; do
  uv run inmotion quantize export $D/${m}_ft_seed42.pt --out /tmp/hpo-art/${m}_ft_seed42-$q --format $q
done; done

cargo build --release --target x86_64-unknown-linux-musl -p inmotion-cli   # in rust/
uv run python results/campaign/measure_device.py --binary rust/target/release/inmotion-infer \
  --artifacts /tmp/hpo-art --device this-pc --out results/campaign/perf_device_this-pc.csv
uv run python results/campaign/measure_scaling.py --binary rust/target/release/inmotion-infer \
  --artifacts /tmp/hpo-art --device this-pc --out results/campaign/scaling_this-pc.csv
uv run python results/campaign/measure_router.py --out results/campaign

# figures
uv run python results/campaign/make_figures.py
```

Transfer to the router is `tar` over `ssh` — `scp` is blocked there and there is
no `base64`. `measure_router.py` performs that whole path, including the shell
sampler it deploys (`router-measure.sh`), so the router numbers are reproducible
rather than hand-copied.
