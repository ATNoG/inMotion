# CPU inference: what was measured, and what it cost

Everything here is regenerable from the committed scripts; nothing is quoted from
memory. `run_bench.sh` reproduces the raw CSVs, `summarise.py` reduces them to
`summary.csv`.

The deployment target is a **wave-router: Intel Celeron J3455 @ 1.50 GHz**, no
AVX, no AVX2, no FMA (`sse4_2`, `aes`, `sha_ni` only), 4 cores, 3.4 GB RAM. All
router binaries are `x86_64-unknown-linux-musl` built at the **baseline**
`x86-64` feature level, statically linked, and transferred by `tar` over `ssh`
(`scp` is blocked and the router has no `base64`). The development machine is an
i5-1135G7 (AVX2 + FMA) and is a *different* machine on purpose: it shows how much
of a crate's win is the crate and how much is the host's ISA.

## Verdict on the crate question

The brief asked for `gemm`/`matrixmultiply` + `rayon`. It was measured rather
than assumed, and the measurement says no:

**On the J3455** — the machine the code has to run on — a hand-written rank-1
kernel beats every candidate, and is the only one that reproduces the reference
bit for bit:

| impl | lejepa proj (10,128→128) | ts_jepa proj (5,256→256) | vs naive | bit-exact |
|---|---|---|---|---|
| `std` naive dot product (old) | 0.3905 ms | 0.7641 ms | 1.00x | yes |
| **rank-1 packed (shipped)** | **0.0608 ms** | **0.1216 ms** | **6.4x** | **yes** |
| `matrixmultiply::sgemm` | 0.1128 ms | 0.2854 ms | 3.4x | no, 3.6e-7 |
| `gemm` (faer's microkernel) | 0.1053 ms | 0.2446 ms | 3.7x | no, 3.6e-7 |
| `faer::linalg::matmul` | 0.0908 ms | 0.1835 ms | 4.3x | no, 3.6e-7 |
| `gemm` + `rayon` | 0.1053 ms | 0.2447 ms | 3.7x | no |
| `faer` + `rayon` | 0.0909 ms | 0.1834 ms | 4.3x | no |

So on the target: **1.50x faster than faer, 1.73x faster than `gemm`, 1.84x
faster than `matrixmultiply`**, and the only entry that is bit-identical. Full
data in `crate_survey.csv` (3 machines x 4 shapes x 7 impls).

Why the crates lose here: their kernels are tuned for AVX2/FMA, which the J3455
does not have, and they pay packing + runtime-dispatch costs that a GEMM this
small (M = 5-10, K = N = 128-256) cannot amortise. The same crates *do* win on
the i5 — `gemm` is 2.1x faster than the hand kernel there — which is exactly the
trap: benchmarking on the development machine would have picked the wrong kernel
for the device. On the dev machine's baseline-musl build, `faer` is also 13x
faster than the same `faer` code built for gnu, so third-party microbenchmarks
are not even stable across hosts, let alone across CPUs.

`std::simd` was **not** tried: it is nightly-only (`portable_simd`), and this
workspace builds on stable 1.93.1. `rayon` was tried and **did not help** — see
below.

## The three kernels that were rewritten

All in `rust/inmotion-core/src/ops.rs`. The thread running through all three:
with `f32` and no fast-math, a reduction is a serial dependency chain (~4 cycles
per add) and the compiler is forbidden from reassociating it. Every rewrite
sweeps a *different* axis with independent accumulators, and leaves the adds that
produce any single output element in their original order. So the results are
identical, not merely close.

1. **`linear_packed` / `PackedWeight`.** Weights are transposed once at load to
   `(in, out)`; the kernel then does `acc[r][o] += x[r][i] * w[i][o]` with the
   output axis contiguous and vectorisable. This was ~97 % of a LeJEPA forward.
2. **`self_attention`.** The score and context reductions ran `... for c { ...
   for j }`, which is one serial chain per `(i, j)`. At T = 182 (T-JEPA) that
   alone was **15.1 ms per layer — 11x all four projections together**. `k`'s
   head slice is transposed once into `(head_dim, T)` so the `j` sweep is
   contiguous; the context loop holds one `head_dim` accumulator row and walks
   `j` outward.
3. **`conv1d_same`.** Same disease, and it was ~100 % of SIGReg's inference time.
   The `(ci, k)` reduction is flattened into one axis and the output-channel axis
   is made innermost; the accumulator is held transposed as
   `(length, channels_out)` so the inner loop is contiguous in both operands.

## Results: before -> after

`before` is the pre-optimisation kernel rebuilt from this tree (see "The before
build"); `after` is the shipped code. Runs are interleaved so machine load drifts
cancel, and each cell is measured 3x — the table takes the best run per cell,
because on a shared machine the tail is load, not kernel. Raw per-run rows are in
`bench_host.csv` and `bench_router.csv`.

### Development machine (i5-1135G7, baseline `x86-64` build)

| model | precision | before median | after median | median speedup | before p95 | after p95 | p95 speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| lejepa | fp32 | 4.3444 ms | 0.7006 ms | **6.20x** | 4.8152 ms | 0.8750 ms | 5.50x |
| lejepa | int8 | 4.3272 ms | 0.6945 ms | **6.23x** | 5.0129 ms | 0.8454 ms | 5.93x |
| lejepa | q4 | 4.3532 ms | 0.7065 ms | **6.16x** | 5.0508 ms | 1.0614 ms | 4.76x |
| t_jepa | fp32 | 125.4556 ms | 23.8793 ms | **5.25x** | 237.1877 ms | 31.3886 ms | 7.56x |
| t_jepa | int8 | 112.9183 ms | 24.1083 ms | **4.68x** | 117.5822 ms | 26.2248 ms | 4.48x |
| t_jepa | q4 | 112.6999 ms | 24.0297 ms | **4.69x** | 118.9137 ms | 26.2976 ms | 4.52x |
| ts_jepa | fp32 | 9.8478 ms | 1.5870 ms | **6.21x** | 10.3810 ms | 1.9962 ms | 5.20x |
| ts_jepa | int8 | 9.8503 ms | 1.5673 ms | **6.28x** | 11.3014 ms | 1.7684 ms | 6.39x |
| ts_jepa | q4 | 9.8698 ms | 1.5967 ms | **6.18x** | 11.0859 ms | 1.8233 ms | 6.08x |
| cf_jepa | fp32 | 5.4293 ms | 0.7765 ms | **6.99x** | 5.9707 ms | 1.0091 ms | 5.92x |
| cf_jepa | int8 | 5.4006 ms | 0.7655 ms | **7.05x** | 5.8381 ms | 0.9074 ms | 6.43x |
| cf_jepa | q4 | 5.4482 ms | 0.7629 ms | **7.14x** | 5.9962 ms | 1.1716 ms | 5.12x |
| sigreg | fp32 | 7.9493 ms | 1.6250 ms | **4.89x** | 9.5902 ms | 2.1337 ms | 4.49x |
| sigreg | int8 | 8.0487 ms | 1.4079 ms | **5.72x** | 8.5296 ms | 1.6438 ms | 5.19x |
| sigreg | q4 | 8.0584 ms | 1.4064 ms | **5.73x** | 9.9836 ms | 1.6186 ms | 6.17x |

### wave-router (Intel Celeron J3455, the deployment target)

| model | precision | before median | after median | median speedup | before p95 | after p95 | p95 speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| lejepa | fp32 | 8.8121 ms | 2.9024 ms | **3.04x** | 9.0358 ms | 3.0112 ms | 3.00x |
| lejepa | int8 | 8.8158 ms | 2.8932 ms | **3.05x** | 9.0751 ms | 2.9730 ms | 3.05x |
| lejepa | q4 | 8.8474 ms | 2.9194 ms | **3.03x** | 9.1291 ms | 2.9881 ms | 3.06x |
| t_jepa | fp32 | 283.6626 ms | 90.3118 ms | **3.14x** | 287.2508 ms | 91.5554 ms | 3.14x |
| t_jepa | int8 | 284.0535 ms | 90.4940 ms | **3.14x** | 287.3963 ms | 92.2567 ms | 3.12x |
| t_jepa | q4 | 283.9073 ms | 90.5419 ms | **3.14x** | 286.8908 ms | 91.7908 ms | 3.13x |
| ts_jepa | fp32 | 16.3576 ms | 5.5338 ms | **2.96x** | 16.8520 ms | 5.6287 ms | 2.99x |
| ts_jepa | int8 | 16.3527 ms | 5.5931 ms | **2.92x** | 16.8501 ms | 5.7033 ms | 2.95x |
| ts_jepa | q4 | 16.3395 ms | 5.5396 ms | **2.95x** | 16.8589 ms | 5.7497 ms | 2.93x |
| cf_jepa | fp32 | 9.1951 ms | 3.1187 ms | **2.95x** | 9.4341 ms | 3.2124 ms | 2.94x |
| cf_jepa | int8 | 9.2106 ms | 3.1798 ms | **2.90x** | 9.4749 ms | 3.2892 ms | 2.88x |
| cf_jepa | q4 | 9.2017 ms | 3.1748 ms | **2.90x** | 9.4593 ms | 3.2540 ms | 2.91x |
| sigreg | fp32 | 24.8637 ms | 6.2082 ms | **4.00x** | 25.6659 ms | 6.4515 ms | 3.98x |
| sigreg | int8 | 24.9100 ms | 6.2695 ms | **3.97x** | 25.6436 ms | 6.4429 ms | 3.98x |
| sigreg | q4 | 24.9145 ms | 6.2136 ms | **4.01x** | 25.6138 ms | 6.4175 ms | 3.99x |

Cross-check against the recorded baseline: the router's `before` lejepa q4 is
**8.8474 ms median** here against the **8.9002 ms** recorded in
`results/quant/edge_router_j3455.json` — 0.6 % apart, from an independent binary
and an independent protocol. That is the evidence that `before` really is the
pre-optimisation code, and it means the router speedups are anchored to a figure
nobody in this task produced.

Absolute levels on the development machine are *not* comparable to that recorded
baseline (its 4.35 ms lejepa-q4 median here vs 6.99 ms recorded): that machine is
shared with other work and the recorded figure was taken under different load.
The interleaved A/B design is what makes the *ratios* trustworthy even so; the
levels are the part that moves.

## What did not help

Reported rather than quietly dropped:

- **`rayon`.** No measurable effect at any of these shapes. On the router,
  `gemm`+`rayon` is 0.1053 ms against `gemm`'s 0.1053 ms and `faer`+`rayon` 0.0909
  against `faer`'s 0.0908 — the thread-pool synchronisation costs exactly what the
  parallel work saves. These are single-inference latencies of 0.02-0.8 ms per
  op; there is nothing for a thread pool to earn. (`rayon` would help batch
  *throughput*, which is a different measurement from the one the brief asks for.)
- **Row blocking the rank-1 kernel** (to keep the accumulator set in L1 for
  T = 182, where it is 93 KB). Measured at block sizes 4/8/16/32/64: 0.3360-0.3426
  ms against `all_rows`' 0.3413 ms for T-JEPA's projection. Within noise; the L1
  theory did not hold and the code stayed simpler without it.
- **Register-tiled microkernels** (`R x C` accumulator tiles of 4x8, 2x8, 4x16,
  5x8, 10x8): 2.0x-6.1x against the shipped kernel's 8.9x at the same shape. The
  constant-size accumulator arrays defeated the vectoriser. Not adopted.
- **`target-cpu=native`.** Not adopted, and not just because of the router: the
  entire point of the exercise is that one binary must run on the J3455, which
  has none of the features `native` would emit.
- **Switching to `faer` for the dev machine** and keeping the hand kernel for the
  router. Rejected: two kernels means two sets of numbers, and the device-
  independent parity result below is worth more than 2.9x on a laptop. It is a
  real tradeoff and it is stated rather than hidden.

## Parity is bit-identical, not merely within tolerance

`inmotion-infer check` replays the golden vectors on both machines. All five
models x three precisions (fp32, int8, q4) produce the **exact same max|diff|
before and after**, to every printed digit, on both machines:

| model | fp32 | int8 | q4 |
|---|---|---|---|
| lejepa | 4.05e-6 | 4.23e-6 | 3.34e-6 |
| t_jepa | 2.62e-6 | 4.14e-6 | 6.20e-6 |
| ts_jepa | 2.74e-6 | 2.92e-6 | 5.01e-6 |
| cf_jepa | 2.21e-6 | 2.95e-6 | 1.91e-6 |
| sigreg | 1.23e-6 | 1.97e-6 | 1.43e-6 |

Tolerance is 2e-4, so every case keeps >30x of headroom and no case moved. The
`before` column here is also the number quoted by the model-port tasks, which is
independent evidence that the `before` build really is the pre-optimisation code.

Regression tests pin the property without needing a checkpoint:
`ops::tests::packed_matches_linear_bit_for_bit`,
`ops::tests::attention_is_bit_exact_after_the_rewrite` (against a retained
copy of the old triple loop), and `ops::tests::packed_linear_adds_the_bias_before_the_products`.

## The router build assumes no AVX/AVX2/FMA

There is no `target-cpu`, `target-feature` or `RUSTFLAGS` anywhere in the
workspace (`rust/Cargo.toml` sets only `opt-level`/`lto`/`codegen-units`), so the
router binary is built at the stock baseline `x86-64` feature level:
`cargo build --release --target x86_64-unknown-linux-musl`, static-pie.

The shipped binary contains **zero VEX-encoded (AVX-class) instructions**:

    $ objdump -d rust/target/x86_64-unknown-linux-musl/release/inmotion-infer \
        | grep -cE '^\s+[0-9a-f]+:\s+[a-z0-9]+\s+%v[0-9]'
    0

and the copy that produced the router numbers is that same binary, byte for byte
(md5 `e557ade75786c1f855ee50b3602b0aec` on both ends), `sha256
29cd6de5b482f3505bfdee2e3837ece5185db00b3e6dd86be58b96f42075faf3`. It ran the
whole benchmark and parity suite on the J3455, which has none of AVX/AVX2/FMA.

This is a deliberate cost of choosing the hand kernel: the candidate crates ship
AVX2/FMA microkernels guarded by runtime CPU detection (`pulp`/`raw-cpuid`), so
their speed depends on the host and their correctness depends on that detection
being right on a Celeron. The hand kernel compiles to whatever the target
supports and needs no detection at all.

## The `before` build

There is no commit to check out that gives five-model `before` numbers: the model
ports are uncommitted in the working tree, and the recorded baseline
(`results/quant/edge_router_j3455.json`, read-only input for this task) covers
lejepa q4 only. So `before` was rebuilt from the *current* tree by reverting only
the bodies of the three kernels to their original loop shapes, in a scratch copy
under `/tmp`:

    cp -r rust /tmp/before-rust           # excluding target/
    # revert linear_packed's body and PackedWeight::pack to the original (out, in)
    # dot-product form; the call sites, loader and artifact format are untouched
    cd /tmp/before-rust && cargo build --release

That build reproduces the recorded lejepa q4 figures and the port tasks' parity
numbers exactly, which is what makes the A/B above trustworthy.

## Commands

Export the artifacts and build the two binaries:

    uv run python -c "
    from pathlib import Path; from inmotion.quant.export import export, write_golden
    for m in ['lejepa','t_jepa','ts_jepa','cf_jepa','sigreg']:
        for q in [None,'int8','q4']:
            out = Path('/tmp/art')/f'{m}-{q or \"fp32\"}'
            export(m, out, quant=q); write_golden(m, out, quant=q)"
    cd rust && cargo build --release
    cd rust && cargo build --release --target x86_64-unknown-linux-musl

Parity, both machines:

    /tmp/perf/after-gnu  check --artifact /tmp/art/<model>-<prec> --golden /tmp/art/<model>-<prec>
    ./after-musl         check --artifact art/<model>-<prec>     --golden art/<model>-<prec>

Benchmark, both machines (this is `run_bench.sh`):

    sh run_bench.sh host      # taskset -c 6, 300 iterations + 40 warmup, 3 reps
    sh run_bench.sh router    # 200 iterations + 30 warmup, 3 reps

Deploy (scp is blocked on the router, so `tar` over `ssh`):

    tar czf deploy.tgz -C /tmp/art . -C /tmp/perf before-musl after-musl
    cat deploy.tgz | ssh wave-router 'cd /mnt/extra/cargo/inmotion/perf && tar xzf -'

Reduce:

    python3 summarise.py

## Files

| file | what |
|---|---|
| `run_bench.sh` | reproduces the raw CSVs on either machine |
| `bench_host.csv` | raw per-run rows, dev machine (before x after, 3 reps) |
| `bench_router.csv` | raw per-run rows, wave-router |
| `summary.csv` | before/after median and p95 per device/model/precision |
| `crate_survey.csv` | 3 machines x 4 shapes x 7 kernels, incl. bit-exactness |
| `crate_survey_*.csv` | the unmerged per-machine captures |
| `summarise.py` | raw CSVs -> `summary.csv` |
