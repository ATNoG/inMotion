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

These are the numbers as first measured, by the benchmark build whose hashes are
recorded under "The router build assumes no AVX/AVX2/FMA". They were re-measured
against the repaired tree; see "Reproduced on the repaired tree" below for what
reproduced and what moved.

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

Checkpoint note: these figures were measured against the sigreg checkpoint that
was current when the kernel work ran. `sigreg` was later retrained on
`dataset.csv` (t12), and its parity is now 8.3e-7 / 8.3e-7 / 9.5e-7, which is
tighter, not worse. `rust/README.md` quotes the retrained numbers. The table below
is the record of the kernel before/after comparison, so it keeps the checkpoint
it was actually measured against.

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

**No `inmotion` code path emits AVX, AVX2 or FMA.** The whole binary is *not*
free of AVX-encoded instructions: the only ones belong to `memchr`, in the three
`find_avx2` symbols that crate reaches exclusively through its own runtime CPUID
dispatch, so they are never taken on the J3455 (which has none of AVX/AVX2/FMA).
Attributing them by symbol:

    $ objdump -d --no-show-raw-insn rust/target/x86_64-unknown-linux-musl/release/inmotion-infer \
        | grep -cE '^\s+[0-9a-f]+:\s+v[a-z0-9]+'
    282
    $ objdump -d --no-show-raw-insn <same binary> \
        | awk '/^[0-9a-f]+ </{s=$0} /^[[:space:]]+[0-9a-f]+:[[:space:]]+v[a-z0-9]+/{c[s]++} END{for(k in c) print c[k], k}' | sort -rn
    250 ...memchr...count_raw...find_avx2...
     27 ...memchr...One...rfind_raw_avx2...
      5 ...memchr...memrchr_raw...find_avx2...

All 282 are in those three `memchr` symbols; none is in an `inmotion*` symbol.

The hashes below identify the **benchmark build** — the binary that produced the
first round of numbers, and the copy that was transferred to and ran on the
J3455 — not the current working tree. Use them to check the *committed* numbers;
the re-run numbers in "Reproduced on the repaired tree" were produced by the
repaired build and carry their own hashes there:

    md5    e557ade75786c1f855ee50b3602b0aec  (on both ends)
    sha256 29cd6de5b482f3505bfdee2e3837ece5185db00b3e6dd86be58b96f42075faf3

That build ran the whole benchmark and parity suite on the J3455. Its `md5` is
recorded for the router copy (musl); the host table came from the gnu build of
the same tree, `md5 132d899460992fcdd24803a2395c4fb6`,
`sha256 83751b30082d8c290933451c71bb653d98c1ea0876059a48af1f6f056ffc9157`.

This is a deliberate cost of choosing the hand kernel: the candidate crates ship
AVX2/FMA microkernels guarded by runtime CPU detection (`pulp`/`raw-cpuid`), so
their speed depends on the host and their correctness depends on that detection
being right on a Celeron. The hand kernel compiles to whatever the target
supports and needs no detection at all.

## Reproduced on the repaired tree

The repair that followed (the round-1 findings, then R2-01 and the round-2 open
items) changed `main.rs`, `artifact.rs` and docs, and
`docs/reports/CODE_REVIEW_RUST_ROUND2.md` argued from `git diff` that the numbers
still held. That argument was re-established by measurement instead: both
binaries were rebuilt from the repaired tree, and the whole A/B was re-run on
both machines with `run_bench.sh`.

The re-run binaries (repaired tree, stock baseline `x86-64`, rustc 1.93.1):

| build | file | md5 | sha256 |
|---|---|---|---|
| host after | `/tmp/perf/after2-gnu` | `33a20e6a29a8c850886e635e877e96c5` | `e2a8b4d4fe4313792f29c8bfa10c8144e581ade2f124df97e62571b3984cbcb3` |
| router after | `/tmp/perf/after2-musl` | `6372b6e1113aecb89dd9739506d88bb4` | `2c326abee708b83743d4a1f192183d0e78223f302ef6f82c82ed89d09b9d7a77` |
| host before | `/tmp/perf/before2-gnu` | `f8e8dd7da82f0526d1dbcbf18aa7bd1a` | `520a7f9e0a3ebc8e32907d8a5edda4ff3a1555b4a9532c2358b201231a8df6e0` |
| router before | `/tmp/perf/before2-musl` | `92f4dca2d3bd7d588cb67829ce9e229c` | `6aa3baef13b8802a23d5702cbe28ab5c3b98d0e923edaec12ffc5f1d4943f684` |

Parity is unchanged by the repair: all five models x three precisions pass on
both the gnu and the musl repaired binaries, and the `before` binary (repaired
tree, three kernels reverted) prints the same max|diff| digits as the `after`
one. The committed `bench_host.csv`/`bench_router.csv` were regenerated by this
re-run, so the raw rows and `summary.csv` now on disk are the repaired build's.

### On the deployment target (the J3455), everything reproduces

All fifteen router cells land within 1.4 % of the committed speedup and 0.8 % of
the committed medians:

| model | precision | committed | re-run | delta |
|---|---:|---:|---:|---:|
| lejepa | fp32 | 3.04x | 3.05x | +0.3% |
| lejepa | int8 | 3.05x | 3.03x | -0.7% |
| lejepa | q4 | 3.03x | 3.03x | +0.0% |
| t_jepa | fp32 | 3.14x | 3.15x | +0.3% |
| t_jepa | int8 | 3.14x | 3.14x | +0.0% |
| t_jepa | q4 | 3.14x | 3.13x | -0.3% |
| ts_jepa | fp32 | 2.96x | 2.92x | -1.4% |
| ts_jepa | int8 | 2.92x | 2.95x | +1.0% |
| ts_jepa | q4 | 2.95x | 2.93x | -0.7% |
| cf_jepa | fp32 | 2.95x | 2.93x | -0.7% |
| cf_jepa | int8 | 2.90x | 2.91x | +0.3% |
| cf_jepa | q4 | 2.90x | 2.90x | +0.0% |
| sigreg | fp32 | 4.00x | 4.02x | +0.5% |
| sigreg | int8 | 3.97x | 4.01x | +1.0% |
| sigreg | q4 | 4.01x | 3.99x | -0.5% |

Worst router cell: `ts_jepa` fp32, 2.96x -> 2.92x, 1.4 %. Medians move at most
0.8 % (`t_jepa` q4 before 283.9073 -> 283.2844 ms). The device-side question the
brief asks — "re test in this machine and in the router" — is answered yes on the
router.

### On the development machine, eight cells move beyond 3 %

The host re-run does **not** reproduce as tightly. On the speedup ratio, **8 of
the 15** host cells exceed 3 % against the committed table, and **7 of 15** do on
the after-median. Ranked, the speedup deltas are:

| cell | committed | re-run | delta |
|---|---:|---:|---:|
| cf_jepa q4 | 7.14x | 6.63x | -7.1% |
| t_jepa fp32 | 5.25x | 4.91x | -6.5% |
| cf_jepa int8 | 7.05x | 6.71x | -4.8% |
| sigreg int8 | 5.72x | 5.45x | -4.7% |
| lejepa fp32 | 6.20x | 5.91x | -4.7% |
| sigreg q4 | 5.73x | 5.53x | -3.5% |
| t_jepa q4 | 4.69x | 4.85x | +3.4% |
| lejepa int8 | 6.23x | 6.03x | -3.2% |

The other seven cells span -2.8 % to +3.0 % (the top of that band is `t_jepa
int8` at +2.99 %, immediately below the threshold). The router is the contrast:
0 of 15 exceed 3 % on either statistic. All of this is reported rather than
dropped.

**Cause, established by measurement, not argument.** An interleaved run of the
old and new *binaries* in one session over the worst cells separates the build
from the machine:

    t_jepa  fp32   old before 111.43 111.29 111.36 111.20 111.42 ms
                   new before 111.38 110.38 111.55 111.07 111.67 ms   (agree)
                   old after   23.02  23.40  23.04  23.01  23.23 ms
                   new after   22.33  22.59  22.26  22.82  22.32 ms   (old faster)
    cf_jepa q4     old before   5.426  5.394  5.377  5.419  5.346 ms
                   new before   5.399  5.354  5.391  5.358  5.390 ms  (agree)
                   old after    0.717  0.728  0.728  0.735  0.745 ms
                   new after    0.799  0.807  0.826  0.793  0.806 ms  (old faster)

Both `before` builds agree to about 1 %; both `after` builds differ by 3-10 %
depending on the cell, in the *same session*, so the difference is in the
binary, not in the machine. Yet the two `after` binaries are the same code: with
addresses masked, `linear_packed` disassembles to the same 1988 instructions in
both, and `nm` gives the same symbol hash (`hd59a657b199fcf20`) and the same
0x722-byte size — the kernels did not change. `ops.rs` and `artifact.rs` are
byte-identical to the round-2 pinned revision (`sha256 92963c1f…` and
`2bb3943f…`), so the symbol maps differ only in pre-existing work outside this
workstream: the loader guards already in `artifact.rs` and the harness code in
`main.rs` (`bench`/`check`/`predict`), nothing the kernels see. The binaries
differ only in where the functions were placed.

That placement matters at these sizes: the LeJEPA projection is a 0.7 ms call
that is alignment-sensitive to a few percent on this microarchitecture, so a
rebuilt binary's *host* cells move with the layout. The router cells do not: the
musl binaries built from the two trees agree within 0.2 % (lejepa int8
2.9890 ms old vs 2.9935 ms new, measured directly), and the full router re-run is
the table above. The device result is layout-insensitive; the laptop is not.

The committed host medians are therefore a valid measurement of *that* binary,
but host medians are reproducible only to the ~7 % that code layout allows, and
the earlier 6-7x host speedups sit at the optimistic end of that band. The
kernel question the table exists to answer is unaffected: the router — the
machine the code has to run on — reproduces within noise, and parity is
bit-identical on both.

### The harness bugs this re-run exposed

The committed `run_bench.sh` was broken in **both** branches, and the re-run
surfaced it:

1. **Host.** It called `line "$json" x "$model" "$prec" "$build" "$run"` while
   its `line()` printed `host,$2,$3,$4,$5,…`. Inside a shell function `$2` is the
   *function's* second argument — the literal `x` — so the device was fine but
   every label moved one column left: `model` received `x`, `precision` received
   the model, `build` the precision, `run` the build. The metrics themselves
   still landed in their right columns.
2. **Router.** Its nested helper `f() { printf '%s\n' "$1" | sed -n
   "s/.*\"$2\":…" }` *shadows* the outer function's positional parameters: a
   function call gets its own frame, so inside `f` `$1` is the field name and
   `$2` is unset. `$(f min_ms)` therefore piped the word `min_ms` into a sed
   pattern of `"":`, and all six metrics printed empty — with the header still
   claiming 11 columns.

Neither defect invalidates the committed numbers. The committed tables have their
own provenance:

| file | producer |
|---|---|
| `bench_router.csv` at HEAD (`sha256 5adc98af…`, 11 columns x 90 rows) | an earlier device-side variant; **its producer is not reproducible today.** It matches neither CSV on wave-router (`bench_router.csv` `8c00f98f…`, 22 data rows; `bench_router2.csv` `cab37a42…`, 90 data rows). It cannot have come from the committed `run_bench.sh`, whose router branch emits blank metrics |
| `bench_router.csv` shipping in this commit (`sha256 cab37a42…`) | `router_bench2.sh` on wave-router, by byte-identity: the shipping file is sha256-identical to that script's default output `bench_router2.csv`. It runs `./before2-musl` / `./after2-musl` (the repaired-build binaries) over the same 11-column rows |
| `bench_host.csv` | a corrected host invocation of the same harness; the producer is not in the repository, but 0 of its 90 rows contain `,x,` and its shape is exactly what the fixed branch emits |
| `summary.csv` | the committed `summarise.py` itself, merging both files above |

`router_bench.sh` on the device (`sha256 cbd0cdb5…`, 875 B, mode 755) is the
right contrast for the broken committed `run_bench.sh`: the same protocol, an
11-field header byte-identical to the committed CSV's, and a working extractor —
`f() { echo "$j" | sed -n "s/.*\"$1\":\([-0-9.]*\).*/\1/p"; }`, JSON in `$j` and
the field name in `$1`, so `$(f min_ms)` reads the value rather than the word
`min_ms`. Its own default output on the device (`8c00f98f…`, 22 data rows)
matches neither the file at HEAD nor the file that ships here, so it is not
evidence for either.

Run the committed `summarise.py` on the committed `bench_host.csv` and
`bench_router.csv` and it reproduces the committed `summary.csv` value for value
— 30 cells, 15 host and 15 router. The only difference is line endings (the
committed CSVs are LF, the csv module's default dialect writes CRLF, which is why
the current `summarise.py` pins `lineterminator="\n"`). So **the committed data
is trustworthy**; what was broken is the committed transcription of the harness
that produced it.

Both branches now pass the JSON to the parser explicitly (argv for the host,
stdin for the router) and take the device as `$2`, so both write the same 11
columns and `summarise.py` merges them. `summarise.py` grew the matching guard: it
skips a trailing status line such as the router runner's `DONE`, but refuses a
partial row, which is what column drift looks like.

## The `before` build

There is no commit to check out that gives five-model `before` numbers: the model
ports are uncommitted in the working tree, and the recorded baseline
(`results/quant/edge_router_j3455.json`, read-only input for this task) covers
lejepa q4 only. So `before` is rebuilt from the *current* tree by reverting only
the bodies of the four kernels to their original loop shapes. That revert is
`make_before_kernels.py` plus `ops_before_optimisation.rs`, both tracked here, so a
clone needs nothing outside the repository:

    # all commands from the repository root; the script and its baseline live in
    # results/perf/ and the target is always explicit — a scratch copy here, never
    # the checkout's own rust/inmotion-core/src/ops.rs
    mkdir -p /tmp/before-build
    cp -r rust /tmp/before-build/rust           # any scratch path *outside* the tree
    python3 results/perf/make_before_kernels.py \
        /tmp/before-build/rust/inmotion-core/src/ops.rs
    cd /tmp/before-build/rust && cargo build --release
    cd /tmp/before-build/rust && cargo build --release --target x86_64-unknown-linux-musl

The scratch copy must be a **different path from the repository**, and this is
the defect the recipe exists to avoid rather than a caution about caution: the
form that used to be documented here was `cp -r rust /tmp/before-rust` followed by
reverting `/tmp/before-rust/inmotion-core/src/ops.rs` against an *untracked*
original. On any host that does not already hold that original, the `cp` simply
creates `/tmp/before-rust` as a copy of the **current** optimised tree, so the
script's target and its pre-optimisation side are the same optimised source, every
replacement is an identity (`13 -> 13`, `32 -> 32`, `59 -> 59`, `45 -> 45`), and
the build produces the optimised binary **labelled `before`** — a false A/B. That
recipe only appeared to work on the one machine whose `/tmp/before-rust` already
held the retained tree, so the claim rested entirely on out-of-band state.

The recipe above has neither problem. It copies the tree to a path that does not
exist yet, so the copy is a fresh tree; it reverts that copy against the tracked
`ops_before_optimisation.rs`, which is *not* the file being written, so no run can
be an identity there; and it is therefore reproducible on a clean host.

Two things to keep avoiding. Do not point the script at
`rust/inmotion-core/src/ops.rs` inside the working tree: it writes its target in
place, so that would edit the checkout. And do not name the target as the
baseline — that is the identity shape above. The script makes the first one
impossible rather than merely warned about: there is no implicit target and no
default destination from any directory, so a bare
`python3 results/perf/make_before_kernels.py` exits non-zero — from the repository
root, from a scratch tree, from anywhere — instead of reverting
`./rust/inmotion-core/src/ops.rs`, and a named target inside this repository is
refused unless `--in-repo` says you mean it. Dropping one argument from a
documented command is therefore a refusal, not an in-place write. To produce the
`before` source without touching the tree, read the checkout and write elsewhere
with `--out`, which is the explicit route:

    python3 results/perf/make_before_kernels.py --out /tmp/ops-before.rs

Running it over a file that already carries those bodies is an identity — it
prints `identity: already reverted` for each item and `nothing changed`, which
`--check` reports without writing anything (exit 0 when the target is already
pre-optimisation, 1 when the optimised bodies are still there):

    python3 results/perf/make_before_kernels.py --check \
        /tmp/before-build/rust/inmotion-core/src/ops.rs     # exits 0 after the recipe

**The script must never target a path inside the working tree.** It rewrites its
target in place, so one run with this checkout's own
`rust/inmotion-core/src/ops.rs` as the target silently converts the tree to the
pre-optimisation bodies — which is exactly what happened on 2026-09-12, and it is
worse than it sounds: the reviewed revision of that file was uncommitted, so a
`git checkout`/`git restore` afterwards appears to "fix" it while in fact
restoring `HEAD`, destroying the repair, and leaving the tree clean so no dirty
worktree check notices. `--out` to a scratch path is mandatory for producing the
`before` source; treat an in-place run inside this checkout as forbidden. The
pre-t13 script had no explicit target and defaulted to `./rust/inmotion-core/src/ops.rs`
under whatever directory the caller stood in, which is the hazard that fired; the
current script refuses that shape (no implicit target, and an in-repo in-place
target requires `--in-repo`).

The script replaces exactly four items and reports each one —
`impl PackedWeight`, `linear_packed`, `self_attention`, `conv1d_same`; the old and
new line counts it prints are the evidence (18 -> 13, 34 -> 32, 83 -> 59, 74 -> 45
on the tree this was measured against). The resulting `ops.rs` is
`sha256 a7fadc8056deabfc…`, not the tree's `92963c1f7e16bbfa…`. That build
reproduces the recorded figures: re-confirmed against the restored revision, a
fresh copy of the tree at `sha256 92963c1f7e16bbfa…` produces
`a7fadc8056deabfc…` (148 lines different from the tree, not an identity) and, built,
measures 4.4306 / 4.3491 ms on lejepa fp32 against the reference `before` build's
4.3368 / 4.9450 ms and the optimised build's 0.7684 / 0.8554 ms in the same
session; `--check` exits 1 on the optimised source and 0 on the reverted one, and
parity is `max |diff| 4.05e-6`. Quiet runs land lower (4.3811 and 4.2832 ms) and
loaded ones higher, against the committed `before` 4.3444 ms. That is the
pre-optimisation kernel, not a mislabelled optimised one: the optimised build
measures 0.77-0.86 ms on the same machine in the same session.

The four bodies that travel are pinned by their own hashes, so a fixture whose
content drifted cannot pass unnoticed even while the line counts still look
right. Each hash is the sha256 of the block's text exactly as it appears in
`ops_before_optimisation.rs`, newlines included, the block running from the first
line that starts with the opener through the first later line whose stripped
content is exactly `}`:

| body | lines | sha256 in `ops_before_optimisation.rs` |
|---|---|---|
| `impl PackedWeight` | 13 | `44362bac3bbf22a52c150e0f9c0bd2e62f884620132d1e33e898460a5c5da891` |
| `pub fn linear_packed` | 32 | `d6b7402438ef3523903655979f30dc30659bba420bfb0f017f973c93bcc7bdae` |
| `pub fn self_attention` | 59 | `44f8cb58990ea2792dc2e859a5f3afb98f3414cb6a2c89cc899e745f8df544f7` |
| `pub fn conv1d_same` | 45 | `5413e90d425e4620c0234f5c36f16f24c46152afa8484d0ea3add2ff393933d7` |

The rule is code, not prose: those are exactly the `OPEN` entries and the
`blocks()` scan whose opener match is `line.rstrip("\n").startswith(open_text)`
and whose close match is `lines[j].rstrip("\n") == "}"`, in
`make_before_kernels.py` at `sha256 9a8e4a9934e73e6f144087b1a55c246e8f1b57834122d88e89f3d57b430f90b6`
(import it and call `blocks()` on the fixture to re-derive the spans; `main()` is
`__main__`-guarded, so importing writes nothing). The fixture itself is
`sha256 9f6b8ab5c1510d05c4998fa966827dc95f93a4d508d7dc7b5725fc6322432c6a`; if the
script or the fixture hash has moved, re-derive the four numbers above before
citing them.

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

Benchmark, both machines (this is `run_bench.sh`; both branches write the same
11 columns, so `summarise.py` merges them):

    sh run_bench.sh host      # taskset -c 6, 300 iterations + 40 warmup, 3 reps
    sh run_bench.sh router    # 200 iterations + 30 warmup, 3 reps

`run_bench.sh` benchmarks the pair named by `BEFORE` and `AFTER`; it does not
build them. `AFTER` is the tree built above (`rust/target/release/inmotion-infer`
on the host, `rust/target/x86_64-unknown-linux-musl/release/inmotion-infer` for
the router) and `BEFORE` is the sibling gnu binary from "The `before` build".
The full set is `BEFORE`, `AFTER`, `ART`, `OUT`, `REPS`, `ITER` and `WARM`, all
overridable by environment variable, so the repaired build's re-run is:

    BEFORE=/tmp/perf/before2-gnu AFTER=/tmp/perf/after2-gnu sh run_bench.sh host

Deploy (scp is blocked on the router, so `tar` over `ssh`):

    tar czf deploy.tgz -C /tmp/art . -C /tmp/perf before-musl after-musl
    cat deploy.tgz | ssh wave-router 'cd /mnt/extra/cargo/inmotion/perf && tar xzf -'

The router writes its CSV on the device; the copy already there is byte-identical
to `/tmp/art` (checked per artifact with `md5sum`), so the two machines measured
the same weights. Fetch it back over `ssh`/`cat` — `base64` and `scp` are both
missing on busybox, and `tar` over the pipe is for uploading only.

Reduce:

    python3 summarise.py

## Files

| file | what |
|---|---|
| `run_bench.sh` | reproduces the raw CSVs on either machine; `host` or `router` branch. Benchmarks the pair in `BEFORE`/`AFTER` (host: gnu, defaults `/tmp/perf/before-gnu` and `/tmp/perf/after-gnu`; router: musl, defaults `./before-musl` and `./after-musl`) — build those first, see "The `before` build" |
| `make_before_kernels.py` | applies the four pre-optimisation bodies from `ops_before_optimisation.rs` to a copy of `ops.rs`, making the `before` build reproducible with no out-of-band input. The target is always explicit: `--out FILE` reads the tree's `ops.rs` and writes elsewhere, a positional target is reverted in place, and with neither the script exits non-zero instead of guessing. A target inside this repository is refused unless `--in-repo` says you mean it |
| `ops_before_optimisation.rs` | the tracked fixture the script reads (the four pre-optimisation bodies, byte-identical to the retained original, permanently in-repo); the `before` source it produces is `ops.rs` with those four bodies reverted, `sha256 a7fadc8056deabfc…` |
| `ops_retained_original.rs` | archival witness for provenance, not a build input: the retained pre-optimisation tree's `ops.rs`, kept as the independent reference for checking that fixture (durable copies at `~/inmotion-recovery/ops.rs.70c19066` and in this file's payload), `sha256 70c190665f73f7a0a680c5c1aeb6860ba1513be228266455c8832fa8c40f2f7a`, 31043 bytes. It is not the committed fixture and it is not what the build consumes; the four pre-optimisation bodies it contains are preserved and verified byte-identical to the fixture's (concatenated in item order: `sha256 489755f8c50c77f2939974a43531844263c842f009a79e117aa7c7fbb0871755`), and since it is a whole stale tree while the fixture is today's tree with only those bodies reverted, the fixture is the one `run_bench.sh`'s `BEFORE` side is built from |
| `bench_host.csv` | raw per-run rows, dev machine (before x after, 3 reps) |
| `bench_router.csv` | raw per-run rows, wave-router |
| `summary.csv` | before/after median and p95 per device/model/precision |
| `crate_survey.csv` | 3 machines x 4 shapes x 7 kernels, incl. bit-exactness |
| `crate_survey_*.csv` | the unmerged per-machine captures |
| `summarise.py` | raw CSVs -> `summary.csv` |
