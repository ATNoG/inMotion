# Rust inference and other devices

How to take a trained world model off this machine: export it, build the Rust
runtime for the target, prove the port reproduces PyTorch **on that target**, and
measure it there. Everything on this page is a command you can run; the measured
numbers behind the design are in `results/perf/README.md` and
`rust/README.md`.

The Rust path exists so inference needs no Python. The ESP32 is a separate C
firmware under `firmware/` and does **not** use this crate — see
`docs/reports/ESP32_FEASIBILITY.md`.

---

## 1. What is being evaluated

Three different questions, three different commands. Do not conflate them.

| question | command | answer |
|---|---|---|
| Does the Rust port compute what PyTorch computes? | `inmotion-infer check` | max absolute deviation, against a recorded tolerance |
| Is the quantized model still accurate? | `inmotion quantize eval` | MCC per model × precision, on `data/processed/dataset.csv` |
| How fast is it **on the target**? | `inmotion-infer bench` | min / median / mean / p95 / max and throughput |

**Models** — the five families served by this crate:

| model | shape |
|---|---|
| `lejepa` | pre-norm transformer + attention pool; per-timestep linear tokenizer |
| `t_jepa` | adds `[REG]` slots and an input projection before the transformer |
| `ts_jepa` | `Conv1d` patch projection tokenizer |
| `cf_jepa` | same stack, different tokenizer/tail |
| `sigreg` | multi-scale CNN with a Gaussian latent bottleneck — shares `ops.rs` and nothing else |

**Precisions** — list them with `uv run inmotion quantize formats`:
`fp32`, `fp16`, `bf16` (cast-only), and `int8`, `q8`, `q6`, `q5`, `q4`, `q3`,
`q2` (different group sizes). The measured sweeps use `fp32`, `int8` and `q4`.

**Dataset** — quantization accuracy is measured on **one** dataset,
`data/processed/dataset.csv` (registry id `ds-base-v2`). The harness refuses to
silently accept a different one: `quantize eval` re-checks its fp32 result
against `BASELINE_MCC` in `src/inmotion/quant/evaluate.py` and fails if the
baseline does not reproduce. The ICAISF lineage scores differently and is a
different experiment.

Baseline fp32 test MCC on `dataset.csv`:

| model | MCC |
|---|---:|
| `lejepa` | 0.9143 |
| `ts_jepa` | 0.9194 |
| `cf_jepa` | 0.9083 |
| `t_jepa` | 0.8982 |
| `sigreg` | 0.6333 |

---

## 2. Produce an artifact (Python side)

An artifact is a flat directory the Rust runtime reads. It is the only interface
between the two languages.

```bash
# fp32 — no quantization
uv run inmotion quantize export lejepa --out /tmp/art-lejepa-fp32 --format fp32

# int8, and the default also writes golden vectors for the parity check
uv run inmotion quantize export lejepa --out /tmp/art-lejepa-int8 --format int8

# q4 with a tighter tolerance recorded for the check
uv run inmotion quantize export lejepa --out /tmp/art-lejepa-q4 --format q4 --tol 1e-4
```

Resulting layout — three files, and nothing else:

```
/tmp/art-lejepa-q4/
  meta.json           architecture, hyperparameters, quantization spec, provenance
  model.safetensors   the weights
  golden.json         N real test windows + the outputs PyTorch produced for them, and `tol`
```

`--golden` defaults to on. **Keep it.** Without `golden.json` there is nothing to
`check` against on the target, and the port becomes unverifiable there.

Useful companions:

```bash
uv run inmotion quantize formats                      # every scheme, with its granularity
uv run inmotion quantize show lejepa --format q4      # size and error for one combination
uv run inmotion quantize eval                         # full sweep -> results/quant/sweep_dataset.csv
uv run inmotion quantize stats                        # the tables the figures read
uv run inmotion plots quantization                    # rebuild the figures from those tables
```

`quantize eval` takes `--models`, `--formats`, `--data`, `--seed` and `--out`;
`quantize stats` adds `--sweep`, `--out-dir`, `--latency-iters` and
`--latency-warmup`. Defaults are all five models, all formats.

---

## 3. Build the runtime for the target

The crate is `rust/inmotion-cli`, and the binary is `inmotion-infer`.

```bash
cd rust

# same machine
cargo build --release -p inmotion-cli
# -> rust/target/release/inmotion-infer

# another x86-64 Linux, static: this is what the wave-router J3455 runs
cargo build --release --target x86_64-unknown-linux-musl -p inmotion-cli
# -> rust/target/x86_64-unknown-linux-musl/release/inmotion-infer
```

**Use the baseline feature level.** There is no `target-cpu`, `target-feature` or
`RUSTFLAGS` anywhere in the workspace, on purpose: the router has no AVX, AVX2 or
FMA, and a binary built with them crashes with `SIGILL` there. Build for the
lowest target you intend to run on, not for this machine.

Add the target once if cargo does not know it:

```bash
rustup target add x86_64-unknown-linux-musl
```

`libm` supplies `erf`/`exp`/`sqrt` so results are identical across targets and
libcs rather than drifting with the platform — that is what makes the parity
numbers reproducible instead of approximate.

---

## 4. The four runtime commands

```
inmotion-infer describe --artifact DIR
inmotion-infer predict  --artifact DIR --input FILE
inmotion-infer bench    --artifact DIR [--iterations N] [--warmup N] [--json]
inmotion-infer check    --artifact DIR --golden DIR
```

`--artifact` is required for all four. (The usage banner printed by a bare
invocation lists only three; `bench` is real and is what the benchmark script
calls.)

**`describe`** — what the artifact declares, with no inference:

```
model        : lejepa
artifact     : /tmp/art-lejepa-q4
tensors      : 79
family       : transformer
seq_len      : 10
in_channels  : 4
num_classes  : 4
d_model      : 128
n_heads      : 4
num_layers   : 4
n_tokens     : 10
n_reg        : 0
tokenizer    : linear (per timestep)
tail         : proj + gelu + batchnorm
quantization : {"axis":0,"bits":4,"granularity":"per_group","group_size":32,...}
```

The lines between `family` and `quantization` are the hyperparameters that shape
this family's graph; another family reports its own.

Use it first on a new device: it loads and validates the artifact, rejecting one
whose declared architecture disagrees with its weights, and it is the fastest way
to tell "wrong artifact" from "wrong device".

**`predict`** — one window in, one classification out:

```bash
inmotion-infer predict --artifact /tmp/art-lejepa-q4 --input window.txt
```

Input is plain text: floats separated by any whitespace or commas, read in row
order, `seq_len × in_channels` of them. Non-finite values (`nan`, `inf`) are
rejected rather than allowed to poison the comparison. Output:

```
logits     : [-0.6575064, 1.9401256, -0.93640953, -0.98155904]
probabilities: [0.062846765, 0.844151, 0.047550686, 0.045451537]
predicted  : class 1 (confidence 0.844151)
```

**`bench`** — latency on this machine, default 200 iterations plus 20 warmup:

```bash
inmotion-infer bench --artifact /tmp/art-lejepa-q4 --iterations 300 --warmup 40
inmotion-infer bench --artifact /tmp/art-lejepa-q4 --json     # for scripts
```

Human output reports min, median, mean, p95, max and throughput. `--json`
emits `min_ms`, `median_ms`, `mean_ms`, `p95_ms`, `max_ms`,
`throughput_per_s` and the artifact's model and quantization — the shape every
consumer of these numbers expects.

**`check`** — the parity harness, and the one command that must be run **on the
target** before you trust it:

```bash
inmotion-infer check --artifact /tmp/art-lejepa-q4 --golden /tmp/art-lejepa-q4
```

It replays the golden windows, recomputes them with the artifact's own quantized
weights and reports:

```
cases      : 16
max |diff| : 0.00000412
tolerance  : 0.000200
PARITY OK
```

A deviation over `tol` prints the offending case and exits non-zero. `tol` comes
from `golden.json`, which the exporter wrote — so a `golden.json` with the field
removed is an error, not a silent pass.

---

## 5. Move it to another device

Copy **two** things: the binary and the artifact directory. Nothing else is
needed — no Python, no shared libraries, no data.

```bash
# on this machine
cd rust
cargo build --release --target x86_64-unknown-linux-musl -p inmotion-cli
tar czf /tmp/inmotion-runtime.tgz \
    -C rust/target/x86_64-unknown-linux-musl/release inmotion-infer
tar czf /tmp/inmotion-artifacts.tgz -C /tmp art-lejepa-q4
```

`scp` is blocked on the wave-router and it has no `base64`, so transfer with
`tar` over `ssh`:

```bash
ssh wave-router 'mkdir -p /mnt/extra/cargo/inmotion/perf'
tar czf - -C rust/target/x86_64-unknown-linux-musl/release inmotion-infer \
  | ssh wave-router 'tar xzf - -C /mnt/extra/cargo/inmotion/perf'
tar czf - -C /tmp art-lejepa-q4 \
  | ssh wave-router 'tar xzf - -C /mnt/extra/cargo/inmotion/perf'
```

Then, **on the device**, prove it before measuring it:

```bash
ssh wave-router 'cd /mnt/extra/cargo/inmotion/perf && \
  ./inmotion-infer describe --artifact art-lejepa-q4 && \
  ./inmotion-infer check    --artifact art-lejepa-q4 --golden art-lejepa-q4'
```

A `PARITY OK` there is the claim that matters: the binary reproducing PyTorch on
*your* machine says nothing about the J3455.

---

## 6. Measure, and keep the measurement honest

One-off:

```bash
./inmotion-infer bench --artifact art/lejepa-q4 --iterations 200 --warmup 30 --json
```

The full before/after grid — five models × three precisions × both builds — is
scripted:

```bash
# on the development machine
sh results/perf/run_bench.sh host      # -> results/perf/bench_host.csv

# on the router, from its deploy directory
sh results/perf/run_bench.sh router    # -> results/perf/bench_router.csv
```

Two design points in that script are the reason its numbers are usable:

* **`before` and `after` are interleaved**, not run in separate batches, so
  machine load drifts cancel instead of landing entirely on one side. Each cell
  is measured `REPS` times (default 3) and `summarise.py` keeps the best run.
* **`before` is rebuilt from this tree**, not a saved binary: the four optimised
  kernels in `rust/inmotion-core/src/ops.rs` are reverted to their original loop
  shapes by `results/perf/make_before_kernels.py` against the tracked baseline
  `results/perf/ops_before_optimisation.rs`.

That script rewrites a source file in place, so it carries a guard: it refuses to
write inside a checkout unless you pass `--in-repo`, and it refuses a bare
invocation outright. The safe forms are a scratch target or `--out`:

```bash
# out of tree
python3 results/perf/make_before_kernels.py --out /tmp/perf/before/ops.rs

# or a scratch copy of the tree, in place
python3 results/perf/make_before_kernels.py /tmp/before-build/rust/inmotion-core/src/ops.rs
```

Writes into the checkout itself need `--in-repo` **and** mean it: the reviewed
`ops.rs` is uncommitted work, and one accidental in-place run is the incident
that guard exists for. `tests/test_make_before_kernels_guard.py` locks the
behaviour.

The development machine is an i5-1135G7 (AVX2 + FMA) and the router is a Celeron
J3455 at 1.5 GHz with none of those. That difference is the point: `gemm` beats
the hand-written kernel on the i5 and loses badly on the J3455, so **a kernel
chosen from development-machine numbers would be the wrong kernel for the
device.** Benchmark on the target, or not at all.

---

## 7. Adding a new device

1. Check its libc and ISA. No AVX/AVX2/FMA → baseline build, and the musl target
   if it is not glibc.
2. Build for that target (`§3`) and transfer the binary plus one artifact (`§5`).
3. `describe` — the artifact loads and declares the architecture you expect.
4. `check` — `PARITY OK` **on that device**. Do not proceed without it.
5. `bench --json` — record the result with the artifact hash and the git revision
   of the binary's source, or the number will not be reproducible later.
6. Only then add it to the sweep.

---

## 8. The rest of the system

This page covers the Rust/device path only. For everything else:

| need | start at |
|---|---|
| every command, with a realistic invocation | [`commands.md`](commands.md) |
| every argument, option and default | [`../reference/cli.md`](../reference/cli.md) |
| using a checkpoint: list, inspect, verify, predict, adopt | [`running-models.md`](running-models.md) |
| training each family | [`training.md`](training.md) |
| what is and is not reproducible | [`../reproducibility.md`](../reproducibility.md) |
| what a reviewer can independently verify | [`../audit/AUDIT_CHECKLIST.md`](../audit/AUDIT_CHECKLIST.md) |

The shortest useful entry points:

```bash
uv run inmotion doctor              # environment and artifact store, before anything else
uv run inmotion model list          # what is registered
uv run inmotion model verify        # strict-load everything, check parameter counts
uv run inmotion datasets verify     # every dataset against its recorded checksum and row count
uv run inmotion model predict NAME --reference data/processed/dataset.csv --input windows.csv
uv run inmotion evaluate checkpoints   # per-class metrics for saved checkpoints
```

`model verify` and `datasets verify` are the two cheapest ways to tell a real
failure from an environment problem; run them before debugging anything else.
