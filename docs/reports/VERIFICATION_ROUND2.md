# Verification report, round 2 (inmotion-edge-close, task `t5`)

Independent verification performed by `verifier` on this checkout. Every number
below was produced by a command I ran myself in this session; nothing is quoted
from another member's output, and no number is carried over from
`docs/reports/VERIFICATION.md`. I edited nothing under `src/`, `rust/`,
`firmware/` or `results/`; the only file I wrote is this report.

## Verdict

**F1, F2, F3, F4, F5, F6, R2-01 and both round-2 open items are genuinely
closed**, each by re-running the demonstration that produced it. The two
harness fixes (`R2-01`, silent `tol`) were verified causally, not by reading the
diff: I rebuilt the same revision *without* the two guards and watched the old
failure come back. Parity is 15/15 on both binaries and every number is
unchanged from the round-2 record. A clone outside this working copy regenerates
the figures and the stats tables byte-for-byte.

Three new findings came out of the verification, none of which reopens a closed
item. The one that matters is **F8 (medium)**: `results/perf/README.md` — a file
that ships in this closure — asserts two things about the *committed* benchmark
tables that `git` contradicts. Its description of the harness bugs is accurate
and their fix is real; the conclusions it draws about the committed CSVs are
not. **F7 (low)** and **F9 (low)** are a wrong-machine row in
`stats_latency.csv` and an understated host-delta count in the same README. A
fourth, **F11 (low)**, is that the README's reproducible `before`-build script is
untracked (§9.6).

§9 answers the captain's follow-up: the two script bugs are real and I re-derived
them by **executing** the committed script (its output does not match the
committed data in either branch), the committed `bench_router.csv` came from the
device-local `router_bench.sh` while the committed `summary.csv` was genuinely
merged from both bench files by the committed `summarise.py` — so the committed
data is sound and the README's "router-only, 15-cell" conclusion is wrong. Note
that the captain's earlier message repeated that conclusion; the captain has
since confirmed it was a false premise, and §9 is the evidence trail either way.

## 1. Enforcement gates (all re-run here)

| gate | command | result |
|---|---|---|
| clippy | `CARGO_TARGET_DIR=/tmp/verif2-clippy cargo clippy --all-targets -- -D warnings` | **exit 0**, no warnings, from an *empty* target dir |
| tests, gnu release | `cargo test --release` | **50 passed; 0 failed**, exit 0 |
| tests, musl release | `cargo test --release --target x86_64-unknown-linux-musl` | **50 passed; 0 failed**, exit 0 |
| models | `uv run inmotion model verify` | `All 18 model(s) verified.` exit 0 |
| datasets | `uv run inmotion datasets verify` | `All 7 dataset version(s) match the manifest.` exit 0 |
| rust-port export tests | `uv run pytest tests/test_quant_export.py -k rust_port -v` | **15 passed**, 24 deselected, exit 0 |

`#[test]` attributes in the tree total 50 (`ops.rs` 24, `quant.rs` 13,
`world_model.rs` 5, `tensor.rs` 4, `artifact.rs` 2, `sigreg.rs` 2), matching the
round-2 count — no test was deleted to make a finding go away. The three tests
round 2 added are present by name: `metadata_len_matches_the_granularity`,
`metadata_len_rejects_declarations_it_cannot_honour`
(`artifact.rs:425,453`) and
`conv1d_same_skips_taps_that_fall_entirely_in_the_padding` (`ops.rs:570`).

### The kernel bodies did not move

I re-hashed every reviewed source file against the revision recorded in
`CODE_REVIEW_RUST_ROUND2.md`:

| file | sha256 vs round-2 record |
|---|---|
| `inmotion-core/src/artifact.rs` | **unchanged** `2bb3943fdf48b8e2` |
| `inmotion-core/src/lib.rs` | **unchanged** `efa9b78e30a2e9c0` |
| `inmotion-core/src/model.rs` | **unchanged** `e406be1dba877378` |
| `inmotion-core/src/ops.rs` | **unchanged** `92963c1f7e16bbfa` |
| `inmotion-core/src/quant.rs` | **unchanged** `e60c5ff0fe0a5386` |
| `inmotion-core/src/sigreg.rs` | **unchanged** `d7e3f1de3c8eba2f` |
| `inmotion-core/src/tensor.rs` | **unchanged** `8d4a8998e0cf16da` |
| `inmotion-core/src/world_model.rs` | **unchanged** `a684b55374da2600` |
| `inmotion-cli/src/main.rs` | changed — the two `t1` guards |

Eight of the nine reviewed files are byte-identical to the revision round 2
verified; the CLI is the only one that moved, and it moved only by the two
guards below. Nothing the benchmark measured was touched.

## 2. Parity, re-derived on both binaries (15/15 each)

I re-exported all fifteen artifacts myself
(`uv run inmotion quantize export <model> -o <dir> -f <fmt>`, `--golden` default
on) into `/tmp/verif2/art`, rebuilt both binaries from this tree, and replayed
every cell with `inmotion-infer check --artifact <d> --golden <d>`:

| model | fp32 | int8 | q4 | cases | vs round-2 record |
|---|---|---|---|---|---|
| lejepa | 4.05e-6 | 4.23e-6 | 3.34e-6 | 16 | identical |
| t_jepa | 2.62e-6 | 4.14e-6 | 6.20e-6 | 16 | identical |
| ts_jepa | 2.74e-6 | 2.92e-6 | 5.01e-6 | 16 | identical |
| cf_jepa | 2.21e-6 | 2.95e-6 | 1.91e-6 | 16 | identical |
| sigreg | 8.30e-7 | 8.30e-7 | 9.50e-7 | 16 | identical |

All 15 cells `PARITY OK`, exit 0, on **both** the gnu binary
(sha256 `e2a8b4d4fe4313792f29c8bfa10c8144e581ade2f124df97e62571b3984cbcb3`) and
the musl/router binary (`2c326abee708b83743d4a1f192183d0e78223f302ef6f82c82ed89d09b9d7a77`).
Every `golden.json` carries `tol = 0.0002` and the checker's own default is 2e-4,
so no tolerance was widened. sigreg's tightened post-retrain row
(8.3e-7 / 8.3e-7 / 9.5e-7) is reproduced, not accepted.

## 3. Per-finding verdicts

Each row is a demonstration I re-ran, with its exit code.

### F1 (medium) — ESP32 clock evidence — **CLOSED**

*Re-measured on the device.* `/dev/ttyUSB0` is attached. I drove the firmware
myself:

```
sg dialout -c 'uv run --extra esp python firmware/tools/run_on_device.py \
  --port /dev/ttyUSB0 --cases /tmp/lejepa_int8/cases.tsv \
  --out /tmp/verif2/esp32_verify_240mhz.json --repeats 3'
-> runs=48 clock=240 MHz median=636.686 ms ... exit 0
```

| field | my run | committed `esp32_lejepa_240mhz.json` | delta |
|---|---|---|---|
| `clock_mhz` | 240 | 240 | 0 |
| `clock_hz` | 240000000 | 240000000 | 0 |
| `median_ms` | 636.686 | 636.6805 | 0.0055 ms (8.6 ppm) |
| `min_ms` / `max_ms` | 630.903 / 640.152 | 630.897 / 640.162 | ≤0.01 ms |

The clock is now an artifact field, not a filename: both `esp32_*.json` carry
`clock_mhz`, `clock_hz` and `clock_source` — the device's own boot line
(`I (426) cpu_start: cpu freq: 240000000 Hz`). The boot-log parser is pinned by
`run_on_device.py --selftest` (exit 0, 4 cases).

*The filename is not consulted any more.* `_clock_mhz(path)` is gone;
`_clock_label(payload)` reads `clock_mhz` and raises when it is absent
(`quant_figures.py:190`). I tested that adversarially by copying the artifacts
under **swapped** names:

```
filename=esp32_lejepa_160mhz.json  payload clock_mhz=240  label='240 MHz'
filename=esp32_lejepa_240mhz.json  payload clock_mhz=160  label='160 MHz'
```

The labels follow the payload, so a rename can no longer move a clock label. A
stray clockless `esp32_*.json` makes the pipeline **fail loudly** instead of
plotting "unknown clock":

```
ValueError: measurement has no clock_mhz field; re-measure with
firmware/tools/run_on_device.py so the artifact names its own clock
```

The byte-identical **unclocked** duplicate is gone: only
`esp32_lejepa_160mhz.json` and `esp32_lejepa_240mhz.json` exist, and
`fig5_latency_rows.csv` labels them "160 MHz" / "240 MHz".

*Not re-measured by me:* **the 160 MHz row.** Re-measuring it means building and
flashing a `CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_160` image and restoring the 240 MHz
one, which I did not do. I verified that row as an *artifact whose internal
clock field is present and cross-checked*: `clock_mhz = 160` with the 160 MHz
boot line, median 888.4805 ms, and its md5 `f7c9f273503e76b6b1639e9d0a08f6af`
matches the `clock_provenance` block in the tracked
`firmware/evidence/device_measurements.json` exactly. Its value also reproduces
the earlier tracked record (888.4765 ms) to 0.004 ms. That is internal
consistency, not a device re-run, and is reported as such.

#### Correction: round-1 F1's *diagnosis* was itself wrong

The captain's guidance, and my own pre-fix evidence, agree that
`docs/reports/VERIFICATION.md` F1 mis-states the defect. F1 says the file named
`..._240mhz.json` "carries the 160 MHz run". The pre-fix state I captured at the
start of this session says otherwise:

| pre-fix file | median | md5 |
|---|---|---|
| `esp32_lejepa_240mhz.json` | 636.6825 ms | `722c5e344eda94828a4fd18bf92d5c59` |
| `esp32_lejepa_160mhz.json` | 888.4765 ms | `d638d2b2249cff7b757ebb2b24cfb41c` |
| `esp32_lejepa.json` | 888.4765 ms | `d638d2b2249cff7b757ebb2b24cfb41c` |

636.68 ms *is* the 240 MHz number and 888.48 ms *is* the 160 MHz number (the
device's own boot log, the tracked `device_measurements.json`, `ESP32_FEASIBILITY.md`
and my own re-measurement all agree). So **the two clocked filenames matched their
contents**; there was no swap, and F1's "rename the two JSONs so each name matches
the run it contains" was aimed at a problem that did not exist.

What was genuinely broken, and is what `t2` fixed, was narrower:

1. `esp32_lejepa.json` was byte-identical to `esp32_lejepa_160mhz.json`, and
   `sorted()` puts `esp32_lejepa.json` *first* (`.` < `_`), so the old
   first-seen-wins dedup kept the **clockless** name and discarded the clocked
   one — which is why the 888.48 ms bar rendered "unknown clock".
2. `_clock_mhz()` derived the clock from the filename stem at all.

Both are closed: the duplicate is deleted, the clock is an artifact field read
from the payload, and the label path no longer consults a filename. F1's
*required fix* was still directionally right — its "or re-measure and write both
with the clock taken from the device" branch is exactly what closed it — but its
diagnosis is superseded by this section. I did not rewrite
`docs/reports/VERIFICATION.md`; this report supersedes it deliberately.

### F2 (medium) — figure inputs gitignored — **CLOSED**

The three device artifacts are now tracked
(`results/quant/edge_router_j3455.json`, `esp32_lejepa_160mhz.json`,
`esp32_lejepa_240mhz.json`, staged as `A`).

*Defect reproduced.* In a clone outside this working copy, with the three JSONs
removed, `plots quantization` fails with an actionable error instead of the bare
OS error of the original finding:

```
uv run inmotion plots quantization
-> FileNotFoundError: missing results/quant/edge_router_j3455.json;
   run `sh run_bench.sh router`                                  exit 1
```

*Closure proven.* With them present, in the same external clone (see §4 for the
clone construction and the one precondition), `uv run inmotion plots
quantization` → **exit 0**, and the regenerated figures are byte-identical:
**18 of 20** compared outputs match the committed files exactly — all 6 PDFs, all
6 PNGs, `figures/data/fig1_device_budgets.csv`,
`figures/data/fig5_latency_rows.csv`, and `stats_size/accuracy/coverage/layers.csv`.
The two that differ are the two the manifest itself declares cannot be
byte-stable:

- `stats_latency.csv` — `tables.latency.md5` is `null` with `varying:
  "machine-timed rows; regenerate with this command"`.
- `stats_manifest.json` — differs only in `/cwd`, `/argv` and `/python`
  (`3.13.7` → `3.13.9`); every table md5, row count, input md5 and
  `latency_artifacts` entry is identical.

### F3 (low) — over-broad AVX claim — **CLOSED**

`results/perf/README.md:202-217` and `docs/reports/CODE_REVIEW_RUST.md:95-97,446-448`
now say no `inmotion` code path emits AVX/AVX2/FMA and that the binary's only AVX
is `memchr`'s runtime-CPUID-dispatched `find_avx2`. I re-measured it myself on
the musl binary:

```
objdump -d --no-show-raw-insn <musl bin> | grep -cE '^\s+[0-9a-f]+:\s+v[a-z0-9]+'
282
attributed by enclosing symbol:
   250  <memchr::arch::x86_64::memchr::count_raw::find_avx2>
    27  <memchr::arch::x86_64::avx2::memchr::One::rfind_raw_avx2>
     5  <memchr::arch::x86_64::memchr::memrchr_raw::find_avx2>
inmotion VEX symbols: 0
```

Exactly 282, all in those three symbols, none in an `inmotion*` symbol — the
qualified claim is the measured one.

### F4 (low) — stale parity quotes — **CLOSED**

The pre-retrain sigreg and cf_jepa figures no longer stand unqualified:

- `docs/reports/CODE_REVIEW_RUST.md:76-82` labels its block "the figures as
  measured in this round-1 review, against the artifacts that existed then",
  names the retrain, gives the current values and points at `rust/README.md` as
  the live record.
- `docs/reports/REFACTOR_SUMMARY.md` no longer carries them (repo-wide sweep for
  `1.23e-6|1.97e-6|1.43e-6|2.15e-6|2.86e-6|1.67e-6` returns only
  `results/perf/README.md:184`).
- `results/perf/README.md:167-177` captions that remaining table as "the record
  of the kernel before/after comparison, so it keeps the checkpoint it was
  actually measured against" and points at `rust/README.md`.

The live numbers were not taken on trust: my own parity run reproduces
`sigreg 8.3e-7 / 8.3e-7 / 9.5e-7` and `cf_jepa 2.21e-6 / 2.95e-6 / 1.91e-6`
exactly (§2).

### F5 (low) — manifest provenance — **CLOSED**

`stats_manifest.json` now records `tables.latency = {"md5": null, "varying":
"machine-timed rows; regenerate with this command"}`, keeps md5s for the other
four tables, and lists all three device inputs in `latency_artifacts`. I checked
the md5s are not decorative: `size`, `accuracy`, `coverage`, `layers` all
**MATCH** their files on disk, every recorded row count matches the file, and both
`inputs` md5s match.

The declaration is honest rather than static: in a clone where I removed the
device artifacts, `latency_artifacts` came back as `[]`, not as a list of three
files that were never read.

*Reproducibility of the table's device content.* Regenerating in the external
clone yields 9 rows with the same `(device, model, precision)` key sequence as
the committed table, and **all four device-derived rows match exactly** in
`median_ms`, `p95_ms`, `min_ms`, `max_ms` and `throughput_per_s`. Only the five
`--latency-iters` torch rows vary — which is precisely what the manifest now
declares.

### F6 (low) — checkpoint count — **CLOSED**

`uv run inmotion model verify` → `All 18 model(s) verified.`, exit 0. The count
is 18.

### R2-01 (low) — wrong-length golden *input* panicked — **CLOSED**

I rebuilt the failing input from the golden fixture rather than reading the
diff (`golden.json`: `seq_len=10`, `in_channels=4`, expected input 40 values),
in two shapes: every row truncated to 2 of 4 channels (20 values), and only case
3 shortened by one value (39 values).

```
gnu  & musl:  --golden shortA -> exit 1  error: case 0: input has 20 values, expected 40
gnu  & musl:  --golden shortB -> exit 1  error: case 3: input has 39 values, expected 40
unmodified golden           -> exit 0  PARITY OK  (control still passes)
```

**The fix is causal, not incidental.** I copied the tree outside the repo
(`/tmp/verif2/prefix-rust`), removed only the input-length guard, rebuilt, and
ran the same fixture:

```
thread 'main' panicked at inmotion-core/src/tensor.rs:24:9:
assertion `left == right` failed: data length 20 does not match shape [10, 4] (40 elements)
  left: 20
 right: 40                                        exit 101
```

The named-case error with exit 1 is produced by the guard and nothing else.

### Round-2 open item 1 — silent `tol` default — **CLOSED**

Same method. With `tol` deleted from `golden.json`:

```
gnu & musl -> exit 1  error: golden.json: no tol (the exporter writes one per artifact)
```

and the guard-removed rebuild prints the old behaviour exactly:

```
cases : 16   max |diff| : 0.00000423   tolerance : 0.000200   PARITY OK   exit 0
```

so a golden with the field deleted can no longer pass. No real golden is
rejected: the exporter writes `tol`, and all fifteen of mine carried it.

### Round-2 open item 3 — stale benchmark-build hash — **CLOSED**

`results/perf/README.md:219-231` now says the md5/sha256 identify the
**benchmark build** that produced the committed numbers and ran on the J3455, not
the current tree. The qualification is necessary and sufficient: I measured the
current musl binary at sha256 `2c326abee708…`, which is *not* the quoted
benchmark build `29cd6de5…`, so a reader is no longer invited to check the wrong
hash.

### `t4`'s claim that the committed benchmark numbers survive the repair — **VERIFIED on the router**

`after2-musl` on the router is sha256 `2c326abee708b837…` — **byte-identical to
the musl binary I built here** — so my parity run in §2 used the repaired
benchmark build.

I re-ran four cells on the device myself (100 iterations, 20 warmup) with
`before2-musl` / `after2-musl`:

| cell | committed before→after | my before→after | committed x | my x |
|---|---|---|---|---|
| lejepa fp32 | 8.7855 → 2.8770 | 8.8026 → 2.9168 | 3.05 | 3.02 |
| t_jepa fp32 | 283.4733 → 90.0966 | 283.5458 → 90.2192 | 3.15 | 3.14 |
| ts_jepa fp32 | 16.3404 → 5.5916 | 16.2817 → 5.6124 | 2.92 | 2.90 |
| sigreg q4 | 24.8950 → 6.2449 | 24.8605 → 6.2329 | 3.99 | 3.99 |

Worst disagreement 1.4 % (lejepa fp32 after, +1.38 %), consistent with t4's
"within 1.4 %" and with my run using half the committed iteration count.
Comparing the committed tables cell by cell, **0 of 15 router cells move by more
than 3 %** (worst −1.4 %, `ts_jepa` fp32). The device answer to the brief's
"re test in this machine and in the router" holds on the router.

I did **not** re-run the host bench. t4 itself documents host medians as
reproducible only to ~7 % because of code placement, so a single re-run cannot
adjudicate the host column either way; I verified the host rows' structure and
the summariser's rule instead (below), and report the host delta question as
F9 rather than claiming to have settled it.

### The harness defects `t4` fixed — real, and correctly fixed

The claims about the *script* are accurate. `git show HEAD:results/perf/run_bench.sh`
has the router branch calling `line "$json" x "$model" ...` (hardcoded device,
10 fields under an 11-column header) with a nested `f()` whose `$2` was the sed
field name rather than the JSON, so every value printed blank. The current
script passes `"$DEVICE"` and feeds `f()` the JSON on stdin. The regenerated
`bench_host.csv` and `bench_router.csv` are both 11 columns × 90 rows with no
short rows (`5 models × 3 precisions × 2 builds × 3 reps`).

`summarise.py` is documented, not accidental: it reports the **best** run per
cell (`min(..., key=median_ms)`), which is why `summary.csv` reproduces run 1 of
3. That is stated in its docstring, so I do not count it as a defect.

## 4. Fresh clone (outside the working copy)

`/tmp` is tmpfs on this machine and a second torch venv filled it (7.7 GB), so I
built the clone at `$HOME/fresh-verify-rt2` instead. It is outside the working
copy, which is what matters for F2:

```
mkdir -p $HOME/fresh-verify-rt2
git -C <worktree> ls-files -z | tar --null -T - -cf - | tar -C $HOME/fresh-verify-rt2 -xf -
```

518 files, i.e. exactly the tracked set with working-tree contents, and **no
ignored file present** — which is the point, since the ignored files here would
mask the defect.

**Precondition I must state.** A tracked-files-only clone cannot run
`quantize stats` at all, because it loads the model checkpoints from the
**gitignored** `20-aug/models/` store (`FileNotFoundError: lejepa: checkpoint not
found at .../lejepa_ft_seed42.pt`, exit 1). That is a documented, pre-existing
repository property, not the F2 defect: `README.md:165` calls `20-aug/`
"checkpoint stores (not tracked)" and `docs/guides/running-models.md:24` says
model binaries are gitignored. It is 42 files / 286 MB and was never in git. With
that local store hardlinked in, both commands succeed:

```
uv run inmotion quantize stats       -> exit 0   (wrote 5 tables + stats_manifest.json)
uv run inmotion plots quantization   -> exit 0   (6 figures x 2 formats)
```

`--latency-iters 0` does **not** lift the dependency — the models are also loaded
for the accuracy/size path — so the checkpoint store is genuinely required.

So the acceptance criterion "a fresh clone regenerates the figures and stats
tables to exit 0" passes **on the figures unconditionally** and on the stats
tables **with the checkpoint store present**; and it never depends on this
working copy's ignored `results/` artifacts, which was the actual defect.

## 5. New findings from this verification

These are reported separately; none of them reopens F1-F6, R2-01 or either open
item, and none of them touches parity, the kernels or the device numbers.

### F7 (low) — the dev-machine row of `stats_latency.csv` carries the router's dispersion

- file: `results/quant/stats_latency.csv`, `src/inmotion/viz/quant_stats.py:550-561`
- problem: `_rows_from_payload` builds the reference row as
  `_bench_row(hostname, {**bench, **reference}, ...)`, where `bench` is the
  artifact's **router** `benchmark` block and `reference_dev_machine` carries only
  `median_ms` and `throughput_per_s`. The merge therefore leaves the router's
  `p95_ms`, `min_ms`, `max_ms` (and the router's musl `runtime` label) on a row
  attributed to the development machine. Reproduced in the committed file and in
  a fresh regeneration:

  | field | `andreribeiro` ref row | `CARRIS` router row |
  |---|---|---|
  | `median_ms` | 6.9937 | 8.9002 (differs, correct) |
  | `p95_ms` | 9.1982 | 9.1982 (**inherited**) |
  | `min_ms` | 8.7456 | 8.7456 (**inherited**) |
  | `max_ms` | 9.3405 | 9.3405 (**inherited**) |

- impact: the row's headline median is right and **no figure reads this table**
  (`quant_figures.py` uses `results/perf/summary.csv`), but a reader taking the
  column named `p95_ms` for the dev machine gets the router's number.
- required fix: build the reference row from `reference` alone, or record only
  the fields the reference actually measured (emitting empty cells for the rest),
  and take its `runtime` from the host build, not the router's target.

### F8 (medium) — `results/perf/README.md` misstates the committed benchmark tables

- file: `results/perf/README.md:349-351`, `:290-296`
- problem: the README claims "The committed `bench_router.csv` did not come from
  `run_bench.sh` (it is ten columns wide, from the ad-hoc on-device script of the
  time) … the committed `summary.csv` is a router-only table of 15 cells." Both
  are false at the commit they will land on:

  | artifact | README says | `git` at `9f3e5fd` / `HEAD` |
  |---|---|---|
  | `bench_router.csv` | 10 columns, ad-hoc script | **11 columns, 90 rows**, `device=router` |
  | `bench_host.csv` | — | **11 columns, 90 rows**, `device=host` |
  | `summary.csv` | router-only, 15 cells | **30 cells** (15 host + 15 router) |

  All three were first committed in that shape in `9f3e5fd`, so the
  "10-column/15-cell" state never existed in git, and `summarise.py` clearly did
  merge both devices. The *script*-level bugs (hardcoded device, `$2` collision,
  blank values) are real and are correctly fixed — my own check of `git show
  HEAD:results/perf/run_bench.sh` confirms them — but the README draws a false
  conclusion about the previous committed data from them.

  **I went looking for a ten-column artifact and could not find one anywhere.**
  The captain's guidance repeats the claim that "the committed `bench_router.csv`
  did not come from `run_bench.sh` at all" and was 10 columns wide, leaving a
  router-only 15-cell `summary.csv`. I checked every copy reachable from here:

  | where | `bench_router.csv` | `summary.csv` |
  |---|---|---|
  | `git show 9f3e5fd:` / `HEAD:` | 11 columns, 90 rows | 30 cells (15 host + 15 router) |
  | this working tree | 11 columns, 90 rows | 30 cells |
  | the router's own `/mnt/extra/cargo/inmotion/perf/` | `bench_router.csv` 11 columns / 23 rows; `bench_router2.csv` 11 columns / 91 rows | — |

  No 10-column file exists under that name in any of them, and the perf CSVs'
  *first* commit (`9f3e5fd`) already carried 11 columns and 30 summary cells.
  What the captain's point 5 gets right is that the script was broken and that
  the committed tables therefore did not come from it; what is not reproducible
  is the claim that the committed tables were **malformed**. A router-only
  15-cell `summary.csv` is a plausible *intermediate* state of t4's own re-run (its
  buggy router branch run alone would produce exactly that), which is the most
  likely origin of the claim — but as a statement about the committed artifact it
  is false, and the README ships it.
- impact: this file ships in the closure and is the evidence record for the
  brief's re-test question; it tells a reviewer the earlier baseline was
  malformed when it was not.
- required fix: drop the two assertions about the committed CSVs (or reword to
  "the committed CSVs are well-formed but could not have been produced by this
  script"), keeping the accurate script-bug description. A router-only
  intermediate `summary.csv` from t4's own run should not be described as the
  committed one. **The full provenance of each committed file, the two script
  bugs re-derived by execution, and the proof that `summarise.py` did merge, are
  in §9 — that section is the authority for this finding.**

### F9 (low) — the host delta count is understated

- file: `results/perf/README.md:288-296`
- problem: "three cells move by 3-7 %" and "Twelve of the fifteen cells land
  within 3 % of the committed speedup; three do not". Recomputing committed vs
  re-run `summary.csv`, **eight** of the fifteen host cells move by more than
  3 % on the speedup ratio, and seven do on `after_median_ms`:
  `cf_jepa` q4 −7.1 %, `t_jepa` fp32 −6.5 %, `cf_jepa` int8 −4.8 %, `sigreg`
  int8 −4.7 %, `lejepa` fp32 −4.7 %, `sigreg` q4 −3.5 %, `t_jepa` q4 +3.4 %,
  `lejepa` int8 −3.2 %. The sentence is also self-contradictory: it says three
  cells miss 3 % and then gives the rest as spanning −3.5 % to +3.4 %.
- impact: low, and it argues *against* t4's own interest — the true picture makes
  the committed host speedups less reproducible than claimed, which is the same
  conclusion t4 reached. But the number is wrong in a shipped file.
- required fix: correct "three" to "eight" (and "Twelve" to "seven"), keeping the
  existing code-layout explanation.

### Observation (informational, not a finding)

`firmware/evidence/device_measurements.json` now carries two different medians
for the same 240 MHz clock: `runs.240mhz.latency_ms_median = 636.6825` (the
original run) and `clock_provenance.artifacts[...240mhz].latency_ms_median =
636.6805` (the re-measurement). Both are real measurements 4 ppm apart, and the
`clock_provenance` block names its artifact, so nothing is wrong — but labelling
the `runs` block as the original and `clock_provenance` as the re-measurement
would stop a reader seeing two numbers for one clock.

## 6. What I could not reproduce, and one thing I chose not to

- **160 MHz ESP32 latency — not re-measured on the device.** That needs a
  160 MHz firmware build, a flash and a restore; I did not do it. Verified as an
  artifact with an internal clock field, cross-checked by md5 against the tracked
  `clock_provenance` (§3/F1).
- **Host benchmark column — not re-benched.** t4 documents host reproducibility
  as ~7 % by construction, so one more run cannot confirm or refute the host
  numbers. I verified their structure and the summariser rule, and recomputed the
  deltas (F9) instead of claiming a measurement I did not make. The router column
  *was* re-measured by me and agrees.
- **`quantize stats` in a tracked-files-only clone** cannot run, for the
  documented checkpoint reason in §4 — reported, not worked around.
- **`/tmp` could not host the clone** (tmpfs, 7.7 GB, exhausted by a second torch
  venv), so the clone lives at `$HOME/fresh-verify-rt2`, still outside the working
  copy. Nothing in the F2 result depends on the path.

One upstream drift is worth the captain's attention: the working tree carries
unstaged edits on top of staged ones (`src/inmotion/viz/quant_figures.py` is
`MM`, and the `rust/*.rs` modifications are unstaged), so the index is not the
state I verified. `t7` must commit the **working tree**, not `git commit` on the
current index alone, or the guards and the figure fix I verified will not be in
the commit.

## 7. Artifacts this verification produced (outside the repo, nothing committed)

- `/tmp/verif2/art/{5 models x fp32,int8,q4}` — artifacts + goldens I exported
- `/tmp/verif2/mal/{shortA,shortB,notol}` — the R2-01 and `tol` fixtures
- `/tmp/verif2/esp32_verify_240mhz.json` — my own device measurement
- `/tmp/verif2/prefix-rust`, `/tmp/verif2/prefix-target` — the guard-removed
  rebuild (deleted after use; its outputs are quoted above)
- `$HOME/fresh-verify-rt2` — the external clone built before `t8`
- `$HOME/fresh-verify-rt3` — the external clone rebuilt *after* `t8` (§8)

## 8. Addendum after captain guidance

### 8.1 The `t8` re-run was concurrent — I re-ran the figure/stats check on the post-`t8` state

My §4 result was taken on the working state as it stood then. `t8` (regenerate
figures/stats after the benchmark re-run) has since **completed** — the working
tree's figures and stats were rewritten at 11:58 from the post-`t4`
`results/perf/summary.csv` (11:56:53). So I rebuilt the external clone from the
current working state (`$HOME/fresh-verify-rt3`, 560 files) and re-ran the whole
check. The conclusion is unchanged, so nothing in §4 was a stale-input artifact:

```
uv run inmotion quantize stats     -> exit 0  (wrote 5 tables + stats_manifest.json)
uv run inmotion plots quantization -> exit 0  (6 figures x 2 formats)
```

**18 of 20** regenerated outputs are byte-identical to the committed files (all 6
PDFs, all 6 PNGs, `figures/data/fig1_device_budgets.csv`,
`figures/data/fig5_latency_rows.csv`, `stats_size/accuracy/coverage/layers.csv`).
The two that differ are again exactly the declared ones: `stats_latency.csv`
(manifest `md5: null`, `varying`) and `stats_manifest.json` (`/cwd`, `/argv`,
`/python` only).

I also verified `t8`'s own stale-input fix rather than taking it on trust.
`fig5_latency_rows.csv` now holds **62 rows**, not 32:

| device | build | rows |
|---|---|---|
| host | before / after | 15 / 15 |
| router | before / after | 15 / 15 |
| esp32 | deployed | 2 |

and the host rows are no longer missing. For all **30** host+router cells the
fig5 `before`/`after` medians and the derived speedup agree with
`results/perf/summary.csv` to the printed digit — **0 mismatches**. The file's
`artifact` column names only `results/perf/summary.csv` and the two clocked ESP32
JSONs, with nothing stale.

#### Full shape-and-content audit of `fig5_latency_rows.csv` (not just md5 equality)

The captain's point that md5 equality alone is circular is right, so I audited
the content directly against `results/perf/summary.csv`:

| check | result |
|---|---|
| total rows | **62** (expected 62) |
| breakdown | host 15 before + 15 after; router 15 + 15; esp32 2 |
| host/router rows checked against `summary.csv` | **60 of 60**, 0 missing keys |
| worst abs diff, `median_ms` vs `before/after_median_ms` | **0.000e+00** |
| worst abs diff, `p95_ms` vs `before/after_p95_ms` | **0.000e+00** |
| duplicate keys `(device, model, precision, build)` | 1 — and it is benign (below) |
| `artifact` column | only `summary.csv` + the two clocked ESP32 JSONs |

The one "duplicate" is the pair of ESP32 rows:
`(esp32, lejepa, int8, deployed)` appears twice because the two rows are two
*different clock measurements of the same model and precision*, separated by
`artifact` (`esp32_lejepa_160mhz.json` / `…240mhz.json`) and labelled 160 / 240
MHz. Adding `artifact` to the key gives **0 duplicates**, and all 60 host/router
keys are unique. So this is my key being too coarse, not a defect — and it is
worth stating precisely rather than reporting a phantom finding.

#### The comparison baseline is the working tree, not `git HEAD`

My md5 comparison in §4 compared the clone's regenerated files against the
**working tree** (`$WT/$f`), which is the correct baseline: `git HEAD` still
carries the pre-`t4` figures until `t7` commits, so comparing against
`git show HEAD:results/quant/figures/...` would report mismatches that are not
defects. Confirmed by `git status`, which shows only four figure paths modified
against HEAD: `figures/data/fig1_device_budgets.csv`,
`figures/data/fig5_latency_rows.csv`, `fig5_latency_by_device.pdf` and
`fig5_latency_by_device.png`.

I also checked `t8`'s claim that the other figures did not move, using HEAD as an
independent baseline for the files that must *not* have changed:

| figure | vs `git HEAD` |
|---|---|
| `fig1_fits_map.pdf/.png` | same as HEAD |
| `fig2_size_accuracy_pareto.pdf/.png` | same as HEAD |
| `fig3_accuracy_vs_bits.pdf/.png` | same as HEAD |
| `fig4_coverage.pdf/.png` | same as HEAD |
| `fig5_latency_by_device.pdf/.png` | **changed** |
| `fig6_size_compression_vs_bits.pdf/.png` | same as HEAD |

`fig1_device_budgets.csv` changed by exactly one annotation string
(`results/quant JSONs` → `results/quant/esp32_*.json`), which is why its rendered
PDF/PNG is byte-identical to HEAD. So `t8`'s "only fig5 moved, plus the two data
CSVs" is independently confirmed.

#### Lesson recorded: a match is only meaningful against an independent baseline

Two of this session's confident claims failed for the same structural reason, and
both are worth carrying forward:

1. **`t3`'s "all 12 figure md5s match the tracked ones"** compared its
   regenerated output against the output it had just staged itself — a circular
   comparison. It passed while `fig5_latency_rows.csv` was missing all 30 host
   rows (32 rows instead of 62), because the baseline had the same defect as the
   artifact. Regenerating from a summary that was mid-write made a 32-row file
   look like a match.
2. **Round-1 F1's clock pairing** used the *filename* as the baseline for a
   property of the *content*. The filename cannot witness what the content holds,
   so a mislabelled bar looked verified.

The general rule this verification applied instead: compare against something the
artefact under test cannot influence — `git HEAD` for files that must not change
(§8.1), the committed `summary.csv` for the figure data, a guard-removed rebuild
for the two harness guards (§3/R2-01), and the payload's own `clock_mhz` rather
than a filename (§3/F1). Counting rows and re-deriving values is what caught the
32-row regression; md5 equality would not have.

### 8.2 The ESP32 part, confirmed directly

The captain is right that the serial verify command in the task needs `sg`:
this session's uid is not in its effective group list (`getent group dialout`
does list `andre`, so `sg` succeeds without a password). With `sg dialout`:

```
sg dialout -c 'uv run --extra esp python -m esptool --port /dev/ttyUSB0 --no-stub chip-id'
-> Chip type: ESP32-D0WD-V3 (revision v3.1), Features: ... 240MHz
   MAC f8:b3:b7:24:06:08                                          exit 0
```

which independently confirms the part the artifacts name (`device:
ESP32-D0WD-V3`). The unprivileged form exits 2 with "Try to add user into
dialout or uucp group" — **a permissions artefact of how the command is invoked,
not a device finding**, and I have not recorded it as one.

### 8.3 The fresh-clone definition

The captain's point 4 matches what I actually did: my clone is built from the
**working state**, not from HEAD (`tar` reads the working-tree contents of every
`git ls-files` path, so all of the uncommitted port and closure work is present,
and I did not conclude the uncommitted tree cannot be cloned). Two independently
built clones agree: `edge-evidence`'s `/home/andre/tmp/clone-final` has the same
md5 for `rust/inmotion-cli/src/main.rs` (`bd81a86c…`), for
`src/inmotion/viz/quant_figures.py` (`b3db27bd…`) and for
`results/quant/esp32_lejepa_240mhz.json` (`a0fb621d…`) as this working tree, and
carries both t1 guards.

### 8.4 The harness bugs, and what "reported honestly" means here

The script-level defects are real and I confirmed them independently in
`git show HEAD:results/perf/run_bench.sh`: the router branch called
`line "$json" x ...` (hardcoded device) with a nested `f()` whose `$2` was the sed
*field name*, so every value printed blank; the host branch passed the JSON first
so `$2` inside `line()` resolved to the script's own empty `$2`. Both branches now
pass the JSON explicitly and take the device as `$2`, and the regenerated CSVs are
11 columns × 90 rows with no short rows. What I cannot reproduce is the *result*
attributed to them (§5/F8): no 10-column `bench_router.csv` and no 15-cell
router-only committed `summary.csv` exists in git, in this tree, or on the router.

On the host outliers: they are **not smoothed** — the README prints the three
worst (`cf_jepa` q4 7.14x → 6.63x, `t_jepa` fp32 5.25x → 4.91x, `cf_jepa` int8
7.05x → 6.71x) and attributes them to code placement. The problem is the
opposite of smoothing: it **under-counts**, because eight of fifteen host cells
exceed 3 %, not three (F9). The cause claim is consistent with what I measured —
the kernel sources are byte-identical to the revision round 2 verified and parity
did not move, so a ~7 % host shift cannot be a kernel change — but I did not
disassemble the two host binaries, so I confirm the measurement and the
implication, not t4's instruction-level attribution.

## 9. Provenance of the committed benchmark tables (requested by the captain)

The captain asked me to verify the harness-bug claim "on its own merits" and to
say which committed file came from where, using the repository as the authority.
I did this by **executing** the committed script rather than reading it, and by
re-deriving the summary with the committed `summarise.py`.

### 9.1 The committed script is genuinely broken in both branches — re-derived by running it

I extracted `git show HEAD:results/perf/run_bench.sh` to a scratch directory,
pointed `BEFORE`/`AFTER` at stub binaries that print a fixed JSON, and ran both
branches. The script's own output:

| branch | row the committed script actually emits |
|---|---|
| host | `host,x,lejepa,fp32,before,4.0584,4.4530,4.5800,5.1175,6.0000,224.6` |
| router | `router,lejepa,fp32,before,1,,,,,,` |

Both rows have **11 fields**; neither is short.

- **Host bug — real, but not the mechanism t4 described.** The call is
  `line "$json" x "$model" "$prec" "$build" "$run"` while the format string
  prints `'host,$2,$3,$4,$5,...'`. Inside a POSIX shell function `$2` is the
  *function's own* second positional parameter, which is the literal `x` — it is
  **not** a leak of the script's `$2` (t4's stated mechanism). The observed defect
  is that the `model` column receives `x` and every label from column 2 onward
  holds its neighbour's meaning (`precision`←model, `build`←precision,
  `run`←build). The metrics still land in the right columns.
- **Router bug — real, but 11 fields, not 10.** `f() { printf '%s\n' "$1" | sed -n "s/.*\"$2\":…" }`
  is called as `f min_ms`, so inside `f` `$1` is the *field name* and `$2` is
  unset; sed is handed an empty pattern and matches nothing, so all six metrics
  print blank. The comment "ten columns" is wrong: the row has 11 fields, six of
  them empty.

### 9.2 Which committed file came from where

| committed file | produced by | evidence |
|---|---|---|
| `bench_router.csv` | the device-local, **untracked** `router_bench.sh` on wave-router | same 11-column header; emits `router,<model>,<prec>,<build>,<run>,<6 real metrics>` via a *correct* `f()` (`echo "$j" | sed … "$1"`); the committed file has no trailing `DONE` line, while the device's own copy of that runner's output does |
| `bench_host.csv` | a **corrected host invocation** of the same harness; the producer is not in the repository | 0 of 90 rows contain `,x,`, all 91 lines have 11 fields, values coherent; its shape is exactly what the *fixed* branch emits (§9.3) |
| `summary.csv` | the committed `summarise.py` itself, merging both bench files | see §9.4 |

`run_bench.sh` could not have produced `bench_router.csv`: its router branch
writes blank metrics, and the committed file has none. The captain's point that
the committed tables "did not come from the committed script" is therefore
**correct**, and the router CSV now has a named producer. The device's own
`bench_router.csv` (23 lines, trailing `DONE`) is a partial earlier run of the
same ad-hoc script; the committed file is the full, cleaned one.

### 9.3 The committed data matches the *fixed* script, not the broken one

Running the current (repaired) `run_bench.sh` with the same stubs:

| source | host row | router row |
|---|---|---|
| HEAD committed data | `host,lejepa,fp32,before,1,3.9944,…` | `router,lejepa,fp32,before,1,8.6915,…` |
| HEAD committed script | `host,x,lejepa,fp32,before,4.0584,…` | `router,lejepa,fp32,before,1,,,,,,` |
| fixed script | `host,lejepa,fp32,before,1,4.0584,…` | `router,lejepa,fp32,before,1,4.0584,…` |

The committed schema and protocol are exactly what the repaired harness produces.
So the committed `run_bench.sh` is best characterised as a **broken transcription
of the harness that did produce the committed data**, not as evidence that the
data is malformed.

### 9.4 `summarise.py` did merge — proven with the committed copy

I ran `git show HEAD:results/perf/summarise.py` on `git show
HEAD:results/perf/bench_host.csv` and `…bench_router.csv`:

```
python3 summarise.py      # exit 0
-> 30 rows; device mix of the committed summary.csv: 15 host, 15 router
-> diff (values only): IDENTICAL to HEAD:results/perf/summary.csv
```

Every value and the row count match. The **only** difference is line endings:
the committed CSVs and `summary.csv` are LF, while the csv module's default
dialect makes `summarise.py` write CRLF — which is exactly why the current
`summarise.py` pins `lineterminator="\n"`.

The committed `summary.csv` is therefore a genuine 30-cell merge of both bench
files, reproducibly regenerated by the committed summariser. It is **not** a
15-cell router-only table, and `summarise.py` did **not** "merge nothing".

### 9.5 Answer: the README's write-up overstates the consequence

`results/perf/README.md:349-351` says the committed `bench_router.csv` "is ten
columns wide, from the ad-hoc on-device script of the time, so the committed
tables were not produced by one harness — and `summarise.py` therefore merged
nothing: the committed `summary.csv` is a router-only table of 15 cells."

| clause | verdict |
|---|---|
| the committed `bench_router.csv` did not come from `run_bench.sh` | **true** — and now has a named producer (`router_bench.sh`) |
| "it is ten columns wide" | **false** — 91 of 91 lines have 11 fields |
| "`summarise.py` therefore merged nothing" | **false** — the committed summariser reproduces the committed summary |
| "the committed `summary.csv` is a router-only table of 15 cells" | **false** — it is 30 cells, 15 host + 15 router |

The write-up is right that the two branches are buggy and that the committed
router CSV came from something else; it is wrong to conclude from that the
committed **data** is malformed. The script was broken, the artifacts were
produced another way, and the committed numbers are trustworthy — three separate
facts that §5/F8 shows were collapsed into one wrong conclusion.

### 9.6 New finding F11 (low) — the `before` build is documented but not tracked

- file: `results/perf/README.md:367-370,428`, `results/perf/make_before_kernels.py`
- problem: the README presents `make_before_kernels.py` as the reproducible way
  to build the `before` binary ("not a remembered hand edit") and lists it in the
  file table, but the file is **untracked and gitignored**
  (`.gitignore:263` → `results/`; `git ls-files results/perf/make_before_kernels.py`
  is empty). This is the same defect class as F2 before it was fixed: a
  documented input that a clone cannot obtain. `run_bench.sh` needs
  `BEFORE`/`AFTER` binaries, so the README's claim that `run_bench.sh`
  "reproduces the raw CSVs on either machine" is not reproducible from a clone.
- impact: low for the figures (they do not need the `before` build), but it
  undercuts the reproducibility claim for the benchmark tables that F2/F5 were
  about, and `t7` is about to commit the tree.
- required fix: `git add -f results/perf/make_before_kernels.py` alongside the
  three device JSONs (the pattern t3 already used), or drop it from the README's
  file table and say plainly that the `before` binary is a local artifact.
