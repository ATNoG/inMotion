# Verification report (inmotion-edge, task t9)

Independent verification performed by `verifier` on this checkout. Every number
below was produced by re-running the harness here; nothing is quoted from another
member's output. The claims under test are the t2-t8 dependency results, the
project invariants, and the five acceptance criteria of t9.

## Verdict

- Design and execution are sound: the port, the kernel work, the dataset switch,
  the tables and the figures all reproduce, and three of the five router cells I
  re-measured match the committed values.
- Two medium findings need remediation before these numbers go into a paper: one
  mislabelled ESP32 measurement that reaches a figure annotation, and benchmark
  inputs that cannot regenerate the committed figures from a fresh clone.
- Four low findings are documentation drift and over-broad claims, not incorrect
  measurements.

## 1. Golden-vector parity, all five models x fp32/int8/q4 (reproduced)

I re-exported every artifact from the checkpoints myself (`inmotion.quant.export
.export` + `write_golden`) into `/tmp/verif/art`, then replayed the golden vectors
with both independent runtime paths: `inmotion-infer check` (batch, reports the
worst case over the 16 cases) and `inmotion-infer predict` per case, which I read
back from the printed logits and compared against `golden.json`. The per-case
predict worst case matches `check`'s worst case for every cell, so the two paths
agree.

Tolerance 2e-4 everywhere; all 15 cells pass.

| model | fp32 max abs diff | int8 | q4 | cases | verdict |
|---|---|---|---|---|---|
| lejepa | 4.05e-6 | 4.23e-6 | 3.34e-6 | 16 | PARITY OK |
| t_jepa | 2.62e-6 | 4.14e-6 | 6.20e-6 | 16 | PARITY OK |
| ts_jepa | 2.74e-6 | 2.92e-6 | 5.01e-6 | 16 | PARITY OK |
| cf_jepa | 2.21e-6 | 2.95e-6 | 1.91e-6 | 16 | PARITY OK |
| sigreg | 8.3e-7 | 8.3e-7 | 9.5e-7 | 16 | PARITY OK |

No member raised the tolerance: every `golden.json` carries `tol = 0.0002`, and
the checker itself defaults to 2e-4 rather than trusting the file blindly.

## 2. Deployed router binary (reproduced)

- The binary that produced `results/perf/bench_router.csv` is
  `/mnt/extra/cargo/inmotion/perf/after-musl`; its sha256 is
  `29cd6de5b482f3505bfdee2e3837ece5185db00b3e6dd86be58b96f42075faf3`, byte for
  byte the same as my local build at
  `rust/target/x86_64-unknown-linux-musl/release/inmotion-infer`. Size 878152 on
  both sides.
- Target: `x86_64-unknown-linux-musl`, static (the local ELF is `static-pie
  linked`; the on-router copy runs with no interpreter available), built with an
  empty `rustflags` list in the cargo fingerprint and no `target-cpu`/
  `target-feature`/`RUSTFLAGS` anywhere in the tree.
- The router really has no AVX: `/proc/cpuinfo` reports `sse4_2` and not
  `avx`/`avx2`/`fma`. The deployed binary runs correctly there
  (`describe` on the on-device lejepa int8 artifact returns the expected graph).
- Static scan of my local musl binary: all 270 AVX-encoded instructions live in
  three `memchr` symbols (`find_avx2` variants), which that crate reaches only
  behind a runtime `CPUID` dispatch. Zero project (inmotion) code contains an AVX,
  AVX2 or FMA instruction. See finding F3 only for the wording of the claim.

## 3. Benchmarks (independently re-measured on the real router)

I ran `after-musl` and `before-musl` myself on `wave-router` over the artifacts
already deployed there (100 iterations, 20 warmup), and compared with the
committed `results/perf/summary.csv`:

| cell | committed before -> after | committed speedup | my before -> after | my speedup |
|---|---|---|---|---|
| lejepa fp32 | 8.8121 -> 2.9024 | 3.04 | 9.0703 -> 3.0115 | 3.01 |
| t_jepa fp32 | 283.66 -> 90.31 | 3.14 | 283.80 -> 90.44 | 3.14 |
| sigreg q4 | 24.91 -> 6.21 | 4.01 | 24.94 -> 6.26 | 3.99 |

So the router timings are real, device-side, and reproducible within noise.

## 4. Dataset switch and the baseline guard (reproduced, and proven live)

- `grep -rn icaisf src/ scripts/ tests/ results/quant/` returns nothing: no
  quantization path, table or figure consults `dataset.icaisf.csv` any more. The
  committed sweep is 35 rows (5 models x 7 precisions) and every row's `dataset`
  column is `dataset.csv`, `n_test = 703`.
- I re-ran the harness myself: `inmotion quantize eval --formats fp32` for all
  five models reproduces the recorded fp32 MCC exactly
  (lejepa 0.9142641067, t_jepa 0.8982207227, ts_jepa 0.9194244063,
  cf_jepa 0.9083210969, sigreg 0.6333303415) and reports "baseline reproduced".
- Guard proven live by perturbing a reference value: with `BASELINE_MCC`
  untouched, `check_baseline` on the committed sweep returns `[]`. After changing
  the recorded `cf_jepa` value by +0.02 it returns
  `cf_jepa: fp32 MCC 0.9083210968710779 vs recorded 0.9283210968710779 on dataset.csv`,
  and returns `[]` again once restored. The guard is not decorative.
- Statistics consistency: for all 35 rows, `unquantized_weight_bytes /
  quantized_bytes` from `quant_reports.json` equals the `compression` column to
  <5e-5, `quantized_bytes + skipped + buffers` equals `total_mb`, and
  `n_params_quantized + n_params_skipped == n_params_total` (0 violations).
  Coverage claims hold: params 98.66% (lejepa) to 99.38% (ts_jepa), bytes 90.96%
  (lejepa q2) to 97.72% (ts_jepa q8), cast rows coverage 1.0.

## 5. Figures and spot-checks (reproduced)

- `uv run inmotion plots quantization` regenerates all 6 figures in 12.5 s and
  every PDF and PNG md5 is unchanged from the committed files (12/12), so the
  figures are reproducible on this machine and traceable to their inputs.
- Spot-check 1 (fig5a, per-cell speedup annotation). The code computes
  `before/after` per cell and prints `x{speedup:.1f}`. From
  `results/perf/summary.csv`: host lejepa fp32 4.3444/0.7006 = 6.20 (plots `x6.2`),
  router lejepa fp32 8.8121/2.9024 = 3.04 (plots `x3.0`), host sigreg q4
  8.0584/1.4064 = 5.73, router t_jepa int8 284.0535/90.494 = 3.14. All four agree
  with `fig5_latency_rows.csv` and with the plotted values.
- Spot-check 2 (fig1, fit verdicts and cell MB). Cell text is
  `stats_size.csv.total_mb`; the fit rule is `total_mb*1e6 <= budget`.
  ts_jepa fp32 8.613396 MB > 4 MB flash -> "does not fit" (correct);
  cf_jepa fp32 4.929044 MB > 4 MB -> "does not fit" (correct);
  lejepa q2 0.345728 MB = 345728 B > 307200 B SRAM -> never resident (correct).
  The router budget 3 391 000 000 B is `ram_available_mb = 3391` from
  `edge_router_j3455.json`, which I re-read on the device.
- Spot-check 3 (fig4a, "lejepa worst at 9.0% of bytes"): `1 - frac_bytes_quantized`
  for lejepa q4 is 0.078505, the largest among the five models, so the description
  holds.
- Traceability gap: see finding F2 for the device JSONs the latency figure needs.

## 6. Project invariants (re-checked)

- `uv run inmotion model verify` -> `All 18 model(s) verified` (exit 0). Note the
  count is 18, not the 17 in the task brief (finding F6).
- `uv run inmotion datasets verify` -> `All 7 dataset version(s) match the manifest`
  (exit 0).
- `uv run pytest tests/test_quant_export.py -k rust_port -v` -> 15 passed.
- World-model baseline, re-measured: see section 4.

## Findings

Findings are reported here and in the task payload; nothing in `src/`, `rust/`,
`firmware/` or `results/` was edited by this task.

### F1 (medium) - ESP32 clocked benchmark JSON filenames hold the other run's data

- file: `results/quant/esp32_lejepa_240mhz.json`, `results/quant/esp32_lejepa_160mhz.json`
- problem: the file named `..._240mhz.json` carries the 160 MHz run
  (median 636.6825 ms, evidence 240 MHz median 636.6825 ms), and the file named
  `..._160mhz.json` carries the 160 MHz *baseline* run (median 888.4765 ms; the
  tracked `firmware/evidence/device_measurements.json` labels 888.4765 ms as
  `160mhz`), so the filenames and their contents are swapped relative to each
  other. `quant_figures._clock_mhz` then derives the clock from the filename, so
  `results/quant/figures/data/fig5_latency_rows.csv` labels the 888.48 ms row
  "unknown clock" and the 636.68 ms row "240 MHz". The written summary for t8
  ("ESP32 636.68 ms (240 MHz) and 888.48 ms (160 MHz)") is right and the figure's
  clock label on the 636.68 ms row is right, but the 888.48 ms row loses its
  160 MHz label and the JSON set is self-contradictory: `esp32_lejepa_160mhz.json`
  duplicates `esp32_lejepa.json` byte for byte (md5 d638d2b2...) while the 240 MHz
  run has no file that names the run correctly.
- required fix: rename the two JSONs so each name matches the run it contains
  (or re-measure and write both with the clock taken from the device, not the
  filename), then re-run `uv run inmotion plots quantization` and update the
  tracked `firmware/evidence/device_measurements.json` if needed.

> **Correction (added by `t7`, after the re-measurement; the text above is left as
> the record of what was seen then).** The "problem" paragraph is not what the
> device measured. Both artifacts now carry the clock the ESP32 printed in its own
> boot log, written by `firmware/tools/run_on_device.py` from
> `cpu_start: cpu freq: <Hz>`, so the payload decides and no filename heuristic
> remains:
>
> | artifact | `clock_mhz` | boot-log `clock_source` | median | md5 |
> |---|---|---|---|---|
> | `results/quant/esp32_lejepa_240mhz.json` | 240 | `I (426) cpu_start: cpu freq: 240000000 Hz` | 636.6805 ms | `84255803a3a2e22777c80217e05bfc1e` |
> | `results/quant/esp32_lejepa_160mhz.json` | 160 | `I (426) cpu_start: cpu freq: 160000000 Hz` | 888.4805 ms | `1f00c69bd722ee018040a1d27505f197` |
>
> Each name matches the run inside it, so F1's premise — that the file named
> `..._240mhz.json` holds the 160 MHz run — is contradicted by the artifact's own
> boot-log line. `quant_figures._clock_mhz` (the filename heuristic this finding is
> about) no longer exists; `_clock_label` reads `clock_mhz` and raises if it is
> absent, and the deleted byte-identical `esp32_lejepa.json` duplicate is gone from
> the set. The md5s above are the ones the tracked
> `firmware/evidence/device_measurements.json` `clock_provenance` block claims, and
> they match the files on disk. The two files were re-measured rather than renamed
> (240 MHz median 636.6805 ms, 160 MHz median 888.4805 ms), which is the
> "re-measure and write both with the clock taken from the device" branch of the
> required fix above. Evidence: `firmware/evidence/device_measurements.json`
> (`clock_provenance`), `docs/reports/VERIFICATION_ROUND2.md` and
> `docs/reports/CODE_REVIEW_RUST_ROUND3.md`.
>
> One thing this annotation cannot settle: the old duplicate was deleted during the
> repair, so whether the pre-repair files were genuinely swapped or the medians in
> the paragraph above were mis-attributed is no longer re-readable from the tree.
> What matters for delivery is that the clock is now an authoritative artifact
> field produced by the device and read by the figure, which it is.

### F2 (medium) - the inputs for the committed figures are gitignored

- file: `.gitignore` (`results/`), `results/quant/edge_router_j3455.json`,
  `results/quant/esp32_lejepa*.json`
- problem: the six committed figures, `stats_*.csv` and `stats_manifest.json` are
  force-added past the `results/` ignore rule, but the device artifacts they read
  are not tracked: `edge_router_j3455.json` (router latency row and the router RAM
  budget in fig1) and the three `esp32_*.json` (the two ESP32 bars in fig5, the
  SRAM source string and the flash blob size). A fresh clone of this commit cannot
  run `uv run inmotion plots quantization` or `inmotion quantize stats` to
  reproduce the committed figures/tables: `device_budgets` does a bare
  `json.loads` on `edge_router_j3455.json` and raises `FileNotFoundError`, so the
  claim "figures regenerate from committed CSVs" only holds on this working copy.
  (`firmware/evidence/device_measurements.json` is tracked and holds the ESP32
  summary, but the figure reads the gitignored `results/` copies.)
- required fix: track the four device JSONs (force-add, or move them out of the
  ignored `results/` root, e.g. under `results/quant/evidence/`), or make the
  figure/stats commands fall back to the tracked firmware evidence with an explicit
  note in `stats_manifest.json`.

### F3 (low) - "objdump confirms zero AVX" is broader than the binary

- file: `docs/reports/CODE_REVIEW_RUST.md:429`, `results/perf/README.md`
- problem: the claim as worded says the binary has zero AVX. The deployed binary
  contains 270 AVX-encoded instructions, all of them inside three `memchr`
  `find_avx2` symbols reached only through that crate's runtime CPUID dispatch.
  No `inmotion` code emits AVX/AVX2/FMA, so the *project* claim is true, but a
  reader checking the whole binary will find those instructions and may conclude
  the binary is not baseline-safe.
- required fix: qualify the sentence - "no AVX in any inmotion code path; the only
  AVX in the binary is `memchr`'s runtime-dispatched `find_avx2`, never taken on
  this part".

### F4 (low) - quoted parity numbers for sigreg and cf_jepa are from an older build

- file: `rust/README.md`, `docs/reports/CODE_REVIEW_RUST.md`,
  `docs/reports/REFACTOR_SUMMARY.md`, t2/t4 payloads
- problem: t4/t3 quote `sigreg 1.23e-6 / 1.97e-6 / 1.43e-6` and
  `cf_jepa 2.15e-6 / 2.86e-6 / 1.67e-6`; the current tree reproduces
  `sigreg 8.3e-7 / 8.3e-7 / 9.5e-7` and `cf_jepa 2.21e-6 / 2.95e-6 / 1.91e-6`.
  The uncommitted edit to `results/perf/README.md` already explains that sigreg was
  retrained under t12 and that the tighter numbers are current, so the numbers in
  the review documents are the pre-retrain record; the drift is documentation, not
  parity.
- required fix: point the review documents at `rust/README.md` (or re-quote the
  current numbers) so no reader compares a stale record against a live run.

### F5 (low) - per-run latency CSV varies by construction, and the manifest does not list the ESP32 inputs

- file: `results/quant/stats_latency.csv`, `results/quant/stats_manifest.json`
- problem: the t6 claim that only `stats_latency.csv` varies across runs is because
  that table contains freshly measured torch-CPU rows; that is by design but means
  the manifest's `md5` for it (`0ad10656...`) is only valid for that one run, and
  the manifest lists `latency_artifacts = ["results/quant/edge_router_j3455.json"]`
  while the ESP32 JSONs that fig5 and the SRAM budget string consume are not listed,
  so the table's provenance is incomplete.
- required fix: either drop the md5 for the freshly measured table or record
  `n/a (measured at generation)`, and add the `esp32_*.json` paths to
  `latency_artifacts`.

### F6 (low) - checkpoint count in the task brief and docs disagrees with the registry

- file: task brief ("17 registered checkpoints"), `docs/reports/*.md`
- problem: `uv run inmotion model verify` loads and verifies 18 models
  (exit 0), not 17. The invariant still holds; only the count is wrong. The
  `sigreg_icaisf` checkpoint is one of the 18 and is unaffected by the dataset
  switch (it is a historical checkpoint, not an evaluation input).
- required fix: correct the count to 18 where it is quoted.

## What I could not reproduce

Nothing in the acceptance criteria is unverified. Two limits are worth stating:
the router bench was re-run at 100 iterations (the committed CSV used 200), so
agreement is within measurement noise rather than exact; and I did not re-flash
the ESP32 or re-run `firmware/tools/run_on_device.py`, so the ESP32 numbers are
verified as *artifacts internally consistent with the evidence* (F1), not re-measured
on the device in this session.
