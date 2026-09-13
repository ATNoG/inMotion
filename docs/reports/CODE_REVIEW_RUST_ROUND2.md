# Round-2 code review — Rust world-model port and CPU kernels

Task `t16` (review round 2), reviewing `t15` (the repair of the fourteen findings
in `docs/reports/CODE_REVIEW_RUST.md`).

**Verdict: pass.** All fourteen round-1 findings are either fixed and
re-verified (thirteen of them, each by re-running the exact demonstration that
found it) or reduced to one sentence in a file the owner had no authority to edit,
which is now also fixed. One new low-severity finding remains — a panic in the
parity harness when the *golden file* (a test fixture, not the artifact) has a
wrong-length input array — with a one-line fix. It does not block delivery: no
finding touches parity, the kernels, the artifact loaders or any production path.

## Revision reviewed

The tree is uncommitted work, so the revision is pinned by hash. `tensor.rs`
changed **while this review was running** (a doc-comment rewording only); the
enforcement gates were re-run afterwards, on the revision below.

| file | sha256 |
|---|---|
| `rust/inmotion-core/src/artifact.rs` | `2bb3943fdf48b8e22975303fafafbd506b0e0e5a7bda31ba97cd13284a86b793` |
| `rust/inmotion-core/src/lib.rs` | `efa9b78e30a2e9c0e9bf1cc32206fe2bd0008cea84adc013f189bf1fa455d6a3` |
| `rust/inmotion-core/src/model.rs` | `e406be1dba87737888c60d7015d49f510822714ebfb7a63e5160e269293d6954` |
| `rust/inmotion-core/src/ops.rs` | `92963c1f7e16bbfa38709c9ab48f5672e4af73f3c9d0187d537499c0ca3ff5ea` |
| `rust/inmotion-core/src/quant.rs` | `e60c5ff0fe0a53864a8c09e4a8f61c06fedd6ccfb5219351fb6194bae7377914` |
| `rust/inmotion-core/src/sigreg.rs` | `d7e3f1de3c8eba2fb70721fce76a2349635a3d4bb6997e6e126e3a5c04f49f87` |
| `rust/inmotion-core/src/tensor.rs` | `8d4a8998e0cf16dac97e4e199fdc83069a2a8a41178fcad1f0bbae682378882f` |
| `rust/inmotion-core/src/world_model.rs` | `a684b55374da2600c38e96c98d1afd3c15f54e26a1cd8ff3a696dc97073d3a2d` |
| `rust/inmotion-cli/src/main.rs` | `048befd977a5d84fa096921fb1f1c035ca8b5458a32d4efb9e9c804d4e838c39` |
| `rust/README.md` | `7e9d5e50d6e3d1527419af15d87110790b997b94a5409cf34bcb08ebc70940e1` |
| `rust/Cargo.toml` | `312c0d3c339c9f3ed230bd7c575698b8a21b9efc7efa35edf2fef0775f8c2f34` |

## The fourteen round-1 findings, re-verified

Each row re-runs the demonstration that produced the original finding. "Before"
is the output quoted in the round-1 report.

| id | claim | re-verification (post-repair) | status |
|---|---|---|---|
| R-01 | parity harness truncates the comparison | golden logits truncated 4 -> 1: `error: case 0: golden has 1 logits, but the model has 4 classes`, exit 1 (was `PARITY OK`, exit 0) | fixed |
| R-02 | quantized metadata unvalidated | `.zp` shrunk to 1 entry: `error: encoder.tokenizer.weight: 128 scales but 1 zero points`, exit 1 (was an index panic, exit 101). `.zp` re-dtyped to `F32`: `error: encoder.tokenizer.weight.zp: expected I32, got F32` | fixed |
| R-03 | NaN input panics the argmax | window with `nan`: `error: /tmp/nan-window.txt: "nan" is not a finite number`, exit 1 (was a panic, exit 101) | fixed |
| R-04 | `unwrap()` on golden rows | `golden.json` with a non-array input row returns `case {i}: ...` errors; `grep -n 'unwrap()' rust/inmotion-cli/src/main.rs` returns nothing | fixed |
| R-05 | `bench --iterations 0` indexes empty | `error: --iterations must be > 0`, exit 1 (was an index panic, exit 101) | fixed |
| R-06 | unchecked header arithmetic | header length `usize::MAX - 3`: `error: weights: header length exceeds file size`, exit 1 (was a slice panic, exit 101) | fixed |
| R-07 | dead + duplicated 3-D transposes | `git diff -U0 -- rust/inmotion-core/src/ops.rs` deletes `transpose_time_channel`, `transpose_channel_time` and their test; `grep -rn` over the repo finds no reference | fixed |
| R-08 | README says "Only `lejepa` is ported" | `rust/README.md` Scope now lists all five families and the `arch.family` selection; the parity table is 5 models x 3 precisions | fixed |
| R-09 | ESP32-S3 / PSRAM / microcontroller wording | `lib.rs:6-9`, `rust/README.md:13-17` and `:85-95`, `artifact.rs:6-7` and `:142-143`, **and** `tensor.rs:4` now state the two baseline x86-64 targets and that the ESP32 path is C firmware; `grep -rn 'microcontroller\|ESP32-S3\|PSRAM'` over `rust/` finds only the corrected sentences | fixed |
| R-10 | `as usize` truncation on 32-bit | `usize_field` and `usize_list_field` now use `usize::try_from(..)` with a "does not fit a usize" error | fixed |
| R-11 | the all-padding `conv1d_same` tap had no test | new `conv1d_same_skips_taps_that_fall_entirely_in_the_padding` (length 1, kernel 3) asserts `[10.0]`, i.e. only the centre tap contributes; test count went 48 -> 50 | fixed |
| R-12 | duplicated doc sentence | removed (`git diff -- rust/inmotion-core/src/quant.rs` is a single deleted line) | fixed |
| R-13 | redundant logits clone | `let probs = inmotion_core::ops::softmax_last(&logits);` | fixed |
| R-14 | "# Enforced in CI" with no Rust CI | comment now names `cargo clippy --all-targets -- -D warnings` and `cargo test` | fixed |

## Enforcement gates, run on the final revision

```
$ CARGO_TARGET_DIR=/tmp/rev2b-clippy cargo clippy --all-targets -- -D warnings
    Checking inmotion-core v0.1.0 (...)
    Checking inmotion-cli v0.1.0 (...)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 4.37s   # exit 0

$ cargo test                                          # 50 passed; 0 failed
$ cargo test --release                                # 50 passed; 0 failed
$ cargo test --release --target x86_64-unknown-linux-musl   # 50 passed; 0 failed
```

Both clippy runs used an empty `CARGO_TARGET_DIR`, so no cached result could mask
a warning. The new tests (`metadata_len_matches_the_granularity`,
`metadata_len_rejects_declarations_it_cannot_honour`,
`conv1d_same_skips_taps_that_fall_entirely_in_the_padding`) are part of the 50.

## Parity and the router target, re-measured after the repair

Both binaries were rebuilt from the repaired tree. The loader now validates every
quantized tensor at load, so a wrong count/dtype assumption would have broken
loading rather than inference — which is exactly what these runs test.

```
gnu   binary: 15/15 PARITY OK   (all five models x fp32/int8/q4, unchanged numbers)
musl  binary: 15/15 PARITY OK   (the router build, baseline x86-64)
```

Nothing moved: `lejepa` 4.05e-6/4.23e-6/3.34e-6, `t_jepa` 2.62e-6/4.14e-6/6.20e-6,
`ts_jepa` 2.74e-6/2.92e-6/5.01e-6, `cf_jepa` 2.21e-6/2.95e-6/1.91e-6.

The new `rust/README.md` sigreg row (`8.3e-7 / 8.3e-7 / 9.5e-7`, which the README
attributes to the dataset.csv retrain) is **verified**, not accepted: replaying
the retrained artifacts gives

```
sigreg_fp32  max |diff| : 0.00000083  PARITY OK
sigreg_int8  max |diff| : 0.00000083  PARITY OK
sigreg_q4    max |diff| : 0.00000095  PARITY OK
```

No router feature was introduced by the repair:

```
$ objdump -d rust/target/x86_64-unknown-linux-musl/release/inmotion-infer \
    | grep -cE '^\s+[0-9a-f]+:\s+[a-z0-9]+\s+%v[0-9]'
0
```

The rebuild hash is `04c9b05c9a2d4c7d1297ef4d732b2a5ad32b076dab8e0d55b10e019329e977cd`.

**The `t5` benchmark numbers remain valid.** `git diff -U0 -- rust/inmotion-core/src/ops.rs`
shows hunk headers only in `mod tests` plus the deletion of the two dead
transpose helpers: `linear_packed`, `self_attention`, `conv1d_same`,
`fold_batch_norm` and `channel_affine` bodies are byte-identical to the code that
was benchmarked, so the before/after A/B and the crate survey still describe the
shipped kernels.

## Residual finding

### R2-01 (low) — a wrong-length golden *input* still panics the harness

Fixed for `logits`, not for `input`. `check` builds the tensor straight from the
parsed array (`rust/inmotion-cli/src/main.rs:154`, `let x = Tensor::new(input, vec![model.seq_len(), model.in_channels()]);`)
with no length check, so a corrupted golden input reaches `Tensor::new`'s
invariant assert:

```
$ python3 -c "... c['input'] = [row[:2] for row in c['input']] ..."   # 10x2 instead of 10x4
$ inmotion-infer check --artifact /tmp/short-input --golden /tmp/short-input
thread 'main' panicked at inmotion-core/src/tensor.rs:24:9:
assertion `left == right` failed: data length 20 does not match shape [10, 4] (40 elements)
  left: 20
 right: 40                                                        # exit 101
```

This is the same class as round-1 R-04 (which the repair fixed for the `unwrap`
calls): a malformed golden file should name the bad case, not panic. It is a test
fixture path — the weights loader (which the acceptance criterion is about) no
longer panics anywhere I could reach — hence low, and it does not block delivery.

**Required fix:** validate the golden's input width next to the existing logit
check, e.g.

```rust
let expected_input = model.seq_len() * model.in_channels();
if input.len() != expected_input {
    return Err(format!("case {i}: input has {} values, expected {expected_input}", input.len()));
}
```

## Open items (not findings — they need a decision or belong to another owner)

1. **`tol` is still a silent default in the harness** (`main.rs:113`, `golden["tol"].as_f64().unwrap_or(2e-4)`).
   Removing `tol` from a golden file still prints `tolerance : 0.000200 ... PARITY OK`,
   exit 0, so a missing field selects a tolerance instead of failing. Worth a
   one-line `ok_or`, but it is a default, not a wrong answer, and it surfaced while
   re-reading the function for R-01 rather than from a reported symptom.
2. **The granularity count rule now has two implementations** — `metadata_len` in
   `artifact.rs` and the `rows / last.div_ceil(group)` arithmetic inside
   `quant::dequantize`. They agree today (the 18 parity runs load through both),
   and the loader's copy is tested, so this is not a defect; if the exporter's
   grouping ever changes, consider having `dequantize` call the shared helper so
   the flattening rule has one definition.
3. **`results/perf/README.md:204-205` still prints the benchmark binary's hash**
   (`md5 e557ade7...`, `sha256 29cd6de5...`). That build is now older than the
   tree (the repaired build is `04c9b05c...`) because the repair changed
   validation and CLI code. The numbers are unaffected (see the kernel-diff note
   above), but the README should say the hash identifies the *benchmark* build so
   it is not read as the current artifact. Out of this review's scope (`results/`).
4. **Resolved since round 1:** `results/perf/*` (README, `bench_host.csv`,
   `bench_router.csv`, `crate_survey*.csv`, `parity_before_after.csv`,
   `run_bench.sh`, `summarise.py`) are now tracked, so the router evidence the
   definition of done asks for is committed rather than only on disk.
5. `rust/inmotion-core/src/tensor.rs:4` — the last stale microcontroller claim —
   is fixed in the revision reviewed here, closing R-09 completely. It required
   the owner to edit a file outside `t15`'s declared scope, which the completion
   validator rejected earlier; the captain's widened scope is what closed it.

## Judgments carried forward (unchanged by the repair)

- **The tokenizer abstraction genuinely removes duplication.** Round-1 judgment
  still holds on these hashes: `world_model.rs` is one `forward` for lejepa,
  t_jepa, ts_jepa and cf_jepa; `lejepa.rs` stays deleted; the only branches are the
  tokenizer and the tail, both of which mirror a real difference in the reference;
  T-JEPA's `[REG]` drop is in the shared path and pinned by a hand-built model.
- **The kernels are correct and bit-exact.** The repair did not touch them; the
  two bit-exactness tests still pass in release on gnu and on the router's musl
  baseline, and parity is unchanged on both binaries.
- **The loaders now reject rather than guess.** `metadata_len` is a good addition:
  it makes the loader's expectations explicit and tested instead of trusting the
  exporter, and it closes the `per_tensor`-with-extra-scales ambiguity that
  `dequantize` would otherwise have accepted silently.

## Modules that are clean (round 2)

`tensor.rs`, `model.rs`, `world_model.rs`, `sigreg.rs`, `ops.rs`, `quant.rs` and
`artifact.rs` have no open findings; `main.rs` has R2-01 only, in `check`, and the
rest of the file (window validation, argmax, `describe`, JSON bench output) is
clean. No new dead code, no new CPU-feature assumption, no new undocumented public
item — clippy with `-D warnings` and `missing_docs` passes from an empty target
directory.
