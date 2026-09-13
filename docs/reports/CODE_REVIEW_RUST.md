# Code review — Rust world-model port and CPU kernels

Task `t10` (review, attempt 1), reviewing `t2` plus the kernels from `t5`.
Verdict: **needs_revision**. There is **no parity or numerics defect**: all 15
model x precision artifacts replay `PARITY OK`, and the bit-exactness claims hold
in optimized builds on both the gnu and the router's `musl` baseline target. The
findings are robustness of the loaders/CLI and documentation accuracy, and two of
them are demonstrated panics that the task's own acceptance criteria forbid.

## Revision reviewed

The tree is uncommitted work, so the exact revision is pinned by hash. Every
number below was produced against these files; they were re-hashed after the
review and did not change.

| file | sha256 |
|---|---|
| `rust/inmotion-core/src/ops.rs` | `484f0cef3c95a14cd8288696fcc990448beb8811679805a30f79b9f250cae4e0` |
| `rust/inmotion-core/src/world_model.rs` | `9a5760749286560ef1a268e87089a4b8d99b4db511138e81ac2179bc856b616e` |
| `rust/inmotion-core/src/sigreg.rs` | `d7e3f1de3c8eba2fb70721fce76a2349635a3d4bb6997e6e126e3a5c04f49f87` |
| `rust/inmotion-core/src/quant.rs` | `ceeaae79ae870bd09b4814d431be079d8ced99dea68a3e7c792c21ef349f9177` |
| `rust/inmotion-core/src/artifact.rs` | `cfaa6679fb5d485102ec3a817d0888091851eefad5a4bb7c2d95dfb45b410254` |
| `rust/inmotion-core/src/model.rs` | `e406be1dba87737888c60d7015d49f510822714ebfb7a63e5160e269293d6954` |
| `rust/inmotion-core/src/tensor.rs` | `9fcc2134124220febd6ef95bea97084a0ab5d5a39139a5dbb91e69debe800080` |
| `rust/inmotion-core/src/lib.rs` | `d88d2d96d9215f55e29698409c6daade9848c0c9193bb2ce5308fbbca4de7dd3` |
| `rust/inmotion-cli/src/main.rs` | `25928df919d7f7c6173756b4d0a4a51aa64caa07b1f46c2c60695d60e2ecb952` |

## The enforcement gates, run for this review

`cargo clippy --all-targets -- -D warnings` was re-run against an **empty target
directory** so a cached "fresh" result could not mask anything:

```
$ CARGO_TARGET_DIR=/tmp/rev-clippy cargo clippy --all-targets -- -D warnings
    Checking inmotion-core v0.1.0 (.../rust/inmotion-core)
    Checking inmotion-cli v0.1.0 (.../rust/inmotion-cli)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 5.29s   # exit 0
```

So pedantic clippy, `unsafe_code = "forbid"` and `missing_docs` all hold:
`grep -rn 'unsafe\|std::arch\|_mm_\|is_x86_feature' rust/inmotion-core/src rust/inmotion-cli/src`
returns nothing, and every public item carries a doc comment.

```
$ cargo test                                             # 48 passed; 0 failed
$ cargo test --release                                    # 48 passed; 0 failed
$ cargo test --release --target x86_64-unknown-linux-musl  # 48 passed; 0 failed
```

The `musl` run matters: it exercises the kernel code at the router's baseline
feature level, and the two bit-exactness tests pass in an optimized build there
too:

```
test ops::tests::packed_matches_linear_bit_for_bit ... ok
test ops::tests::attention_is_bit_exact_after_the_rewrite ... ok
```

## Evidence reproduced independently

Parity, replaying the golden vectors with the shipped binary against the
artifacts the port tasks exported (`/tmp/art/<model>-<precision>`) — 15/15
`PARITY OK`, numbers identical to those quoted by `t2`, `t3` and `t4`:

```
cf_jepa-fp32 :: max |diff| : 0.00000221  PARITY OK     lejepa-fp32 :: 0.00000405
cf_jepa-int8 :: max |diff| : 0.00000295  PARITY OK     lejepa-int8 :: 0.00000423
cf_jepa-q4   :: max |diff| : 0.00000191  PARITY OK     lejepa-q4   :: 0.00000334
t_jepa-fp32  :: max |diff| : 0.00000262  PARITY OK     ts_jepa-fp32 :: 0.00000274
t_jepa-int8  :: max |diff| : 0.00000414  PARITY OK     ts_jepa-int8 :: 0.00000292
t_jepa-q4    :: max |diff| : 0.00000620  PARITY OK     ts_jepa-q4   :: 0.00000501
sigreg-fp32  :: max |diff| : 0.00000123  PARITY OK     sigreg-int8  :: 0.00000197
sigreg-q4    :: max |diff| : 0.00000143  PARITY OK     (tolerance 0.000200)
```

These are the figures as measured in this round-1 review, against the artifacts
that existed then. The `sigreg` row is a **pre-dataset.csv-retrain record**: after
the retrain the current tree re-exports and reproduces `sigreg 8.3e-7 / 8.3e-7 /
9.5e-7`, which `rust/README.md`'s parity table carries, and `cf_jepa` moved to
`2.21e-6 / 2.95e-6 / 1.91e-6`. `rust/README.md` is the live record; this block is
history (finding F4 in `docs/reports/VERIFICATION.md`).

No CPU feature the router lacks in any `inmotion` code path. The check as
originally run was narrower than the binary, so read it with the qualification:

```
$ objdump -d rust/target/x86_64-unknown-linux-musl/release/inmotion-infer \
    | grep -cE '^\s+[0-9a-f]+:\s+[a-z0-9]+\s+%v[0-9]'
0
$ sha256sum rust/target/x86_64-unknown-linux-musl/release/inmotion-infer
29cd6de5b482f3505bfdee2e3837ece5185db00b3e6dd86be58b96f42075faf3
```

The pattern above only matches a VEX instruction whose *first* operand is a
`%v`-register, so its `0` does not mean the binary is AVX-free. Counting VEX
mnemonics instead gives 282, all of them inside `memchr`'s three
runtime-CPUID-dispatched `find_avx2` symbols, which this CPU never selects — no
`inmotion` symbol contains one (F3; the corrected statement and the commands live
in `results/perf/README.md`).

The hash above identifies the deployed binary, which matches
`results/perf/README.md`'s recorded hash. There is no `.cargo/config.toml`
in `rust/` and no `RUSTFLAGS` in the environment, so the baseline feature level is
the default rather than an accident of the host.

## Findings

Summary (one row per finding; details and required fixes follow):

| id | severity | file:line | problem |
|---|---|---|---|
| R-01 | medium | `rust/inmotion-cli/src/main.rs:127` | parity harness compares `got`/`expected` with `zip`, so a short output passes |
| R-02 | medium | `rust/inmotion-core/src/artifact.rs:283` | quantized metadata count/dtype never validated; `quant.rs:123,157` then panics |
| R-03 | medium | `rust/inmotion-cli/src/main.rs:84` | `partial_cmp(..).unwrap()` panics on a NaN logit, and NaN input is accepted |
| R-04 | low | `rust/inmotion-cli/src/main.rs:113` | `unwrap()` on golden-file rows panics instead of returning `Err` |
| R-05 | low | `rust/inmotion-cli/src/main.rs:176` | `bench --iterations 0` indexes an empty vector |
| R-06 | low | `rust/inmotion-core/src/artifact.rs:191` | unchecked `usize` arithmetic on untrusted header fields |
| R-07 | low | `rust/inmotion-core/src/ops.rs:925` | `transpose_time_channel`/`transpose_channel_time` are dead + duplicate `transpose_2d` |
| R-08 | low | `rust/README.md:46` | "Only `lejepa` is ported" is false; parity table covers one model |
| R-09 | low | `rust/inmotion-core/src/lib.rs:7` | ESP32-S3 / PSRAM / "microcontroller build" claims contradicted by the verified part |
| R-10 | low | `rust/inmotion-core/src/artifact.rs:80` | `as usize` silently truncates arch fields on a 32-bit target |
| R-11 | low | `rust/inmotion-core/src/ops.rs:884` | the all-padding guard in `conv1d_same` is never exercised by a test |
| R-12 | low | `rust/inmotion-core/src/quant.rs:22` | duplicated doc sentence (pre-existing) |
| R-13 | low | `rust/inmotion-cli/src/main.rs:78` | logits cloned and reshaped for a softmax that accepts them as-is |
| R-14 | low | `rust/Cargo.toml:11` | "# Enforced in CI" — there is no Rust CI in this repo |

### R-01 (medium) — the parity harness can print `PARITY OK` on a short comparison

`rust/inmotion-cli/src/main.rs:127`:

```rust
let diff = got
    .iter()
    .zip(&expected)
    .map(|(a, b)| (a - b).abs())
    .fold(0.0f32, f32::max);
```

`zip` stops at the shorter side, and nothing before it compares lengths or
shapes. This is the command that certifies the team's definition of done, and
`tests/test_quant_export.py::test_rust_port_reproduces_pytorch` asserts only
`returncode == 0` and `"PARITY OK" in stdout`.

Demonstrated: I copied `lejepa-fp32`'s golden file and truncated every case's
`logits` from 4 elements to 1, then ran the harness against it —

```
$ inmotion-infer check --artifact /tmp/trunc-golden --golden /tmp/trunc-golden
cases      : 16
max |diff| : 0.00000226
tolerance  : 0.000200
PARITY OK          # exit 0, against a golden with 25% of the expected output
```

Required fix: before the fold, reject a length mismatch (and compare
`got.len()`/`expected.len()` against the model's `num_classes()`), e.g.
`if got.len() != expected.len() { return Err(format!("case {i}: expected {} logits, got {}", expected.len(), got.len())); }`,
so a wrong-shaped output can never be reported as verified.

### R-02 (medium) — quantized metadata is not validated, so a malformed artifact panics

`Artifact::quantized_weight` (`rust/inmotion-core/src/artifact.rs:283`) checks
that the tensor is quantized, that `bits` and `granularity` exist, and then
copies `scale` and `zero_point` across without ever comparing their lengths or
the `.zp` dtype:

```rust
let scale = self.raw_f32(&format!("{name}.scale"))?;
let zero_point = self.raw_i32(&format!("{name}.zp"))?;
```

`raw_i32` (`artifact.rs:339`) also performs no dtype check at all (unlike
`raw_f32`, which rejects anything but `F32`), so an `F32` `.zp` buffer would be
reinterpreted as `i32` and silently produce wrong weights. With `scale` longer
than `zero_point`, `dequantize` indexes out of bounds:

```rust
// rust/inmotion-core/src/quant.rs:123 (per_channel)
(f32::from(u) - qw.zero_point[channel] as f32) * qw.scale[channel]
// rust/inmotion-core/src/quant.rs:157 (per_group)
(f32::from(u) - qw.zero_point[meta] as f32) * qw.scale[meta]
```

Demonstrated: I rebuilt `model.safetensors` for `lejepa-int8` with the
`encoder.tokenizer.weight.zp` buffer shrunk from 128 entries to 1 (header shape
and offsets rewritten consistently, `meta.json` untouched — i.e. exactly what a
truncated export looks like), then ran the CLI:

```
$ inmotion-infer describe --artifact /tmp/bad-art
thread 'main' panicked at inmotion-core/src/quant.rs:123:50:
index out of bounds: the len is 1 but the index is 1        # exit 101
```

Required fix: validate in `quantized_weight` (at load, not at first use) that
`scale.len() == zero_point.len()`, that `.zp` has dtype `I32` (add the check to
`raw_i32`), and that the counts match the declared granularity — `1` for
`per_tensor`, `shape[axis]` for `per_channel`, `rows * last.div_ceil(group)` for
`per_group` — returning `Err` as the neighbouring code already does for the
`per_channel` scale count in `dequantize`. A malformed artifact must not be able
to reach an indexing panic.

### R-03 (medium) — `predict` panics on a non-finite logit, and NaN input is accepted

`rust/inmotion-cli/src/main.rs:80`:

```rust
let (best, &confidence) = probs
    .data
    .iter()
    .enumerate()
    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
    .unwrap();
```

`partial_cmp` returns `None` for `NaN`, and `read_window`
(`main.rs:34`) validates only the *count* of values, so a window containing
`nan` or `inf` — both of which `str::parse::<f32>` accepts — flows through
`layer_norm`/`gelu`/`softmax` as `NaN` and panics here. Demonstrated with a
40-value `lejepa-fp32` window whose 8th value is `nan`:

```
$ inmotion-infer predict --artifact /tmp/art/lejepa-fp32 --input /tmp/nan-window.txt
thread 'main' panicked at inmotion-cli/src/main.rs:84:45:
called `Option::unwrap()` on a `None` value                  # exit 101
```

Required fix: use the total order (`max_by(|a, b| a.1.total_cmp(b.1))`) *and*
reject non-finite input in `read_window` (`if !v.is_finite() { return Err(...) }`
while mapping), since a `NaN` window is a broken measurement, not a prediction
request.

### R-04 (low) — golden-file rows are unwrapped instead of returned as errors

`rust/inmotion-cli/src/main.rs:113`, `:115`, `:122` use `row.as_array().unwrap()`,
`v.as_f64().unwrap()`, while the *same* expression at `:109` carefully returns
`Err("case has no input")`. A truncated `golden.json` therefore panics with an
`Option::unwrap()` message rather than naming the bad field.

Required fix: replace each with `ok_or_else(|| format!("case {i}: ..."))?`, as
the surrounding code already does.

### R-05 (low) — `bench --iterations 0` panics

`rust/inmotion-cli/src/main.rs:234` accepts any parsed `usize`, including `0`;
`samples` is then empty and `:179`/`:181` index it:

```
$ inmotion-infer bench --artifact /tmp/art/lejepa-fp32 --iterations 0
thread 'main' panicked at inmotion-cli/src/main.rs:179:25:
index out of bounds: the len is 0 but the index is 0        # exit 101
```

Required fix: `if iterations == 0 { return Err("--iterations must be > 0".into()); }`
before timing (the value could equally be rejected at parse time in `main`).

### R-06 (low) — unchecked arithmetic on untrusted header fields

`rust/inmotion-core/src/artifact.rs:191` and `:208`:

```rust
let header_len = u64::from_le_bytes(blob[..8].try_into().unwrap()) as usize;
let header_end = 8 + header_len;
...
if end < start || header_end + end > blob.len() {
```

`8 + header_len` and `header_end + end` are unchecked. Demonstrated with an
8-byte header length of `usize::MAX - 3` (release binary, so wrapping, not a
debug-assertion panic):

```
$ inmotion-infer describe --artifact /tmp/badlen
thread 'main' panicked at inmotion-core/src/artifact.rs:197:41:
slice index starts at 8 but ends at 4                        # exit 101
```

Required fix: use `checked_add` (and `usize::try_from` for the u64 fields) and
return the existing "header length exceeds file size" / "data_offsets ... lie
outside the buffer" errors when the arithmetic is not representable.

### R-07 (low) — two dead helpers duplicating `transpose_2d`

`rust/inmotion-core/src/ops.rs:925-959` defines `transpose_time_channel` and
`transpose_channel_time` as `(B, T, C) <-> (B, C, T)`. Their only callers in the
whole repository are their own round-trip test (`ops.rs:992-998`); every real
tokenizer moves the axis with `transpose_2d` instead
(`world_model.rs:615`,`:617`, `sigreg.rs:295`), which is the same operation for
the batch-of-1 the crate supports and is what the artifact format actually
produces. `grep -rn 'transpose_time_channel\|transpose_channel_time'` over the
repository returns only `ops.rs` (definition + test).

Required fix: delete both functions and their test (routing the tokenizers
through them would re-introduce a 3-D batch axis the graph does not have).

### R-08 (low) — `rust/README.md` still describes the port as lejepa-only

`rust/README.md:46`: "Only `lejepa` is ported. The other three reported JEPA
models have structurally different encoders — T-JEPA has a feature-index
tokenizer, TS-JEPA and CF-JEPA are patch-based with a `context_encoder` — so each
needs its own `forward`, sharing the ops in `ops.rs`." The current tree serves
all five families from one stack (`world_model.rs` + `sigreg.rs`), and the parity
table above it (`:36`) lists one model at three precisions.

Required fix: rewrite the Scope section to state the five families served by the
single shared stack and either extend the parity table to 5 models x 3 precisions
or point at `results/perf/README.md` for the full table. (A reader relying on
this file would conclude the deliverable is incomplete.)

### R-09 (low) — stale hardware premise in the crate docs

The verified part is an **ESP32-D0WD-V3 (classic ESP32, no PSRAM)**
(`docs/reports/ESP32_FEASIBILITY.md:13,16,19-20`: "This contradicts the earlier
plan's premise of an ESP32-S3 with 8 MB PSRAM"), and the working ESP32 firmware is
C (`firmware/main/jepa_model.c`), not this crate. The Rust docs still say:

- `rust/inmotion-core/src/lib.rs:7` — "The target includes an ESP32-S3";
- `rust/README.md:13` — "the target includes an ESP32-S3";
- `rust/inmotion-core/src/artifact.rs:6` — "small enough for a microcontroller build", and `:140` — "On a microcontroller served by slow PSRAM";
- `rust/README.md:66` — "can build without `std` math for a microcontroller target".

The crate has no `#![no_std]`, uses `std::fs`/`HashMap`/`PathBuf`
(`artifact.rs:175-187`) and is only ever built for `x86_64` (gnu + musl); there is
no MCU target build in the tree and no `esp`/`xtensa` reference in `rust/`.

Required fix: state the actual targets (x86_64 gnu + musl baseline; the ESP32 demo
lives in `firmware/` as C) and drop the S3/PSRAM wording. This is documentation
accuracy, not a design constraint: `libm` remains justified by the portable math
it provides.

### R-10 (low) — `as usize` hides truncation on a 32-bit target

`rust/inmotion-core/src/artifact.rs:80` (`usize_field`) and `:109`
(`usize_list_field`) do `.map(|v| v as usize)`. The workspace lints allow
`cast_possible_truncation`, so on any 32-bit target (which the docs in R-09 claim
to support) a corrupted `arch` value above `u32::MAX` becomes a small number
instead of an error — a silent architecture guess, the exact failure class the
new `Meta` readers were introduced to remove.

Required fix: `usize::try_from(v).map_err(|_| format!("`{key}` does not fit a usize"))`
in both readers.

### R-11 (low) — the all-padding guard in `conv1d_same` has no test

`rust/inmotion-core/src/ops.rs:880-887`:

```rust
let t0 = pad.saturating_sub(k);
let t1 = (length + pad).saturating_sub(k).min(length);
if t0 >= t1 {
    continue;
}
```

The two tests at `ops.rs:543-567` both use `length >= kernel`, where `t0 < t1`
always holds, so this branch (a tap entirely inside the padding) is never taken.
The parity vectors do not reach it either (`seq_len = 10`, kernels 3/5/7).

Required fix: add one test with a signal shorter than the kernel (e.g.
`length = 1`, `kernel = 3`, asserting the two fully-padded taps are skipped and
the centre tap is used), or delete the branch and document that `length >= kernel`
is a precondition.

### R-12 (low) — duplicated doc sentence

`rust/inmotion-core/src/quant.rs:22` repeats "Parse the name the exporter writes,
or `None` if it is unknown." on two consecutive lines (present at `HEAD` too, so
pre-existing rather than introduced by this work).

Required fix: delete one line.

### R-13 (low) — redundant clone + reshape before softmax

`rust/inmotion-cli/src/main.rs:78` clones the logits and reshapes them into a
tensor they already are (`forward` returns `(1, num_classes)`, per
`world_model.rs:737`/`sigreg.rs:327`):

```rust
let probs = inmotion_core::ops::softmax_last(&Tensor::new(logits.data.clone(), vec![1, logits.len()]));
```

Required fix: `let probs = inmotion_core::ops::softmax_last(&logits);`.

### R-14 (low) — "# Enforced in CI" names a CI that does not exist

`rust/Cargo.toml:11` says the lints are enforced in CI. The only workflow in the
repository is `.github/workflows/astro.yaml` (the website), and the git history
records the removal (`46dca2b get rid of ci`, `1137cbd chore: run every quality
gate locally instead of in CI`).

Required fix: reword to the local gate that actually enforces them
(`cargo clippy --all-targets -- -D warnings`, `cargo test`).

## Judgment: does the tokenizer abstraction genuinely remove duplication?

**Yes.** It removes it rather than hiding it.

- The single stack is real: `world_model.rs:639-738` is one `forward` for lejepa,
  t_jepa, ts_jepa and cf_jepa, and `lejepa.rs` is deleted (`git status` shows
  `D rust/inmotion-core/src/lejepa.rs`) with `grep -rn 'lejepa::'` returning
  nothing; `lib.rs` no longer exports a per-family model type.
- The variation is confined to two matches, and both name things that genuinely
  differ in the reference: the tokenizer
  (`TokenizerWeights` + one dispatch, `world_model.rs:604-635`) and the tail
  (`world_model.rs:692-711`). The tail branch is required, not duplicated logic:
  `lejepa`'s `encoder.proj`+`bn` is `Linear + GELU + BatchNorm1d`, while
  `t/ts/cf_jepa` end in a `LayerNorm`, which is why `EncoderTail` has exactly two
  variants and `tail_from_artifact` (`:262`) cross-checks the tensors.
- T-JEPA's one genuinely different rule — `[REG]` slots are not pooled — sits in
  the shared `forward` (`world_model.rs:716-731`) instead of forking a code path,
  and it is pinned by a hand-built mini model whose pooled value would differ if
  the slot were kept (`pooling_drops_the_reg_tokens`, `:841`).
- The variant and the weights cannot disagree, because both come from the same
  validated `Config` (`TokenizerWeights` is built inside the match that consumes
  `config.tokenizer`, `world_model.rs:507-531`).
- The remaining derived value, the patch stride (`patch_stride`, `:111`), is
  recovered from the artifact's `n_tokens` and errors unless the mapping is exact
  (`span % steps != 0` fails), so it is not a silent guess; if the exporter can
  record it, recording it would be marginally better, but nothing here guesses.

## Judgment: the kernels (`t5`)

Reviewed hardest, as instructed, because that is where hardware and numerics risk
concentrate.

- `linear_packed`/`PackedWeight` (`ops.rs:31-111`) seeds each accumulator with the
  bias and adds `x[i] * w[i]` in ascending `i` — the same operands in the same
  order as `linear` (`ops.rs:16-27`), which is what makes the bit-exactness
  plausible rather than lucky. It is **verified**, not asserted: both
  `packed_matches_linear_bit_for_bit` and `attention_is_bit_exact_after_the_rewrite`
  (the latter against a retained copy of the old triple loop, `ops.rs:659-696`)
  pass in release on gnu *and* on the router's musl baseline target. That kills
  the two risks I looked for first — FMA contraction differing between kernels, and
  the rewrite being merely epsilon-equal.
- The attention rewrite allocates `kh` (`ops.rs:366`) and the context accumulator
  per head; both preserve the original per-element add order (`c` ascending, then
  `j` ascending), which the parity numbers confirm end-to-end (t_jepa q4 6.20e-6
  is the worst case, 32x inside tolerance).
- `conv1d_same`'s tap-interval arithmetic (`ops.rs:882-887`) is correct for
  `length >= kernel`: `t0 = max(0, pad-k)`, `t1 = min(length, length+pad-k)`,
  `src0 = t0+k-pad`, and the `(ci,k)` reduction order is preserved (ci-major,
  k-minor) exactly as the replaced triple loop summed. Its one uncovered branch is
  R-11.
- No ISA assumption in project code: no `std::arch`, no `is_x86_feature_detected!`, no
  `target_feature`, no `RUSTFLAGS`/`target-cpu`, and no `inmotion` code path emits
  AVX, AVX2 or FMA. The deployed musl binary is *not* free of AVX-encoded
  instructions as a whole — it carries 282 VEX mnemonics, all in `memchr`'s three
  runtime-CPUID-dispatched `find_avx2` symbols, which the J3455 (no AVX/AVX2/FMA)
  never selects (see `results/perf/README.md`'s "router build" section, which
  F3 corrected to say exactly this). So the binary runs there, and
  `results/perf/README.md`'s crate survey is
  consistent with the hand kernel winning there and losing on the AVX2 host.

## Modules that are clean

Stated explicitly, because the findings above are a minority of the code:

- `rust/inmotion-core/src/tensor.rs` — a 60-line dense type with every invariant
  asserted in the constructor; nothing to flag.
- `rust/inmotion-core/src/model.rs` — family dispatch that reads `arch.family`
  and errors on anything unknown (`:24-30`); no defaults, no name-based branching,
  per-family `describe()`.
- `rust/inmotion-core/src/world_model.rs` — every architecture field is required
  and cross-checked against the named tensor shapes (`:134-370`), the layer count
  is read rather than probed (`:337-349`), missing/extra `pos_embed` and
  `input_proj` are both errors (`:234-258`, `:311-329`), and the two tests that do
  not need a checkpoint pin the tokenizer and pooling rules.
- `rust/inmotion-core/src/sigreg.rs` — the CNN branch declares its kernels as data
  and compares them against what it implements (`front_kernels`/`block_kernels`,
  `:74-87`), validates every conv width against `num_filters` before loading, and
  folds each `BatchNorm1d` once via the single shared
  `fold_batch_norm`+`channel_affine` path (`ops.rs:156-239`) instead of
  re-deriving a second fold.
- `rust/inmotion-core/src/ops.rs` — numerics (see the kernel judgment); the only
  complaints are R-07 and R-11.
- `rust/inmotion-core/src/quant.rs` — the dequantization formula is uniform
  (`(u - zp) * scale`), `unpack` validates its bit width and buffer length
  (`:60-70`), and each granularity validates the counts it needs; the gap is that
  the *loader* does not validate what `quant.rs` then trusts (R-02).
- `rust/inmotion-core/src/artifact.rs` — the blob is read once and indexed, not
  re-read per tensor (`:135-150`), dtype/byte-count integrity is checked at load
  (`:214-231`), and the new `Meta` readers make a missing architecture field an
  error (`:69-124`); R-02, R-06 and R-10 are the gaps.
- `rust/inmotion-cli/src/main.rs` — `describe` now asks the model to describe
  itself (`model.describe()`, `:62`) instead of printing transformer
  hyperparameters for a CNN, and the JSON bench output is built with
  `serde_json::to_string` so model names and paths cannot break the document
  (`:190-200`); R-01, R-03, R-04, R-05 and R-13 are the CLI gaps.

## Observations for the captain (not Rust findings, no required fix here)

1. `t5`'s task record has **no output recorded**, yet its evidence is real:
   `results/perf/README.md`, `bench_host.csv`, `bench_router.csv`,
   `crate_survey*.csv`, `summary.csv`. Those files are **gitignored**
   (`git check-ignore -v results/perf/README.md` -> `.gitignore:263:results/`), so
   the router numbers that the brief asks for are not committed anywhere. The
   claim in `ops.rs:42-45` ("1.7x faster than the `gemm` crate and 1.9x faster
   than `matrixmultiply`") is backed by `crate_survey_router_j3455.csv`, which
   records the three J3455 rows for that shape:

   ```
   lejepa d_model proj,10,128,128,rank1_packed,0.060800,0e0
   lejepa d_model proj,10,128,128,gemm_crate,0.105323,3.5762787e-7
   lejepa d_model proj,10,128,128,matrixmultiply_sgemm,0.112805,3.5762787e-7
   ```

   i.e. 0.105323 / 0.060800 = 1.73x and 0.112805 / 0.060800 = 1.86x (the perf
   README reduces the same three runs to 1.73x / 1.84x; the shipped comment's
   "1.7x / 1.9x" is consistent with both). So the code comment is backed by an
   artifact — just an untracked one.
2. The bit-exactness property that the whole rewrite rests on is currently
   pinned only by unit tests on synthetic inputs plus the 16-case golden vectors
   per model. That is adequate today; a golden vector with `t > 100` tokens would
   also exercise the `T = 182` path the attention rewrite targets. Not a finding:
   `t_jepa`'s exported windows do use that path (`results/perf/README.md`
   "At T = 182 (T-JEPA)"), so it is already covered end to end.

## Verdict

`needs_revision`. No finding touches parity, the kernel numerics or the measured
speedups — those are verified independently above, on both the host and the
router's baseline target. What must change before this port is signed off:

- the three medium findings, because R-02 and R-03 are panics reachable from a
  malformed artifact and from user input (the acceptance criteria that t10 must
  judge say a malformed artifact may not reach an `unwrap`/index panic), and R-01
  weakens the harness that certifies everyone else's parity claims;
- R-07 (dead, duplicated helpers), which is an explicit acceptance criterion;
- the documentation in R-08/R-09/R-14, which currently misstates what the port
  contains and what it targets;
- R-04/R-05/R-06/R-10/R-11/R-12/R-13 as one-line robustness and hygiene fixes.

Everything was verified with `cargo clippy --all-targets -- -D warnings` and
`cargo test` (debug and release, gnu and musl) plus the parity, `objdump` and
malformed-artifact reproductions quoted above.
