# Round-3 code review — final verdict (t11, review round 2)

Task `t11`, the live review gate for this closure. The environment recovered, the pass
was executed rather than read, and this revision records its result.

## Verdict: **pass**

Every acceptance criterion is verified **by execution** at the pin below. The pass also
found two genuine defects — a drifted digest record and a stale cited hash — and both
were corrected and re-verified before this verdict was written; §3 records them, because
the check that caught them is the point of the gate rather than a footnote to it.

The guard is now settled as behaviour, not as text: in a `.git`-seeded disposable clone,
an in-repo write destination is refused with exit 1, the `--in-repo` escape hatch still
works, and the bare default still refuses.

## 1. Revision judged

| file | sha256 | size |
|---|---|---|
| `results/perf/make_before_kernels.py` | `9a8e4a99…` | 11213 B (259 lines) |
| `results/perf/ops_before_optimisation.rs` | `9f6b8ab5…` | 35748 B (952 lines) |
| `results/perf/ops_retained_original.rs` | `99a63798…` | 32402 B |
| `results/perf/README.md` | `bf8c9d73…` | 35506 B |
| `results/quant/stats_manifest.json` | `b8646f90…` | 1445 B |
| `rust/inmotion-core/src/ops.rs` | `92963c1f…` | 36316 B |

The witness was edited again at 02:21:23, after the verdict was first drafted, so the pin
above was re-taken and every value-dependent check in §3 was re-run against it; the
verdict is recorded against *this* revision, not the earlier one. A whole-tree sweep for
every superseded value (`9c26a2d0`, `a48b5328`, `c0ba3d7c`, `4029b65e`, `cf9900d6`,
`9b453a08`, `6caa95a7`, `e613d66a`, `914c0f01`, `d3f2664e`, and the string `20-876`)
returns no hit in any tracked file. Because the tree has demonstrated that it can move
between two consecutive measurements, the verdict is only meaningful for the pinned
revision; a further edit means re-taking the digest and hash checks rather than trusting
this page.

Rust §1 set, re-measured with `ops.rs` last: `artifact.rs` `2bb3943f`, `lib.rs`
`efa9b78e`, `model.rs` `e406be1d`, `quant.rs` `e60c5ff0`, `sigreg.rs` `d7e3f1de`,
`tensor.rs` `8d4a8998`, `world_model.rs` `a684b553`, `Cargo.toml` `312c0d3c`,
`rust/README.md` `7e9d5e50`, `main.rs` `e61d6b79`, `ops.rs` `92963c1f`. All eleven match
the round-2 pins, so round 2's judgments carry forward by construction. `HEAD` is
`8266b08`; nothing was committed by this review.

The script is byte-identical to the revision the guard matrix was executed against, so
those results apply unchanged to this pin.

## 2. Acceptance criteria, verified by execution

| criterion | result | evidence |
|---|---|---|
| README's three false clauses gone; outlier counts match a fresh recomputation | **passed** | greps for `ten columns`, `10 columns`, `router-only`, `merged nothing`, `not produced by one harness` all return 0; recomputed from `HEAD:summary.csv` vs the current table: **8/15** host cells exceed 3 % on the speedup ratio and **7/15** on the after-median, exactly as stated |
| no regression in the previously passed items | **passed** | clippy `--all-targets -- -D warnings` exit 0 from an empty `CARGO_TARGET_DIR`; `cargo test --release` and `--target x86_64-unknown-linux-musl` both 50 passed / 0 failed / 0 ignored; **15/15 PARITY OK on gnu and 15/15 on musl**, every `max |diff|` equal to the committed digits; F1 clocks 160/240 MHz with medians 888.4805/636.6805, no escape bytes, `_clock_mhz` absent; F2 fresh clone of the tracked set runs `plots quantization` exit 0 with **14/14** outputs byte-identical; F5 four manifest md5s MATCH on re-hash with latency `md5: null` and 3 `latency_artifacts`; F6 `All 18 model(s) verified.`; N-1 checklist reads 18; N-3 `.vscode/` ignored with `CITATION.cff` present at this pin (removed later in `37e3983`) |
| all ten Rust files other than `main.rs` byte-identical to the round-2 pin; `ops.rs` still `92963c1f` | **passed** | the eleven hashes above |
| the guard refuses an in-repo write and still honours `--in-repo` | **passed** | the matrix below |

### 2.1 The guard matrix, in a `.git`-seeded disposable clone

Seeded with `~/inmotion-recovery/ops.rs.92963c1f`; the live tree hashed before and after
every case and never touched.

| invocation (cwd = clone root) | exit | clone `ops.rs` |
|---|---|---|
| (a) bare, no argument | 1 | unchanged `92963c1f…` |
| (b) `--out /tmp/rv-o1.rs` | 0 | unchanged; out `a7fadc80…` |
| (c) `--out rust/inmotion-core/src/ops.rs` | **1** | **unchanged** — "refusing to write … it is inside the tree this run would edit" |
| (c-IN) same, plus `--in-repo` | 0 | `a7fadc80…` (positive control) |
| (d) positional `/tmp/rv-scratch.rs` | 0 | scratch `a7fadc80…` |

Both directions of the three-way rule are covered: a later change that removes the
refusal fails (c), and one that makes it unconditional fails (c-IN).

### 2.2 Per-item verdict

One entry per item the gate is required to rule on. "Re-run" means this pass executed it
at the pin in §1; "carried" means an earlier review in this round executed it and the
inputs are unchanged since — every one of the eleven Rust §1 hashes still equals its
round-2 pin, and `results/quant/` and the three perf CSVs are unchanged, so a carried
verdict rests on byte-identical inputs rather than on trust. The distinction is stated
rather than flattened because "verified earlier" and "verified at this revision" are not
the same claim.

| item | verdict | basis | evidence |
|---|---|---|---|
| F1 — clock recorded in the artifact, not the filename | pass | re-run | clocks 160/240 MHz, medians 888.4805 / 636.6805, no escape bytes, `def _clock_mhz` absent from the source |
| F2 — fresh clone reproduces the figures | pass | re-run | tracked set cloned (527 files), `plots quantization` exit 0, **14/14** outputs byte-identical |
| F3 — reference/disassembly claims | pass | carried | round-2 verification (282 VEX mnemonics, 3 `memchr` symbols, 0 `inmotion`); `REFERENCE`-side inputs unchanged |
| F4 — F4 reconciliation | pass | carried | round-1/round-2 reconciliation; unchanged inputs |
| F5 — stats tables reproducible | pass | re-run | size/accuracy/coverage/layers md5s MATCH on re-hash; latency `md5: null`; 3 `latency_artifacts` |
| F6 — model registry | pass | re-run | `All 18 model(s) verified.` |
| F7 — non-monotone claim | pass | carried | 0/9 non-monotone as re-scoped under t12; declaration-only, no artifact dependency |
| F8 — false provenance clauses | pass | re-run | all five false-clause greps return 0; provenance re-derived from git history (HEAD `bench_router.csv` 11 cols × 90 rows, one commit `9f3e5fd`; shipping file byte-identical to the device's `bench_router2.csv`, `cab37a42…`; `summary.csv` 30 cells reproduced by `summarise.py`) |
| F9 — outlier count | pass | re-run | recomputed 8/15 on the speedup ratio and 7/15 on the after-median, matching the README |
| F11 — reproducible `before` build, packaging, Files table, **and the guard** | pass | re-run | baseline tracked and repo-relative; the guard matrix in §2.1 executed here — (c) refuses with exit 1 and (c-IN) permits — plus the `.git`-less copy writing by design as the documented scratch behaviour |
| N-1 | pass | re-run | checklist reads 18 |
| N-2 | pass | carried | declaration-only re-scope under t12 |
| N-3 | pass | re-run | `.vscode/` ignored at `.gitignore:213`; `CITATION.cff` present *at the revision judged* — it was removed from the repository afterwards, in `37e3983`, so this clause reads as history. The ignore half still holds. |
| N-4, N-5, N-6 | pass | carried | round-2 verification; unchanged inputs |
| R2-01 — input-width guard | pass | carried | round-2 open item closed under t1/t5: a 10×2 input exits 1 with "input has 20 values, expected 40"; `main.rs` is `e61d6b79`, byte-identical to the verified revision |
| round-2 open item 1 (`tol` no longer defaults) | pass | carried | missing `tol` exits 1; a perturbed logit still reports the out-of-tolerance case; same byte-identical `main.rs` |
| round-2 open item 3 | pass | carried | closed by t4 and re-verified under t5; the three CSVs are md5-identical to that delivery |
| R3-01 — digest window | pass | re-run | re-derived from the durable anchor with the 0-based `[s,e]` window: `44362bac…`/`d6b74024…`/`44f8cb58…`/`5413e90d…`, concatenation `489755f8…`; `grep -c 44362bac README.md` = 1 and the other three likewise; the concatenation appears in the README, the fixture header and the witness header; §3 records the cause and the fix |
| R3-02 — fixture-hash citation | pass | re-run | `sha256sum results/perf/ops_before_optimisation.rs` = `9f6b8ab5c1510d05c4998fa966827dc95f93a4d508d7dc7b5725fc6322432c6a`, exactly what `README.md:514` quotes |
| R3-03 — witness payload range | pass | re-run | header reads "lines 20-875 inclusive" in all four places; brute force returns exactly (20, 875) at 31043 B / `70c19066…` |


## 3. The two defects this pass found, and their resolution

Both were real, both were corrected during the pass, and both are re-verified at the pin
in §1. They are recorded because they are exactly the class of failure this gate exists
to catch — a record that disagrees with the artefact while the artefact is sound.

### R3-01 — the digest record was computed over a one-line-shifted window *(fixed)*

At the pin I first judged, the four body digests and their concatenation — in the README
table, the README prose, the fixture header, the witness header and the frozen-pass note
— read `9c26a2d0…`, `a48b5328…`, `c0ba3d7c…`, `4029b65e…`, concatenated `cf9900d6…`.
Those values are not reproducible under the extraction rule the same documents state,
which is the script's own: openers from its `OPEN`, spans from `blocks()` — start at the
line whose `rstrip("\n")` begins with the opener, end at the first subsequent line whose
`rstrip("\n")` is exactly `}` — with the splice `new[d0:d1+1] = old[s0:s1+1]`, i.e. the
window `[s, e]`.

Two measurements settled it, and the second does not depend on either party's extraction:

- **over `[s, e]`**, with newlines preserved, the anchor
  `~/inmotion-recovery/ops.rs.70c19066`, the fixture and the witness agree byte-for-byte:
  `impl PackedWeight` 13 lines `44362bac…`, `linear_packed` 32 `d6b74024…`,
  `self_attention` 59 `44f8cb58…`, `conv1d_same` 45 `5413e90d…`, concatenated
  `489755f8…`;
- **differential**: applying the script to a copy of the repository's `ops.rs` and
  hashing *what it wrote* yields `[54, 66]` = 382 bytes = `44362bac…` for that item —
  equal to the anchor's `[s, e]` block. Whatever either of us believes about extraction,
  those are the bytes the script moves.

A brute force over start/end offsets and join styles showed the previously recorded
values reproduce **only** under `[s-1, e-1]`, a window that opens with the blank line
above the opener and ends at an inner `}` — dropping the item's own closing brace
(`ops_before_optimisation.rs:91-104`: starts at line 91, blank; ends at line 103,
dropping line 104). It was an off-by-one, not a convention.

Consequence, had it stood: the frozen-pass cross-check ("if the concatenation disagrees
while the four per-item digests match, the discrepancy is in the header") would have
fired against a *correct* fixture — the check built to catch a drifted record would have
reported one that was not there, and the digest table would have failed a consistent
artefact.

**Resolved:** at the pin in §1 the README table, the README prose, the fixture header and
the witness header all carry the `[s, e]` values, the four bodies are byte-identical
between anchor and fixture, and `489755f8…` appears in both the README and the fixture
header. The frozen-pass note's expectations (its lines 105-113) also carry the corrected
values, and its lines 260-268 record the defect itself, quoting the wrong values as
wrong and the measured ones as measured. The durable lesson is the one the captain and I
had agreed in advance of it happening: the artefact never moved, the sentences about it
did, and only re-deriving caught it.

**The cause, and why it survived its author's own check** — recorded because it is the
sharpest of this round's five instances. `blocks()` returns **0-based** indices
(`enumerate(lines)`) with an inclusive end, and the correct window is therefore
`lines[s:e+1]`. The wrong values came from writing `lines[s-1:e]`: a one-line shift, not a
misread rule. Two properties made it invisible to inspection:

- the shifted window has the **same length** as the true one — 13/32/59/45 either way — so
  a line-count check cannot separate them, and the fixtures, the anchor and the witness
  all *agreed* on the wrong window because they share the layout that made the shift look
  plausible (a blank line above the opener, an inner `}` one line before the item's own);
- every other invariant held. Only the hash could tell the two windows apart.

That is why the useful instrument was brute-forcing `(start, end)` and join style against
the required digest rather than re-reading a stated rule — the rule was right, and the
implementation of it in the measuring code was off by one. The general form: when a value
is derived by slicing, the slice's *boundaries* are part of the claim, and a boundary
error is invisible to any check that only counts what is inside them.

### R3-02 — `README.md:514` cited a stale fixture hash *(fixed)*

At that earlier pin the README cited the fixture as `9f6b8ab5…` while the file then
measured `9b453a08…`, because the fixture had moved after the citation was written.
**Resolved:** at the pin in §1 the cited value equals the measured value
(`9f6b8ab5c1510d05c4998fa966827dc95f93a4d508d7dc7b5725fc6322432c6a`).

### R3-03 — the witness header claimed the wrong payload range *(fixed; found by another reviewer, not by this pass)*

`results/perf/ops_retained_original.rs` claimed its payload was "lines 20-876 inclusive"
on three separate lines. Brute-forcing every `(start, end)` against the required
31043 bytes and `70c19066…` yields exactly one range — **(20, 875)** — so 876 is a blank
line and 877 is the closing `*/`. The claim was false in a tracked file, and it is the
same drifted-record class as R3-01: the payload was correct while the sentence describing
it was not.

Two things are worth recording about how this one surfaced. First, **this pass did not
find it**: I verified the digest values and the README's cited hash at the pin I judged
but did not re-check the witness's own range claim, which I had encountered earlier in
the round and had left as a §6 note rather than folding into the pass's checks. It was
reported by rust-repair against an intermediate revision, and the gap in my own pass is
the honest part of the record: a claim verified once is not a claim verified at the
revision under judgement. Second, it was fixed between two of my measurements — the
witness moved at 02:21:23, seconds before the pin in §1 was taken — which is why the pin
was re-taken and every dependent check re-run.

**Resolved:** at the pin in §1 the header reads "lines 20-875 inclusive" in all three
places, states the payload as `70c19066…` at 31043 bytes, and its concatenation claim is
`489755f8…`. Measured: payload `[20, 875]` is 31043 bytes with sha256
`70c190665f73f7a0a680c5c1aeb6860ba1513be228266455c8832fa8c40f2f7a`, and every superseded
value including the string `20-876` returns no hit in any tracked file.

## 4. F8's provenance table, verified against the device

Established earlier in this round over SSH on `wave-router` (`CARRIS`,
`Intel(R) Celeron(R) CPU J3455 @ 1.50GHz`, matching the artifact's `device` field): HEAD's
`bench_router.csv` is 11 columns x 90 rows with exactly one commit (`9f3e5fd`) and
matches neither CSV on the device; the shipping `bench_router.csv` is byte-identical to
the device's `bench_router2.csv` (`cab37a42…`, 90 rows), written by `router_bench2.sh`
running `./before2-musl`/`./after2-musl`; HEAD's `summary.csv` is 30 cells (15 host, 15
router) and is reproduced value-for-value by HEAD's own `summarise.py`. The three false
clauses are absent. The unsupported `DONE` inference was withdrawn once the device showed
no bench CSV carries a `DONE` line — a claim settled by probing the device rather than by
reasoning about the prose.

## 5. How the round got here

Four claims during this closure turned out to be behaviour claims stated as artifact
claims, and each was settled by execution rather than by reading: F11's first closure
(`git ls-files` returned the path, but the documented recipe was a self-comparison
no-op); the guard's documented scope (it refused in-repo positional targets while
exempting `--out`); the digest record above; and the guard's second scope question (a
`.git`-less clone yields `None` from `checkout_root()`, so the exemption there is by
construction rather than by accident). The first three are closed and executed; the
fourth is documented behaviour with the positive control that keeps it honest.

The paired-check lesson is worth keeping: a block that only tests refusal cannot detect a
guard that refuses too much, and one that only tests permission cannot detect a guard that
allows too much. That is why the matrix carries (c) and (c-IN) together.

## 6. What this pass permits

`t7` may proceed to its pre-staging set, subject to the conditions its own note already
carries: the recurring headroom gate (`df -h` and `df -i` on `/opt/chroot/tmp`, taken
again at each commit step rather than once at the start), the Rust §1 hash set re-read
with `ops.rs` last, `cargo test --release`, and `bench_router.csv` held to `cab37a42…`.
No code, test, kernel, figure or parity number was implicated by the two defects above;
both were sentences.

## 7. Verification halted — environment failure, not a finding

Kept because a reader would otherwise infer that the review simply stopped or completed
without incident. Earlier in this round the environment lost the ability to execute
anything: `bash` failed at its output-write step, `glob` and `grep` could not spawn their
search processes, and a read-only `ssh 127.0.0.1 'df -h'` returned `spawn 失败: ENOSPC`.
Only `read` worked, which is the sole reason the cause was established rather than
guessed.

Cause, by probe rather than inference: the chroot's scratch filesystem
`/opt/chroot/tmp` was out of **space (bytes)**, not inodes. Two independent probe pairs
established it — a 0-byte create succeeds (an inode is allocated, no data page needed)
while a 1-2 byte write to the same directory returns ENOSPC (tmpfs allocates a 4 KiB
page), and both 0-byte files were read back to confirm they existed. The same 2-byte
write succeeded on `/tmp`, `/home/andre` and `/dev/shm`, which excluded those filesystems
and located the failure. Total `Shmem` is only about 400 MiB across every tmpfs, so the
instance must have carried a `size=` cap the host mount table does not show — a separate
mount instance in the chroot's namespace, or a unit-mounted limit. The captain later
recovered 5.8 GB on it; during this pass it reads 7.7 G with 5.8 G free.

The consequence was a hold, not a change of verdict, and it cost time rather than
correctness. What it taught is recorded above: a mitigation is not a closure, and a limit
stated in one place and denied in another is the same defect as the rest of this round.
