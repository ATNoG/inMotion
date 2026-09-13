# Reports

The record: what was found, what was measured, and who verified it. Mostly
history — for the current interface read [`../guides/`](../guides/) and
[`../reference/`](../reference/) instead.

| file | what it is |
|---|---|
| [`ARTIFACTS.md`](ARTIFACTS.md) | where models, logs and results live; which are tracked in git and which are not; the mamba3 provenance gap; why a checkpoint can appear "missing" |
| [`MIGRATION.md`](MIGRATION.md) | old command → new command, and what has not been reorganised yet |
| [`REFACTOR_SUMMARY.md`](REFACTOR_SUMMARY.md) | what changed in the reorganisation, with before/after measurements |
| [`IMPLEMENTATION_SUMMARY.md`](IMPLEMENTATION_SUMMARY.md) | how the mega-ensemble harness was built (historical design note) |
| [`ESP32_FEASIBILITY.md`](ESP32_FEASIBILITY.md) | whether the world models fit and run on an ESP32-D0WD-V3, measured on the detected part rather than assumed |
| [`VERIFICATION.md`](VERIFICATION.md) | independent verification of the first closure — every number re-run by the verifier, none quoted from the authors |
| [`VERIFICATION_ROUND2.md`](VERIFICATION_ROUND2.md) | the round-2 independent verification, after the repairs |
| [`CODE_REVIEW_RUST.md`](CODE_REVIEW_RUST.md) | round-1 adversarial review of the Rust port and CPU kernels. Verdict: **needs_revision** — packaging and provenance findings, no numerics defect |
| [`CODE_REVIEW_RUST_ROUND2.md`](CODE_REVIEW_RUST_ROUND2.md) | round-2 review of the repair of those fourteen findings. Verdict: **pass** |
| [`CODE_REVIEW_RUST_ROUND3.md`](CODE_REVIEW_RUST_ROUND3.md) | round-3 final gate, with a per-item verdict table for every finding across the rounds. Verdict: **pass** |

## How to read these

* **A verdict belongs to a revision.** Each review and verification document
  states the revision it judged. If a file it names has since moved, the verdict
  still holds for that revision and must be re-taken for the current one; it does
  not silently extend to the newer bytes.
* **Wrong values here are usually deliberate.** These documents record what was
  found and corrected, so a superseded hash or a retracted diagnosis appears
  described **as** wrong. Check the surrounding sentence before "fixing" one.
* **Superseded documents are kept, not deleted.** A reader is better served by
  the sequence of what was believed and why it changed than by a tidy final state
  with the reasoning removed.
