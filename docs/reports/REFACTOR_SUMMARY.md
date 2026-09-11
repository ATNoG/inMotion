# Refactor summary

Baseline is the `pre-refactor` tag (`bde65ca`). A full-history git bundle and
sha256 manifests of every artifact are in `~/inmotion-freeze/`.

## Measured

| metric | before | after |
|---|---|---|
| tracked files | 798 | 407 |
| entries at repository root | 70 | 20 |
| Python modules at root | 17 | **0** |
| datasets at root | 7 | **0** |
| tracked images (pdf/png) | 473 | 74 |
| top-level importable packages | 3 | **1** |
| duplicate model builders | 27 | **0** |
| argparse option definitions | ~180 | **0** |
| ruff findings | 268 | **0** |
| tests | 0 | **1 777** |

The image drop is generated figures being untracked, not deleted: they remain
on disk in `plots/` and `combined/`.

## What changed

### Model loading

A checkpoint's architecture is now data — a `ModelSpec` recording factory,
keyword arguments, state-dict prefix and expected parameter count — instead of a
`_build_*` function whose constants had been recovered by comparing tensor
shapes. 17 curated specs plus shape-driven adoption cover every checkpoint in
the store: the 12 paper models, 5 pre-HPO/retrained variants, and 23 `dl/`
classifiers and ensembles.

`load_state_dict(..., strict=False)` is gone. It had been hiding real breakage:
`mamba3_tcn` loaded while missing 13 keys, and two different builders each
claimed the same `lejepa` checkpoint with 59 shape mismatches between them.
Loading now fails loudly, and a forward-pass fingerprint per model catches
changes that still load.

### Preprocessing

The `StandardScaler` was never saved by any training path, and two incompatible
conventions existed. A checkpoint therefore could not be re-run without its exact
training CSV, which is why the demo contained a heuristic that refits a scaler
and guesses. `FeaturePipeline` now persists the statistics, and reconstructed
statistics are labelled as reconstructed.

### A CLI that answers "how do I run this"

`inmotion` is a package of native typer commands (`model`, `datasets`, `data`,
`train`, `evaluate`, `analyze`, `plots`, `doctor`). Every pipeline exposes a
typed `RunConfig` and a `run(args) -> int`, with no argument parsing, so the same
work is drivable from a notebook. `inmotion model verify` strict-loads every
registered model, and `docs/reference/pipelines.md` documents every option by
reading the dataclasses.

### Dataset versioning

Datasets are tracked in git, so history is the version history.
`data/DATASETS.json` adds checksums, row and class counts, device counts,
lineage and the commit that introduced each version. The pre-recollection lineage
is marked `known_incorrect` per the project owner's report and warns on resolve.

### Layout

Everything importable is under `src/inmotion/`. Data is tiered under
`data/{raw,interim,processed}/`, with a resolver so bare filenames still work.
Shell scripts are under `scripts/{slurm,local,archive,oneoff}/`. Generated output
is untracked and regenerable.

### Documentation

`docs/` now holds an architecture guide, a reproducibility statement, two task
guides, four generated references, model cards and an audit checklist. The
reference documents are generated from the code and CI fails when they are stale.

### Auditability

CI runs lint, formatting, doc-staleness, the root suite, the demo suite and a
secret guard. `docs/audit/AUDIT_CHECKLIST.md` maps every claim to the command
that verifies it.

## Findings that need a decision

1. **The `mamba3_*` checkpoints do not match the paper.** `MASTER_RESULTS.md`
   §2.3 publishes 0.843M / 0.842M / 1.701M parameters; the stored files are
   695,812 / 561,796 / 1,193,349. The published figures match builds with one
   more block or layer, so those three MCCs cannot be reproduced from `backup/`.
   Recorded and asserted by tests; no artifact was touched.
2. **The W&B key is in git history.** Ten scripts leaked it; they no longer do,
   but the value must be rotated.
3. **The Mamba-3 HPO search space was partly inert.** Three of five reported
   dimensions could not affect the model. The code is fixed; existing HPO logs
   predate the fix.
4. **The `ds-icaisf` labels are unverified.** If it belongs to the older lineage
   whose labels the owner reports as wrong, the reported ensemble numbers are
   affected.
5. **Three cited result files are absent:** `results/mega_ensemble_results.csv`,
   `results/ensemble_size_frontier.csv` and
   `models/dl/augmented/DeepStackEnsemble_seed42.pt`. They were missing before
   this work; nothing was fabricated.
6. **No `LICENSE` and no `CITATION.cff` reference to a camera-ready paper yet.**

## Not done

- `inmotion/dl/` and `inmotion/classification/` still duplicate responsibilities
  on data loading, optimisation and visualisation. Merging them is a behavioural
  change and is left as a follow-up; `inmotion/pipelines/exotic/hpo_jepa.py`
  (432 lines) is the largest remaining module.
- The four `mamba3_*` hybrids are registered and loadable but were explored
  rather than reported; they carry no metric claim.
- `*_pretrain_best.pt` bundles are recognised and explained, not adopted.
