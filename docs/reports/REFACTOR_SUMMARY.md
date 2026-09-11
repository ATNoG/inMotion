# Refactor summary

Baseline is the `pre-refactor` tag (`bde65ca`). Full-history bundle and artifact
manifests are in `~/inmotion-freeze/`.

## Measured

| metric | before | after |
|---|---|---|
| tracked files | 798 | 385 |
| tracked images (pdf/png) | 473 | 74 |
| tracked size | 88.6 MB | 81.6 MB |
| entries at repo root | 70 | 45 |
| declared dependencies | 20 | 18 |
| `_build_*` model definitions | 27 (in 3 files) | 0 (one registry) |
| test count | 0 (root) | 1,591 |

The image drop is generated figures being untracked, not deleted — they remain
on disk in `plots/` and `combined/`.

## What changed

**Model loading.** A checkpoint's architecture is now data (`ModelSpec`: factory
name + kwargs + state-dict prefix) instead of a `_build_*` function whose
constants had been recovered by comparing tensor shapes. 17 curated specs plus
shape-driven adoption cover every checkpoint in the store: 12 paper models
(§2.3), 5 base/retrained variants, and 23 `src/inmotion/dl/` classifiers and ensembles.

**Strict loading.** `load_state_dict(..., strict=False)` is gone. It had been
hiding real breakage: `mamba3_tcn` "loaded" while missing 13 keys, and two
different builders each claimed the same `lejepa` checkpoint correctly with
59 shape mismatches between them. Loading now fails loudly, and a forward-pass
fingerprint per model catches behaviour changes that still load.

**Preprocessing.** The `StandardScaler` was never saved by any training path, and
two incompatible conventions existed (`src/inmotion/dl/data_loader.py` fit on the whole CSV before
splitting; `mega_ensemble` fit on the training fold). A checkpoint therefore
could not be re-run without also having its exact training CSV, which is why the
demo contained a heuristic that refits a scaler and guesses. `FeaturePipeline`
now persists the statistics; predictions built on reconstructed statistics say so.

**A CLI that answers "how do I run this".** `inmotion model
list|info|verify|predict|adopt|inventory` requires no knowledge of the training
scripts. `inmotion model verify` strict-loads all 17 registered models.

**Dataset versioning.** Datasets stay in git, so history is the version history;
`data/DATASETS.json` adds checksums, row/class counts, device counts, lineage,
and the commit that introduced each version. The pre-recollection lineage is
marked `known_incorrect` per the project owner's report and warns on resolve.

## Findings that need a decision

1. **The `mamba3_*` checkpoints do not match the paper.** MASTER_RESULTS §2.3
   publishes 0.843M / 0.842M / 1.701M parameters; the stored files are
   695,812 / 561,796 / 1,193,349. The published figures match builds with one
   more block or layer. Those three MCCs cannot be reproduced from `backup/`.
   See `ARTIFACTS.md`. No artifact has been touched — the discrepancy is recorded
   and asserted by tests.
2. **The W&B key is in git history.** Ten scripts leaked it; they no longer do,
   but the value is still in history and must be rotated.
3. **The `ds-icaisf` labels are unverified.** This is the dataset the ISAC
   ensemble was scored on. If it belongs to the older lineage whose labels the
   owner reports as wrong, the paper's numbers are affected. The manifest records
   it as `unverified` rather than assuming either way.
4. **Three cited result files are missing:** `results/mega_ensemble_results.csv`,
   `results/ensemble_size_frontier.csv`, and
   `models/dl/augmented/DeepStackEnsemble_seed42.pt`. MASTER_RESULTS cites them.
   They were absent before this work; nothing was fabricated to fill the gap.

## Not done

- `run_dl.py` (1,635 lines) and `run_exotic.py` (1,562 lines) are intact and
  wrapped by the CLI rather than split. Splitting them is a behavioural change
  and was lower value than the model-loading work.
- The pipelines still parse `sys.argv` internally; `inmotion run pipeline`
  delegates to them unchanged. See `MIGRATION.md`.
- `src/inmotion/classification/` has not moved under `src/inmotion/`; it still overlaps
  `src/inmotion/dl/` on `DataLoader`, `optimization` and `visualization`.
- Legacy `src/inmotion/dl/` has 58 ruff findings, left alone deliberately.
- `*_pretrain_best.pt` bundles are recognised and explained, not adopted.
