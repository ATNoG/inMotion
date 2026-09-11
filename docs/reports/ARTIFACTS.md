# Artifacts: where models and results live

## Models (not in git)

Checkpoint binaries are large and are **not** tracked. They live in two stores:

| store | contents |
|---|---|
| `20-aug/models/` | the ISAC 2026 paper's models: base and after-HPO JEPA/SIGReg runs, plus the `dl/` classifier store |
| `backup/` | the members that only exist here: `sigreg_seed{3,5,42}`, the four `mamba3_*` variants, `DeepStackEnsemble_seed42` |
| `models/` | classical-ML joblibs used by the demo (`GaussianProcess*.joblib`) |

A fresh clone therefore has **no models**. The root suite skips the model tests
with an explicit reason rather than reporting false failures.

`inmotion model list` shows which are present; `inmotion model verify`
strict-loads each one and checks it against its recorded parameter count.

### Reproducing the store

`inmotion model inventory <dir>` scans a directory, recovers each checkpoint's
architecture from its weights, and reports what it cannot recover. Every
checkpoint currently in the stores is recoverable except the pre-training
bundles (below).

## Pre-training bundles

`*_pretrain_best.pt` are **not** single-module state dicts. Each holds
`predictor` plus `encoder` (LeJEPA) or `context_encoder`/`target_encoder`
(T-JEPA, TS-JEPA, CF-JEPA), along with `epoch` and `probe_mcc`.

They are deliberately not adopted: the predictor's configuration is not
determined by the encoder's, so a guess would load and then silently resume with
a mis-shaped predictor. Use them through the pipeline that wrote them:

```bash
inmotion run pipeline exotic -- --model lejepa --checkpoint 20-aug/models/exotic/normal-new-ds/lejepa_pretrain_best.pt
```

## Provenance gap: the mamba3 checkpoints

`docs/isac/results/MASTER_RESULTS.md` §2.3 publishes parameter counts for
`mamba3_tcn` (0.843M), `mamba3_transformer` (0.842M) and `mamba3_multiview`
(1.701M). The stored files are **smaller**: 695,812 / 561,796 / 1,193,349 module
parameters respectively. The published figures match builds with one more block
or Mamba layer than the stored artifacts have.

The MCCs attributed to those three (`0.8844`, `0.8581`, `0.8640`) therefore
cannot be reproduced from the files in `backup/`. Either the artifacts were
overwritten by a later, smaller retrain, or the table's parameter column was
computed from a rebuilt configuration rather than the file.

This is recorded in `registry.PROVENANCE_GAPS` and asserted by
`tests/test_checkpoint_contract.py`, which fails if the discrepancy is
"resolved" by editing either side without restoring the right artifact.
`inmotion model info mamba3_tcn` surfaces it.

## Results and figures

Generated outputs are **not** tracked. They stay on disk and are regenerable:

| path | contents | tracked? |
|---|---|---|
| `plots/` | 469 figure files from the ML/DL runs | no |
| `results/` | per-seed ML/DL result CSVs | only `results/dl/dl_results_seed42.csv`, which the paper cites |
| `combined/` | multi-seed aggregate plots and statistics | no |
| `catboost_info/` | CatBoost training telemetry | no |
| `20-aug/logs/` | HPO logs | **yes** — cited as provenance by MASTER_RESULTS |
| `20-aug/results/` | exotic-pipeline result CSVs | **yes** — cited by MASTER_RESULTS |
| `docs/isac/results/` | curated paper numbers (`MASTER_RESULTS.md`, raw CSVs) | **yes** |

The rule: if a document cites it as provenance, it is tracked; if it is
regenerable output, it is not.
