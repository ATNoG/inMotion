# Artifacts: where models and results live

## Models (not in git)

Checkpoint binaries are large and are **not** tracked. They live in two stores:

| store            | contents                                                                                                          |
| ---------------- | ----------------------------------------------------------------------------------------------------------------- |
| `20-aug/models/` | the FNWF 2026 paper's models: base and after-HPO JEPA/SIGReg runs, plus the `src/inmotion/dl/` classifier store   |
| `backup/`        | the members that only exist here: `sigreg_seed{3,5,42}`, the four `mamba3_*` variants, `DeepStackEnsemble_seed42` |
| `models/`        | classical-ML joblibs used by the demo (`GaussianProcess*.joblib`)                                                 |

A fresh clone therefore has **no models**. The root suite skips the model tests
with an explicit reason rather than reporting false failures.

`inmotion model list` shows which are present; `inmotion model verify`
strict-loads each one and checks it against its recorded parameter count.

### Reproducing the store

`inmotion model inventory <dir>` scans a directory, recovers each checkpoint's
architecture from its weights, and reports what it cannot recover. Every
checkpoint currently in the stores is recoverable except the pre-training
bundles (below).

## Quantized artifacts

The Rust runtime's artifacts for the five after-HPO classifiers live **beside the
checkpoints they came from**, one directory per (model, precision):

```
20-aug/models/exotic/normal-new-ds/full-after-hpo/artifacts/<model>-<precision>/
```

Three producers write here, and they are named apart:

```
<model>-fp32, <model>-int8, <model>-q1..q8    the in-house quantizer
<model>-torchao-int8                           torch.ao's per-channel int8
<model>-bnb-nf4 | -bnb-fp4 | -bnb-int8         bitsandbytes
```

The in-house quantizer writes 45 (five models x nine precisions), torch.ao 5 and
bitsandbytes 15. `artifacts/MANIFEST.md` records what each producer gives, and
`results/campaign/approach_comparison.csv` puts them side by side on accuracy,
size, weight-level fidelity and whether the runtime can load them.

Only the in-house and torch.ao artifacts are runtime-loadable. Every scheme in
the runtime dequantizes as one affine map, `(u - zp) * scale`; bitsandbytes'
codebooks are not affine — NF4 by construction, and its 8-bit path despite
looking like plain int8 (its 256 entries are non-uniformly spaced, leaving a 62%
residual against the best straight line). Teaching the reader a codebook is a
format change and is not attempted here.

Five models x nine precisions: `fp32`, `int8` and `q1`..`q8`. Each directory
holds `meta.json` (architecture and quantization spec), `model.safetensors` (the
packed weights) and `golden.json` (the vectors the runtime's parity check
replays).

`fp16`, `bf16` and `fp8` are deliberately absent. They are cast precisions that
`inmotion quantize eval` measures for accuracy, but an artifact is read by the
Rust runtime, which accepts F32, I32 and U8 only — so no artifact can carry them.
`inmotion quantize export --format fp16` refuses with that reason rather than
writing something the runtime could not load.

The name is `<model>-<precision>` deliberately: that is what the campaign's
measurement tooling parses, so `results/campaign/measure_device.py` and
`measure_scaling.py` read this layout directly and `measure_router.py` ships it
to the router unchanged.

They are **not tracked**, for the same reason the checkpoints are not: each is
derived from a checkpoint plus a scheme.
`results/campaign/export_artifacts.py` regenerates all sixty and parity-checks
each one as it writes, so a stale directory cannot silently outlive its
checkpoint. The manifest of what exists, with sizes and parity status, is
`results/campaign/artifacts_manifest.csv`.

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

The paper's table publishes parameter counts for
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

| path                 | contents                                              | tracked?                                                       |
| -------------------- | ----------------------------------------------------- | -------------------------------------------------------------- |
| `plots/`             | 469 figure files from the ML/DL runs                  | no                                                             |
| `results/`           | per-seed ML/DL result CSVs                            | only `results/dl/dl_results_seed42.csv`, which the paper cites |
| `combined/`          | multi-seed aggregate plots and statistics             | no                                                             |
| `catboost_info/`     | CatBoost training telemetry                           | no                                                             |
| `20-aug/logs/`       | HPO logs                                              | **yes** — cited as provenance by the paper tables                |
| `20-aug/results/`    | exotic-pipeline result CSVs                           | **yes** — cited by the paper tables                              |
| `docs/isac/results/` | curated paper numbers and raw CSVs | **yes**                                                        |

The rule: if a document cites it as provenance, it is tracked; if it is
regenerable output, it is not.
