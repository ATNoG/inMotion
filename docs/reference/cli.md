# CLI reference

Generated from the typer application in `src/inmotion/cli.py` by walking the
click command tree, so it lists exactly what the program accepts.

```
inmotion --help
```

## `data`

Build and clean datasets.

### `data augment`

Augment a dataset with noise injection and intra-class mixup.

### `data clean`

Audit a dataset per device and drop rows that poison training.

### `data export`

Turn a raw Wavecom capture into a per-route CSV.

Options:

- `--input`, `-i` - Wavecom capture file.
- `--output`, `-o` - Destination CSV.
- `--group` (default: `[]`) - label:mac1,mac2,... (repeatable).
- `--group-label` (default: `[]`) - group_name:final_label (repeatable).

### `data merge`

Merge per-route CSVs into one dataset.

Options:

- `--input`, `-i` - Directory of per-route CSVs.
- `--output`, `-o` - Merged dataset CSV.
- `--pattern` (default: `*.csv`) - Glob for input files.

## `datasets`

Dataset versions and provenance.

### `datasets list`

List recorded dataset versions, newest lineage last.

Options:

- `--json` (default: `False`) - 

### `datasets manifest`

Regenerate the manifest's measured fields, or print it.

Options:

- `--write` (default: `False`) - Rewrite data/DATASETS.json.

### `datasets resolve`

Print the path of a dataset version, warning when it is not trustworthy.

Arguments:

- `IDENTIFIER` (required) - Dataset id or file name.

### `datasets show`

Show lineage and label provenance for one dataset version.

Arguments:

- `IDENTIFIER` (required) - Dataset id or file name.

### `datasets verify`

Re-check every dataset against its recorded checksum and row count.

## `doctor`

Report on the environment and the artifact store.

## `model`

Inspect, verify and run trained models.

### `model adopt`

Recover a checkpoint's architecture and write a sidecar.

Use --factory to resolve structural ambiguity such as TS-JEPA versus CF-JEPA
at the same width, which are indistinguishable from weights alone.

Arguments:

- `CHECKPOINT` (required) - Path to a .pt checkpoint.

Options:

- `--name` - Name for the adopted model.
- `--factory` - Pin the architecture when ambiguous.
- `--write` - Write a sidecar-tagged copy to this directory.

### `model info`

Show the architecture and provenance of one model.

Arguments:

- `NAME` (required) - Registered model name.

### `model inventory`

Adopt every checkpoint under a directory and report what is recoverable.

Arguments:

- `DIRECTORY` (required) - Directory to scan for .pt files.

Options:

- `--json` (default: `False`) - 

### `model list`

List every registered model.

Options:

- `--json` (default: `False`) - Machine-readable output.

### `model predict`

Run a trained model over a CSV of RSSI windows.

The model's preprocessing statistics were never persisted by the original
training code, so scaling is reconstructed from ``--reference`` (the same CSV
the model trained on) unless a saved pipeline exists next to the weights.

Arguments:

- `NAME` (required) - Registered model name.

Options:

- `--input`, `-i` - CSV of RSSI rows.
- `--reference` - CSV used to fit scaling statistics.
- `--out`, `-o` - 

### `model verify`

Strict-load every model and check it against its recorded parameter count.

Arguments:

- `NAME` (optional) - Verify one model.

## `run`

Run a training or evaluation pipeline.

### `run list`

List the available pipelines.

### `run pipeline`

Run one pipeline, passing remaining arguments straight through.

Example:  inmotion run pipeline exotic -- --model ts_jepa --seed 42

Arguments:

- `PIPELINE` (required) - Pipeline name; see `inmotion run list`.

## Pipelines

`inmotion run pipeline <name> -- <args>` forwards everything after `--` to the
pipeline's own argument parser. Each pipeline also runs standalone as
`python -m inmotion.<module>`.

| name | module | purpose |
|---|---|---|
| `classification` | `inmotion.pipelines.classification` | classical ML classifiers (EDA, CV, Optuna, feature importance) |
| `dl` | `inmotion.pipelines.dl` | supervised deep learning: baselines, ensembles, MoE, deep stacking, NAS |
| `exotic` | `inmotion.pipelines.exotic` | JEPA world models and SIGReg: SSL pretraining + fine-tuning |
| `ensemble` | `inmotion.pipelines.ensemble` | greedy ensemble selection over existing checkpoints |
| `mega-ensemble` | `inmotion.pipelines.mega_ensemble` | OOF stacking over every model family, with calibration |
| `hpo-paper` | `inmotion.pipelines.hpo_paper` | retrain the paper's HPO-best configs (Optuna DBs were deleted) |
| `evaluate-checkpoints` | `inmotion.evaluation.checkpoints` | retroactive per-class metrics for saved DL checkpoints |
| `shap` | `inmotion.evaluation.shap` | SHAP channel attribution for the world models |
| `feature-importance` | `inmotion.evaluation.feature_importance` | permutation feature importance for DL models |
| `analyze-seeds` | `inmotion.analysis.seeds` | combine and plot multi-seed results |
| `analyze-interference` | `inmotion.analysis.interference` | cross-route interference analysis from concurrent_noise_path |
| `regenerate-plots` | `inmotion.viz.regenerate` | rebuild classical-ML figures from saved CSVs |
| `regenerate-dl-plots` | `inmotion.viz.regenerate_dl` | rebuild DL figures from saved CSVs |
