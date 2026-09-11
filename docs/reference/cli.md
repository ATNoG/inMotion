# CLI reference

Generated from the typer application in `src/inmotion/cli/` by walking the
click command tree, so it lists exactly what the program accepts.

```
inmotion --help
```

## `analyze`

Analyse results.

### `analyze interference`

Cross-route interference from the concurrent_noise_path column.

Options:

- `--data` (default: `data/processed/dataset.csv`) - Dataset CSV with concurrent_noise_path.
- `--output-dir` (default: `plots/interference`) - Figure output directory.

### `analyze seeds`

Combine and plot results from multi-seed runs.

Options:

- `--results-dir` (default: `results`) - Directory of per-seed result CSVs.
- `--output-dir` (default: `combined`) - Where to write the combined analysis.
- `--seeds` (default: `3,5,42`) - Comma-separated seeds to combine.

## `data`

Build and clean datasets.

### `data augment`

Augment a dataset with noise injection and intra-class mixup.

Options:

- `--data` - Dataset CSV to augment.
- `--output`, `-o` - Destination CSV.

### `data clean`

Audit a dataset per device and drop rows that poison training.

Options:

- `--data` - Dataset CSV to audit.
- `--output` - Write the cleaned CSV here.
- `--keep-shifted` (default: `False`) - Keep device-shifted but self-consistent rows.
- `--drop-ambiguous` (default: `False`) - Also drop samples the model rejects.
- `--ambiguity-thresh` (default: `0.9`) - Confidence above which a wrong prediction is dropped.

### `data export`

Turn a raw Wavecom capture into a per-route CSV.

Options:

- `--input`, `-i` - Raw Wavecom capture (.txt).
- `--output`, `-o` - Destination CSV.
- `--group` - label:mac1,mac2,... (repeatable).
- `--group-label` - group_name:final_label (repeatable).

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

## `evaluate`

Evaluate and combine models.

### `evaluate checkpoints`

Retroactive per-class metrics for saved DL checkpoints.

Options:

- `--data` (default: `/home/andre/Desktop/UA/IT/inMotion/inMotion/data/processed/dataset.csv`) - Dataset CSV.
- `--seed` (default: `42`) - Random seed.
- `--checkpoints-dir` (default: `models/dl`) - Directory of .pt checkpoints.
- `--output-dir` (default: `results/dl`) - Where to write the results.
- `--optuna-db` - Optuna storage URL, for HPO checkpoints.
- `--no-wandb` (default: `False`) - Disable WandB logging.

### `evaluate ensemble`

Greedy ensemble selection over existing checkpoints.

Options:

- `--data` (default: `/home/andre/Desktop/UA/IT/inMotion/inMotion/data/processed/dataset.csv`) - Dataset CSV.
- `--seed` (default: `42`) - Random seed.
- `--rich` (default: `False`) - Use the 18-channel feature set.

### `evaluate feature-importance`

Permutation feature importance for the DL models.

Options:

- `--data` (default: `/home/andre/Desktop/UA/IT/inMotion/inMotion/data/processed/dataset.csv`) - Dataset CSV.
- `--seed` (default: `42`) - Random seed.
- `--epochs` (default: `100`) - Epochs per model.
- `--patience` (default: `20`) - Early-stopping patience.
- `--output-dir` (default: `docs/icaisf/paper/images`) - Figure output directory.
- `--checkpoints-dir` (default: `models/dl`) - Directory of DL checkpoints.

### `evaluate mega-ensemble`

OOF stacking over every model family, with calibration.

Options:

- `--data` (default: `/home/andre/Desktop/UA/IT/inMotion/inMotion/data/processed/dataset.csv`) - Dataset CSV.
- `--seed` (default: `42`) - Random seed.
- `--size-frontier` (default: `False`) - Report the ensemble-size frontier.
- `--no-tabpfn` (default: `False`) - Exclude TabPFN.
- `--no-classical` (default: `False`) - Exclude the classical ML members.
- `--output` - Write the result table here.

### `evaluate shap`

SHAP channel attribution for the world models.

Options:

- `--checkpoint` (default: `backup/ts_jepa_ft_seed42.pt`) - Model checkpoint to explain.
- `--data` (default: `/home/andre/Desktop/UA/IT/inMotion/inMotion/data/processed/dataset.csv`) - Dataset CSV.
- `--model` (default: `ts_jepa`) - Model family: t_jepa, ts_jepa, lejepa, cf_jepa or sigreg.
- `--samples` (default: `50`) - Number of samples to explain.
- `--nsamples` (default: `150`) - SHAP evaluation budget.
- `--seed` (default: `42`) - Random seed.
- `--output-dir` (default: `plots/shap`) - Figure output directory.

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

## `plots`

Regenerate figures from saved results.

### `plots classical`

Rebuild the classical-ML figures from saved CSVs.

Options:

- `--csv` - A single classification_results.csv.
- `--results-dir` - A directory of result CSVs.
- `--plots-dir` (default: `plots`) - Figure output directory.
- `--eda` (default: `False`) - Also regenerate the EDA figures.
- `--data` - Dataset CSV, needed with --eda.

### `plots dl`

Rebuild the DL figures from saved CSVs.

Options:

- `--results-csv` (default: `results/dl/dl_detailed_seed42.csv`) - Extended DL results CSV.
- `--output-dir` (default: `plots/dl`) - Figure output directory.
- `--eda` (default: `False`) - Also regenerate the EDA figures.
- `--data` - Dataset CSV, needed with --eda.

## `quantize`

Quantize models and measure the cost.

### `quantize eval`

Measure accuracy under quantization on the paper's split.

Options:

- `--models` - Comma-separated; default: the JEPA family.
- `--formats` - Comma-separated; default: all.
- `--data` - Dataset CSV; default: dataset.icaisf.csv.
- `--seed` (default: `42`) - Split seed.
- `--out` (default: `results/quant/sweep.csv`) - Where to write the results CSV.

### `quantize formats`

List the available quantization schemes.

### `quantize show`

Quantize one model and report size and error.

Arguments:

- `MODEL` (required) - Registered model name.

Options:

- `--format`, `-f` (default: `int8`) - Scheme, e.g. q4.

## `train`

Train models.

### `train classification`

Classical ML classifiers: EDA, CV, Optuna, importance.

Options:

- `--data` (default: `/home/andre/Desktop/UA/IT/inMotion/inMotion/data/processed/dataset.csv`) - Dataset CSV.
- `--seed` (default: `42`) - Random seed.
- `--optimize` (default: `False`) - Run the Optuna tuning stage.
- `--dashboard` (default: `False`) - Launch optuna-dashboard after tuning.
- `--models-dir` (default: `models`) - Model output directory.
- `--results-dir` (default: `results`) - Results output directory.
- `--plots-dir` (default: `plots`) - Figure output directory.

### `train dl`

Supervised deep learning: baselines, HPO/NAS, MoE, DeepStack.

Options:

- `--data` (default: `/home/andre/Desktop/UA/IT/inMotion/inMotion/data/processed/dataset.csv`) - Dataset CSV.
- `--seed` (default: `42`) - Random seed.
- `--epochs` (default: `150`) - Max epochs per model.
- `--trials` (default: `50`) - Optuna trials per study.
- `--batch-size` (default: `64`) - Batch size.
- `--num-gpus` (default: `0`) - GPUs to use; 0 = auto-detect.
- `--models-dir` - Checkpoint output directory.
- `--results-dir` - Results CSV directory.
- `--plots-dir` - Figure output directory.
- `--wandb-project` - Override the WandB project.
- `--no-wandb` (default: `False`) - Disable WandB logging.
- `--no-optuna` (default: `False`) - Skip HPO and NAS studies.
- `--no-moe` (default: `False`) - Skip the mixture-of-experts phase.
- `--no-deepstack` (default: `False`) - Skip the DeepStack phase.
- `--no-meta` (default: `False`) - Skip the metadata-fusion phases.
- `--no-resume` (default: `False`) - Retrain even when a checkpoint exists.

### `train exotic`

JEPA world models and SIGReg: SSL pretraining and fine-tuning.

Options:

- `--model` (default: `t_jepa`) - t_jepa, ts_jepa, lejepa, cf_jepa, sigreg or mamba3_<variant>.
- `--data` (default: `/home/andre/Desktop/UA/IT/inMotion/inMotion/data/processed/dataset.csv`) - Dataset CSV.
- `--seed` (default: `42`) - Random seed.
- `--batch-size` (default: `128`) - Batch size.
- `--gpu` (default: `0`) - GPU device index.
- `--rich-features` (default: `False`) - Use the 18-channel feature set.
- `--pretrain-epochs` (default: `300`) - Stage-1 epochs.
- `--finetune-epochs` (default: `50`) - Stage-2 epochs.
- `--epochs` (default: `150`) - Epochs for directly-supervised models.
- `--patience` (default: `25`) - Early-stopping patience.
- `--lr` (default: `0.001`) - Learning rate for directly-supervised models.
- `--pretrain-lr` (default: `0.0003`) - Stage-1 learning rate.
- `--finetune-lr` (default: `0.001`) - Stage-2 learning rate.
- `--probe-cadence` (default: `10`) - Epochs between linear probes.
- `--probe-patience` (default: `15`) - Probe-based patience.
- `--checkpoint` - Resume from a pretrained encoder.
- `--finetune-only` (default: `False`) - Skip pretraining.
- `--d-model` (default: `256`) - Model width.
- `--nhead` (default: `8`) - Attention heads.
- `--num-layers` (default: `4`) - Transformer layers.
- `--dim-ff` (default: `512`) - Feed-forward width.
- `--pred-dim` (default: `128`) - Predictor width.
- `--pred-layers` (default: `2`) - Predictor depth.
- `--d-state` (default: `16`) - Mamba-3 state width.
- `--dropout` (default: `0.2`) - Dropout for the Mamba-3 hybrids.
- `--mimo-rank` (default: `4`) - Mamba-3 MIMO rank.
- `--latent-dim` (default: `128`) - SIGReg latent width.
- `--sigreg-lambda` (default: `0.01`) - SIGReg regulariser weight.
- `--hpo` (default: `False`) - Run an Optuna study first.
- `--hpo-trials` (default: `30`) - Trials per study.
- `--hpo-objective` (default: `mcc`) - mcc, efficiency or pareto.
- `--models-dir` (default: `models/exotic`) - Checkpoint output directory.
- `--results-dir` (default: `results/exotic`) - Results CSV directory.
- `--viz-dir` (default: `models/exotic/viz`) - Figure output directory.
- `--no-wandb` (default: `False`) - Disable WandB logging.
- `--wandb-project` (default: `inMotion-exotic-hpo`) - WandB project name.

### `train hpo-paper`

Retrain the paper's HPO-best configurations.

Options:

- `--data` (default: `/home/andre/Desktop/UA/IT/inMotion/inMotion/data/processed/dataset.csv`) - Dataset CSV.
- `--seed` (default: `42`) - Random seed.
- `--epochs` (default: `150`) - Epochs per model.
- `--patience` (default: `20`) - Early-stopping patience.
- `--models` - Comma-separated subset, e.g. gru,tcn.
- `--models-dir` (default: `models/dl`) - Checkpoint output directory.
- `--results-dir` (default: `results/dl`) - Results output directory.

## Pipeline modules

Every training, evaluation and analysis pipeline exposes the same shape:
a typed `RunConfig` dataclass and a `run(args) -> int` function. The CLI
builds the config from typed options, so the pipelines contain no argument
parsing and can be driven from Python directly:

```python
from inmotion.pipelines.exotic import RunConfig, run

run(RunConfig(model='ts_jepa', seed=42, pretrain_epochs=500))
```

The two largest pipelines are packages with one module per responsibility:

| pipeline | package | modules |
|---|---|---|
| `inmotion train dl` | `inmotion.pipelines.dl` |
| | | config, devices, builders, workers, pipeline |
| `inmotion train exotic` | `inmotion.pipelines.exotic` |
| | | config, devices, data, builders, training, |
| | | hpo, hpo_jepa, hpo_sigreg, hpo_mamba3, pipeline |

`python -m inmotion.pipelines.<name>` runs the same CLI for those packages.
