# Training guide

Every command runs from the repository root. `uv sync --all-packages` first.

`MAMBA_SSM_AVAILABLE=0` is recommended unless you have the CUDA Mamba extension
installed: importing `mamba_ssm` can hang for minutes while it builds kernels.

## Classical machine learning

```bash
uv run inmotion run pipeline classification -- \
    --data data/processed/dataset.csv
```

Runs EDA, cross-validates a set of scikit-learn classifiers, optionally tunes
them with Optuna, computes feature importance, and writes plots plus a results
CSV. Add `--optimize` for the Optuna stage and `--dashboard` to launch
`optuna-dashboard`.

## Deep learning baselines

```bash
MAMBA_SSM_AVAILABLE=0 uv run inmotion run pipeline dl -- \
    --data data/processed/dataset.csv --seed 42 --trials 50
```

Phases, in order, each skippable by flag:

| phase | flag to skip | what it does |
|---|---|---|
| 1 | `--no-optuna` | trains each single model on its own GPU (round-robin) |
| 2 | `--no-optuna` | Optuna HPO per model |
| 3 | `--no-optuna` | NAS |
| 4 | `--no-moe` | mixture-of-experts variants |
| 5 | `--no-deepstack` | DeepStackEnsemble, a 10-base stack with a learned meta-model |
| — | `--no-meta` | metadata-fusion variants |

Checkpoints land in `--models-dir`, results in `--results-dir`. An existing
checkpoint is reused unless `--force-retrain` is given.

## World models (SSL)

```bash
# find a configuration first
MAMBA_SSM_AVAILABLE=0 uv run inmotion run pipeline exotic -- \
    --model ts_jepa --data data/processed/dataset_augmented3.csv \
    --hpo --hpo-trials 50

# pretrain plus fine-tune
MAMBA_SSM_AVAILABLE=0 uv run inmotion run pipeline exotic -- \
    --model ts_jepa --data data/processed/dataset_augmented3.csv --seed 42 \
    --pretrain-epochs 500 --finetune-epochs 80 --batch-size 512
```

Two-stage by nature: self-supervised pretraining, then supervised fine-tuning.
The JEPA models (`t_jepa`, `ts_jepa`, `lejepa`, `cf_jepa`) use both stages;
`sigreg` and the `mamba3_*` hybrids are directly supervised and take `--epochs`.

Stage-1 checkpoint selection is driven by a **linear-probe MCC**, not by the
masked-prediction loss, because the loss does not track representation quality.
`--probe-cadence` and `--probe-patience` control that probe.

Resume from a pretrained encoder:

```bash
inmotion run pipeline exotic -- --model ts_jepa \
    --checkpoint 20-aug/models/exotic/normal-new-ds/ts_jepa_pretrain_best.pt \
    --finetune-only
```

## The paper's HPO-best configs

The Optuna databases were deleted, so the winning configurations are re-declared
from the documented values and retrained:

```bash
MAMBA_SSM_AVAILABLE=0 uv run inmotion run pipeline hpo-paper -- \
    --data data/processed/dataset.csv --seed 42
```

Values come from `docs/icaisf/hpo_supplementary.md`.

## Ensembles

```bash
# greedy selection over existing checkpoints (Caruana forward selection)
uv run inmotion run pipeline ensemble -- \
    --data data/processed/dataset_augmented3.csv --seed 42

# OOF stacking over every family, with calibration
MAMBA_SSM_AVAILABLE=0 uv run inmotion run pipeline mega-ensemble -- \
    --data data/processed/dataset_augmented3.csv --seed 42
```

`mega-ensemble` combines DL checkpoints, the JEPA family, classical ML and
TabPFN. It needs the member checkpoints present; `inmotion model list` shows
which are.

## On a cluster

`scripts/slurm/*.sh` carry `#SBATCH` headers:

```bash
sbatch scripts/slurm/job_full_train.sh
```

`scripts/README.md` says which script does what and which are superseded.
Credentials come from the environment (`env.example`), never from the scripts.

## Reproducing a run

Record these four things and the run is repeatable: the command, `--seed`, the
dataset (`inmotion datasets show <id>` gives its checksum), and the git commit.
`inmotion doctor` prints the environment. See `docs/reproducibility.md`.
