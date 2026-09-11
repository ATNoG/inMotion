# Command guide

Every command, when to reach for it, and a realistic invocation. Run everything
from the repository root. For the full option list see
[`../reference/cli.md`](../reference/cli.md) (the interface) and
[`../reference/pipelines.md`](../reference/pipelines.md) (what each option does
and its default).

```bash
uv sync --all-packages
uv run inmotion --help
```

Most pipelines touch PyTorch. Unless you have the CUDA Mamba extension
installed, prefix with `MAMBA_SSM_AVAILABLE=0`: importing `mamba_ssm` can hang
for minutes while it builds kernels.

---

## First: is everything intact?

```bash
uv run inmotion doctor          # versions, CUDA, which models are on disk
uv run inmotion datasets verify # dataset checksums against the manifest
uv run inmotion model verify    # strict-load every registered model
```

`doctor` is the fastest way to see whether a fresh clone is usable — model
binaries are gitignored, so a clone has none until the artifact store is
restored.

---

## Working with models

```bash
uv run inmotion model list                      # what exists, and where
uv run inmotion model list --json               # for a script
uv run inmotion model info ts_jepa              # architecture, params, provenance
uv run inmotion model verify ts_jepa            # strict-load just this one
```

Predict from a CSV of 10-second RSSI windows:

```bash
uv run inmotion model predict ts_jepa \
    --input new_routes.csv \
    --reference data/processed/dataset_augmented3.csv \
    --out predictions.csv
```

`--reference` is the CSV the model was trained on; it supplies the
standardisation statistics, which the original training code never saved. Omit
it and the default for the model's feature width is used, with a warning.

A checkpoint with no sidecar:

```bash
uv run inmotion model adopt path/to/checkpoint.pt --name my_model
uv run inmotion model inventory 20-aug/models     # scan a whole tree
uv run inmotion model adopt ckpt.pt --name m --write artifacts/m   # keep the result
```

`adopt` reports ambiguity rather than guessing when two architectures are
structurally identical, and names pre-training bundles instead of attempting
them.

---

## Data

The three tiers are `data/raw/` (captures), `data/interim/` (per-session CSVs)
and `data/processed/` (training sets). `inmotion datasets show <id>` explains
each processed version.

```bash
# raw capture -> one CSV for one collection session
uv run inmotion data export \
    --input data/raw/routeAtoB.txt \
    --output data/interim/AB.csv \
    --group G1:aa:bb:cc:dd:ee:ff \
    --group-label G1:AB

# many session CSVs -> one dataset
uv run inmotion data merge --input data/interim --output data/processed/dataset.csv

# audit which devices or samples carry inconsistent labels
uv run inmotion data clean --data data/processed/dataset.csv \
    --output data/processed/dataset_clean.csv

# add class-conditioned noise and intra-class mixup, flagged `synthetic`
uv run inmotion data augment \
    --data data/processed/dataset.csv \
    --output data/processed/dataset_augmented3.csv
```

`clean` exists because absolute RSSI level identifies *which device* recorded a
sequence more strongly than *which route* it took. It quantifies that per device
and can drop the poisoned parts.

Dataset versions:

```bash
uv run inmotion datasets list
uv run inmotion datasets show ds-augmented-v3
uv run inmotion datasets resolve ds-pure    # warns: labels reported incorrect
uv run inmotion datasets manifest --write   # refresh checksums after editing
```

---

## Training

```bash
uv run inmotion train classification --data data/processed/dataset.csv --optimize
uv run inmotion train dl --data data/processed/dataset.csv --seed 42 --trials 50
uv run inmotion train exotic --model ts_jepa --seed 42 \
    --pretrain-epochs 500 --finetune-epochs 80 --batch-size 512
uv run inmotion train hpo-paper --data data/processed/dataset.csv --seed 42
```

`train dl` runs phases in order — single models, HPO, NAS, MoE, DeepStack — each
skippable (`--no-optuna`, `--no-moe`, `--no-deepstack`, `--no-meta`). Existing
checkpoints are reused unless you pass `--no-resume`.

`train exotic` is two-stage: self-supervised pretraining, then fine-tuning
(`t_jepa`, `ts_jepa`, `lejepa`, `cf_jepa`). `sigreg` and the `mamba3_*` hybrids
are directly supervised and take `--epochs`. Resume from a pretrained encoder:

```bash
uv run inmotion train exotic --model ts_jepa \
    --checkpoint 20-aug/models/exotic/normal-new-ds/ts_jepa_pretrain_best.pt \
    --finetune-only
```

See [`training.md`](training.md) for what each phase does and why stage-1
selection uses a linear probe.

---

## Evaluating and combining

```bash
# per-class metrics for every saved checkpoint
uv run inmotion evaluate checkpoints --data data/processed/dataset.csv --seed 42

# greedy ensemble selection over existing checkpoints
uv run inmotion evaluate ensemble --data data/processed/dataset_augmented3.csv

# OOF stacking over DL + JEPA + classical + TabPFN
uv run inmotion evaluate mega-ensemble --data data/processed/dataset_icaisf.csv --seed 42

# which feature channels matter, by SHAP and by permutation
uv run inmotion evaluate shap --checkpoint backup/ts_jepa_ft_seed42.pt --model ts_jepa
uv run inmotion evaluate feature-importance --data data/processed/dataset.csv
```

`mega-ensemble` needs its member checkpoints present; `model list` shows which.
`--no-tabpfn` and `--no-classical` restrict the member library.

---

## Analysis and figures

```bash
uv run inmotion analyze seeds --results-dir results --seeds 3,5,42
uv run inmotion analyze interference --data data/processed/dataset.csv
uv run inmotion plots classical --results-dir results/results_42
uv run inmotion plots dl --results-csv results/dl/dl_detailed_seed42.csv
```

Figures are regenerable output and are not tracked; the result CSVs they read
are. `--eda` on either `plots` command rebuilds the exploratory figures too and
needs `--data`.

---

## On a cluster

`scripts/slurm/*.sh` carry `#SBATCH` headers and call the same commands:

```bash
sbatch scripts/slurm/job_full_train.sh
```

Credentials come from the environment (`env.example`), never from the scripts.
`scripts/README.md` says which script does what and which are superseded.

---

## Driving a pipeline from Python

The CLI is a thin layer over `run(args) -> int`, so a notebook or a sweep can
bypass it:

```python
from inmotion.pipelines.exotic import RunConfig, run

for seed in (42, 7, 123):
    code = run(RunConfig(model="ts_jepa", seed=seed, pretrain_epochs=300))
    if code:
        raise SystemExit(code)
```

Every field name is documented in
[`../reference/pipelines.md`](../reference/pipelines.md).
