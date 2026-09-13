# Reproducibility

What an independent reader needs in order to re-run this work and compare
numbers. Every claim below is checkable against a file in this repository.

## Environment

| item | value | source |
|---|---|---|
| Python | `>=3.13` (`.python-version` pins 3.13) | `pyproject.toml`, `.python-version` |
| package manager | `uv` | `uv.lock` (fully pinned, including transitive) |
| PyTorch | 2.9.1+cu126, from the explicit `pytorch-cu126` index | `pyproject.toml` |
| CUDA | required for the reported DL runs; CPU is fine for tests | `torch.cuda.is_available()` |

`uv.lock` is committed, so `uv sync --all-packages` reproduces the exact
dependency graph. `uv run inmotion doctor` reports the resolved versions and
whether CUDA is visible.

## Determinism

| source of variation | how it is controlled |
|---|---|
| Python / NumPy / PyTorch seeds | `inmotion.models.registry`-independent; each pipeline calls `set_seed(seed)` at entry |
| split into train/val/test | stratified, fixed seed, defined once in `_stratified_indices` |
| Optuna sampler | seeded where the study is created |
| dataset selection | explicit path; `inmotion datasets verify` proves the file matches its recorded checksum |

Seeds used in the reported results: **42** (primary), plus 3 and 5 for
multi-seed diversity in the ensembles.

**Known non-determinism.** cuDNN convolution kernels are not bit-reproducible by
default. Re-running a DL training script with the same seed reproduces the
*procedure* and lands within floating-point noise of the reported metric, but not
the identical float. `tests/test_model_outputs_golden.py` accounts for this by
rounding before hashing, so it detects logic changes without failing on
kernel-level differences.

## The split convention

`_stratified_indices(y, 42)` in `inmotion.pipelines.mega_ensemble` implements it:

| item | value |
|---|---|
| primary dataset | `dataset.icaisf.csv` for the ensemble; `dataset_augmented3.csv` for training the world models |
| split | 80/20 train-test, stratified by label |
| validation | 12.5% of the train portion |
| split seed | 42 |
| metric | **Matthews Correlation Coefficient (MCC)**, macro-averaged |

MCC is the reported metric throughout because the classes are only approximately
balanced and MCC is robust to that where accuracy is not.

## Reproducing the paper numbers

```bash
uv sync --all-packages

# 1. the numbers are loadable at all
uv run inmotion model verify

# 2. the datasets have not changed under git
uv run inmotion datasets verify

# 3. member-by-member MCC, then the ensemble
uv run inmotion evaluate mega-ensemble \
    --data data/processed/dataset.icaisf.csv --seed 42
```

## Provenance gaps

Recorded rather than papered over. `inmotion model info <name>` prints each one.

### 1. The Mamba-3 hybrids are not part of the reported results

Four Mamba-3 hybrids are registered and loadable, but they were explored rather
than reported, so no MCC or parameter count is claimed for them and none is
checked against the paper table. They carry
`note="experimental; not in the reported results"`.

### 2. Result files cited by MASTER_RESULTS are absent

`results/mega_ensemble_results.csv`, `results/ensemble_size_frontier.csv` and
`models/dl/augmented/DeepStackEnsemble_seed42.pt` were already missing before
this work. Nothing was fabricated to fill the gap. The equivalent DeepStack
artifact is `backup/DeepStackEnsemble_seed42.pt`.

### 3. A hyperparameter search once reported inactive dimensions

The Mamba-3 search suggested three parameters that its builder hardcoded, so the
study reported a five-dimensional space while three dimensions could not affect
the model. The code is fixed. Any Mamba-3 HPO log predates the fix; nothing in
the reported results depends on those logs.

### 4. The pre-recollection dataset labels are reported incorrect

`ds-pure`, `ds-noise` and `ds-augmented-v1` carry labels the project owner
reports as wrong. They are marked `known_incorrect` in the manifest and
`inmotion datasets resolve` warns. They are retained for provenance only.
`ds-icaisf`, the dataset the ensemble was scored on, sits in the same 8-device
lineage and is marked `unverified` — if it is the older lineage, the paper's
numbers are affected. This needs a decision, not a code change.

## What is not reproducible

| item | why |
|---|---|
| the deleted Optuna databases (`optuna_dl_*.db`) | never committed; `train_hpo_paper.py` exists to rebuild the winning configs from the documented values |
| the pre-fix Mamba-3 HPO search | see gap 3 |
| `20-aug/logs/` | committed, so the *reported* runs are traceable, but the GPU jobs are not re-runnable without the cluster |
| pretraining bundles | `*_pretrain_best.pt` hold a predictor plus encoder rather than one module's weights; resume them through the pipeline, see `docs/reports/ARTIFACTS.md` |
