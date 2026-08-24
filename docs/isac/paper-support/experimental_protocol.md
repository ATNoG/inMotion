# ISAC 2026 — Experimental Protocol

## 1. Data

| Item | Value |
|---|---|
| Primary evaluation dataset | `dataset.icaisf.csv` (ICAISF variant, 4 classes AA/AB/BA/BB) |
| JEPA training dataset | `dataset_augmented3.csv` (class-conditional Gaussian noise + intra-class Mixup augmentation of the corrected-July-labels base) |
| Source collection | 10 RSSI readings at 1 s intervals per observation; pure (single device) + noise (4 concurrent devices) scenarios; same 4 devices as the prior EUCNC study |
| Split | 80/20 stratified train-test; 12.5% of train held as validation (≈10% of total) |
| Split seed | 42, fixed across all runs (`_stratified_indices(y, 42)` in `mega_ensemble.py`) |
| Sequence length | 10 timesteps |
| Feature channels | 4 (raw, Δ, Δ², window-dev) or 18 (rich: +spectral/stats/shape/STFT) |

## 2. Multi-seed protocol (IMPORTANT — must be stated in the paper)

> All experiments reported here were run with **seed 42** on this machine. The
> same training pipelines were additionally run on a second machine with seeds
> **3, 5, 7, 123, and 2024** (multi-seed runs, `job_multi_seed*.sh`).
> **Results across all seeds converge to the same values** — per-seed test MCCs
> vary by less than the run-to-run noise of a single training run (see the
> sigreg_s3/sigreg_s5 members, §5 of MASTER_RESULTS.md, and the multi-seed
> finetune-only runs in §6). Because the numbers are statistically
> indistinguishable, the results tables report the seed-42 runs only; the
> multi-seed convergence is stated here and in the results section rather than
> duplicating near-identical tables.

**Paper wording suggestion:**
"All results were obtained with a fixed split seed (42). To confirm stability,
each pipeline was additionally trained with seeds 3, 5, 7, 123 and 2024 on a
second machine; the resulting test MCCs converge to the values reported here
(within run-to-run variance), so the paper reports the seed-42 runs throughout."

## 3. Training protocols

### 3.1 JEPA pretraining (run_exotic.py `pretrain_jepa`)
- Optimizer: AdamW, `pretrain_lr` (per-model HPO winner), weight decay 1e-4,
  cosine annealing to 1% of peak LR.
- Gradient clipping at 1.0.
- **EMA target encoder**: momentum cosine-scheduled `ema_start → ema_end`
  (e.g. 0.99 → 0.996; CF-JEPA 0.996 → 0.999).
- Masking: T-JEPA 30% contiguous block; TS-JEPA patch-block ratio annealed
  0.4→0.7; LeJEPA next-step (no mask); CF-JEPA mask-free multi-horizon.
- **Checkpoint selection**: best linear-probe MCC (evaluated every 10 epochs) —
  NOT raw val loss.
- Epochs: ~300–500 full pretraining for paper models (finetune-only runs from
  short 9-epoch pretrains collapse to 0.49–0.60 test MCC and are excluded).

### 3.2 JEPA fine-tuning (`finetune_jepa`)
- Two-stage progressive unfreezing:
  - Stage 1: encoder frozen, train head only, `finetune_epochs/4` epochs at
    `finetune_lr`.
  - Stage 2: encoder unfrozen, `finetune_epochs − stage1` epochs at
    `finetune_lr × 0.1`.
- Mixup (α=0.4), label smoothing (ε=0.1), early stopping (patience 25, Trainer
  default), best-val-MCC checkpoint selection.
- Head: learned-attention pool → MLP (see architectures file).

### 3.3 SIGReg classifier
- Direct supervised training, cross-entropy + λ·SIGReg(latents), AdamW, cosine,
  early stopping.

### 3.4 Classical DL models (run_dl.py)
- Per-model default configs; HPO_* variants searched by Optuna (study
  `optuna_dl_9344.db`, 50 TPE trials per model), best trial rebuilt and
  retrained.

### 3.5 DeepStackEnsemble
- Two-stage: train 10 bases, freeze, train 4 per-class Level2Nets on the
  concatenated 40-dim base probabilities, then DeepMetaMLP on the 16-dim
  Level2Net outputs. Trained Jun 7 on `dataset_augmented.csv`.

## 4. Hyperparameter optimization (HPO)

- JEPA family: Optuna studies in `optuna_exotic.db` / `optuna_dl_2616.db` /
  `optuna_dl_9344.db`; search over d_model, nhead, layers, dim_ff, pred_dim,
  pred_layers, lrs, EMA schedule.
- **Pareto frontier**: NSGA-II over MCC × parameter count; the Pareto winners
  are **fully retrained** (`--hpo-pareto-train`, "full-after-hpo" runs) at the
  winner configs and those retrained checkpoints are the paper's headline
  models (§2.2 of MASTER_RESULTS.md).
- HPO proxy scores (40 epochs) are reported separately from the full-run test
  MCCs and must not be conflated.

## 5. Ensemble construction (mega_ensemble.py)

1. **Members**: 21 DL models (JEPA family + Mamba-3 family + HPO DL + DeepStack
   + recurrents) plus optional classical (kNN/LR/GP/CatBoost/ROCKET/catch22/
   TabPFN). Each member loads its own checkpoint (prefix handling for
   checkpoint-key mismatches) and predicts softmax probabilities on val/test.
2. **Feature pipelines**: JEPA models with `rich=True` consume the 18-channel
   features; all others consume 4-channel. (sigreg, mamba3, DL, classical →
   plain 4; t_jepa/ts_jepa/cf_jepa → rich 18.)
3. **Temperature calibration**: per-member scalar T fit to minimize validation
   NLL; probabilities rescaled before ensembling.
4. **Combiners compared**:
   - **LR-stack** (winner): multinomial LogisticRegression over the
     concatenated calibrated probabilities `(n_members × 4)`-dim; C swept over
     [0.005 … 10] (11 values), best C chosen on validation MCC, evaluated on
     test. `max_iter=5000`.
   - Equal-weight mean, entropy-weighted mean, rank average, bagged-Caruana
     selection (20 bags × 50 steps on 60% val subsamples).
5. **Subset selection**: fixed member subsets (world-only, world+best-DL,
   best-world, best-world+best-DL) and greedy selection on validation MCC
   (config 6). **Size frontier**: greedy addition of the member with best
   MCC-per-added-param, reporting total params vs test MCC.

## 6. Metrics

- **Primary**: Matthews Correlation Coefficient (MCC) on the held-out test
  split (seed-42 split fixed for all members and ensembles).
- Accuracy reported alongside. Param counts from checkpoint `state_dict`
  (`sum(p.numel())`).
- Model selection always on **validation**; test used once for reporting.

## 7. Reproducibility

- Split: `_stratified_indices(y, 42)` in `mega_ensemble.py`.
- Training: `uv run python run_exotic.py ...` with per-model Pareto-winner
  flags in `docs/isac/results/MASTER_RESULTS.md` §7 (example given) and
  `job_*.sh` scripts in the repo root.
- Ensemble: `MAMBA_SSM_AVAILABLE=0 uv run python mega_ensemble.py --data
  dataset.icaisf.csv --seed 42 --no-tabpfn --no-classical --size-frontier`.
- Every number in the results tables has a provenance pointer (CSV/log/ckpt)
  in `docs/isac/results/MASTER_RESULTS.md`.
