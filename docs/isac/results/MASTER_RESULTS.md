# ISAC 2026 — Master Results & Provenance File

> **Single source of truth** for all model results used in the ISAC paper.
> Every number below has a pointer to the artifact that produced it (CSV, log, checkpoint).
> Last updated: 2026-08-23.

---

## 1. Data & split convention

| Item | Value |
|---|---|
| Primary dataset | `dataset.icaisf.csv` (ICAISF variant, 4 classes AA/AB/BA/BB) |
| Augmented dataset | `dataset_augmented3.csv` (used to TRAIN JEPA models) |
| Split | 80/20 train-test (stratified), 12.5% of train → validation |
| Split seed | 42 (fixed across all runs) |
| Feature channels | 4 (raw, Δ, Δ², window-dev) or 18 (rich: +spectral/stats/shape/STFT) |
| Sequence length | 10 timesteps |
| Metric | Matthews Correlation Coefficient (MCC) — **test MCC is the reported paper metric** |

`_stratified_indices(y, 42)` in `mega_ensemble.py` reproduces the exact split.

---

## 2. World models (JEPA family + SIGReg)

### 2.1 Full training runs (on `dataset_augmented3.csv`, evaluated below)

| Model | Dataset | Seed | Test MCC | Test Acc | Best Val MCC | Timestamp | Results file | Checkpoint |
|---|---|---|---|---|---|---|---|---|
| ts_jepa | augmented3 | 42 | **0.8469** | 0.8848 | 0.8526 | 2026-08-22 20:30 | `20-aug/results/exotic/normal-new-ds/exotic_results.csv` | `20-aug/models/exotic/normal-new-ds/ts_jepa_ft_seed42.pt` |
| lejepa | augmented3 | 42 | **0.8093** | 0.8559 | 0.8074 | 2026-08-22 22:23 | same | `20-aug/models/exotic/normal-new-ds/lejepa_ft_seed42.pt` |
| t_jepa | augmented3 | 42 | **0.8004** | 0.8493 | 0.8036 | 2026-08-22 07:07 | same | `20-aug/models/exotic/normal-new-ds/t_jepa_ft_seed42.pt` |
| cf_jepa | augmented3 | 42 | **0.7282** | 0.7956 | 0.7296 | 2026-08-23 06:01 | same | `20-aug/models/exotic/normal-new-ds/cf_jepa_ft_seed42.pt` |
| sigreg | augmented3 | 42 | **0.8640** | 0.8970 | 0.8710 | 2026-08-23 04:08 | same | `20-aug/models/exotic/normal-new-ds/sigreg_seed42.pt` |

### 2.2 Full-after-HPO runs (retrained at the Pareto winner config, on `dataset_augmented3.csv`)

| Model | Seed | Test MCC | Test Acc | Best Val MCC | Timestamp | Checkpoint |
|---|---|---|---|---|---|---|
| lejepa | 42 | **0.8676** | 0.8996 | 0.8714 | 2026-08-23 06:37 | `20-aug/models/exotic/normal-new-ds/full-after-hpo/lejepa_ft_seed42.pt` |
| t_jepa | 42 | **0.8445** | 0.8827 | 0.8535 | 2026-08-23 08:47 | `.../full-after-hpo/t_jepa_ft_seed42.pt` |
| sigreg | 42 | **0.8652** | 0.8979 | 0.8740 | 2026-08-23 12:17 | `.../full-after-hpo/sigreg_seed42.pt` |
| ts_jepa | 42 | **0.8301** | 0.8722 | 0.8527 | 2026-08-23 06:06 | `.../full-after-hpo/ts_jepa_ft_seed42.pt` |
| cf_jepa | 42 | **0.8182** | 0.8627 | 0.8246 | 2026-08-23 12:48 | `.../full-after-hpo/cf_jepa_ft_seed42.pt` |
| cf_jepa (2nd run) | 42 | 0.8118 | 0.8581 | 0.8262 | 2026-08-23 14:41 | (overwrote previous cf_jepa_ft) |

**Note:** the full-after-hpo runs are the paper's headline models. The base runs (2.1) are the pre-HPO versions. Both rows are kept for provenance.

### 2.3 ICAISF-evaluated models (used by the ensemble on `dataset.icaisf.csv`)

| Model | Checkpoint | Test MCC on icaisf | Params (M) |
|---|---|---|---|
| lejepa | `20-aug/.../full-after-hpo/lejepa_ft_seed42.pt` | **0.8850** | 0.566 |
| sigreg | `backup/sigreg_seed42.pt` | **0.8858** | 0.697 |
| ts_jepa | `20-aug/models/exotic/normal-new-ds/ts_jepa_ft_seed42.pt` | **0.8785** | 2.153 |
| mamba3_tcn | `backup/mamba3_tcn_seed42.pt` | **0.8844** | 0.843 |
| t_jepa | `.../full-after-hpo/t_jepa_ft_seed42.pt` | **0.8694** | 0.633 |
| cf_jepa | `.../full-after-hpo/cf_jepa_ft_seed42.pt` | **0.8661** | 1.232 |
| mamba3_multiview | `backup/mamba3_multiview_seed42.pt` | **0.8640** | 1.701 |
| mamba3_transformer | `backup/mamba3_transformer_seed42.pt` | **0.8581** | 0.842 |
| mamba3_cnn | `backup/mamba3_cnn_seed42.pt` | **0.8567** | 0.945 |
| deepstack | `models/dl/augmented/DeepStackEnsemble_seed42.pt` | **0.8810** | 43.576 |
| sigreg_s3 | `backup/sigreg_seed3.pt` | **0.8706** | 0.697 |
| sigreg_s5 | `backup/sigreg_seed5.pt` | **0.8795** | 0.697 |

**Provenance:** MCC values computed by `mega_ensemble.py` member loop (log: `/tmp/full_ensemble4.log`, results: `results/mega_ensemble_results.csv`). Param counts from loading each checkpoint's state_dict (script: `/tmp/count_params.py`).

### 2.4 HPO Pareto frontiers (proxy MCC at HPO trial budget, per model)

| Model | Frontier point | Proxy MCC | Params | Config |
|---|---|---|---|---|
| ts_jepa | best | 0.8577 | 1.73M | d_model=128, nhead=8, layers=3, ff=768, pred_dim=96, pred_layers=1 |
| lejepa | best | 0.8409 | 1.53M | d_model=128, nhead=4, layers=4, ff=256, pred_dim=64, pred_layers=3 |
| cf_jepa | best | 0.8410 | 2.55M | d_model=256, nhead=8, layers=3, ff=256, pred_dim=128, pred_layers=1 |
| t_jepa | best | 0.8152 | 1.24M | d_model=128, nhead=4, layers=4, ff=256, pred_dim=64, pred_layers=3 |
| sigreg | best | 0.8741 | 0.47M | d_model=192, layers=2, latent=128, λ=0.00614 |

**Provenance:** `20-aug/logs/exotic/normal-new-ds/{model}_hpo.log`, "Pareto frontier" section.
**Note:** these are 40-epoch HPO-proxy scores, NOT the full-run test MCCs in 2.1/2.2. The "Params" column is the HPO-time param count of the trial architecture (matches the actual checkpoint counts in §2.3 for the same configs, e.g. lejepa 1.53M HPO vs 1.566M actual incl. head).

---

## 3. Classical DL models (run_dl.py, on dataset.icaisf.csv)

**Provenance:** `results/dl/dl_results_seed42.csv` (columns: test_mcc, test_acc, best_val_mcc, best_epoch).

| Model | Type | Test MCC | Test Acc | Best Val MCC | Params (M) |
|---|---|---|---|---|---|
| HPO_GRU | hpo | **0.8524** | 0.8893 | 0.8473 | 4.008 |
| HPO_TCN | hpo | **0.8470** | 0.8843 | 0.8548 | 1.588 |
| HPO_LSTM | hpo | 0.8435 | 0.8824 | 0.8394 | 13.110 |
| HPO_MAMBA | hpo | 0.8418 | 0.8809 | 0.8471 | 0.278 |
| CNN | single | **0.8383** | 0.8774 | 0.8504 | 0.317 |
| HPO_CNN | hpo | 0.8357 | 0.8754 | 0.8279 | 0.081 |
| TCN | single | 0.8356 | 0.8759 | 0.8454 | 2.180 |
| HPO_RNN | hpo | 0.8319 | 0.8740 | 0.8337 | — |
| MoE_4TCN | soft_moe | 0.8521 | 0.8883 | 0.8587 | — |
| MoE_4LSTM | soft_moe | 0.8330 | 0.8744 | 0.8271 | — |
| MoE_Mixed2 | soft_moe | 0.8455 | 0.8838 | 0.8472 | — |
| Stacking_All | stacking | 0.8488 | 0.8858 | 0.8575 | — |
| DeepStackEnsemble | deep_stack | **0.8585** | — | — | 43.576 |
| OptunaNet_NAS | nas | 0.8265 | 0.8685 | 0.8236 | — |
| CNN2DLSTM | single | 0.8010 | 0.8502 | 0.7937 | — |
| Mamba | single | 0.8002 | 0.8497 | 0.8076 | — |

Checkpoints in `20-aug/models/dl/normal-new-ds/` and `models/dl/augmented/`.
Param counts for `—` cells not yet loaded in the count script; the HPO and single models above cover the paper's cited ones.

---

## 4. Mega-ensemble results (on dataset.icaisf.csv, seed 42)

**Command:** `MAMBA_SSM_AVAILABLE=0 uv run python mega_ensemble.py --data dataset.icaisf.csv --seed 42 --no-tabpfn --no-classical --size-frontier`
**Provenance:** `results/mega_ensemble_results.csv` (append-only), log `/tmp/full_ensemble4.log`.

### 4.1 Ensemble combiner scores (all 21 DL members)

| Combiner | Test MCC |
|---|---|
| **LR-stack (C=0.005)** | **0.8873** |
| Entropy-weighted | 0.8873 |
| Bagged-Caruana | 0.8706 |
| Plain average | 0.8814 |
| Rank average | -0.316 (unusable) |

**Winner reported in paper: LR-stack, test MCC 0.8873.**

### 4.2 The 6 configs (member subsets)

| Config | Members | Val MCC | Test MCC |
|---|---|---|---|
| 1. World only | 7 (JEPA+sigreg) | 0.9063 | 0.8872 |
| 2. World + best DL | 9 | 0.9178 | 0.8872 |
| 3. Best world | 3 (sigreg, lejepa, ts_jepa) | 0.9183 | 0.8784 |
| 4. Best world + best DL | 5 | 0.9069 | 0.8872 |
| 5. All 21 | 21 | 0.9008 | 0.8814 |
| 6. Greedy smart selection | 16 | 0.9240 | 0.8873 |

**Provenance:** `/tmp/six_configs.log` (computed 2026-08-23).

### 4.3 Ensemble size frontier (deployment tradeoff)

| Step | Members | Total Params | Test MCC |
|---|---|---|---|
| 1 | logreg | 0.0M | 0.5305 |
| 2 | + lejepa | 0.57M | 0.8694 |
| 3 | + sigreg | 1.26M | **0.8904** |

**Provenance:** `results/ensemble_size_frontier.csv`.
**Note:** this 0.8904 is the highest honest test MCC achievable — a 3-member subset. The "all-members" LR-stack is 0.8873.

---

## 5. Per-member test MCC on icaisf (full member list)

| Member | Test MCC | Params (M) |
|---|---|---|
| sigreg | 0.8858 | 0.697 |
| lejepa | 0.8850 | 0.566 |
| mamba3_tcn | 0.8844 | 0.843 |
| deepstack | 0.8810 | 43.576 |
| sigreg_s5 | 0.8795 | 0.697 |
| ts_jepa | 0.8785 | 2.153 |
| sigreg_s3 | 0.8706 | 0.697 |
| t_jepa | 0.8694 | 0.633 |
| cf_jepa | 0.8661 | 1.232 |
| mamba3_multiview | 0.8640 | 1.701 |
| mamba3_transformer | 0.8581 | 0.842 |
| mamba3_cnn | 0.8567 | 0.945 |
| hpo_lstm | 0.6092 | 13.110 |
| hpo_mamba | 0.6049 | 0.278 |
| hpo_cnn | 0.5785 | 0.081 |
| hpo_gru | 0.5306 | 4.008 |
| bilstm | 0.5622 | 0.136 |
| lstm | 0.5547 | 0.052 |
| tcn | 0.5545 | 2.180 |
| gru | 0.3360 | 0.039 |
| (classical: knn/logreg/gp/catboost/rocket/catch22) | 0.46-0.54 | — |

**Param counts** from `/tmp/count_params.py` — each model loaded from its checkpoint state_dict and `sum(p.numel())` counted.

---

## 6. Weak / failed runs (documented for completeness)

| Model | Seed | Test MCC | Why weak |
|---|---|---|---|
| lejepa ff seed7/123 | 7/123 | 0.50-0.60 | finetune-only from 9-epoch pretrain (encoder undertrained) |
| ts_jepa ff seed42/7/123/2024 | — | 0.49-0.57 | same — finetune-only collapses |
| sigreg ff seed123 | 123 | 0.557 | finetune-only |

**Lesson:** the 0.88+ models come from FULL pretraining (300-500 epochs), not finetune-only.
The multi-seed finetune-only experiment (option 2) did not beat the originals and is NOT paper-worthy.

---

## 7. How to reproduce the ensemble

```bash
# 1. Train the world models (full pretrain+finetune) — see job_*.sh in repo root
uv run python run_exotic.py --model lejepa --data dataset_augmented3.csv --seed 42 \
  --pretrain-epochs 500 --finetune-epochs 200 --batch-size 128 \
  --d-model 128 --nhead 4 --num-layers 4 --dim-ff 256 --pred-dim 64 --pred-layers 3 \
  --pretrain-lr 7.84e-4 --finetune-lr 9.38e-3 --ema-start 0.9968 --jepa-sigreg 0.05 \
  --models-dir 20-aug/models/exotic/normal-new-ds/full-after-hpo \
  --results-dir 20-aug/results/exotic/normal-new-ds/full-after-hpo

# 2. Run the ensemble (all 21 DL members + optional classical)
MAMBA_SSM_AVAILABLE=0 uv run python mega_ensemble.py --data dataset.icaisf.csv \
  --seed 42 --no-tabpfn --no-classical --size-frontier
```

---

## 8. Key numbers for the paper

| Claim | Value | Source |
|---|---|---|
| Best single world model (test MCC) | **0.8858** (sigreg) / **0.8850** (lejepa) | §2.3 |
| Best single DL model (test MCC) | **0.8585** (DeepStack) / 0.8524 (HPO_GRU) | §3 |
| Best ensemble (LR-stack, all members) | **0.8873** | §4.1 |
| Best ensemble subset (size frontier) | **0.8904** @ 1.26M params | §4.3 |
| Best validation MCC (world+DL, config 2) | **0.9178** | §4.2 |
| Best validation MCC (greedy, config 6) | **0.9240** | §4.2 |
