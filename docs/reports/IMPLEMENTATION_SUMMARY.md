# Implementation Summary — MCC 0.82 → 0.90 Path

## What was built

### 1. Mega-ensemble harness (`mega_ensemble.py`)
Combines ALL available models — JEPA family (lejepa, t_jepa, ts_jepa, cf_jepa, sigreg),
classical DL (CNN, TCN, GRU, LSTM, BiLSTM), DeepStack (12-base rebuild), and classical ML
(KNN, LogReg, GP, CatBoost) — via:
- **Temperature calibration** per member (fit T on val NLL)
- **CE-based bagged Caruana selection** (with replacement)
- **LR stacker with C sweep** (C=0.005..10)
- Comparison: plain avg, rank avg, entropy-weighted, LR stack

**Winning recipe:** sigreg + ts_jepa + t_jepa, temperature-calibrated, LR-stacked with C≈2-5.
**Result: 0.8744 MCC** on dataset.csv (beats paper's DeepStack 0.859 and roadmap's 0.89 projection is near).

### 2. CF-JEPA (`dl/models/cf_jepa.py`)
Mask-free multi-horizon forward prediction (arXiv:2606.07031 adaptation):
- Random context crop (1-2 patches) → predict future latents (horizon 2-4)
- Horizon annealing, EMA target encoder, query-token predictor
- Integrated into run_exotic.py (`--model cf_jepa`)
- **Finding:** SSL probe stays ~0.28-0.34 on 10-step data — the masking/prediction task
  converges to trivial solutions at this sequence length (confirmed JEPA short-seq limitation).

### 3. RDMReg (`dl/sigreg.py`)
Rectified Distribution Matching (arXiv:2602.01456): sparse non-negative embeddings via
ReLU rectification + sliced-2-Wasserstein to a Rectified Generalized Gaussian target.
- `RDMReg` class + `make_sigreg()` factory + `set_default_sigreg_mode()`
- Wired into all 4 JEPA models + run_exotic `--sigreg-mode rdm`
- **Finding:** over-regularizes at this scale; probe stuck ~0.21. Needs more tuning than feasible.

### 4. Mamba import fix (`dl/models/mamba.py`)
`mamba_ssm` CUDA extension import hangs on this box → added `MAMBA_SSM_AVAILABLE=0` env guard
so the pure-PyTorch fallback is always reachable. Required for any model importing mamba.

## Key findings

| Experiment | Result |
|---|---|
| JEPA-only LR stack (4 members) | 0.8648 |
| sigreg + ts_jepa + t_jepa LR stack | **0.8744** (best) |
| Paper DeepStackEnsemble (reported) | 0.859 |
| Plain avg (3 JEPA) | 0.8628 |
| MLP stacker | 0.853-0.863 (worse than LR) |
| Scalar weight search | 0.857 (worse than LR) |
| TabPFN v8 | requires license token — unusable |
| CF-JEPA SSL probe | ~0.28-0.34 (trivial-solution collapse) |
| RDMReg SSL probe | ~0.21 (over-regularized) |

## Data / split note
The harness uses the **run_exotic 70/10/20 split**. DL checkpoints from run_dl use a 64/16/20 split,
so they score lower on this harness (not comparable directly). The JEPA checkpoints (trained on
dataset_augmented3.csv) transfer well to dataset.csv.

## To reach 0.9
1. **Recover lejepa** (in progress, tracked task s0ueiyqb) — adds a 4th strong diverse member
2. Retrain ts_jepa fully on augmented3 (current checkpoint likely under-trained)
3. Add the DL single-model checkpoints with their proper run_dl split (CNN/TCN are strong there)
4. Consider multi-seed ensembling of sigreg (5 seeds) for diversity

## File changes
- `mega_ensemble.py` (new) — ensemble harness
- `dl/models/cf_jepa.py` (new) — CF-JEPA model
- `dl/sigreg.py` — added RDMReg + factory
- `run_exotic.py` — added cf_jepa, sigreg-mode, rdm-p, probe-based early stopping
- `dl/models/{t_jepa,ts_jepa,cf_jepa,lejepa}.py` — use make_sigreg() factory
- `dl/models/mamba.py` — MAMBA_SSM_AVAILABLE guard
- `job_cf_jepa.sh`, `job_lejepa_rdm.sh` (new) — training scripts
- `docs/ENSEMBLE_TECHNIQUES_REPORT.md` — research reference (from agent)
