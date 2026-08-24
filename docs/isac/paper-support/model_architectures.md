# ISAC 2026 — Model Architectures

Precise architectural description of every model used, extracted from the code
(`dl/models/*.py`, `run_exotic.py`, `mega_ensemble.py`). Parameter counts are the
measured checkpoint counts (see `docs/isac/results/MASTER_RESULTS.md` §5).

Common inputs: 10-timestep RSSI windows, shape `(N, 10, C)` with C=4 (plain) or
C=18 (rich). 4 classes: AA, AB, BA, BB.

---

## 1. JEPA family (self-supervised, then fine-tuned)

All JEPA variants share the JEPA recipe: an **online/context encoder** trained by
gradient descent, a **target encoder** updated by exponential moving average (EMA,
no gradients), and a **predictor** that must be shallower and narrower than the
encoder (the JEPA bottleneck). A linear probe MCC during pretraining selects the
best checkpoint.

### 1.1 T-JEPA (`t_jepa`, `dl/models/t_jepa.py`)
- **Tokenizer**: per-(timestep,channel) "feature" tokens; a per-feature linear
  projection with feature-index embeddings, plus **2 learnable [REG] tokens**
  prepended (never masked, never predicted) — the paper's collapse-prevention
  mechanism.
- **Masking**: contiguous temporal block of `mask_ratio=0.3` of the timesteps
  masked per sample; 1 target mask per context.
- **Encoder**: Transformer encoder, `d_model`, `nhead`, `num_layers`,
  `dim_feedforward`, GELU, pre-LayerNorm, `input_proj` linear + final LayerNorm.
- **Predictor**: `pred_dim` bottleneck, `pred_num_layers` transformer layers at
  half `dim_feedforward`, fixed sin/cos positional embedding, mask tokens.
- **Loss**: L2 on unit-normalized latents (target vs prediction) at masked
  positions; optional SIGReg anti-collapse term on pooled context latents
  (`jepa_sigreg`).
- **Classifier head**: learned-attention pooling over context-encoder latents →
  Linear(d_model→128) → GELU → Dropout(0.3) → Linear(128→4).
- **Paper config (full-after-HPO)**: d_model=128, nhead=4, layers=4, ff=256,
  pred_dim=64, pred_layers=3, 18 channels, 0.633 M params.

### 1.2 TS-JEPA (`ts_jepa`, `dl/models/ts_jepa.py`)
- **Patch tokenizer**: Conv1D over channels, `patch_size=2`, `seq_len=10` → 5
  patches; learnable positional embedding per patch.
- **Masking**: contiguous temporal block of patches (`mask_ratio` annealed
  0.4→0.7 during training).
- **Encoder/predictor**: Transformer (d_model=embed_dim, nhead, layers, ff);
  predictor bottleneck at `pred_dim`, `pred_num_layers`.
- **Loss**: L2 on normalized latents; EMA target encoder.
- **Paper config (full run)**: embed_dim=256, nhead=8, layers=4, ff=512,
  pred_dim=128, pred_layers=2, 18 channels, 2.153 M params.

### 1.3 LeJEPA (`lejepa`, `dl/models/lejepa.py`)
- **End-to-end next-step latent prediction**: no EMA, no stop-grad, no REG
  tokens. Encoder → causal Transformer predictor predicting `z_{t+1}` from
  `z_{1..t}`.
- **Loss**: MSE between predicted next-step latents and target latents plus
  **SIGReg** (ECF-based Gaussian regularizer) on all latents, weight
  `sigreg_lambda`.
- **Paper config (full-after-HPO)**: d_model=128, nhead=4, layers=4, ff=256,
  pred_layers=3, 4 channels, 0.566 M params.

### 1.4 CF-JEPA (`cf_jepa`, `dl/models/cf_jepa.py`)
- **Mask-free multi-horizon forward prediction**: random context crop of the
  encoded patch sequence; predictor maps context latents to future latents at
  horizons 1→3 (annealed during training, `horizon_start=1`, `horizon_end=3`;
  context crop length 1–3 patches).
- Conv1D patch tokenizer (patch_size=2), EMA target encoder, optional SIGReg.
- **Paper config (full-after-HPO)**: embed_dim=256, nhead=8, layers=3, ff=256,
  pred_dim=128, pred_layers=1, 18 channels, 1.232 M params.

### 1.5 SIGReg classifier (`sigreg`, `dl/models/sigreg_classifier.py`)
- **Directly supervised**, not a JEPA: CNN encoder (`_CNNEncoder`,
  `num_filters`, `num_blocks`) → global pooling → Linear(num_filters→latent_dim)
  → LayerNorm → Linear(latent_dim→4).
- **Loss**: cross-entropy + `sigreg_lambda × SIGReg(latents)` (ECF Gaussian
  regularizer on the latent bottleneck, 256 random 1-D slices).
- **Paper config**: num_filters=192, blocks=3, latent_dim=128, λ=0.01, 4
  channels, 0.697 M params.

---

## 2. Mamba-3 family (`dl/models/mamba3_*.py`)

State-space (Mamba) backbones with different front-end fusion; all take 4
channels and use `d_state=16`, dropout 0.2, `mimo_rank=4`.

| Variant | Front-end | d_model | Params (M) |
|---|---|---|---|
| `mamba3_cnn` | CNN (cnn_channels=192) | 192 | 0.945 |
| `mamba3_tcn` | TCN (tcn_channels=128) | 128 | 0.843 |
| `mamba3_transformer` | Transformer (nhead=4) | 128 | 0.842 |
| `mamba3_multiview` | Multi-view fusion | 256 | 1.701 |

## 3. Classic deep-sequence models (`dl/models/`)

| Model | Architecture | Params (M) |
|---|---|---|
| `cnn` | Multi-kernel CNN, kernels [3,5,7], 128 filters | 0.317 |
| `tcn` | Dilated causal TCN, 256 ch, depth 3, dilation 6, k=4 | 2.180 |
| `gru` | 2-layer GRU hidden 64 + attention | 0.039 |
| `lstm` | 2-layer LSTM hidden 64 + attention | 0.052 |
| `bilstm` | 2-layer BiLSTM hidden 64 + attention | 0.136 |
| `cnn2d_rnn` | Conv2d (3,3),(5,3) → LSTM/GRU + attention | — |
| HPO_GRU / HPO_LSTM / HPO_CNN / HPO_MAMBA / HPO_RNN | Optuna-searched variants (study `optuna_dl_9344.db`, rebuilt from `best_trial.params`) | 4.008 / 13.110 / 0.081 / 0.278 / — |
| MoE_4TCN / MoE_4LSTM / MoE_Mixed2 | Soft mixture of experts, top-2 gating | — |
| Stacking_All | Stacked generalization over base models | — |
| OptunaNet_NAS | NAS-searched network | — |
| Autoformer | FFT auto-correlation decomposition transformer | — |

## 4. DeepStackEnsemble (`deepstack`, `dl/models/deep_stack.py`)

- **10 base models** (2 configs each of CNN2DRNN, CNN, GRU, Mamba, RNN, TCN —
  see `_build_ds_ensemble` in `mega_ensemble.py`), each emitting 4-class
  softmax probabilities.
- **4 per-class Level2Nets** (`Level2Net(40, 4)`, small MLP) each receive the
  concatenated 10×4 = 40-dim base probabilities and output 4-class logits.
- **DeepMetaMLP**: receives the 4 Level2Net outputs (4×4 = 16-dim) → final
  4-class logits, dropout 0.3.
- **Params**: 43.576 M (the paper's "DeepStackEnsemble"). Note: the paper's
  original table listed 18 bases / 1.2 M; the actual checkpoint is 10 bases /
  43.6 M — the paper text must match the measured checkpoint.

## 5. Classical ML members (`mega_ensemble.py`)

Operate on the **raw 10 RSSI columns** (standard-scaled on train), not the
4/18-channel tensors:

- kNN (k=5), LogisticRegression (C=0.5, max_iter=2000),
  GaussianProcessClassifier (RBF(1.0)), CatBoost (300 iterations).
- **ROCKET**: 512 random convolutional kernels over the 4-channel tensors →
  LR (C=1.0).
- **catch22**: 22 canonical time-series features over the 4-channel tensors →
  LR (C=1.0).
- TabPFN (optional): prior-fitting transformer over raw columns.

---

## 6. Feature channels

**Plain (4)** — deterministic transforms of the single RSSI signal:
`raw`, `Δ` (velocity), `Δ²` (acceleration), window deviation `r_t − μ_w`.

**Rich (18)** — adds, per timestep: local spectral content (2: rolling-FFT band
DC/AC energies over 3-sample windows), rolling statistics (5: mean, range, std,
skew, kurtosis over 3/5-sample windows), shape (3: peak location, slope sign,
sign-change count), and full-sequence STFT (4 log-energy frequency bands
broadcast to every timestep). See `_expand_features_rich` in `dl/data_loader.py`.
JEPA models that use rich features consume 18 channels; all others consume 4.

## 7. Head design (shared)

All fine-tuned JEPA heads: learned-attention pooling over encoder latents →
Linear(d_model→128) → GELU → Dropout(0.3) → Linear(128→4). SIGReg and the
classical DL models use their native heads.
