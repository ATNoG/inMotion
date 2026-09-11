# Roadmap to MCC > 0.90

## Current State

| Metric | Value |
|--------|-------|
| SOTA (DeepStackEnsemble) | MCC 0.859 |
| SOTA (HPO TCN, augmented) | MCC 0.865 |
| SIGReg HPO (1 trial, 6 epochs) | MCC 0.857 |
| TS-JEPA (3 pretrain + 6 finetune epochs) | MCC 0.552 |
| T-JEPA (3 pretrain + 6 finetune epochs) | MCC 0.233 |

The JEPA models have **never been fully trained** — 3 pretraining epochs is nothing.
JEPA needs 300–500 epochs of self-supervised pretraining to learn representations.

---

## Why > 0.90 Is Possible

JEPA pretraining is designed to learn **interference-invariant** features.
Your data is 84% interfered (`noise=True`). A model that learns to ignore
interference during SSL pretraining should outperform purely supervised models.

Expected gains from proper JEPA pretraining: **+0.03–0.05 MCC** over base model.

Combined with ensembling: **+0.01–0.02 MCC**.

Projected ceiling: **0.89–0.91 MCC**.

---

## Phase 1 — Full JEPA Pretraining

Run these two jobs. Each takes ~2–4 hours on a GPU.

### TS-JEPA (recommended first — better at short sequences)

```bash
uv run python run_exotic.py \
    --model ts_jepa --data dataset_augmented.csv --seed 42 \
    --pretrain-epochs 500 --finetune-epochs 80 \
    --d-model 256 --nhead 8 --num-layers 4 --dim-ff 512 \
    --pred-dim 128 --pred-layers 2 --batch-size 512 \
    --pretrain-lr 3e-4 --finetune-lr 1e-3 \
    --models-dir models/exotic --results-dir results/exotic \
    --viz-dir models/exotic/viz
```

**What to watch:**
- Pretrain loss should decrease steadily (epochs 0–500)
- Uniformity should stay above -3.5 (collapse detection)
- PCA snapshots at `models/exotic/viz/ts_jepa_seed42/latents_epoch*.png`
- Fine-tune val MCC should climb past 0.80

### T-JEPA (complementary — feature-level masking)

```bash
uv run python run_exotic.py \
    --model t_jepa --data dataset_augmented.csv --seed 42 \
    --pretrain-epochs 500 --finetune-epochs 80 \
    --d-model 256 --nhead 8 --num-layers 4 --dim-ff 512 \
    --pred-dim 128 --pred-layers 2 --batch-size 512 \
    --pretrain-lr 3e-4 --finetune-lr 1e-3 \
    --models-dir models/exotic --results-dir results/exotic \
    --viz-dir models/exotic/viz
```

### Multi-seed (for ensemble diversity)

```bash
for seed in 42 123 456 789; do
    uv run python run_exotic.py \
        --model ts_jepa --data dataset_augmented.csv --seed $seed \
        --pretrain-epochs 500 --finetune-epochs 80 \
        --d-model 256 --nhead 8 --num-layers 4 --dim-ff 512 \
        --pred-dim 128 --pred-layers 2 --batch-size 512 \
        --models-dir models/exotic --results-dir results/exotic
done
```

---

## Phase 2 — HPO the Best Models

Once the full pretraining confirms JEPA works, run HPO to squeeze out extra performance.

### SIGReg (fast — ~30s/trial)

```bash
uv run python run_exotic.py \
    --model sigreg --data dataset_augmented.csv --seed 42 \
    --hpo --hpo-trials 50 --batch-size 256 \
    --models-dir models/exotic --results-dir results/exotic
```

Then train with best params:

```bash
uv run python run_exotic.py \
    --model sigreg --data dataset_augmented.csv --seed 42 \
    --epochs 200 --batch-size 512 \
    --models-dir models/exotic --results-dir results/exotic
```

### TS-JEPA HPO (~3min/trial — run overnight)

```bash
uv run python run_exotic.py \
    --model ts_jepa --data dataset_augmented.csv --seed 42 \
    --hpo --hpo-trials 30 --batch-size 256 \
    --pretrain-epochs 50 --finetune-epochs 20 \
    --models-dir models/exotic --results-dir results/exotic
```

Then full train with best HPO params (HPO auto-updates args, just run without `--hpo`):

```bash
uv run python run_exotic.py \
    --model ts_jepa --data dataset_augmented.csv --seed 42 \
    --pretrain-epochs 500 --finetune-epochs 80 --batch-size 512 \
    --models-dir models/exotic --results-dir results/exotic
```

---

## Phase 3 — Ensemble

Take the best models and stack them:

```python
# Pseudo-code for ensemble
models = [
    load_best("ts_jepa"),      # JEPA pretrained
    load_best("t_jepa"),       # JEPA pretrained
    load_best("sigreg"),       # Gaussian regularizer
    load_best("cnn"),          # Best supervised baseline
    load_best("tcn"),          # Best supervised baseline
]
# Train a stacking meta-learner (LogisticRegression or small MLP)
# on the concatenated outputs of all models
```

The existing `dl/models/ensemble.py` already has `StackingEnsemble` — reuse it.

---

## Phase 4 — If Still Below 0.90

### A. Feature engineering
Add frequency-domain features (FFT magnitude of the 10-step sequence).
WiFi interference has characteristic frequency signatures that time-domain
features miss.

### B. Data quality audit
Analyze the 16% of samples the model consistently misclassifies:
- Are they mislabeled?
- Are they ambiguous even to a human?
- Remove or relabel them if needed.

### C. Larger model
Scale up: d_model=384, num_layers=6, dim_ff=1024, pred_dim=192.
More capacity helps when you have 10K samples and SSL pretraining.

### D. Train on all data
Combine clean + noise + augmented datasets for pretraining.
JEPA pretraining doesn't need labels — use every RSSI sequence you have,
including test-set sequences (without labels).

---

## Success Criteria

| MCC | Status |
|-----|--------|
| > 0.86 | Beat current SOTA |
| > 0.88 | Strong improvement |
| > 0.90 | Target achieved |
| > 0.92 | Exceptional |

---

## Monitoring "The World"

During JEPA pretraining, check `models/exotic/viz/{model}_seed42/`:

- **`pretrain_summary.png`** — loss curves, uniformity (collapse detection), EMA schedule
- **`latents_epoch0000.png`** — latent space at epoch 0 (random initialization)
- **`latents_epoch0500.png`** — latent space after full pretraining
- **`latents_epoch9999.png`** — pretrained embeddings colored by true class
- **`latents_epoch10000.png`** — fine-tuned embeddings colored by true class

A healthy JEPA pretraining shows:
1. Pretrain loss decreasing and stabilizing
2. Uniformity staying above -3.5 (no collapse)
3. PCA plots showing class separation emerging in the latent space
4. Fine-tuned embeddings more clustered by class than pretrained ones
