#!/bin/bash
#SBATCH --partition=gpuPartition
#SBATCH --nodelist=atari
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=2
#SBATCH --job-name=inMotion_full_train
#SBATCH --output=logs/inMotion_full_train_%j.out
#SBATCH --gres=gpu:1

export CUDA_VISIBLE_DEVICES=0
export WANDB_ENTITY="andreribeiro87-universidade-de-aveiro"
export WANDB_API_KEY="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"
PYTHONUNBUFFERED=1

# ═══════════════════════════════════════════════════════════════════
# FULL TRAIN v2 — best HPO values + all pipeline fixes
#
# Fixes since v1:
#   1. Pretrain val loss now masked to target positions only
#      (TS-JEPA's was a broadcast bug: 485-687 instead of ~0.4)
#   2. Early stopping + checkpoint selection now driven by a LINEAR
#      PROBE MCC (representation quality), not the raw val loss
#   3. T-JEPA gets SIGReg anti-collapse (--jepa-sigreg 0.1):
#      latent space was collapsing (uniformity 2.4 -> 0.2), killing
#      pretraining. Verified: uniformity now stable ~2.5-2.8
#   4. SIGReg classifier gets higher patience (150) — it was stopping
#      at epoch 62 of 300 while still improving
#
# Expected: TS-JEPA probe MCC climbs past 0.7 -> stage-1 MCC > 0.7
# ═══════════════════════════════════════════════════════════════════

DATA=dataset_augmented3.csv

# ── SIGReg: direct supervised (HPO best: 0.8671) ──────────────────
uv run python run_exotic.py --model sigreg --data $DATA --seed 42 \
    --epochs 300 --patience 150 --batch-size 512 \
    --d-model 192 --num-layers 3 --latent-dim 128 \
    --sigreg-lambda 0.008625079907536571 --lr 0.00014719177455031863 \
    --models-dir models/exotic --results-dir results/exotic

# ── TS-JEPA: full pretrain + finetune (HPO best: 0.8312) ──────────
uv run python run_exotic.py --model ts_jepa --data $DATA --seed 42 \
    --pretrain-epochs 500 --finetune-epochs 80 \
    --d-model 192 --nhead 4 --num-layers 4 --dim-ff 256 \
    --pred-dim 64 --pred-layers 3 \
    --pretrain-lr 0.000619705216092559 --finetune-lr 0.003505849839898099 \
    --ema-start 0.996391789202448 \
    --probe-cadence 10 --probe-patience 5 \
    --batch-size 512 \
    --models-dir models/exotic --results-dir results/exotic \
    --viz-dir models/exotic/viz

# ── T-JEPA: full pretrain + finetune (HPO best: 0.7492) ────────────
uv run python run_exotic.py --model t_jepa --data $DATA --seed 42 \
    --pretrain-epochs 500 --finetune-epochs 80 \
    --d-model 256 --nhead 8 --num-layers 4 --dim-ff 768 \
    --pred-dim 96 --pred-layers 3 \
    --pretrain-lr 0.0005977512671097678 --finetune-lr 0.003140525716793761 \
    --ema-start 0.9981856924030291 \
    --jepa-sigreg 0.1 \
    --probe-cadence 10 --probe-patience 5 \
    --batch-size 512 \
    --models-dir models/exotic --results-dir results/exotic \
    --viz-dir models/exotic/viz

# ═══════════════════════════════════════════════════════════════════
# MULTI-SEED (for ensemble diversity) — uncomment after first run
# ═══════════════════════════════════════════════════════════════════
# for seed in 123 456 789; do
#     uv run python run_exotic.py --model ts_jepa --data $DATA --seed $seed \
#         --pretrain-epochs 500 --finetune-epochs 80 \
#         --d-model 192 --nhead 4 --num-layers 4 --dim-ff 256 \
#         --pred-dim 64 --pred-layers 3 \
#         --pretrain-lr 0.000619705216092559 --finetune-lr 0.003505849839898099 \
#         --ema-start 0.996391789202448 \
#         --probe-cadence 10 --probe-patience 5 \
#         --batch-size 512 \
#         --models-dir models/exotic --results-dir results/exotic
# done
