#!/bin/bash
# ============================================================================
# Master training v2 — fixes from v1 analysis:
#   1. Probe-based early stopping was killing JEPA at epoch 60 (probe stuck ~0.26
#      is NORMAL for this 10-step task — the MCC comes from FINE-TUNING).
#      → --probe-patience 40 so pretraining runs its full course.
#   2. More fine-tuning epochs (200 vs 80) — this is where MCC actually comes from.
#   3. LeJEPA recovery with the ORIGINAL working config (batch 512, lr 5e-4).
#      → saves to models/exotic (the ensemble path), NOT 20-aug.
#
# Submit:  sbatch job_master_train_v2.sh
# ============================================================================
#SBATCH --partition=gpuPartition
#SBATCH --nodelist=atari
#SBATCH --job-name=inMotion_v2
#SBATCH --output=logs/master_v2_%j.out
#SBATCH --error=logs/master_v2_%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00

set -euo pipefail

cd ~/inMotion
export MAMBA_SSM_AVAILABLE=0
export PYTHONUNBUFFERED=1
mkdir -p logs models/exotic results/exotic

echo "=== Master v2 start: $(date) ==="

# ── Phase 1: LeJEPA recovery → models/exotic (the ensemble path) ───────────
echo ""
echo "=== [1/4] LeJEPA recovery (original config, full pretrain) ==="
CUDA_VISIBLE_DEVICES=0 uv run python -u run_exotic.py --model lejepa --data dataset_augmented.csv \
    --gpu 0 --pretrain-epochs 400 --finetune-epochs 200 \
    --d-model 256 --num-layers 4 --pred-dim 128 \
    --pretrain-lr 5e-4 --finetune-lr 1e-3 --batch-size 512 \
    --probe-cadence 10 --probe-patience 40 \
    --seed 42 --no-wandb 2>&1 | tee logs/master_v2_lejepa.log

# ── Phase 2: TS-JEPA full retrain (finetune-heavy, rich features) ──────────
echo ""
echo "=== [2/4] TS-JEPA seed42 retrain (finetune 200, high probe patience) ==="
CUDA_VISIBLE_DEVICES=1 uv run python -u run_exotic.py --model ts_jepa --data dataset_augmented3.csv \
    --seed 42 --pretrain-epochs 500 --finetune-epochs 200 \
    --d-model 192 --nhead 4 --num-layers 4 --dim-ff 256 \
    --pred-dim 64 --pred-layers 3 \
    --pretrain-lr 0.00062 --finetune-lr 0.0035 --ema-start 0.9964 \
    --probe-cadence 10 --probe-patience 40 --batch-size 512 \
    --no-wandb 2>&1 | tee logs/master_v2_ts_jepa.log

# ── Phase 3: T-JEPA seed42 retrain (rich features — matches harness!) ──────
echo ""
echo "=== [3/4] T-JEPA seed42 retrain (RICH features, finetune 200) ==="
CUDA_VISIBLE_DEVICES=1 uv run python -u run_exotic.py --model t_jepa --data dataset_augmented3.csv \
    --seed 42 --pretrain-epochs 500 --finetune-epochs 200 \
    --d-model 256 --nhead 8 --num-layers 4 --dim-ff 768 \
    --pred-dim 96 --pred-layers 3 \
    --pretrain-lr 0.00060 --finetune-lr 0.0031 --ema-start 0.9982 \
    --jepa-sigreg 0.1 --probe-cadence 10 --probe-patience 40 --batch-size 512 \
    --rich-features \
    --no-wandb 2>&1 | tee logs/master_v2_t_jepa.log

# ── Phase 4: T-JEPA multi-seed (rich features, seeds 3/5) ──────────────────
echo ""
echo "=== [4/4] T-JEPA seeds 3/5 (rich features, for diversity) ==="
for seed in 3 5; do
    echo "--- T-JEPA seed=$seed ---"
    CUDA_VISIBLE_DEVICES=0 uv run python -u run_exotic.py --model t_jepa --data dataset_augmented3.csv \
        --seed $seed --pretrain-epochs 500 --finetune-epochs 200 \
        --d-model 256 --nhead 8 --num-layers 4 --dim-ff 768 \
        --pred-dim 96 --pred-layers 3 \
        --pretrain-lr 0.00060 --finetune-lr 0.0031 --ema-start 0.9982 \
        --jepa-sigreg 0.1 --probe-cadence 10 --probe-patience 40 --batch-size 512 \
        --rich-features \
        --no-wandb 2>&1 | tee logs/master_v2_t_jepa_${seed}.log
done

echo ""
echo "=== Master v2 done: $(date) ==="
echo "Checkpoints:"
ls -la models/exotic/lejepa_ft_seed42.pt models/exotic/ts_jepa_ft_seed42.pt \
      models/exotic/t_jepa_ft_seed42.pt models/exotic/t_jepa_ft_seed3.pt \
      models/exotic/t_jepa_ft_seed5.pt 2>/dev/null | awk '{print $6,$7,$8,$9}'
