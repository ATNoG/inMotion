#!/bin/bash
# ============================================================================
# Master long-run training script — rebuild ALL ensemble members.
# Submit with:  sbatch job_master_train.sh
# Runs on atari (2× RTX A4000), sequential phases so checkpoints land cleanly.
#
# Phase 1 — LeJEPA recovery (currently BROKEN by RDMReg experiment, 0.48)
# Phase 2 — SIGReg multi-seed (seeds 3,5 — seed42 already 0.886)
# Phase 3 — Mamba-3 variants (tcn/transformer/multiview — only cnn exists)
# Phase 4 — TS-JEPA + T-JEPA multi-seed (seeds 3,5 for ensemble diversity)
# Phase 5 — HPO paper models on dataset.icaisf.csv (for paper-data ensemble)
# ============================================================================
#SBATCH --partition=gpuPartition
#SBATCH --nodelist=sega
#SBATCH --job-name=inMotion_master
#SBATCH --output=logs/master_train_%j.out
#SBATCH --error=logs/master_train_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G


cd ~/inMotion
export MAMBA_SSM_AVAILABLE=0   # avoid mamba_ssm CUDA-extension hang
export PYTHONUNBUFFERED=1

# mkdir -p 20-aug/logs 20-aug/models/exotic 20-aug/results/exotic

# echo "=== Master training start: $(date) ==="
# nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

# # ── Phase 1: LeJEPA recovery (the critical missing member) ─────────────────
# # Config from job_exotic.sh that produced the original 0.8495 checkpoint.
# echo ""
# echo "=== [1/5] LeJEPA recovery (dataset_augmented.csv) ==="
# CUDA_VISIBLE_DEVICES=0 uv run python -u run_exotic.py --model lejepa --data dataset_augmented.csv \
#     --gpu 0 --pretrain-epochs 400 --finetune-epochs 80 \
#     --d-model 256 --num-layers 4 --pred-dim 128 \
#     --pretrain-lr 5e-4 --finetune-lr 1e-3 --batch-size 512 \
#     --seed 42 --no-wandb --results-dir 20-aug/results/exotic --models-dir 20-aug/models/exotic 2>&1 | tee 20-aug/logs/master_lejepa.log

# # ── Phase 2: SIGReg multi-seed (diversity) ─────────────────────────────────
# # Seed 42 already scores 0.886 on dataset.icaisf.csv; add 3 and 5.
# echo ""
# echo "=== [2/5] SIGReg multi-seed (dataset_augmented3.csv) ==="
# for seed in 3 5; do
#     echo "--- SIGReg seed=$seed ---"
#     CUDA_VISIBLE_DEVICES=0 uv run python -u run_exotic.py --model sigreg --data dataset_augmented3.csv \
#         --seed $seed --epochs 300 --patience 150 --batch-size 512 \
#         --d-model 192 --num-layers 3 --latent-dim 128 \
#         --sigreg-lambda 0.0086 --lr 0.000147 \
#         --no-wandb --results-dir 20-aug/results/exotic --models-dir 20-aug/models/exotic 2>&1 | tee 20-aug/logs/master_sigreg_${seed}.log
# done

# # ── Phase 3: Mamba-3 variants (diversity) ──────────────────────────────────
# echo ""
# echo "=== [3/5] Mamba-3 variants (dataset_augmented3.csv) ==="
# for m in mamba3_tcn mamba3_transformer mamba3_multiview; do
#     echo "--- $m ---"
#     CUDA_VISIBLE_DEVICES=0 uv run python -u run_exotic.py --model $m --data dataset_augmented3.csv \
#         --seed 42 --epochs 150 --batch-size 512 --d-model 192 --num-layers 3 \
#         --no-wandb --results-dir 20-aug/results/exotic --models-dir 20-aug/models/exotic 2>&1 | tee 20-aug/logs/master_${m}.log
# done

# ── Phase 4: TS-JEPA + T-JEPA multi-seed (diversity) ───────────────────────
echo ""
echo "=== [4/5] JEPA multi-seed (dataset_augmented3.csv) ==="
for seed in 3 5; do
    echo "--- TS-JEPA seed=$seed ---"
    CUDA_VISIBLE_DEVICES=0 uv run python -u run_exotic.py --model ts_jepa --data dataset_augmented3.csv \
        --seed $seed --pretrain-epochs 500 --finetune-epochs 80 \
        --d-model 192 --nhead 4 --num-layers 4 --dim-ff 256 \
        --pred-dim 64 --pred-layers 3 \
        --pretrain-lr 0.00062 --finetune-lr 0.0035 --ema-start 0.9964 \
        --probe-cadence 10 --probe-patience 5 --batch-size 512 \
        --no-wandb --results-dir 20-aug/results/exotic --models-dir 20-aug/models/exotic 2>&1 | tee 20-aug/logs/master_ts_jepa_${seed}.log
done
for seed in 3 5; do
    echo "--- T-JEPA seed=$seed ---"
    CUDA_VISIBLE_DEVICES=0 uv run python -u run_exotic.py --model t_jepa --data dataset_augmented3.csv \
        --seed $seed --pretrain-epochs 500 --finetune-epochs 80 \
        --d-model 256 --nhead 8 --num-layers 4 --dim-ff 768 \
        --pred-dim 96 --pred-layers 3 \
        --pretrain-lr 0.00060 --finetune-lr 0.0031 --ema-start 0.9982 \
        --jepa-sigreg 0.1 --probe-cadence 10 --probe-patience 5 --batch-size 512 \
        --no-wandb --results-dir 20-aug/results/exotic --models-dir 20-aug/models/exotic 2>&1 | tee 20-aug/logs/master_t_jepa_${seed}.log
done

# ── Phase 5: HPO paper models on the paper dataset ─────────────────────────
echo ""
echo "=== [5/5] HPO paper models (dataset.icaisf.csv) ==="
CUDA_VISIBLE_DEVICES=0 uv run python -u train_hpo_paper.py --data dataset.icaisf.csv \
    --seed 42 --epochs 150 --patience 20 \
    --models gru tcn cnn lstm bilstm --results-dir 20-aug/results/exotic --models-dir 20-aug/models/exotic \
    2>&1 | tee 20-aug/logs/master_hpo_paper.log

echo ""
echo "=== Master training done: $(date) ==="
echo "Checkpoints ready for mega_ensemble:"
ls -la 20-aug/models/exotic/*.pt 20-aug/models/dl/HPO_*.pt 2>/dev/null | grep -E "lejepa|sigreg|mamba3|ts_jepa|t_jepa|HPO" | head -30
