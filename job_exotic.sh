#!/bin/bash
#SBATCH --partition=gpuPartition
#SBATCH --nodelist=xbox
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=2
#SBATCH --job-name=inMotion_exotic
#SBATCH --output=inMotion_exotic_%j.out
#SBATCH --error=inMotion_exotic_%j.err
#SBATCH --gres=gpu:1

export CUDA_VISIBLE_DEVICES=0
export WANDB_ENTITY="andreribeiro87-universidade-de-aveiro"
export WANDB_API_KEY="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"
PYTHONUNBUFFERED=1

# ═══════════════════════════════════════════════════════════════
# LeJEPA — SSL pretrain + fine-tune (paper implementation)
# ═══════════════════════════════════════════════════════════════

uv run python run_exotic.py --model lejepa --data dataset_augmented.csv \
    --gpu 0 --pretrain-epochs 400 --finetune-epochs 80 \
    --d-model 256 --num-layers 4 --pred-dim 128 \
    --pretrain-lr 5e-4 --finetune-lr 1e-3 --batch-size 512

# ═══════════════════════════════════════════════════════════════
# SIGReg — HPO + train (LeJEPA SIGReg + dynamic augmentation)
# ═══════════════════════════════════════════════════════════════

uv run python run_exotic.py --model sigreg --data dataset_augmented.csv \
    --gpu 0 --hpo --hpo-trials 50 --epochs 200 --batch-size 512
