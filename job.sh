#!/bin/bash
#SBATCH --partition=gpuPartition
#SBATCH --nodelist=atari
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=2
#SBATCH --job-name=inMotion_dl
#SBATCH --output=logs/inMotion_dl_%j.out
#SBATCH --gres=gpu:1




export WANDB_ENTITY="andreribeiro87-universidade-de-aveiro"
export WANDB_API_KEY="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"


PYTHONUNBUFFERED=1

# uv run python run_dl.py --seed 42 --trials 50 --wandb-project inMotion-dl-simple-2 --models-dir models/dl/simple-2 2>&1

# SIGReg (fastest — ~30s/trial)
# uv run python run_exotic.py --model sigreg --data dataset_augmented3.csv \
#     --hpo --hpo-trials 50 --batch-size 256

# # TS-JEPA (~3min/trial)
# uv run python run_exotic.py --model ts_jepa --data dataset_augmented3.csv \
#     --hpo --hpo-trials 30 --batch-size 256

# # T-JEPA (~3min/trial)
uv run python run_exotic.py --model t_jepa --data dataset_augmented3.csv \
    --hpo --hpo-trials 30 --batch-size 256
