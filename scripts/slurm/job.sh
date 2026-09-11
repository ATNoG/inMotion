#!/bin/bash
#SBATCH --partition=gpuPartition
#SBATCH --nodelist=atari
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=2
#SBATCH --job-name=inMotion_dl
#SBATCH --output=logs/inMotion_dl_%j.out
#SBATCH --gres=gpu:1




# Credentials come from the environment; see env.example. Never commit a key.
: "${WANDB_API_KEY:?WANDB_API_KEY is not set - see env.example}"
export WANDB_ENTITY="${WANDB_ENTITY:-andreribeiro87-universidade-de-aveiro}"


PYTHONUNBUFFERED=1

# uv run inmotion run pipeline dl -- --seed 42 --trials 50 --wandb-project inMotion-dl-simple-2 --models-dir models/dl/simple-2 2>&1

# SIGReg (fastest — ~30s/trial)
# uv run inmotion run pipeline exotic -- --model sigreg --data data/processed/dataset_augmented3.csv \
#     --hpo --hpo-trials 50 --batch-size 256

# # TS-JEPA (~3min/trial)
# uv run inmotion run pipeline exotic -- --model ts_jepa --data data/processed/dataset_augmented3.csv \
#     --hpo --hpo-trials 30 --batch-size 256

# # T-JEPA (~3min/trial)
uv run inmotion run pipeline exotic -- --model t_jepa --data data/processed/dataset_augmented3.csv \
    --hpo --hpo-trials 30 --batch-size 256
