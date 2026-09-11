#!/bin/bash
#SBATCH --partition=gpuPartition
#SBATCH --nodelist=atari
#SBATCH --cpus-per-task=20
#SBATCH --threads-per-core=2
#SBATCH --job-name=inMotion_dl
#SBATCH --output=inMotion_dl_%j.out
#SBATCH --error=inMotion_dl_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# Credentials come from the environment; see env.example. Never commit a key.
: "${WANDB_API_KEY:?WANDB_API_KEY is not set - see env.example}"
export WANDB_ENTITY="${WANDB_ENTITY:-andreribeiro87-universidade-de-aveiro}"

PYTHONUNBUFFERED=1


echo Seed=5  Variant=augmented  Data=dataset_augmented.csv
uv run python run_dl.py --seed 5 --trials 15 --data dataset_augmented.csv --wandb-project inMotion-dl-augmented --models-dir models/dl/augmented --results-dir results/dl/augmented --plots-dir plots/dl/augmented
echo  Done: Seed=5 Variant=augmented
