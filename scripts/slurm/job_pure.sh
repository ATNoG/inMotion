#!/bin/bash
#SBATCH --partition=gpuPartition
#SBATCH --nodelist=xbox
#SBATCH --cpus-per-task=2
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

echo Seed=42  Variant=pure  Data=data/processed/dataset_only_pure.csv
uv run inmotion run pipeline dl -- --seed 42 --trials 30 --data data/processed/dataset_only_pure.csv --wandb-project inMotion-dl-pure --models-dir models/dl/pure --results-dir results/dl/pure --plots-dir plots/dl/pure
echo  Done: Seed=42 Variant=pure

echo Seed=3  Variant=pure  Data=data/processed/dataset_only_pure.csv
uv run inmotion run pipeline dl -- --seed 3 --trials 30 --data data/processed/dataset_only_pure.csv --wandb-project inMotion-dl-pure --models-dir models/dl/pure --results-dir results/dl/pure --plots-dir plots/dl/pure
echo  Done: Seed=3 Variant=pure

echo Seed=5  Variant=pure  Data=data/processed/dataset_only_pure.csv
uv run inmotion run pipeline dl -- --seed 5 --trials 30 --data data/processed/dataset_only_pure.csv --wandb-project inMotion-dl-pure --models-dir models/dl/pure --results-dir results/dl/pure --plots-dir plots/dl/pure
echo  Done: Seed=5 Variant=pure