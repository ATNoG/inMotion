#!/bin/bash
#SBATCH --partition=gpuPartition
#SBATCH --nodelist=atari
#SBATCH --cpus-per-task=2
#SBATCH --threads-per-core=2
#SBATCH --job-name=inMotion_dl
#SBATCH --output=inMotion_dl_%j.out
#SBATCH --error=inMotion_dl_%j.err
#SBATCH --mem=64G

# Credentials come from the environment; see env.example. Never commit a key.
: "${WANDB_API_KEY:?WANDB_API_KEY is not set - see env.example}"
export WANDB_ENTITY="${WANDB_ENTITY:-andreribeiro87-universidade-de-aveiro}"

PYTHONUNBUFFERED=1

uv run inmotion run pipeline analyze-interference -- --data data/processed/dataset.csv --output-dir plots/dl/normal/interference

echo Variant=noise  Data=data/processed/dataset_only_noise.csv → plots/dl/noise/interference
uv run inmotion run pipeline analyze-interference -- --data data/processed/dataset_only_noise.csv --output-dir plots/dl/noise/interference
echo  Done: interference plots → plots/dl/noise/interference

echo Variant=pure  Data=data/processed/dataset_only_pure.csv → plots/dl/pure/interference
uv run inmotion run pipeline analyze-interference -- --data data/processed/dataset_only_pure.csv --output-dir plots/dl/pure/interference
echo  Done: interference plots → plots/dl/pure/interference

echo Variant=augmented  Data=data/processed/dataset_augmented.csv → plots/dl/augmented/interference
uv run inmotion run pipeline analyze-interference -- --data data/processed/dataset_augmented.csv --output-dir plots/dl/augmented/interference
echo  Done: interference plots → plots/dl/augmented/interference