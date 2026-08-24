#!/bin/bash
#SBATCH --partition=gpuPartition
#SBATCH --nodelist=atari
#SBATCH --cpus-per-task=2
#SBATCH --threads-per-core=2
#SBATCH --job-name=inMotion_dl
#SBATCH --output=inMotion_dl_%j.out
#SBATCH --error=inMotion_dl_%j.err
#SBATCH --mem=64G

export WANDB_ENTITY="andreribeiro87-universidade-de-aveiro"
export WANDB_API_KEY="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"

PYTHONUNBUFFERED=1

uv run python analyze_interference.py --data dataset.csv --output-dir plots/dl/normal/interference

echo Variant=noise  Data=dataset_only_noise.csv → plots/dl/noise/interference
uv run python analyze_interference.py --data dataset_only_noise.csv --output-dir plots/dl/noise/interference
echo  Done: interference plots → plots/dl/noise/interference

echo Variant=pure  Data=dataset_only_pure.csv → plots/dl/pure/interference
uv run python analyze_interference.py --data dataset_only_pure.csv --output-dir plots/dl/pure/interference
echo  Done: interference plots → plots/dl/pure/interference

echo Variant=augmented  Data=dataset_augmented.csv → plots/dl/augmented/interference
uv run python analyze_interference.py --data dataset_augmented.csv --output-dir plots/dl/augmented/interference
echo  Done: interference plots → plots/dl/augmented/interference