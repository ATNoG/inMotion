#!/bin/bash
#SBATCH --partition=gpuPartition
#SBATCH --nodelist=atari
#SBATCH --cpus-per-task=2
#SBATCH --threads-per-core=2
#SBATCH --job-name=inMotion_dl
#SBATCH --output=inMotion_dl_%j.out
#SBATCH --error=inMotion_dl_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

export WANDB_ENTITY="andreribeiro87-universidade-de-aveiro"
export WANDB_API_KEY="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"

PYTHONUNBUFFERED=1

echo Seed=42  Variant=normal  Data=dataset.csv
uv run python run_dl.py --seed 42 --trials 30 --data dataset.csv --wandb-project inMotion-dl-normal --models-dir models/dl/normal --results-dir results/dl/normal --plots-dir plots/dl/normal
echo  Done: Seed=42 Variant=normal

echo Seed=3  Variant=normal  Data=dataset.csv
uv run python run_dl.py --seed 3 --trials 30 --data dataset.csv --wandb-project inMotion-dl-normal --models-dir models/dl/normal --results-dir results/dl/normal --plots-dir plots/dl/normal
echo  Done: Seed=3 Variant=normal

echo Seed=5  Variant=normal  Data=dataset.csv
uv run python run_dl.py --seed 5 --trials 30 --data dataset.csv --wandb-project inMotion-dl-normal --models-dir models/dl/normal --results-dir results/dl/normal --plots-dir plots/dl/normal
echo  Done: Seed=5 Variant=normal