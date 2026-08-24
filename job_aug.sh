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

export WANDB_ENTITY="andreribeiro87-universidade-de-aveiro"
export WANDB_API_KEY="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"

PYTHONUNBUFFERED=1


echo Seed=42  Variant=augmented  Data=dataset_augmented.csv
uv run python run_dl.py --seed 42 --trials 15 --data dataset_augmented.csv --wandb-project inMotion-dl-augmented --models-dir models/dl/augmented --results-dir results/dl/augmented --plots-dir plots/dl/augmented
echo  Done: Seed=42 Variant=augmented

echo Seed=3  Variant=augmented  Data=dataset_augmented.csv
uv run python run_dl.py --seed 3 --trials 15 --data dataset_augmented.csv --wandb-project inMotion-dl-augmented --models-dir models/dl/augmented --results-dir results/dl/augmented --plots-dir plots/dl/augmented
echo  Done: Seed=3 Variant=augmented

echo Seed=5  Variant=augmented  Data=dataset_augmented.csv
uv run python run_dl.py --seed 5 --trials 15 --data dataset_augmented.csv --wandb-project inMotion-dl-augmented --models-dir models/dl/augmented --results-dir results/dl/augmented --plots-dir plots/dl/augmented
echo  Done: Seed=5 Variant=augmented
