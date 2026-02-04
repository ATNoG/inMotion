#!/bin/bash
#SBATCH --partition=gpuPartition
#SBATCH --cpus-per-task=12
#SBATCH --job-name=inMotion
#SBATCH --output=inMotion_%j.out
#SBATCH --error=inMotion_%j.err
#SBATCH --gres=gpu:1


# Optional but extremely wise:
set -euo pipefail

PYTHONUNBUFFERED=1

uv run python run_classification.py --optimize --seed 5 --suffix-after-seed pure --data dataset_only_pure.csv
uv run python run_classification.py --optimize --seed 42 --suffix-after-seed pure --data dataset_only_pure.csv

uv run python run_classification.py --optimize --seed 3 --suffix-after-seed noise --data dataset_only_noise.csv
uv run python run_classification.py --optimize --seed 5 --suffix-after-seed noise --data dataset_only_noise.csv
uv run python run_classification.py --optimize --seed 42 --suffix-after-seed noise --data dataset_only_noise.csv
