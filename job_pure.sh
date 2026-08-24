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

export WANDB_ENTITY="andreribeiro87-universidade-de-aveiro"
export WANDB_API_KEY="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"

PYTHONUNBUFFERED=1

echo Seed=42  Variant=pure  Data=dataset_only_pure.csv
uv run python run_dl.py --seed 42 --trials 30 --data dataset_only_pure.csv --wandb-project inMotion-dl-pure --models-dir models/dl/pure --results-dir results/dl/pure --plots-dir plots/dl/pure
echo  Done: Seed=42 Variant=pure

echo Seed=3  Variant=pure  Data=dataset_only_pure.csv
uv run python run_dl.py --seed 3 --trials 30 --data dataset_only_pure.csv --wandb-project inMotion-dl-pure --models-dir models/dl/pure --results-dir results/dl/pure --plots-dir plots/dl/pure
echo  Done: Seed=3 Variant=pure

echo Seed=5  Variant=pure  Data=dataset_only_pure.csv
uv run python run_dl.py --seed 5 --trials 30 --data dataset_only_pure.csv --wandb-project inMotion-dl-pure --models-dir models/dl/pure --results-dir results/dl/pure --plots-dir plots/dl/pure
echo  Done: Seed=5 Variant=pure