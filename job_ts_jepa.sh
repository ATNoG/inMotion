#!/usr/bin/env bash
# TS-JEPA on augmented dataset — overnight run.
#   nohup bash job_ts_jepa.sh & disown
#   tail -f logs/ts_jepa_aug_*.log
set -euo pipefail

cd ~/inMotion
export CUDA_VISIBLE_DEVICES=1

LOG="logs/ts_jepa_aug_$(date +%Y%m%d_%H%M).log"
mkdir -p logs models/exotic results/exotic

echo "=== TS-JEPA Augmented Run ===" | tee "$LOG"
echo "Start: $(date)" | tee -a "$LOG"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader -i 1 2>/dev/null || echo 'N/A')" | tee -a "$LOG"

uv run python run_exotic.py \
  --model ts_jepa \
  --data dataset_augmented.csv \
  --pretrain-epochs 300 \
  --finetune-epochs 80 \
  --batch-size 128 \
  --pretrain-lr 3e-4 \
  --finetune-lr 1e-3 \
  --unfreeze-encoder \
  --d-model 128 \
  --num-layers 2 \
  --pred-dim 64 \
  --pred-layers 2 \
  --pooling mean \
  --seed 42 \
  --no-wandb \
  2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Done: $(date)" | tee -a "$LOG"

# Also run on normal (non-augmented) dataset for comparison
echo "" | tee -a "$LOG"
echo "=== TS-JEPA Normal Dataset ===" | tee -a "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

uv run python run_exotic.py \
  --model ts_jepa \
  --data dataset.csv \
  --pretrain-epochs 300 \
  --finetune-epochs 80 \
  --batch-size 128 \
  --pretrain-lr 3e-4 \
  --finetune-lr 1e-3 \
  --unfreeze-encoder \
  --d-model 128 \
  --num-layers 2 \
  --pred-dim 64 \
  --pred-layers 2 \
  --pooling mean \
  --seed 42 \
  --no-wandb \
  2>&1 | tee -a "$LOG"

echo "All done: $(date)" | tee -a "$LOG"
