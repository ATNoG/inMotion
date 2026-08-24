#!/usr/bin/env bash
# T-JEPA on augmented dataset — overnight run.
#   nohup bash job_t_jepa.sh & disown

cd ~/inMotion
export CUDA_VISIBLE_DEVICES=1

LOG="logs/t_jepa_aug_$(date +%Y%m%d_%H%M).log"
mkdir -p logs models/exotic results/exotic

echo "=== T-JEPA Augmented Run ===" | tee "$LOG"
echo "Start: $(date)" | tee -a "$LOG"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader -i 1)" | tee -a "$LOG"

uv run python run_exotic.py \
  --model t_jepa \
  --data dataset_augmented.csv \
  --pretrain-epochs 300 \
  --finetune-epochs 80 \
  --batch-size 128 \
  --pretrain-lr 3e-4 \
  --finetune-lr 1e-3 \
  --unfreeze-encoder \
  --d-model 128 \
  --num-layers 2 \
  --n-reg-tokens 2 \
  --pooling mean \
  --seed 42 \
  --no-wandb \
  2>&1 | tee -a "$LOG"

echo "Done: $(date)" | tee -a "$LOG"
