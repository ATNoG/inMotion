#!/usr/bin/env bash
# Multi-seed finetune-only runs for ensemble diversity (option 2).
# Uses existing pretrained encoders; finetunes each seed on dataset.icaisf.csv.
set -euo pipefail

DATA="dataset.icaisf.csv"
MODELS_DIR="20-aug/models/exotic/normal-new-ds/full-after-hpo"
RESULTS_DIR="20-aug/results/exotic/normal-new-ds/full-after-hpo"
LOGS_DIR="20-aug/logs/exotic/normal-new-ds/full-after-hpo"
mkdir -p "$LOGS_DIR"

export MAMBA_SSM_AVAILABLE=0

run_lejepa() {
  local seed=$1
  uv run python run_exotic.py --model lejepa --data "$DATA" \
    --seed "$seed" --finetune-only \
    --checkpoint "$MODELS_DIR/lejepa_pretrain_best.pt" \
    --finetune-epochs 200 --batch-size 128 \
    --d-model 128 --nhead 4 --num-layers 4 --dim-ff 256 \
    --pred-dim 64 --pred-layers 3 \
    --pretrain-lr 0.0007843269804336331 --finetune-lr 0.00937566600133442 \
    --ema-start 0.9967804036673305 --jepa-sigreg 0.05 \
    --models-dir "$MODELS_DIR" --results-dir "$RESULTS_DIR" --no-wandb \
    2>&1 | tee "$LOGS_DIR/lejepa_ff_seed${seed}.log"
}

run_ts_jepa() {
  local seed=$1
  uv run python run_exotic.py --model ts_jepa --data "$DATA" \
    --seed "$seed" --finetune-only --rich-features \
    --checkpoint "$MODELS_DIR/ts_jepa_pretrain_best.pt" \
    --finetune-epochs 200 --batch-size 128 \
    --d-model 128 --nhead 8 --num-layers 3 --dim-ff 768 \
    --pred-dim 96 --pred-layers 1 \
    --pretrain-lr 0.00024744252044213834 --finetune-lr 0.0018108634100884495 \
    --ema-start 0.9978972481217435 --jepa-sigreg 0.05 \
    --models-dir "$MODELS_DIR" --results-dir "$RESULTS_DIR" --no-wandb \
    2>&1 | tee "$LOGS_DIR/ts_jepa_ff_seed${seed}.log"
}

run_sigreg() {
  local seed=$1
  uv run python run_exotic.py --model sigreg --data "$DATA" \
    --seed "$seed" --epochs 300 --batch-size 512 \
    --d-model 192 --num-layers 2 --latent-dim 128 \
    --sigreg-lambda 0.00614454378558747 --lr 0.0005404103854647331 \
    --models-dir "$MODELS_DIR" --results-dir "$RESULTS_DIR" --no-wandb \
    2>&1 | tee "$LOGS_DIR/sigreg_ff_seed${seed}.log"
}

for seed in 42 7 123 2024; do
  run_lejepa "$seed" &
  run_ts_jepa "$seed" &
  run_sigreg "$seed" &
done

wait
echo "=== all multi-seed runs done ==="
