#!/usr/bin/env bash
# CF-JEPA (mask-free multi-horizon) + RDMReg — full pretraining run, v2.
# Harder task: shorter context (1-2 patches), longer horizon (up to 4).
#   nohup bash job_cf_jepa.sh > logs/cf_jepa_rdm.log 2>&1 &

cd ~/inMotion
export CUDA_VISIBLE_DEVICES=0
export MAMBA_SSM_AVAILABLE=0

echo "=== CF-JEPA + RDMReg Full Pretrain v2 ===" | tee -a logs/cf_jepa_rdm.log
echo "Start: $(date)" | tee -a logs/cf_jepa_rdm.log

uv run python run_exotic.py \
  --model cf_jepa \
  --data dataset.csv \
  --pretrain-epochs 400 \
  --finetune-epochs 60 \
  --probe-cadence 10 \
  --probe-patience 20 \
  --batch-size 128 \
  --d-model 256 \
  --nhead 8 \
  --num-layers 4 \
  --dim-ff 512 \
  --pred-dim 128 \
  --pred-layers 2 \
  --cf-horizon-start 2 \
  --cf-horizon-end 4 \
  --cf-ctx-min 1 \
  --cf-ctx-max 2 \
  --jepa-sigreg 0.1 \
  --sigreg-mode rdm \
  --rdm-p 1.5 \
  --pretrain-lr 3e-4 \
  --finetune-lr 1e-3 \
  --seed 42 \
  --no-wandb \
  2>&1 | tee -a logs/cf_jepa_rdm.log

echo "Done: $(date)" | tee -a logs/cf_jepa_rdm.log
