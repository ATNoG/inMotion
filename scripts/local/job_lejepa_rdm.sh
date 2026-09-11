#!/usr/bin/env bash
# LeJEPA + RDMReg — full pretraining run (RDMReg replaces SigReg).
#   nohup bash job_lejepa_rdm.sh > logs/lejepa_rdm.log 2>&1 &

cd ~/inMotion
export CUDA_VISIBLE_DEVICES=0
export MAMBA_SSM_AVAILABLE=0

echo "=== LeJEPA + RDMReg Full Pretrain ===" | tee -a logs/lejepa_rdm.log
echo "Start: $(date)" | tee -a logs/lejepa_rdm.log

uv run python run_exotic.py \
  --model lejepa \
  --data dataset.csv \
  --pretrain-epochs 300 \
  --finetune-epochs 60 \
  --probe-cadence 10 \
  --batch-size 128 \
  --d-model 256 \
  --nhead 8 \
  --num-layers 4 \
  --dim-ff 512 \
  --pred-dim 128 \
  --pred-layers 2 \
  --jepa-sigreg 0.05 \
  --sigreg-mode rdm \
  --rdm-p 1.5 \
  --pretrain-lr 3e-4 \
  --finetune-lr 1e-3 \
  --seed 42 \
  --no-wandb \
  2>&1 | tee -a logs/lejepa_rdm.log

echo "Done: $(date)" | tee -a logs/lejepa_rdm.log
