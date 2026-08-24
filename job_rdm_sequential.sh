#!/usr/bin/env bash
# Sequential RDMReg training: LeJEPA+RDMReg first, then CF-JEPA+RDMReg.
#   nohup bash job_rdm_sequential.sh > logs/rdm_sequential.log 2>&1 &

cd ~/inMotion
export CUDA_VISIBLE_DEVICES=0
export MAMBA_SSM_AVAILABLE=0

echo "=== Sequential RDMReg Training ===" | tee -a logs/rdm_sequential.log
echo "Start: $(date)" | tee -a logs/rdm_sequential.log

echo "--- [1/2] LeJEPA + RDMReg ---" | tee -a logs/rdm_sequential.log
bash job_lejepa_rdm.sh >> logs/rdm_sequential.log 2>&1

echo "--- [2/2] CF-JEPA + RDMReg ---" | tee -a logs/rdm_sequential.log
bash job_cf_jepa.sh >> logs/rdm_sequential.log 2>&1

echo "All done: $(date)" | tee -a logs/rdm_sequential.log
