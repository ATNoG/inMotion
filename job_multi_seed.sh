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

SEEDS=(42 3 5)

# ── Mapping: variant label → dataset CSV + extra flags ──────────────────────
declare -A DATASETS
DATASETS["normal"]="dataset.csv"
DATASETS["augmented"]="dataset_augmented.csv"
DATASETS["noise"]="dataset_only_noise.csv"
DATASETS["pure"]="dataset_only_pure.csv"

for variant in "${!DATASETS[@]}"; do
    # if dataset is normal and seed is 42 skip
    for seed in "${SEEDS[@]}"; do

        read -r data_file extra_flags <<< "${DATASETS[$variant]}"

        echo "============================================================"
        echo "  Seed=${seed}  Variant=${variant}  Data=${data_file}"
        echo "============================================================"

        model_dir="models/dl/${variant}"
        results_dir="results/dl/${variant}"
        plots_dir="plots/dl/${variant}"

        echo uv run python run_dl.py \
            --seed "${seed}" \
            --trials 50 \
            --data "${data_file}" \
            --wandb-project "inMotion-dl-${variant}" \
            --models-dir "${model_dir}" \
            --results-dir "${results_dir}" \
            --plots-dir "${plots_dir}" \
            ${extra_flags:-} \
            2>&1

        echo "  Done: Seed=${seed} Variant=${variant}"
        echo ""
    done
done

echo "All 4 variants × 3 seeds = 12 runs complete."

# ── Interference analysis (once per dataset, seed-independent) ──────────────
echo ""
echo "============================================================"
echo "  Interference Analysis — cross-route noise impact"
echo "============================================================"
echo ""

for variant in "${!DATASETS[@]}"; do
    read -r data_file extra_flags <<< "${DATASETS[$variant]}"

    IFS_DIR="plots/dl/${variant}/interference"
    echo "  Variant=${variant}  Data=${data_file} → ${IFS_DIR}"

    echo uv run python analyze_interference.py \
        --data "${data_file}" \
        --output-dir "${IFS_DIR}" \
        2>&1

    echo "  Done: interference plots → ${IFS_DIR}"
    echo ""
done

echo "All jobs + interference analysis complete."
