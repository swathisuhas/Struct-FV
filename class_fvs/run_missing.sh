#!/bin/bash
#SBATCH -p gpu17,gpu22,gpu24
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --array=0-133
#SBATCH --output=/BS/feature_viz/work/logs/convnext_base_graphMatching_rem_%A_%a.out
#SBATCH --error=/BS/feature_viz/work/logs/convnext_base_graphMatching_rem_%A_%a.err

setup_folder="/BS/feature_viz/work/results/convnext_reverse/"
missing_file="/BS/feature_viz/work/code/GraphMatchingFV/class_fvs/missing_runs_by_class.txt"

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$missing_file")
cls=$(echo "$line" | awk '{print $1}')
runs=$(echo "$line" | cut -d' ' -f2-)

echo "Job started on $(hostname)"
echo "Processing class ${cls}"
echo "Missing runs: ${runs}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || { echo "No GPU visible"; exit 1; }

for run_id in $runs
do
    echo "Rewriting target=${cls}, run_id=${run_id}"

    outdir="${setup_folder}/cls${cls}/t${run_id}"

    rm -rf "$outdir"
    mkdir -p "$outdir"
    python -u class_neurons_fvis.py \
          --run_id ${run_id} \
          --target ${cls} \
          --setup_folder ${setup_folder} \
          --arch_name convnext_base || echo "Failed target=${cls}, run_id=${run_id}"

done

wait