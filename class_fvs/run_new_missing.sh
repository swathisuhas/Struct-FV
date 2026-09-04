#!/bin/bash
#SBATCH -p gpu17,gpu22,gpu24
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --mem=64G
#SBATCH --array=0-1
#SBATCH --output=/BS/feature_viz/work/logs/vitl32_missing_%A_%a.out
#SBATCH --error=/BS/feature_viz/work/logs/vitl32_missing_%A_%a.err

setup_folder="/BS/feature_viz/work/results/new_lambda_ablation/resnet50_structFV_dino_0.25lambda/"
missing_file="/BS/feature_viz/work/code/GraphMatchingFV/missing_runs.txt"
arch_name="resnet50"

# Read corresponding line
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$missing_file")

# First column = class, second column = trial
cls=$(echo "$line" | awk '{print $1}')
run_id=$(echo "$line" | awk '{print $2}')

echo "========================================"
echo "Generating missing run"
echo "Class: ${cls}"
echo "Trial: ${run_id}"
echo "Architecture: ${arch_name}"
echo "Node: $(hostname)"
echo "========================================"

# Must match the folder structure constructed by get_config()
outdir="${setup_folder}/${arch_name}/cls${cls}/t${run_id}"

echo "Output directory: ${outdir}"

# Remove partial/incomplete result
rm -rf "$outdir"
mkdir -p "$outdir"

python -u class_neurons_fvis.py \
    --run_id "${run_id}" \
    --target "${cls}" \
    --setup_folder "${setup_folder}" \
    --arch_name "${arch_name}"

status=$?

if [ "$status" -eq 0 ]; then
    echo "SUCCESS: cls=${cls}, run=${run_id}"
else
    echo "FAILED: cls=${cls}, run=${run_id}"
    exit "$status"
fi