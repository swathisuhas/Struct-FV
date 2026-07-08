#!/bin/bash
#SBATCH -p gpu22,gpu24
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --array=0-99
#SBATCH --output=/BS/feature_viz/work/logs/dataset_%A_%a.out
#SBATCH --error=/BS/feature_viz/work/logs/dataset_%A_%a.err

setup_folder="/BS/feature_viz/work/results/resnet_reverse/"

chunk_size=10
base_start=0

start=$(( base_start + SLURM_ARRAY_TASK_ID * chunk_size ))
end=$(( start + chunk_size - 1 ))

echo "Job started on $(hostname)"
echo "Processing classes ${start} to ${end}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || { echo "No GPU visible"; exit 1; }

for cls in $(seq ${start} ${end})
do
    if [ "$cls" -gt 999 ]; then
        break
    fi

    for run_id in 1 2 3
    do
        echo "Checking target=${cls}, run_id=${run_id}"
        python -u class_neurons_fvis.py \
          --run_id ${run_id} \
          --target ${cls} \
          --setup_folder ${setup_folder} \
          --arch_name resnet50 || echo "Failed target=${cls}, run_id=${run_id}"
    done
done

wait