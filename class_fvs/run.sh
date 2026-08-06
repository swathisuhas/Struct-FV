#!/bin/bash
#SBATCH -p gpu17,gpu22,gpu24
#SBATCH --gres=gpu:1
#SBATCH -o logs/run_vital_%j.out
#SBATCH -e logs/run_vital_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=64G

setup_folder="class_neurons"

python -u class_neurons_fvis.py \
  --run_id 1 \
  --target 670  \
  --setup_folder ${setup_folder} \
  --arch_name convnext_base
