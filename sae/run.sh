#!/bin/bash
#SBATCH -p gpu17,gpu22,gpu24
#SBATCH --gres=gpu:1
#SBATCH -o logs/run_%j.out
#SBATCH -e logs/run_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=64G

# python -u vital_sae_main.py
# python -u top_activating_images.py
wandb agent swsu-project/GraphMatchingFV-sae/l02ma2n7