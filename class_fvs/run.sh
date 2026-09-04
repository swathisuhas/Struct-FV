#!/bin/bash
#SBATCH -p gpu17
#SBATCH --gres=gpu:1
#SBATCH -o logs/run_vital_%j.out
#SBATCH -e logs/run_vital_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G

# micromamba activate /BS/feature_viz/work/micromamba/envs/envs/myenv/

# setup_folder="/BS/feature_viz/work/results/lambda_ablation/vitl16_structFV_dino_2lambda/"

# python -u class_neurons_fvis.py \
#   --run_id 1 \
#   --target 669  \
#   --setup_folder ${setup_folder} \
#   --arch_name resnet50

# python -u eval_dino.py \
#     --struct_image /BS/feature_viz/work/code/GraphMatchingFV/class_fvs/class_neurons/convnext_base/cls670/t1/c1_10_l1_1_l2_1_l3_1_l4_1.png \
#     --vital_image /BS/feature_viz/work/code/GraphMatchingFV/class_fvs/class_neurons_vital/convnext_base/cls670/t1/c1_10_l1_1_l2_1_l3_1_l4_1.png \
#     --nodes_file /BS/feature_viz/work/code/GraphMatchingFV/class_fvs/class_neurons/convnext_base/cls670/t1/graph_eval_nodes.pt \
#     --output_json dino_graph_eval.json

python -u eval_dino_dataset.py \
  --struct_root /BS/feature_viz/work/results/lambda_ablation/vitl32_structFV_dino_2lambda/vit_l_32 \
  --vital_root /BS/feature_viz/work/results/VITAL_ViT/vit-l-16/VITAL \
  --output_dir dino_graph_eval_vitl32_lambda=2lambda.json