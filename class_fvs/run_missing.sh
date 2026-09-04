#!/bin/bash
#SBATCH -p gpu17,gpu22,gpu24
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --array=0-199
#SBATCH --output=/BS/feature_viz/work/logs/convnext_base_graphMatching_rem_%A_%a.out
#SBATCH --error=/BS/feature_viz/work/logs/convnext_base_graphMatching_rem_%A_%a.err

set -u

setup_folder="/BS/feature_viz/work/results/lambda_ablation/convnext_structFV_dino_1lambda"

# 5 classes per array task
# 200 tasks x 5 classes = 1000 classes
chunk_size=5

start=$((SLURM_ARRAY_TASK_ID * chunk_size))
end=$((start + chunk_size - 1))

echo "=========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Classes: ${start}-${end}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"
echo "=========================================="

nvidia-smi || {
    echo "ERROR: No GPU visible"
    exit 1
}

for cls in $(seq "$start" "$end")
do
    if [ "$cls" -gt 999 ]; then
        break
    fi

    for run_id in 1 2 3
    do
        results_folder="${setup_folder}/convnext_base"
        outdir="${results_folder}/cls${cls}/t${run_id}"

        # ------------------------------------------------
        # Check whether this run is already complete
        # ------------------------------------------------

        png_file=""

        if [ -d "$outdir" ]; then
            png_file=$(find "$outdir" \
                -maxdepth 1 \
                -type f \
                -name "*.png" \
                -print \
                -quit)
        fi

        nodes_file="${outdir}/graph_eval_nodes.pt"

        if [ -n "$png_file" ] && [ -f "$nodes_file" ]; then
            echo "COMPLETE: cls=${cls}, run=${run_id} -- skipping"
            continue
        fi

        # ------------------------------------------------
        # Run is incomplete
        # ------------------------------------------------

        echo ""
        echo "------------------------------------------"
        echo "MISSING/INCOMPLETE"
        echo "Class: ${cls}"
        echo "Run:   ${run_id}"
        echo "Folder: ${outdir}"

        if [ -z "$png_file" ]; then
            echo "PNG: MISSING"
        else
            echo "PNG: ${png_file}"
        fi

        if [ ! -f "$nodes_file" ]; then
            echo "graph_eval_nodes.pt: MISSING"
        else
            echo "graph_eval_nodes.pt: PRESENT"
        fi

        echo "Regenerating..."
        echo "------------------------------------------"

        # Remove partial output completely
        rm -rf "$outdir"
        mkdir -p "$outdir"

        # ------------------------------------------------
        # Generate
        # ------------------------------------------------

        python -u class_neurons_fvis.py \
            --run_id "$run_id" \
            --target "$cls" \
            --setup_folder "$setup_folder" \
            --arch_name convnext_base

        status=$?

        # ------------------------------------------------
        # Check Python exit status
        # ------------------------------------------------

        if [ "$status" -ne 0 ]; then
            echo ""
            echo "=========================================="
            echo "FAILED"
            echo "cls=${cls}, run=${run_id}"
            echo "Python exit code: ${status}"
            echo "=========================================="

            exit "$status"
        fi

        # ------------------------------------------------
        # Verify output actually exists
        # ------------------------------------------------

        generated_png=$(find "$outdir" \
            -maxdepth 1 \
            -type f \
            -name "*.png" \
            -print \
            -quit)

        if [ -z "$generated_png" ]; then
            echo "ERROR: Python completed but PNG was not created."
            echo "cls=${cls}, run=${run_id}"
            exit 1
        fi

        if [ ! -f "$nodes_file" ]; then
            echo "ERROR: Python completed but graph_eval_nodes.pt was not created."
            echo "cls=${cls}, run=${run_id}"
            exit 1
        fi

        echo "SUCCESS: cls=${cls}, run=${run_id}"
        echo "PNG: ${generated_png}"
        echo "Nodes: ${nodes_file}"
        echo ""

    done
done

echo "=========================================="
echo "Array task ${SLURM_ARRAY_TASK_ID} completed"
echo "Classes ${start}-${end}"
echo "Finished: $(date)"
echo "=========================================="