import os

SETUP_FOLDER = "/BS/feature_viz/work/results/new_lambda_ablation/resnet50_structFV_dino_0.25lambda/resnet50"

NUM_CLASSES = 1000
RUN_IDS = [1, 2, 3]

missing = []

for cls in range(NUM_CLASSES):
    for run_id in RUN_IDS:
        outdir = os.path.join(
            SETUP_FOLDER,
            f"cls{cls}",
            f"t{run_id}"
        )

        # Check folder
        if not os.path.isdir(outdir):
            missing.append((cls, run_id, "folder missing"))
            continue

        # Check generated PNG
        png_files = [
            f for f in os.listdir(outdir)
            if f.endswith(".png")
        ]

        if len(png_files) == 0:
            missing.append((cls, run_id, "PNG missing"))
            continue

        # Check graph nodes file
        nodes_file = os.path.join(
            outdir,
            "graph_eval_nodes.pt"
        )

        if not os.path.isfile(nodes_file):
            missing.append(
                (cls, run_id, "graph_eval_nodes.pt missing")
            )


# --------------------------------------------------
# Summary
# --------------------------------------------------

total_expected = NUM_CLASSES * len(RUN_IDS)
total_missing = len(missing)
total_present = total_expected - total_missing

print("=" * 50)
print("Dataset Check")
print("=" * 50)

print(f"Expected runs : {total_expected}")
print(f"Complete runs : {total_present}")
print(f"Missing runs  : {total_missing}")

if total_missing == 0:
    print("\nEverything is complete.")
else:
    print("\nMissing:")

    for cls, run_id, reason in missing:
        print(
            f"cls={cls}, run={run_id}: {reason}"
        )


# --------------------------------------------------
# Save missing runs
# --------------------------------------------------

if missing:
    output_file = "missing_runs.txt"

    with open(output_file, "w") as f:
        for cls, run_id, reason in missing:
            f.write(
                f"{cls} {run_id} {reason}\n"
            )

    print(f"\nSaved missing runs to: {output_file}")