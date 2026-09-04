#!/usr/bin/env python3
"""
Evaluate VITAL vs Struct-FV over an entire architecture dataset using
DINOv2 ViT-L/14 and the Struct-FV node correspondences saved at the end
of optimization.

Expected layout (default):
  STRUCT_ROOT/
    cls0/t1/<generated.png>
             graph_eval_nodes.pt
         t2/...
         t3/...
    cls1/...
    ...
  VITAL_ROOT/
    cls0/t1/<generated.png>
         t2/...
         t3/...
    ...

For each class/trial:
  - load graph_eval_nodes.pt from Struct-FV
  - use the SAME saved synthetic/reference node locations for both methods
  - extract DINOv2 dense patch features
  - compute DINO graph loss against every saved reference
  - softmin over references
  - save paired Struct-FV and VITAL scores

Final outputs:
  per_sample.csv
  summary.csv
  summary.json

Lower DINO graph loss is better.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--struct_root", required=True,
                   help="Root containing Struct-FV cls*/t* folders.")
    p.add_argument("--vital_root", required=True,
                   help="Root containing VITAL cls*/t* folders.")
    p.add_argument("--output_dir", required=True)

    p.add_argument("--num_classes", type=int, default=1000)
    p.add_argument("--trials", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--class_prefix", default="cls")
    p.add_argument("--trial_prefix", default="t")

    # If omitted, the script auto-detects the single PNG in each trial folder.
    p.add_argument("--image_name", default=None)
    p.add_argument("--nodes_name", default="graph_eval_nodes.pt")

    p.add_argument("--dino_model", default="dinov2_vitl14")
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--sinkhorn_eps", type=float, default=0.05)
    p.add_argument("--sinkhorn_iters", type=int, default=30)

    p.add_argument("--device", default="cuda")
    p.add_argument("--reference_batch_size", type=int, default=8)
    p.add_argument("--skip_missing", action="store_true")
    return p.parse_args()


def find_generated_image(folder: Path, image_name: Optional[str]) -> Path:
    if image_name is not None:
        p = folder / image_name
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    pngs = sorted([
        p for p in folder.glob("*.png")
        if not p.name.endswith("_masked.png")
    ])
    if len(pngs) == 0:
        raise FileNotFoundError(f"No PNG found in {folder}")
    if len(pngs) > 1:
        raise RuntimeError(
            f"Multiple PNGs found in {folder}. "
            f"Pass --image_name explicitly. Found: {[p.name for p in pngs]}"
        )
    return pngs[0]


def load_dino(device: torch.device, model_name: str):
    # Requires internet/cache access to torch.hub the first time.
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)

    transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    return model, transform


@torch.inference_mode()
def dino_dense_features(model, images: torch.Tensor) -> torch.Tensor:
    """
    Returns normalized dense DINO patch features:
      [B, C, H, W]
    For ViT-L/14 at 224x224, H=W=16.
    """
    out = model.forward_features(images)

    if "x_norm_patchtokens" in out:
        patches = out["x_norm_patchtokens"]  # [B, N, C]
    elif "x_prenorm" in out:
        tokens = out["x_prenorm"]
        patches = tokens[:, 1:, :]
    else:
        raise KeyError(
            "Could not find DINO patch tokens in forward_features output. "
            f"Available keys: {list(out.keys())}"
        )

    B, N, C = patches.shape
    side = int(round(math.sqrt(N)))
    if side * side != N:
        raise ValueError(f"DINO patch count {N} is not a square grid.")

    patches = F.normalize(patches.float(), dim=-1)
    return patches.transpose(1, 2).reshape(B, C, side, side)


def sample_features_at_points(
    feature_map: torch.Tensor,
    points_xy: torch.Tensor,
) -> torch.Tensor:
    """
    feature_map: [1, C, H, W]
    points_xy:   [K, 2] in [-1, 1], ordered (x, y)

    Bilinearly samples DINO features at the saved Struct-FV node positions.
    Returns [K, C].
    """
    points_xy = points_xy.to(
        device=feature_map.device,
        dtype=feature_map.dtype,
    )
    grid = points_xy.view(1, -1, 1, 2)
    sampled = F.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    sampled = sampled[0, :, :, 0].T
    return F.normalize(sampled, dim=-1)


def normalized_distance_matrix(points_xy: torch.Tensor) -> torch.Tensor:
    points_xy = points_xy.float()
    diffs = points_xy.unsqueeze(1) - points_xy.unsqueeze(0)
    D = torch.linalg.norm(diffs, dim=-1)
    nz = D[D > 0]
    med = torch.median(nz) if nz.numel() else D.new_tensor(1.0)
    return D / (med + 1e-6)


def sinkhorn(
    C: torch.Tensor,
    eps: float,
    iters: int,
) -> torch.Tensor:
    kernel = torch.exp(-C / max(eps, 1e-6)) + 1e-9
    n, m = kernel.shape

    a = torch.full(
        (n,), 1.0 / n, device=C.device, dtype=C.dtype
    )
    b = torch.full(
        (m,), 1.0 / m, device=C.device, dtype=C.dtype
    )

    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(iters):
        u = a / (kernel @ v + 1e-8)
        v = b / (kernel.T @ u + 1e-8)

    return torch.diag(u) @ kernel @ torch.diag(v)


def dino_graph_loss(
    syn_features: torch.Tensor,
    ref_features: torch.Tensor,
    syn_points: torch.Tensor,
    ref_points: torch.Tensor,
    sinkhorn_eps: float,
    sinkhorn_iters: int,
) -> torch.Tensor:
    """
    Same structural loss form as Struct-FV, but feature correspondence
    costs are computed from DINO features at the saved node locations.
    """
    syn_features = F.normalize(syn_features, dim=-1)
    ref_features = F.normalize(ref_features, dim=-1)

    C_feat = 1.0 - syn_features @ ref_features.T

    # Match the normalization used in the Struct-FV implementation.
    sf = C_feat.median()
    C = C_feat / (sf + 1e-6)

    Pi = sinkhorn(C, sinkhorn_eps, sinkhorn_iters)
    mass = Pi.sum() + 1e-8

    D_s = normalized_distance_matrix(syn_points).to(C.device)
    D_r = normalized_distance_matrix(ref_points).to(C.device)

    Ds = D_s.unsqueeze(1).unsqueeze(3)
    Dr = D_r.unsqueeze(0).unsqueeze(2)
    diff2 = (Ds - Dr) ** 2

    Pij = Pi.unsqueeze(2).unsqueeze(3)
    Pkl = Pi.unsqueeze(0).unsqueeze(1)

    return (diff2 * Pij * Pkl).sum() / (mass ** 2)


def normalized_softmin(losses: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Normalized softmin.

    Unlike -tau*logsumexp(-L/tau), this does not become more negative
    merely because more references are included:
        -tau log( mean_r exp(-L_r/tau) )

    Since every sample here should have the same number of references,
    either version preserves pairwise ordering, but this form is easier
    to interpret across experiments.
    """
    n = losses.numel()
    return -tau * (
        torch.logsumexp(-losses / max(tau, 1e-6), dim=0)
        - math.log(n)
    )


def load_rgb(path: Path, transform, device: torch.device) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        return transform(im).unsqueeze(0).to(device)


def extract_reference_paths(graph_data: Dict) -> List[str]:
    """
    Accept a few sensible key names so the evaluator is not brittle.
    The modified Struct-FV saver should ideally use `reference_paths`.
    """
    for key in (
        "reference_paths",
        "ref_paths",
        "reference_image_paths",
        "real_image_paths",
    ):
        if key in graph_data:
            return [str(x) for x in graph_data[key]]

    raise KeyError(
        "graph_eval_nodes.pt does not contain reference image paths. "
        "Expected one of: reference_paths, ref_paths, "
        "reference_image_paths, real_image_paths."
    )


def extract_syn_points(graph_data):
    for key in (
        "syn_nodes_xy",
        "syn_nodes",
        "syn_points",
        "synthetic_nodes",
    ):
        if key in graph_data:
            return torch.as_tensor(graph_data[key]).float()

    raise KeyError(
        f"Could not find saved synthetic node positions. "
        f"Available keys: {list(graph_data.keys())}"
    )

def extract_ref_points(graph_data):
    for key in (
        "ref_nodes_xy",
        "ref_nodes",
        "ref_points",
        "reference_nodes",
        "reference_points",
    ):
        if key in graph_data:
            value = graph_data[key]

            if isinstance(value, torch.Tensor):
                return [value[i].float() for i in range(value.shape[0])]

            return [torch.as_tensor(x).float() for x in value]

    raise KeyError(
        f"Could not find saved reference node positions. "
        f"Available keys: {list(graph_data.keys())}"
    )

@torch.inference_mode()
def extract_reference_feature_maps(
    model,
    transform,
    reference_paths: List[str],
    device: torch.device,
    batch_size: int,
) -> List[torch.Tensor]:
    """
    Returns CPU feature maps so a class's 50 references do not need to
    remain resident on GPU.
    """
    output = []

    for start in range(0, len(reference_paths), batch_size):
        batch_paths = reference_paths[start:start + batch_size]
        tensors = []

        for p in batch_paths:
            with Image.open(p) as im:
                tensors.append(transform(im.convert("RGB")))

        batch = torch.stack(tensors).to(device)
        feat = dino_dense_features(model, batch).cpu()
        output.extend([feat[i:i + 1] for i in range(feat.shape[0])])

    return output


def evaluate_method(
    generated_feature_map: torch.Tensor,
    reference_feature_maps: List[torch.Tensor],
    syn_points: torch.Tensor,
    ref_points: List[torch.Tensor],
    device: torch.device,
    tau: float,
    sinkhorn_eps: float,
    sinkhorn_iters: int,
) -> Tuple[float, List[float]]:
    syn_points_dev = syn_points.to(device)
    syn_features = sample_features_at_points(
        generated_feature_map, syn_points_dev
    )

    losses = []

    if len(reference_feature_maps) != len(ref_points):
        raise ValueError(
            f"{len(reference_feature_maps)} reference feature maps but "
            f"{len(ref_points)} reference node sets."
        )

    for ref_map_cpu, ref_pts_cpu in zip(reference_feature_maps, ref_points):
        ref_map = ref_map_cpu.to(device)
        ref_pts = ref_pts_cpu.to(device)

        ref_features = sample_features_at_points(ref_map, ref_pts)

        loss = dino_graph_loss(
            syn_features,
            ref_features,
            syn_points_dev,
            ref_pts,
            sinkhorn_eps,
            sinkhorn_iters,
        )
        losses.append(loss)

    losses_t = torch.stack(losses)
    score = normalized_softmin(losses_t, tau)

    return float(score.item()), [float(x.item()) for x in losses_t]


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def median(xs: List[float]) -> float:
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        return float("nan")
    if n % 2:
        return ys[n // 2]
    return 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def summarize(rows: List[Dict]) -> Dict:
    struct_scores = [r["struct_fv"] for r in rows]
    vital_scores = [r["vital"] for r in rows]
    differences = [s - v for s, v in zip(struct_scores, vital_scores)]

    struct_wins = sum(d < 0 for d in differences)
    vital_wins = sum(d > 0 for d in differences)
    ties = sum(abs(d) < 1e-12 for d in differences)

    # Per-class averages first, useful because there are 3 trials/class.
    by_class: Dict[int, List[Dict]] = {}
    for r in rows:
        by_class.setdefault(r["class"], []).append(r)

    class_rows = []
    for cls, rs in sorted(by_class.items()):
        s = mean([x["struct_fv"] for x in rs])
        v = mean([x["vital"] for x in rs])
        class_rows.append({
            "class": cls,
            "struct_fv": s,
            "vital": v,
            "difference": s - v,
        })

    class_wins = sum(x["difference"] < 0 for x in class_rows)

    return {
        "n_pairs": len(rows),
        "n_classes": len(class_rows),
        "struct_fv": {
            "mean": mean(struct_scores),
            "std": std(struct_scores),
            "median": median(struct_scores),
        },
        "vital": {
            "mean": mean(vital_scores),
            "std": std(vital_scores),
            "median": median(vital_scores),
        },
        "paired_difference_struct_minus_vital": {
            "mean": mean(differences),
            "std": std(differences),
            "median": median(differences),
        },
        "pairwise_wins": {
            "struct_fv_lower": struct_wins,
            "vital_lower": vital_wins,
            "ties": ties,
            "struct_fv_win_rate": (
                struct_wins / len(rows) if rows else float("nan")
            ),
        },
        "class_level": {
            "struct_fv_lower_classes": class_wins,
            "struct_fv_class_win_rate": (
                class_wins / len(class_rows)
                if class_rows else float("nan")
            ),
            "per_class": class_rows,
        },
    }


def main():
    args = parse_args()

    struct_root = Path(args.struct_root)
    vital_root = Path(args.vital_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Loading {args.dino_model} ...")
    model, transform = load_dino(device, args.dino_model)

    rows = []

    

    for cls in range(args.num_classes):
        for trial in args.trials:
            struct_dir = (
                struct_root
                / f"{args.class_prefix}{cls}"
                / f"{args.trial_prefix}{trial}"
            )
            vital_dir = (
                vital_root
                / f"{args.class_prefix}{cls}"
                / f"{args.trial_prefix}{trial}"
            )

            try:
                nodes_path = struct_dir / args.nodes_name
                if not nodes_path.exists():
                    raise FileNotFoundError(nodes_path)

                struct_img = find_generated_image(
                    struct_dir, args.image_name
                )
                vital_img = find_generated_image(
                    vital_dir, args.image_name
                )

                graph_data = torch.load(
                    nodes_path, map_location="cpu", weights_only=False
                )

                syn_points = extract_syn_points(graph_data)
                ref_points = extract_ref_points(graph_data)
                reference_paths = extract_reference_paths(graph_data)

                if len(reference_paths) != len(ref_points):
                    raise ValueError(
                        f"{nodes_path}: {len(reference_paths)} reference "
                        f"paths but {len(ref_points)} node sets."
                    )

                print(
                    f"[cls={cls} t={trial}] extracting DINO features "
                    f"for {len(reference_paths)} references..."
                )

                reference_maps = extract_reference_feature_maps(
                    model,
                    transform,
                    reference_paths,
                    device,
                    args.reference_batch_size,
                )

                struct_tensor = load_rgb(
                    struct_img, transform, device
                )
                vital_tensor = load_rgb(
                    vital_img, transform, device
                )

                # Batch both generated images in one DINO forward pass.
                generated_batch = torch.cat(
                    [struct_tensor, vital_tensor], dim=0
                )
                generated_maps = dino_dense_features(
                    model, generated_batch
                )

                struct_score, struct_ref_losses = evaluate_method(
                    generated_maps[0:1],
                    reference_maps,
                    syn_points,
                    ref_points,
                    device,
                    args.tau,
                    args.sinkhorn_eps,
                    args.sinkhorn_iters,
                )

                vital_score, vital_ref_losses = evaluate_method(
                    generated_maps[1:2],
                    reference_maps,
                    syn_points,
                    ref_points,
                    device,
                    args.tau,
                    args.sinkhorn_eps,
                    args.sinkhorn_iters,
                )

                row = {
                    "class": cls,
                    "trial": trial,
                    "struct_fv": struct_score,
                    "vital": vital_score,
                    "difference": struct_score - vital_score,
                    "struct_better": struct_score < vital_score,
                    "n_references": len(reference_paths),
                    "struct_image": str(struct_img),
                    "vital_image": str(vital_img),
                    "nodes_file": str(nodes_path),
                    "struct_reference_losses": struct_ref_losses,
                    "vital_reference_losses": vital_ref_losses,
                }
                rows.append(row)

                print(
                    f"[cls={cls:04d} t={trial}] "
                    f"Struct-FV={struct_score:.6f} | "
                    f"VITAL={vital_score:.6f} | "
                    f"delta={struct_score - vital_score:+.6f}"
                )

                # Free memory before processing the next class/trial.
                del reference_maps
                del generated_maps
                del generated_batch
                del struct_tensor
                del vital_tensor
                del graph_data
                del syn_points
                del ref_points

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as exc:
                message = f"[cls={cls:04d} t={trial}] ERROR: {exc}"
                if args.skip_missing:
                    print(message)
                    continue
                raise RuntimeError(message) from exc

    if not rows:
        raise RuntimeError("No class/trial pairs were evaluated.")

    # Detailed JSON preserves the 50 individual reference losses.
    with open(output_dir / "per_sample.json", "w") as f:
        json.dump(rows, f, indent=2)

    # Compact CSV for analysis/plotting.
    with open(output_dir / "per_sample.csv", "w", newline="") as f:
        fieldnames = [
            "class",
            "trial",
            "struct_fv",
            "vital",
            "difference",
            "struct_better",
            "n_references",
            "struct_image",
            "vital_image",
            "nodes_file",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})

    summary = summarize(rows)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "mean", "std", "median"])
        writer.writerow([
            "Struct-FV",
            summary["struct_fv"]["mean"],
            summary["struct_fv"]["std"],
            summary["struct_fv"]["median"],
        ])
        writer.writerow([
            "VITAL",
            summary["vital"]["mean"],
            summary["vital"]["std"],
            summary["vital"]["median"],
        ])

    print("\n================ FINAL SUMMARY ================")
    print(f"Pairs evaluated: {summary['n_pairs']}")
    print(f"Classes evaluated: {summary['n_classes']}")
    print(
        "Struct-FV: "
        f"{summary['struct_fv']['mean']:.6f} "
        f"+/- {summary['struct_fv']['std']:.6f} "
        f"(median {summary['struct_fv']['median']:.6f})"
    )
    print(
        "VITAL:     "
        f"{summary['vital']['mean']:.6f} "
        f"+/- {summary['vital']['std']:.6f} "
        f"(median {summary['vital']['median']:.6f})"
    )
    print(
        "Mean paired delta (Struct-FV - VITAL): "
        f"{summary['paired_difference_struct_minus_vital']['mean']:+.6f}"
    )
    print(
        "Struct-FV pairwise win rate: "
        f"{100 * summary['pairwise_wins']['struct_fv_win_rate']:.2f}%"
    )
    print(
        "Struct-FV class-level win rate: "
        f"{100 * summary['class_level']['struct_fv_class_win_rate']:.2f}%"
    )
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()