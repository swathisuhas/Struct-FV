import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_dino_vitl14(device: torch.device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    model = model.to(device).eval()
    return model


def build_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image(path: str, transform: T.Compose) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return transform(image)


@torch.inference_mode()
def dino_patch_features(
    model,
    images: torch.Tensor,
) -> torch.Tensor:
    """Return DINOv2 ViT-L/14 patch features as [B, 16, 16, C]."""
    outputs = model.forward_features(images)
    patches = outputs["x_norm_patchtokens"]
    b, n, c = patches.shape
    side = int(round(n ** 0.5))
    if side * side != n:
        raise RuntimeError(f"Expected square patch grid, got {n} tokens")
    return patches.view(b, side, side, c)


def sample_patch_features(
    dino_grid: torch.Tensor,
    coords: torch.Tensor,
) -> torch.Tensor:
    """Bilinearly sample DINO patch features at normalized node coordinates.

    dino_grid: [1, H, W, C]
    coords: [K, 2] in [-1, 1], ordered as (x, y)
    returns: [K, C]
    """
    if dino_grid.shape[0] != 1:
        raise ValueError("dino_grid must have batch size 1")

    feat = dino_grid.permute(0, 3, 1, 2).contiguous()
    grid = coords.to(feat.device).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        feat,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[0, :, :, 0].transpose(0, 1)


def build_dnorm(coords: torch.Tensor) -> torch.Tensor:
    diffs = coords.unsqueeze(1) - coords.unsqueeze(0)
    d = torch.linalg.norm(diffs, dim=-1)
    nz = d > 0
    med = torch.median(d[nz]) if nz.any() else d.new_tensor(1.0)
    return d / (med + 1e-6)


def sinkhorn(cost: torch.Tensor, eps: float = 0.05, iters: int = 30) -> torch.Tensor:
    kernel = torch.exp(-cost / max(eps, 1e-6)) + 1e-9
    n, m = kernel.shape
    a = torch.full((n,), 1.0 / n, dtype=kernel.dtype, device=kernel.device)
    b = torch.full((m,), 1.0 / m, dtype=kernel.dtype, device=kernel.device)
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(iters):
        u = a / (kernel @ v + 1e-8)
        v = b / (kernel.t() @ u + 1e-8)

    return torch.diag(u) @ kernel @ torch.diag(v)


def dino_graph_loss(
    syn_feat: torch.Tensor,
    ref_feat: torch.Tensor,
    syn_coords: torch.Tensor,
    ref_coords: torch.Tensor,
    sink_eps: float = 0.05,
    sink_iters: int = 30,
) -> float:
    """Same graph loss used by Struct-FV, evaluated in DINO feature space."""
    if syn_feat.shape != ref_feat.shape:
        raise ValueError(f"Feature shapes differ: {syn_feat.shape} vs {ref_feat.shape}")
    if syn_feat.shape[0] != syn_coords.shape[0] or ref_feat.shape[0] != ref_coords.shape[0]:
        raise ValueError("Number of features and coordinates must agree")

    # Match the feature preprocessing used in Struct-FV graph matching.
    syn_std = (syn_feat - syn_feat.mean(0, keepdim=True)) / (syn_feat.std(0, keepdim=True) + 1e-6)
    ref_std = (ref_feat - ref_feat.mean(0, keepdim=True)) / (ref_feat.std(0, keepdim=True) + 1e-6)

    syn_n = F.normalize(syn_std, dim=-1)
    ref_n = F.normalize(ref_std, dim=-1)

    cost_feat = 1.0 - (syn_n @ ref_n.t())
    scale = cost_feat.median()
    cost = cost_feat / (scale + 1e-6)

    pi = sinkhorn(cost, eps=sink_eps, iters=sink_iters)
    mass = pi.sum() + 1e-8

    d_syn = build_dnorm(syn_coords)
    d_ref = build_dnorm(ref_coords)

    ds = d_syn.unsqueeze(1).unsqueeze(3)
    dr = d_ref.unsqueeze(0).unsqueeze(2)
    diff2 = (ds - dr) ** 2

    pij = pi.unsqueeze(2).unsqueeze(3)
    pkl = pi.unsqueeze(0).unsqueeze(1)

    return float((diff2 * pij * pkl).sum().div(mass ** 2).item())


def softmin(losses: torch.Tensor, tau: float = 0.5) -> float:
    return float((-tau * torch.logsumexp(-losses / max(tau, 1e-6), dim=0)).item())


def extract_reference_paths(nodes_payload) -> List[str]:
    paths = nodes_payload.get("reference_paths")
    if not paths:
        raise KeyError("graph_eval_nodes.pt does not contain reference_paths")
    return [str(p) for p in paths]


def load_feature_batch(model, paths: List[str], transform, device, batch_size: int):
    outputs = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        batch = torch.stack([load_image(p, transform) for p in batch_paths], dim=0).to(device)
        outputs.append(dino_patch_features(model, batch).cpu())
    return torch.cat(outputs, dim=0)


def main():
    parser = argparse.ArgumentParser(description="External DINO graph evaluation for Struct-FV and VITAL")
    parser.add_argument("--struct_image", required=True, help="Final Struct-FV generated image")
    parser.add_argument("--vital_image", required=True, help="Final VITAL generated image")
    parser.add_argument("--nodes_file", required=True, help="graph_eval_nodes.pt saved by Struct-FV")
    parser.add_argument("--output_json", default="dino_graph_eval.json")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--sink_eps", type=float, default=0.05)
    parser.add_argument("--sink_iters", type=int, default=30)
    parser.add_argument("--ref_batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    nodes_file = Path(args.nodes_file)
    payload = torch.load(nodes_file, map_location="cpu")
    syn_coords = payload["syn_nodes_xy"].float()
    ref_coords = payload["ref_nodes_xy"].float()
    ref_paths = extract_reference_paths(payload)

    if ref_coords.shape[0] != len(ref_paths):
        raise ValueError(
            f"Reference coordinate count ({ref_coords.shape[0]}) does not match path count ({len(ref_paths)})."
        )

    transform = build_transform()
    model = load_dino_vitl14(device)

    # Reference DINO features are computed once and reused for both methods.
    ref_feat_maps = load_feature_batch(
        model=model,
        paths=ref_paths,
        transform=transform,
        device=device,
        batch_size=args.ref_batch_size,
    )

    def load_single(path: str) -> torch.Tensor:
        image = load_image(path, transform).unsqueeze(0).to(device)
        return dino_patch_features(model, image).cpu()[0]

    struct_grid = load_single(args.struct_image)
    vital_grid = load_single(args.vital_image)

    struct_losses = []
    vital_losses = []

    for r in range(len(ref_paths)):
        ref_grid = ref_feat_maps[r]
        ref_nodes = ref_grid[None]

        struct_nodes = sample_patch_features(struct_grid[None].to(device), syn_coords.to(device)).cpu()
        ref_nodes_sampled = sample_patch_features(ref_nodes.to(device), ref_coords[r].to(device)).cpu()

        vital_nodes = sample_patch_features(vital_grid[None].to(device), syn_coords.to(device)).cpu()

        struct_loss = dino_graph_loss(
            struct_nodes,
            ref_nodes_sampled,
            syn_coords,
            ref_coords[r],
            sink_eps=args.sink_eps,
            sink_iters=args.sink_iters,
        )
        vital_loss = dino_graph_loss(
            vital_nodes,
            ref_nodes_sampled,
            syn_coords,
            ref_coords[r],
            sink_eps=args.sink_eps,
            sink_iters=args.sink_iters,
        )

        struct_losses.append(struct_loss)
        vital_losses.append(vital_loss)

    struct_loss_tensor = torch.tensor(struct_losses)
    vital_loss_tensor = torch.tensor(vital_losses)

    result = {
        "struct_fv_softmin_dino_graph_loss": softmin(struct_loss_tensor, tau=args.tau),
        "vital_softmin_dino_graph_loss": softmin(vital_loss_tensor, tau=args.tau),
        "struct_fv_reference_losses": struct_losses,
        "vital_reference_losses": vital_losses,
        "num_references": len(ref_paths),
        "tau": args.tau,
        "sinkhorn_eps": args.sink_eps,
        "sinkhorn_iters": args.sink_iters,
        "nodes_file": str(nodes_file),
        "struct_image": args.struct_image,
        "vital_image": args.vital_image,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=== DINO Graph Evaluation ===")
    print(f"References: {len(ref_paths)}")
    print(f"Struct-FV softmin DINO graph loss: {result['struct_fv_softmin_dino_graph_loss']:.6f}")
    print(f"VITAL    softmin DINO graph loss: {result['vital_softmin_dino_graph_loss']:.6f}")
    print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()