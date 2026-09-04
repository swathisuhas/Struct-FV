from __future__ import division, print_function, absolute_import, unicode_literals

import math
import os
import random
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
import torchvision.transforms as Tr
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.vision_transformer import EncoderBlock

from utils.utils import clip, denormalize, lr_cosine_policy

IMAGENET_DIR = "/BS/Databases23/imagenet/original/train/"  # Imagenet directory


# -------------------------------------- #
#   VISUALIZATION HELPERS                #
# -------------------------------------- #

def visualize_points(iter_idx, syn_img, ref_img, syn_pts, ref_pts, save_dir):
    """
    Plots Synthetic and Real images with the selected K points overlaid.
    syn_pts, ref_pts: (K, 2) tensors in range [-1, 1] (x, y)
    """
    def _prepare_img(tensor):
        img = tensor.detach().cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img

    img_s = _prepare_img(syn_img)
    img_r = _prepare_img(ref_img)

    H, W, _ = img_s.shape

    def _map_pts(pts):
        pts = pts.detach().cpu().numpy()
        xs = (pts[:, 0] + 1) * 0.5 * (W - 1)
        ys = (pts[:, 1] + 1) * 0.5 * (H - 1)
        return xs, ys

    sx, sy = _map_pts(syn_pts)
    rx, ry = _map_pts(ref_pts)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(img_s)
    axs[0].scatter(sx, sy, c='red', s=100, marker='x', linewidths=2, label='Top-K Syn')
    axs[0].set_title(f"Synthetic (Iter {iter_idx})")
    axs[0].axis('off')

    axs[1].imshow(img_r)
    axs[1].scatter(rx, ry, c='lime', s=60, marker='o', facecolors='none', linewidths=2, label='Matched Real')
    axs[1].set_title(f"Best Real Match (Iter {iter_idx})")
    axs[1].axis('off')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"viz_points_{iter_idx:05d}.png"), dpi=100)
    plt.close()

def visualize_graph_all_cases(
    iter_idx,
    syn_img,
    ref_img,
    syn_graph,
    ref_graph,
    hard_pairs,
    save_dir,
    title_suffix="",
    viz_k=10,
):
    """
    Visualize only the top-k most important SYNTHETIC graph nodes
    and their corresponding matched reference nodes.

    IMPORTANT:
    - The full graph_K nodes are still used during optimization.
    - This function only changes visualization.
    - Top-k is determined from synthetic node importance.
    - Reference nodes are the nodes matched to those synthetic nodes.
    """

    def _prepare_img(tensor):
        img = tensor.detach().cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (
            img.max() - img.min() + 1e-8
        )
        return img

    def _map_pts(pts, H, W):
        pts = pts.detach().cpu().numpy()

        xs = (pts[:, 0] + 1) * 0.5 * (W - 1)
        ys = (pts[:, 1] + 1) * 0.5 * (H - 1)

        return xs, ys

    def _draw_complete_graph(
        ax,
        xs,
        ys,
        colors,
        edge_alpha=0.18,
    ):
        n = len(xs)

        # Draw edges between all displayed nodes
        for i in range(n):
            for j in range(i + 1, n):
                ax.plot(
                    [xs[i], xs[j]],
                    [ys[i], ys[j]],
                    color="white",
                    alpha=edge_alpha,
                    linewidth=1.0,
                    zorder=1,
                )

        # Draw nodes
        ax.scatter(
            xs,
            ys,
            c=colors,
            s=150,
            edgecolors="black",
            linewidths=1.6,
            zorder=3,
        )

    # --------------------------------------------------
    # Prepare images
    # --------------------------------------------------

    img_s = _prepare_img(syn_img)
    img_r = _prepare_img(ref_img)

    H, W, _ = img_s.shape

    # --------------------------------------------------
    # Find top-k most important SYNTHETIC nodes
    # --------------------------------------------------

    importance = syn_graph["node_importance"]

    k = min(
        viz_k,
        importance.numel(),
        len(hard_pairs),
    )

    top_syn_indices = torch.topk(
        importance,
        k=k,
        largest=True,
    ).indices.tolist()

    # --------------------------------------------------
    # Get the reference node matched to each top node
    # --------------------------------------------------

    syn_to_ref = {
        int(syn_i): int(ref_j)
        for syn_i, ref_j in hard_pairs
    }

    selected_pairs = []

    for syn_i in top_syn_indices:
        if syn_i in syn_to_ref:
            selected_pairs.append(
                (syn_i, syn_to_ref[syn_i])
            )

    if len(selected_pairs) == 0:
        print(
            f"Warning: no matched nodes available "
            f"for visualization at iteration {iter_idx}"
        )
        return

    # --------------------------------------------------
    # Extract exactly those locations
    # --------------------------------------------------

    syn_indices = torch.tensor(
        [i for i, _ in selected_pairs],
        dtype=torch.long,
        device=syn_graph["nodes_p_raw"].device,
    )

    ref_indices = torch.tensor(
        [j for _, j in selected_pairs],
        dtype=torch.long,
        device=ref_graph["nodes_p"].device,
    )

    syn_pts_viz = syn_graph["nodes_p_raw"][syn_indices]

    # ref_graph may not contain nodes_p_raw in your refined graph,
    # so nodes_p is used here.
    ref_pts_viz = ref_graph["nodes_p"][ref_indices]

    # --------------------------------------------------
    # Convert normalized coordinates to image coordinates
    # --------------------------------------------------

    sx, sy = _map_pts(
        syn_pts_viz,
        H,
        W,
    )

    rx, ry = _map_pts(
        ref_pts_viz,
        H,
        W,
    )

    # --------------------------------------------------
    # Give every correspondence a distinct color
    # --------------------------------------------------

    import matplotlib.cm as cm

    cmap = cm.get_cmap(
        "tab10",
        max(len(selected_pairs), 1),
    )

    colors = [
        cmap(i)
        for i in range(len(selected_pairs))
    ]

    # Same color ordering on both sides means:
    #
    # synthetic red   <-> reference red
    # synthetic blue  <-> reference blue
    # etc.
    syn_colors = colors
    ref_colors = colors

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(12, 7),
    )

    axs[0].imshow(img_s)
    axs[0].set_title(
        f"Synthetic (Iter {iter_idx}) {title_suffix}"
    )
    axs[0].axis("off")

    axs[1].imshow(img_r)
    axs[1].set_title(
        f"Reference (Iter {iter_idx}) {title_suffix}"
    )
    axs[1].axis("off")

    _draw_complete_graph(
        axs[0],
        sx,
        sy,
        syn_colors,
    )

    _draw_complete_graph(
        axs[1],
        rx,
        ry,
        ref_colors,
    )

    plt.tight_layout()

    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    plt.savefig(
        os.path.join(
            save_dir,
            f"viz_graph_{iter_idx:05d}.png",
        ),
        dpi=120,
    )

    plt.close()

def hard_one_to_one_matching(score_mat: torch.Tensor, maximize: bool = True):
    """
    Compute a hard 1-to-1 assignment with Hungarian matching.

    Args:
        score_mat: (K_syn, K_ref) tensor.
                   If maximize=True, larger scores are better.
                   If maximize=False, smaller costs are better.
        maximize:   Set True if score_mat is a similarity/probability matrix.
                    Set False if score_mat is a cost matrix.

    Returns:
        hard_pairs: list of (syn_i, ref_j)
        syn_to_ref: dict {syn_i: ref_j}
        ref_to_syn: dict {ref_j: syn_i}
    """
    mat = score_mat.detach().cpu().numpy()

    if maximize:
        row_ind, col_ind = linear_sum_assignment(-mat)
    else:
        row_ind, col_ind = linear_sum_assignment(mat)

    hard_pairs = [(int(i), int(j)) for i, j in zip(row_ind, col_ind)]
    syn_to_ref = {int(i): int(j) for i, j in zip(row_ind, col_ind)}
    ref_to_syn = {int(j): int(i) for i, j in zip(row_ind, col_ind)}

    return hard_pairs, syn_to_ref, ref_to_syn

# -------------------------------------- #
#   VIT ATTRIBUTION (Grad x Attn Rollout)
# -------------------------------------- #

def compute_rollout_attention(attn_mats, start_layer=0):
    """attn_mats: list of (B, N, N) matrices (heads already averaged)."""
    if len(attn_mats) == 0:
        raise ValueError("attn_mats must not be empty")

    B, N, _ = attn_mats[0].shape
    eye = torch.eye(N, device=attn_mats[0].device).unsqueeze(0).expand(B, N, N)

    mats = [m + eye for m in attn_mats]
    mats = [m / (m.sum(dim=-1, keepdim=True) + 1e-8) for m in mats]

    joint = mats[start_layer]
    for i in range(start_layer + 1, len(mats)):
        joint = mats[i].bmm(joint)
    return joint


class ViTAttentionCaptureManual:
    """
    Patch each torchvision EncoderBlock to compute manual self-attention and retain grad on attn.
    Stores:
      self.attn[layer] = (B, H, N, N) attention probs
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.blocks = [m for m in model.modules() if isinstance(m, EncoderBlock)]
        self.attn = [None] * len(self.blocks)
        self._orig_forwards = []

    def patch(self):
        self._orig_forwards = []
        for i, blk in enumerate(self.blocks):
            self._orig_forwards.append(blk.forward)

            def forward_manual(this_blk, x, layer_idx=i):
                y = this_blk.ln_1(x)

                mha = this_blk.self_attention
                H = mha.num_heads

                if getattr(mha, "batch_first", False):
                    B, N, D = y.shape
                    y_bn = y
                else:
                    N, B, D = y.shape
                    y_bn = y.permute(1, 0, 2)

                head_dim = D // H
                scale = head_dim ** -0.5

                qkv = F.linear(y_bn, mha.in_proj_weight, mha.in_proj_bias)
                q, k, v = qkv.chunk(3, dim=-1)

                q = q.view(B, N, H, head_dim).permute(0, 2, 1, 3)
                k = k.view(B, N, H, head_dim).permute(0, 2, 1, 3)
                v = v.view(B, N, H, head_dim).permute(0, 2, 1, 3)

                dots = torch.matmul(q, k.transpose(-2, -1)) * scale
                attn = torch.softmax(dots, dim=-1)

                self.attn[layer_idx] = attn
                attn.retain_grad()

                out = torch.matmul(attn, v)
                out = out.permute(0, 2, 1, 3).contiguous().view(B, N, D)
                out = F.linear(out, mha.out_proj.weight, mha.out_proj.bias)

                if not getattr(mha, "batch_first", False):
                    out = out.permute(1, 0, 2)

                out = this_blk.dropout(out)
                x = x + out
                z = this_blk.ln_2(x)
                z = this_blk.mlp(z)
                return x + z

            blk.forward = forward_manual.__get__(blk, EncoderBlock)

    def unpatch(self):
        for blk, orig in zip(self.blocks, self._orig_forwards):
            blk.forward = orig


def vit_patch_attribution_rollout(
    model: torch.nn.Module,
    x_normed: torch.Tensor,
    target_idx: int,
    start_layer: int = 0,
) -> torch.Tensor:
    """
    Compute patch importance for torchvision ViT using Grad x Attn rollout.
    Returns:
      patch_scores: (B, P) where P = N-1 (excluding CLS)
    """
    capture = ViTAttentionCaptureManual(model)
    capture.patch()

    logits = model(x_normed)
    score = logits[:, target_idx].sum()

    model.zero_grad(set_to_none=True)
    score.backward(retain_graph=True)

    attn_mats = []
    for attn in capture.attn:
        if attn is None:
            continue
        grad = attn.grad
        if grad is None:
            grad = torch.zeros_like(attn)
        cam = (attn * grad).clamp(min=0.0).mean(dim=1)  # (B, N, N)
        attn_mats.append(cam.detach())

    if len(attn_mats) == 0:
        capture.unpatch()
        raise RuntimeError("No attention matrices captured for attribution.")

    joint = compute_rollout_attention(attn_mats, start_layer=start_layer)  # (B, N, N)
    N = joint.shape[-1]
    if N <= 1:
        capture.unpatch()
        raise RuntimeError(f"Token count N={N} (no patches).")

    patch_scores = joint[:, 0, 1:]
    patch_scores = patch_scores - patch_scores.min(dim=1, keepdim=True).values
    patch_scores = patch_scores / (patch_scores.max(dim=1, keepdim=True).values + 1e-8)

    capture.unpatch()
    return patch_scores


# -------------------------------------- #
#   GRAPH HELPERS
# -------------------------------------- #

def _grid_positions(h: int, w: int, device: torch.device) -> torch.Tensor:
    # patch / cell centers in [-1, 1]
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h * 2.0 - 1.0
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w * 2.0 - 1.0
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([xx, yy], dim=-1).view(-1, 2)


def _flatten_feats_positions(
    feat_map: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], Optional[torch.Tensor]]:
    cls_token = None

    if feat_map.dim() == 4:
        B, C, H, W = feat_map.shape
        feats = feat_map.permute(0, 2, 3, 1).reshape(B, H * W, C)
        pos = _grid_positions(H, W, feat_map.device).unsqueeze(0).repeat(B, 1, 1)
        return feats, pos, (H, W), cls_token

    if feat_map.dim() == 3:
        B, d1, d2 = feat_map.shape

        # For ViTs, token length L is usually the smaller dimension and embed dim C is the larger one.
        if d1 <= d2:
            tokens = feat_map
            L, C = d1, d2
        else:
            tokens = feat_map.transpose(1, 2)
            L, C = d2, d1

        sqrtL = int(round((L - 1) ** 0.5))
        has_cls = (sqrtL * sqrtL + 1 == L)

        if has_cls:
            cls_token = tokens[:, 0:1, :]
            tokens = tokens[:, 1:, :]
            H = W = sqrtL
        else:
            H = W = int(round(L ** 0.5))
            tokens = tokens[:, :H * W, :]

        feats = tokens
        pos = _grid_positions(H, W, feat_map.device).unsqueeze(0).repeat(B, 1, 1)
        return feats, pos, (H, W), cls_token

    raise ValueError("Unsupported feature map shape for flattening.")



def _build_graph_from_featmap(
    feat_map: torch.Tensor,
    K_parts: int = 32,
    importance_vec: Optional[torch.Tensor] = None
):
    feats_flat, pos_flat, (H, W), cls_token = _flatten_feats_positions(feat_map)
    feats_flat = feats_flat[0]
    pos_flat = pos_flat[0]

    f = feats_flat
    f_std = (f - f.mean(0, keepdim=True)) / (f.std(0, keepdim=True) + 1e-6)
    p = pos_flat
    N, C = f_std.shape

    K_eff = min(K_parts, N)

    if importance_vec is not None:
        mags = importance_vec.to(feats_flat.device).float()
        mags = mags[:N]
        mags = F.relu(mags)
    else:
        if cls_token is not None:
            cls_vec = cls_token[0, 0, :]
            f_norm = F.normalize(feats_flat, dim=1)
            c_norm = F.normalize(cls_vec, dim=0)
            mags = F.relu(f_norm @ c_norm)
        else:
            mags = feats_flat.norm(p=2, dim=1)
        
    _, top_indices = torch.topk(mags, k=K_eff)

    nodes_f = f_std[top_indices]
    nodes_p = p[top_indices]
    nodes_p_raw = pos_flat[top_indices]

    diffs = nodes_p.unsqueeze(1) - nodes_p.unsqueeze(0)
    D = torch.linalg.norm(diffs + 1e-8, dim=-1)
    med = torch.median(D[D > 0]) if (D > 0).any() else D.new_tensor(1.0)
    Dnorm = D / (med + 1e-6)

    return dict(
        nodes_f=nodes_f,
        nodes_p=nodes_p,
        nodes_p_raw=nodes_p_raw,
        Dnorm=Dnorm,
        node_importance=mags[top_indices],
    )


def _sinkhorn(
    C: torch.Tensor,
    eps: float = 0.05,
    iters: int = 30
) -> torch.Tensor:
    K = torch.exp(-C / max(eps, 1e-6)) + 1e-9
    n, m = K.shape

    u = torch.ones(n, device=K.device)
    v = torch.ones(m, device=K.device)

    for _ in range(iters):
        u = (1.0 / n) / (K @ v + 1e-8)
        v = (1.0 / m) / (K.t() @ u + 1e-8)

    return torch.diag(u) @ K @ torch.diag(v)


def graph_matching_loss(
    syn_graph,
    ref_graph
):
    f_s, D_s = syn_graph["nodes_f"], syn_graph["Dnorm"]
    f_r, D_r = ref_graph["nodes_f"], ref_graph["Dnorm"]

    f_s_n = F.normalize(f_s, dim=-1)
    f_r_n = F.normalize(f_r, dim=-1)

    C_feat = 1.0 - (f_s_n @ f_r_n.t())
    sf = C_feat.median()
    C = (C_feat / (sf + 1e-6))

    Pi = _sinkhorn(C)
    mass = Pi.sum() + 1e-8

    Ds = D_s.unsqueeze(1).unsqueeze(3)
    Dr = D_r.unsqueeze(0).unsqueeze(2)
    diff2 = (Ds - Dr) ** 2
    Pij = Pi.unsqueeze(2).unsqueeze(3)
    Pkl = Pi.unsqueeze(0).unsqueeze(1)
    L_edges_dist = (diff2 * Pij * Pkl).sum() / (mass ** 2)

    return L_edges_dist


def _select_reference_indices(C_full: torch.Tensor, graph_K: int, arch_name: str, mnn_k: int = 3):
    """Select exactly graph_K unique reference spatial locations using the same
    correspondence logic used during Struct-FV optimization."""
    K_syn, N_real = C_full.shape

    if 'vit' in arch_name:
        k_mnn = min(mnn_k, N_real, K_syn)
        syn_topk = torch.topk(-C_full, k=k_mnn, dim=1).indices
        real_topk = torch.topk(-C_full.t(), k=k_mnn, dim=1).indices

        selected_indices = []
        seen = set()

        for i in range(K_syn):
            for j in syn_topk[i].tolist():
                if i in real_topk[j].tolist() and j not in seen:
                    selected_indices.append(j)
                    seen.add(j)
                    break

        if len(selected_indices) < graph_K:
            C_flat = C_full.view(-1)
            _, indices = torch.sort(C_flat)
            real_candidates = indices % N_real
            for j in real_candidates.tolist():
                if j not in seen:
                    selected_indices.append(j)
                    seen.add(j)
                if len(selected_indices) == graph_K:
                    break
    else:
        C_flat = C_full.view(-1)
        _, indices = torch.sort(C_flat)
        real_candidates = indices % N_real

        selected_indices = []
        seen = set()
        for j in real_candidates.tolist():
            if j not in seen:
                selected_indices.append(j)
                seen.add(j)
            if len(selected_indices) == graph_K:
                break

    return selected_indices


def save_final_graph_eval_nodes(
    model2,
    model,
    inputs,
    real_out,
    ref_paths,
    targets,
    graph_layer_key,
    graph_K,
    arch_name,
    graph_mnn_k,
    folder_name,
):
    """After optimization, recompute the final Struct-FV graph correspondences
    against all saved reference images and store only spatial coordinates/paths.

    These saved coordinates are later reused for the external DINO evaluation.
    No DINO computation happens here.
    """
    model2.eval()

    with torch.no_grad():
        syn_out_final = model2(inputs)
        syn_feat_map = syn_out_final[graph_layer_key]

    importance_vec = None
    if 'vit' in arch_name:
        with torch.enable_grad():
            patch_scores = vit_patch_attribution_rollout(
                model=model,
                x_normed=inputs,
                target_idx=int(targets),
                start_layer=0,
            )
        if inputs.grad is not None:
            inputs.grad.zero_()
        importance_vec = patch_scores[0].detach()

    syn_graph = _build_graph_from_featmap(
        syn_feat_map[0:1],
        K_parts=graph_K,
        importance_vec=importance_vec,
    )

    ref_coords = []
    for r_idx in range(real_out[graph_layer_key].shape[0]):
        rf_map = real_out[graph_layer_key][r_idx:r_idx + 1]
        real_feats_flat, real_pos_flat, _, _ = _flatten_feats_positions(rf_map)
        real_feats_flat = real_feats_flat[0]
        real_pos_flat = real_pos_flat[0]

        f_r_std = (
            (real_feats_flat - real_feats_flat.mean(0, keepdim=True))
            / (real_feats_flat.std(0, keepdim=True) + 1e-6)
        )
        real_vecs_all = F.normalize(f_r_std, dim=-1)
        syn_vecs = F.normalize(syn_graph["nodes_f"], dim=-1)

        C_full = 1.0 - (syn_vecs @ real_vecs_all.t())
        selected_indices = _select_reference_indices(
            C_full=C_full,
            graph_K=graph_K,
            arch_name=arch_name,
            mnn_k=graph_mnn_k,
        )

        if len(selected_indices) != graph_K:
            raise RuntimeError(
                f"Reference {r_idx}: selected {len(selected_indices)} nodes, "
                f"expected {graph_K}."
            )

        ref_coords.append(real_pos_flat[
            torch.tensor(selected_indices, device=real_pos_flat.device)
        ].detach().cpu())

    payload = {
        "syn_nodes_xy": syn_graph["nodes_p_raw"].detach().cpu(),
        "ref_nodes_xy": torch.stack(ref_coords, dim=0),
        "reference_paths": ref_paths,
        "target_class": int(targets),
        "graph_K": int(graph_K),
        "graph_layer_key": graph_layer_key,
        "arch_name": arch_name,
    }

    os.makedirs(folder_name, exist_ok=True)
    out_path = os.path.join(folder_name, "graph_eval_nodes.pt")
    torch.save(payload, out_path)
    print(f"Saved final graph evaluation nodes to: {out_path}")
    return out_path


def softmin_over_refs(losses: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    return -tau * torch.logsumexp(-losses / max(tau, 1e-6), dim=0)


# def obtain_real_imgs_random(targets, num_real_img):
#     normalize = Tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#     transforms_all = Tr.Compose([
#         Tr.Resize(256),
#         Tr.CenterCrop(224),
#         Tr.PILToTensor(),
#         Tr.ConvertImageDtype(torch.float),
#         normalize,
#     ])

#     class_number = targets
#     class_path = sorted(os.listdir(IMAGENET_DIR))[class_number]
#     class_files = os.listdir(os.path.join(IMAGENET_DIR, class_path))
#     random.shuffle(class_files)

#     img_real = []
#     ref_paths = []
#     inc_img = 0
#     for file in class_files:
#         if inc_img == num_real_img:
#             break
#         try:
#             img_path = os.path.join(IMAGENET_DIR, class_path, file)
#             img = Image.open(img_path)
#             if len(img.split()) != 3:
#                 continue
#             inc_img += 1
#             img_p = transforms_all(img)
#             img_real.append(img_p)
#             ref_paths.append(img_path)
#         except Exception:
#             continue

#     img_real = torch.stack(img_real, 0)
#     return img_real.to('cuda'), ref_paths

def obtain_real_imgs_random(
    targets,
    num_real_img,
    split_seed,
    reference_split,
    run_seed,
):
    normalize = Tr.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    transforms_all = Tr.Compose([
        Tr.Resize(256),
        Tr.CenterCrop(224),
        Tr.PILToTensor(),
        Tr.ConvertImageDtype(torch.float),
        normalize,
    ])

    class_number = targets
    class_path = sorted(os.listdir(IMAGENET_DIR))[class_number]
    class_dir = os.path.join(IMAGENET_DIR, class_path)

    # Always start from a deterministic ordering.
    class_files = sorted(os.listdir(class_dir))

    # --------------------------------------------------
    # Fixed 50/50 selection/held-out split.
    # This MUST NOT depend on run_id.
    # --------------------------------------------------
    split_rng = random.Random(split_seed)
    split_rng.shuffle(class_files)

    split_idx = len(class_files) // 2

    if reference_split == "selection":
        candidate_files = class_files[:split_idx]
    elif reference_split == "heldout":
        candidate_files = class_files[split_idx:]
    else:
        raise ValueError(
            f"Unknown reference_split: {reference_split}. "
            "Expected 'selection' or 'heldout'."
        )

    # --------------------------------------------------
    # Trial-specific reference images.
    #
    # run 0 gets one reference subset
    # run 1 gets another
    # run 2 gets another
    #
    # Same run_seed => same refs across all lambdas.
    # --------------------------------------------------
    ref_rng = random.Random(run_seed)
    ref_rng.shuffle(candidate_files)

    img_real = []
    ref_paths = []

    for file in candidate_files:
        if len(img_real) == num_real_img:
            break

        try:
            img_path = os.path.join(class_dir, file)
            img = Image.open(img_path)

            if len(img.split()) != 3:
                continue

            img_p = transforms_all(img)

            img_real.append(img_p)
            ref_paths.append(img_path)

        except Exception:
            continue

    if len(img_real) < num_real_img:
        raise RuntimeError(
            f"Only found {len(img_real)} valid reference images, "
            f"but requested {num_real_img}."
        )

    img_real = torch.stack(img_real, 0)

    print(
        f"Reference split: {reference_split} | "
        f"split_seed: {split_seed} | "
        f"run_seed: {run_seed} | "
        f"references: {len(ref_paths)}"
    )

    return img_real.to("cuda"), ref_paths

def sort_matching(target, input):
    B, C = target.size(0), target.size(1)
    input_mean = input.mean(0)
    _, index_content = torch.sort(input_mean.view(C, -1), dim=-1)
    inverse_index = index_content.argsort(dim=-1)
    value_style, _ = torch.sort(target.view(B, C, -1), dim=-1)
    value_style = value_style.mean(0)
    mse_loss = (input_mean.view(C, -1) - value_style.gather(-1, inverse_index)) ** 2
    return mse_loss.mean()


def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (
        (diff1.abs() / 255.0).mean()
        + (diff2.abs() / 255.0).mean()
        + (diff3.abs() / 255.0).mean()
        + (diff4.abs() / 255.0).mean()
    )
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2


# -------------------------------------- #
#   MAIN CLASS
# -------------------------------------- #

class DeepFeaturesClass(object):
    def __init__(
        self,
        model=None,
        parameters=dict(),
        coefficients=dict(),
        folder_name=None,
    ):
        self.model = model

        self.image_resolution = parameters["resolution"]
        self.do_flip = parameters["do_flip"]
        self.jitter = parameters["jitter"]
        self.num_real_img = parameters["num_real_img"]
        self.epochs = parameters['epochs']
        self.arch_name = parameters['arch_name']
        self.graph_layer_key = parameters['graph_layer_key']
        self.graph_num_refs_per_step = parameters['graph_num_refs_per_step']
        self.graph_mnn_k = parameters['graph_mnn_k']
        
        self.split_seed = parameters["split_seed"]
        self.reference_split = parameters["reference_split"]
        self.seed = parameters["seed"]

        self.print_every = 100

        self.var_scale_l1 = coefficients["tv_l1"]
        self.var_scale_l2 = coefficients["tv_l2"]
        self.l2_scale = coefficients["l2"]
        self.lr = coefficients["lr"]
        self.feat_dist = coefficients["feat_dist"]
        self.layer_weights = coefficients["layer_weights"]
        self.lambda_graph = coefficients["lambda_graph"]
        self.graph_K = coefficients["graph_K"]
        self.folder_name = folder_name

    def get_images(self, targets=None):
        print("Get Struct-FV call")
        
        # Reproducible trial: same run_id => same randomness across lambdas
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        model = self.model
        data_type = torch.float

        # Feature taps for all supported backbones
        if 'resnet' in self.arch_name:
            return_nodes = {
                "conv1": "conv1",
                "layer1": "layer1",
                "layer2": "layer2",
                "layer3": "layer3",
                "layer4": "layer4",
                "fc": "fc",
            }
        elif 'convnext' in self.arch_name:
            return_nodes = {
                "features.0": "conv1",
                "features.2.1": "layer1",
                "features.4.1": "layer2",
                "features.6.1": "layer3",
                "features.7.2.add": "layer4",
                "classifier.2": "fc",
            }
        elif 'vit_l' in self.arch_name:
            return_nodes = {
                "conv_proj": "conv1",
                "encoder.layers.encoder_layer_2": "layer1",
                "encoder.layers.encoder_layer_9": "layer2",
                "encoder.layers.encoder_layer_20": "layer3",
                "encoder.layers.encoder_layer_23": "layer4",
                "heads.head": "fc",
            }
        else:
            raise ValueError(f"Unsupported arch_name: {self.arch_name}")

        model2 = create_feature_extractor(model, return_nodes=return_nodes)
        model2.eval()

        # img_real, ref_paths = obtain_real_imgs_random(
        #     targets=targets,
        #     num_real_img=self.num_real_img
        # )
        
        img_real, ref_paths = obtain_real_imgs_random(
            targets=targets,
            num_real_img=self.num_real_img,
            split_seed=self.split_seed,
            reference_split=self.reference_split,
            run_seed=self.seed,
        )
        inputs = torch.randn(
            (1, 3, self.image_resolution, self.image_resolution),
            requires_grad=True,
            device='cuda',
            dtype=data_type,
        )

        iteration = 0
        iterations_per_layer = self.epochs
        
        lim_0, lim_1 = self.jitter, self.jitter
        optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, 0.9], eps=1e-8)
        lr_scheduler = lr_cosine_policy(self.lr, 100, iterations_per_layer)
        real_out = None

        for iteration_loc in range(iterations_per_layer):
                iteration += 1
                lr_scheduler(optimizer, iteration_loc, iteration_loc)

                inputs_jit = inputs
                img_real_jit = img_real

                off1 = random.randint(-lim_0, lim_0)
                off2 = random.randint(-lim_1, lim_1)
                inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))
                if random.random() > 0.5 and self.do_flip:
                    inputs_jit = torch.flip(inputs_jit, dims=(3,))

                optimizer.zero_grad()
                syn_out = model2(inputs_jit)

                if real_out is None:
                    with torch.no_grad():
                        real_out = model2(img_real_jit)
                        real_feat_for_graph = real_out[self.graph_layer_key]

                # Feature matching loss
                if 'vit' in self.arch_name:
                    loss_conv1 = sort_matching(input=syn_out["conv1"], target=real_out["conv1"])
                    loss_layer1 = sort_matching(input=syn_out["layer1"].permute(0, 2, 1), target=real_out["layer1"].permute(0, 2, 1))
                    loss_layer2 = sort_matching(input=syn_out["layer2"].permute(0, 2, 1), target=real_out["layer2"].permute(0, 2, 1))
                    loss_layer3 = sort_matching(input=syn_out["layer3"].permute(0, 2, 1), target=real_out["layer3"].permute(0, 2, 1))
                    loss_layer4 = sort_matching(input=syn_out["layer4"].permute(0, 2, 1), target=real_out["layer4"].permute(0, 2, 1))
                else:
                    loss_conv1 = sort_matching(input=syn_out["conv1"], target=real_out["conv1"])
                    loss_layer1 = sort_matching(input=syn_out["layer1"], target=real_out["layer1"])
                    loss_layer2 = sort_matching(input=syn_out["layer2"], target=real_out["layer2"])
                    loss_layer3 = sort_matching(input=syn_out["layer3"], target=real_out["layer3"])
                    loss_layer4 = sort_matching(input=syn_out["layer4"], target=real_out["layer4"])

                loss_add = (
                    self.layer_weights[0] * loss_conv1
                    + self.layer_weights[1] * loss_layer1
                    + self.layer_weights[2] * loss_layer2
                    + self.layer_weights[3] * loss_layer3
                    + self.layer_weights[4] * loss_layer4
                )

                loss_graph = torch.tensor(0.0, device=inputs.device)

                viz_syn_graph = None
                viz_ref_graph = None
                viz_syn_pts = None
                viz_ref_pts = None
                viz_ref_img = None
                viz_hard_pairs = None
                graph_active = self.lambda_graph > 0

                if graph_active:
                    syn_feat_map = syn_out[self.graph_layer_key]
                    importance_list = None

                    if 'vit' in self.arch_name:
                        with torch.enable_grad():
                            patch_scores = vit_patch_attribution_rollout(
                                model=self.model,
                                x_normed=inputs_jit,
                                target_idx=int(targets),
                                start_layer=0
                            )
                        if inputs.grad is not None:
                            inputs.grad.zero_()

                        importance_list = [patch_scores[b].detach() for b in range(patch_scores.shape[0])]

                    syn_graphs = []
                    B_syn = syn_feat_map.shape[0]
                    for b in range(B_syn):
                        imp_b = importance_list[b] if importance_list is not None else None
                        g = _build_graph_from_featmap(
                            syn_feat_map[b:b+1],
                            K_parts=self.graph_K,
                            importance_vec=imp_b
                        )
                        syn_graphs.append(g)

                    per_syn_losses = []
                    B_syn = len(syn_graphs)
                    R_total = real_feat_for_graph.size(0)

                    R_take = min(self.graph_num_refs_per_step, R_total)
                    idx_refs_iter = random.sample(range(R_total), k=R_take)

                    for b in range(B_syn):
                        g_syn = syn_graphs[b]
                        syn_vecs = F.normalize(g_syn["nodes_f"], dim=-1)

                        batch_ref_losses = []

                        best_loss_val = float('inf')
                        best_real_pts_raw = None
                        best_ref_img_idx = -1

                        for r_idx in idx_refs_iter:
                            rf_map = real_feat_for_graph[r_idx:r_idx+1]
                            real_feats_flat, real_pos_flat, _, _ = _flatten_feats_positions(rf_map)
                            real_feats_flat = real_feats_flat[0]
                            real_pos_flat = real_pos_flat[0]

                            real_pos_can = real_pos_flat
                            f_r = real_feats_flat
                            f_r_std = (f_r - f_r.mean(0, keepdim=True)) / (f_r.std(0, keepdim=True) + 1e-6)
                            real_vecs_all = F.normalize(f_r_std, dim=-1)

                            C_full = 1.0 - (syn_vecs @ real_vecs_all.t())

                            K_syn, N_real = C_full.shape

                            if 'vit' in self.arch_name:
                                k_mnn = min(self.graph_mnn_k, N_real, K_syn)

                                syn_topk = torch.topk(-C_full, k=k_mnn, dim=1).indices
                                real_topk = torch.topk(-C_full.t(), k=k_mnn, dim=1).indices

                                selected_indices = []
                                seen = set()

                                for i in range(K_syn):
                                    for j in syn_topk[i].tolist():
                                        if i in real_topk[j].tolist() and j not in seen:
                                            selected_indices.append(j)
                                            seen.add(j)
                                            break

                                if len(selected_indices) < self.graph_K:
                                    C_flat = C_full.view(-1)
                                    _, indices = torch.sort(C_flat)
                                    real_candidates = indices % N_real

                                    for j in real_candidates.tolist():
                                        if j not in seen:
                                            selected_indices.append(j)
                                            seen.add(j)
                                        if len(selected_indices) == self.graph_K:
                                            break
                            else:
                                C_flat = C_full.view(-1)
                                _, indices = torch.sort(C_flat)
                                real_candidates = indices % N_real

                                selected_indices = []
                                seen = set()
                                for idx in real_candidates.tolist():
                                    if idx not in seen:
                                        selected_indices.append(idx)
                                        seen.add(idx)
                                    if len(selected_indices) == self.graph_K:
                                        break

                            if len(selected_indices) > 0:
                                idx_tensor = torch.tensor(selected_indices, device=inputs.device)

                                nodes_f_refined = f_r_std[idx_tensor]
                                nodes_p_refined = real_pos_can[idx_tensor]

                                diffs_r = nodes_p_refined.unsqueeze(1) - nodes_p_refined.unsqueeze(0)
                                D_r = torch.linalg.norm(diffs_r + 1e-8, dim=-1)
                                med_r = torch.median(D_r[D_r > 0]) if (D_r > 0).any() else D_r.new_tensor(1.0)
                                Dnorm_r = D_r / (med_r + 1e-6)

                                g_real_refined = {
                                    "nodes_f": nodes_f_refined,
                                    "nodes_p": nodes_p_refined,
                                    "Dnorm": Dnorm_r
                                }

                                Lg = graph_matching_loss(
                                    g_syn, g_real_refined
                                )
                                batch_ref_losses.append(Lg)

                                if b == 0 and Lg.item() < best_loss_val:
                                    best_loss_val = Lg.item()
                                    best_real_pts_raw = real_pos_flat[idx_tensor]
                                    best_ref_img_idx = r_idx

                                    viz_syn_graph = g_syn
                                    viz_ref_graph = g_real_refined
                                    viz_syn_pts = g_syn["nodes_p_raw"]
                                    viz_ref_pts = best_real_pts_raw
                                    viz_ref_img = img_real_jit[best_ref_img_idx].detach()

                        if len(batch_ref_losses) > 0:
                            losses_stack = torch.stack(batch_ref_losses)
                            loss_b = softmin_over_refs(losses_stack, tau=0.5)
                            per_syn_losses.append(loss_b)

                        # if (iteration) % self.print_every == 0 and b == 0 and best_real_pts_raw is not None:
                        #     syn_img_viz = inputs_jit[0].detach()
                        #     ref_img_viz = img_real_jit[best_ref_img_idx].detach()

                        #     visualize_points(
                        #         iter_idx=iteration,
                        #         syn_img=syn_img_viz,
                        #         ref_img=ref_img_viz,
                        #         syn_pts=g_syn["nodes_p_raw"],
                        #         ref_pts=best_real_pts_raw,
                        #         save_dir=os.path.join(self.folder_name, "points_debug")
                        #     )

                    if len(per_syn_losses) > 0:
                        loss_graph = torch.stack(per_syn_losses).mean()

                
                if not graph_active:
                    syn_feat_map = syn_out[self.graph_layer_key]
                    ref_feat_map = real_feat_for_graph[0:1]

                    viz_syn_graph = _build_graph_from_featmap(
                        syn_feat_map[0:1],
                        K_parts=self.graph_K,
                        importance_vec=None
                    )
                    viz_ref_graph = _build_graph_from_featmap(
                        ref_feat_map,
                        K_parts=self.graph_K,
                        importance_vec=None
                    )

                    syn_vecs_viz = F.normalize(viz_syn_graph["nodes_f"], dim=-1)
                    ref_vecs_viz = F.normalize(viz_ref_graph["nodes_f"], dim=-1)
                    C_viz = 1.0 - (syn_vecs_viz @ ref_vecs_viz.t())
                    viz_hard_pairs, _, _ = hard_one_to_one_matching(C_viz, maximize=False)

                    viz_syn_pts = viz_syn_graph["nodes_p_raw"]
                    viz_ref_pts = viz_ref_graph["nodes_p_raw"]
                    viz_ref_img = img_real_jit[0].detach()
                else:
                    if viz_syn_graph is None or viz_ref_graph is None:
                        viz_syn_graph = g_syn
                        viz_ref_graph = g_real_refined
                        viz_syn_pts = g_syn["nodes_p_raw"]
                        viz_ref_pts = best_real_pts_raw
                        viz_ref_img = img_real_jit[best_ref_img_idx].detach()

                # if (iteration % self.print_every == 0) and (viz_syn_graph is not None) and (viz_ref_graph is not None):
                if (iteration == iterations_per_layer) and (viz_syn_graph is not None) and (viz_ref_graph is not None):
                    syn_img_viz = inputs_jit[0].detach()

                    if viz_hard_pairs is None:
                        syn_vecs_viz = F.normalize(viz_syn_graph["nodes_f"], dim=-1)
                        ref_vecs_viz = F.normalize(viz_ref_graph["nodes_f"], dim=-1)
                        C_viz = 1.0 - (syn_vecs_viz @ ref_vecs_viz.t())
                        viz_hard_pairs, _, _ = hard_one_to_one_matching(C_viz, maximize=False)

                    visualize_graph_all_cases(
                        iter_idx=iteration,
                        syn_img=syn_img_viz,
                        ref_img=viz_ref_img,
                        syn_graph=viz_syn_graph,
                        ref_graph=viz_ref_graph,
                        hard_pairs=viz_hard_pairs,
                        save_dir=os.path.join(
                            self.folder_name,
                            "graph_debug"
                        ),
                        title_suffix=(
                            "(graph active)"
                            if graph_active
                            else "(graph inactive)"
                        ),
                        viz_k=10,
                    )

                # Priors
                loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)
                loss_l2 = torch.norm(inputs_jit.view(1, -1), dim=1).mean()

                loss = (
                    self.var_scale_l2 * loss_var_l2
                    + self.var_scale_l1 * loss_var_l1
                    + self.l2_scale * loss_l2
                    + self.feat_dist * loss_add
                    + self.lambda_graph * loss_graph
                )

                if (iteration - 1) % self.print_every == 0:
                    print(
                        f"Iter {iteration} | Loss: {loss.item():.4f} | Graph: {loss_graph.item():.4f} | Dist: {loss_add.item():.4f}"
                    )

                loss.backward()
                optimizer.step()
                inputs.data = clip(inputs.data, use_fp16=False)

        best_inputs = denormalize(inputs.data.clone())
        self.save_images(best_inputs)

        save_final_graph_eval_nodes(
            model2=model2,
            model=self.model,
            inputs=inputs.detach().requires_grad_(True),
            real_out=real_out,
            ref_paths=ref_paths,
            targets=targets,
            graph_layer_key=self.graph_layer_key,
            graph_K=self.graph_K,
            arch_name=self.arch_name,
            graph_mnn_k=self.graph_mnn_k,
            folder_name=self.folder_name,
        )

    def save_images(self, images):
        for idx in range(images.shape[0]):
            name = (
                f"c1_{self.layer_weights[0]}_"
                f"l1_{self.layer_weights[1]}_"
                f"l2_{self.layer_weights[2]}_"
                f"l3_{self.layer_weights[3]}_"
                f"l4_{self.layer_weights[4]}.png"
            )
            place_to_store = os.path.join(self.folder_name, name)
            
            image_np = images[idx].data.cpu().numpy().transpose((1, 2, 0))
            pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
            pil_image.save(place_to_store)