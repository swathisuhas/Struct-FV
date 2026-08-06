from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import copy
import os
import random
import wandb


import numpy as np
from PIL import Image
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as Tr

from utils.utils import lr_cosine_policy, clip, denormalize

# Removed torchvision feature_extractor and LRPModel imports

def sort_matching(target, input):
    B, C, W, H = target.size()
    input = input.squeeze(0)
    _, index_content = torch.sort(input.view(C, -1))
    inverse_index = index_content.argsort(-1)
    value_style, _ = torch.sort(target.view(B, C, -1))
    value_style = value_style.mean(0)
    mse_loss = (input.view(C, -1) - value_style.gather(-1, inverse_index)) ** 2
    return mse_loss.unsqueeze(0)


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


def obtain_real_imgs(channel, num_real_img, json_file):
    normalize = Tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transforms_all = Tr.Compose([
        Tr.Resize(256),
        Tr.CenterCrop(224),
        Tr.PILToTensor(),
        Tr.ConvertImageDtype(torch.float),
        normalize,
    ])

    with open(json_file, 'r') as f:
        references = json.load(f)

    channel_str = str(channel)
    if channel_str not in references:
        raise ValueError(f"Neuron {channel_str} not found in {json_file}")

    datafiles = references[channel_str]

    img_real = []
    inc_img = 0
    for file in datafiles:
        if inc_img == num_real_img:
            break
        try:
            img = Image.open(file).convert('RGB') 
        except Exception as e:
            print(f"Skipping {file} due to error: {e}")
            continue
            
        inc_img += 1
        img_p = transforms_all(img)
        img_real.append(img_p)
        
    img_real = torch.stack(img_real, 0).to('cuda')
    return img_real

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def check_format(arr):
    if isinstance(arr, torch.Tensor):
        arr = to_numpy(arr)
    if arr.shape[0] == 3:
        return np.moveaxis(arr, 0, -1)
    return arr


def normalize(image):
    image = np.array(image, dtype=np.float32)
    image -= image.min()
    image /= (image.max() + 1e-12)
    return image


def clip_percentile(img, percentile=0.1):
    return np.clip(img, np.percentile(img, percentile), np.percentile(img, 100 - percentile))


def save_maco(image, alpha, percentile_image=1.0, percentile_alpha=95, filename='image.png'):
    image, alpha = check_format(image), check_format(alpha)
    image = clip_percentile(image, percentile_image)
    image = normalize(image)

    alpha = np.mean(alpha, -1, keepdims=True)
    alpha = np.clip(alpha, None, np.percentile(alpha, percentile_alpha))
    alpha = alpha / alpha.max()

    image_np = np.concatenate([image, alpha], -1)
    pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
    pil_image.save(filename)


# ------------------
# Helpers for graph-region masking only
# ------------------

def _normalize_map(x):
    x = x - x.amin(dim=(-2, -1), keepdim=True)
    x = x / (x.amax(dim=(-2, -1), keepdim=True) + 1e-8)
    return x


def _largest_component_from_mask(mask_2d):
    mask = mask_2d.detach().bool().cpu()
    H, W = mask.shape
    visited = torch.zeros((H, W), dtype=torch.bool)

    best_comp = []
    best_size = 0
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(H):
        for x in range(W):
            if not mask[y, x] or visited[y, x]:
                continue

            queue = [(y, x)]
            visited[y, x] = True
            comp = [(y, x)]

            while queue:
                cy, cx = queue.pop(0)
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))
                            comp.append((ny, nx))

            if len(comp) > best_size:
                best_size = len(comp)
                best_comp = comp

    out = torch.zeros((H, W), dtype=torch.bool)
    for y, x in best_comp:
        out[y, x] = True
    return out.to(mask_2d.device)


def _best_component_by_activation(spatial_map, q=0.85, min_area=1):
    if spatial_map.dim() == 3:
        spatial_map = spatial_map.unsqueeze(1)

    x = spatial_map.clone()
    x = _normalize_map(x)

    flat = x.flatten()
    thr = torch.quantile(flat, q)
    binary = (x[0, 0] >= thr)

    if binary.sum() == 0:
        idx = torch.argmax(x[0, 0])
        H, W = x.shape[-2:]
        yy = idx // W
        xx = idx % W
        binary = torch.zeros((H, W), dtype=torch.bool, device=x.device)
        binary[yy, xx] = True

    mask_cpu = binary.detach().bool().cpu()
    x_cpu = x[0, 0].detach().cpu()
    H, W = mask_cpu.shape
    visited = torch.zeros((H, W), dtype=torch.bool)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    best_score = -1.0
    best_comp = []

    for y in range(H):
        for x0 in range(W):
            if not mask_cpu[y, x0] or visited[y, x0]:
                continue

            queue = [(y, x0)]
            visited[y, x0] = True
            comp = [(y, x0)]
            score = float(x_cpu[y, x0])

            while queue:
                cy, cx = queue.pop(0)
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        if mask_cpu[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))
                            comp.append((ny, nx))
                            score += float(x_cpu[ny, nx])

            if len(comp) >= min_area and score > best_score:
                best_score = score
                best_comp = comp

    if len(best_comp) == 0:
        best_mask = _largest_component_from_mask(binary)
    else:
        best_mask = torch.zeros((H, W), dtype=torch.bool)
        for y, x0 in best_comp:
            best_mask[y, x0] = True
        best_mask = best_mask.to(spatial_map.device)

    return best_mask.unsqueeze(0).unsqueeze(0).float()


# ----------------------------
# Graph helpers on layer3 map
# ----------------------------

def visualize_points(iter_idx, syn_img, ref_img, syn_pts, ref_pts, save_dir):
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
    axs[0].scatter(sx, sy, c='red', s=80, marker='x', linewidths=2)
    axs[0].set_title(f"Synthetic (Iter {iter_idx})")
    axs[0].axis('off')

    axs[1].imshow(img_r)
    axs[1].scatter(rx, ry, c='lime', s=60, marker='o', facecolors='none', linewidths=2)
    axs[1].set_title(f"Best Real Match (Iter {iter_idx})")
    axs[1].axis('off')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"viz_points_{iter_idx:05d}.png"), dpi=100)
    plt.close()


def _grid_positions(h, w, device):
    ys = torch.linspace(-1, 1, steps=h, device=device)
    xs = torch.linspace(-1, 1, steps=w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([xx, yy], dim=-1).view(-1, 2)


def _flatten_feats_positions(feat_map):
    B, C, H, W = feat_map.shape
    feats = feat_map.permute(0, 2, 3, 1).reshape(B, H * W, C)
    pos = _grid_positions(H, W, feat_map.device).unsqueeze(0).repeat(B, 1, 1)
    return feats, pos, (H, W)


def _build_graph_from_featmap_importance(feat_map, spatial_mask=None, K_parts=8):
    feats_flat, pos_flat, _ = _flatten_feats_positions(feat_map)
    feats_flat = feats_flat[0]
    pos_flat = pos_flat[0]

    f = feats_flat
    f_std = (f - f.mean(0, keepdim=True)) / (f.std(0, keepdim=True) + 1e-6)

    N, C = f_std.shape
    mags = f.norm(p=2, dim=1)

    if spatial_mask is not None:
        if spatial_mask.dim() == 4:
            mask_flat = spatial_mask[0, 0].reshape(-1) > 0
        elif spatial_mask.dim() == 3:
            mask_flat = spatial_mask[0].reshape(-1) > 0
        else:
            raise ValueError("spatial_mask should be [1,1,H,W] or [1,H,W]")

        valid_count = int(mask_flat.sum().item())
        if valid_count > 0:
            mags = mags.masked_fill(~mask_flat, -1e9)
            K_eff = min(K_parts, valid_count)
        else:
            K_eff = min(K_parts, N)
    else:
        K_eff = min(K_parts, N)

    _, top_indices = torch.topk(mags, k=K_eff)

    nodes_f = f_std[top_indices]
    nodes_p = pos_flat[top_indices]

    diffs = nodes_p.unsqueeze(1) - nodes_p.unsqueeze(0)
    D = torch.linalg.norm(diffs + 1e-8, dim=-1)
    med = torch.median(D[D > 0]) if (D > 0).any() else D.new_tensor(1.0)
    Dnorm = D / (med + 1e-6)

    return {
        "nodes_f": nodes_f,
        "nodes_p": nodes_p,
        "nodes_p_raw": pos_flat[top_indices],
        "Dnorm": Dnorm
    }


def _sinkhorn(C, eps=0.005, iters=30):
    K = torch.exp(-C / max(eps, 1e-6)) + 1e-9
    n, m = K.shape

    a = torch.full((n,), 1.0 / n, device=K.device)
    b = torch.full((m,), 1.0 / m, device=K.device)

    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(iters):
        u = a / (K @ v + 1e-8)
        v = b / (K.t() @ u + 1e-8)

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
    C = C_feat / (sf + 1e-6)

    Pi = _sinkhorn(C)
    mass = Pi.sum() + 1e-8

    Ds = D_s.unsqueeze(1).unsqueeze(3)
    Dr = D_r.unsqueeze(0).unsqueeze(2)
    diff2 = (Ds - Dr) ** 2

    Pij = Pi.unsqueeze(2).unsqueeze(3)
    Pkl = Pi.unsqueeze(0).unsqueeze(1)

    L_edges_dist = (diff2 * Pij * Pkl).sum() / (mass ** 2)
    return L_edges_dist


def softmin_over_refs(losses, tau=0.5):
    return -tau * torch.logsumexp(-losses / max(tau, 1e-6), dim=0)


class DeepFeaturesClass(object):
    def __init__(self,
                 parameters=dict(),
                 coefficients=dict(),
                 exp_name=None,
                 folder_name=None):
                 
        # Updated to unpack Models directly from parameters
        self.clip_model = parameters['clip_model']
        self.sae_model = parameters['sae_model']

        self.image_resolution = parameters["resolution"]
        self.do_flip = parameters["do_flip"]
        self.jitter = parameters["jitter"]
        self.num_real_img = parameters["num_real_img"]
        self.epochs = parameters['epochs']
        self.channel = parameters["channel"]
        self.layer = parameters["layer"]
        self.arch_name = parameters.get('arch_name', 'clip_vit')
        self.topk_json = parameters["topk_json"]
        self.graph_lambda = parameters["graph_lambda"]
        self.graph_K = parameters["graph_K"]

        self.print_every = 100

        self.var_scale_l1 = coefficients["tv_l1"]
        self.var_scale_l2 = coefficients["tv_l2"]
        self.l2_scale = coefficients["l2"]
        self.lr = coefficients["lr"]
        self.feat_dist = coefficients["feat_dist"]
        self.layer_weights = coefficients["layer_weights"]
        self.graph_num_refs = coefficients["graph_num_refs_per_step"]

        self.folder_name = folder_name
        
    def _forward_and_attr(self, img_tensor):
        """
        Runs image through CLIP -> SAE, extracts intermediate feature maps from ViT,
        and computes Attribution Patching (Feature x Gradient) for relevance maps.
        """
        # Forward CLIP, return intermediate layers
        clip_out = self.clip_model(img_tensor, output_hidden_states=True)
        
        # We need the full hidden states (which are part of the main forward graph)
        # ViT-Base has 12 layers.
        mapping = {2: "conv1", 5: "layer1", 8: "layer2", 11: "layer3"}
        names_list = ["layer3", "layer2", "layer1", "conv1"]
        target_hs = []
        
        for name in names_list:
            idx = [k for k, v in mapping.items() if v == name][0]
            target_hs.append(clip_out.hidden_states[idx])
            

        clip_emb = clip_out.pooler_output
        x_centered = clip_emb - self.sae_model.b_dec
        pre_acts = (x_centered @ self.sae_model.W_enc) + self.sae_model.b_enc
        
        target = pre_acts[:, self.channel].sum()
        
        # Calculate gradients 
        grads = torch.autograd.grad(target, target_hs, retain_graph=True, create_graph=True)
        
        spatial_feats = {}
        attention_maps = []
        
        for name, hs, grad in zip(names_list, target_hs, grads):
            B, L, D = hs.shape
            H = int((L - 1)**0.5) 
            
            # Slice the spatial tokens for the feature map
            spatial_f = hs[:, 1:, :].permute(0, 2, 1).view(B, D, H, H)
            spatial_feats[name] = spatial_f
            
            # Slice the spatial tokens for the gradient map
            spatial_g = grad[:, 1:, :].permute(0, 2, 1).view(B, D, H, H)
                
            # Input x Gradient (Attribution patching)
            attention_maps.append(spatial_f * spatial_g)
            
        return spatial_feats, attention_maps
            
        return spatial_feats, attention_maps

    def get_images(self):
        print("Struct-FV inner SAE latent visualization call")

        print_every = self.print_every
        img_original = self.image_resolution
        data_type = torch.float

        img_real = obtain_real_imgs(
            channel=self.channel,
            num_real_img=self.num_real_img,
            json_file=self.topk_json
        )

        inputs = torch.randn(
            (1, 3, img_original, img_original),
            requires_grad=True,
            device='cuda',
            dtype=data_type
        )

        transparency_accumulator = torch.zeros(
            (1, 3, img_original, img_original),
            device='cuda',
            dtype=data_type
        )

        iteration = 0
        iterations_per_layer = self.epochs
        lim_0, lim_1 = self.jitter , self.jitter 
        optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, 0.9], eps=1e-8)
        do_clip = True
        lr_scheduler = lr_cosine_policy(self.lr, 100, iterations_per_layer)

        for iteration_loc in range(iterations_per_layer):
            iteration += 1
            lr_scheduler(optimizer, iteration_loc, iteration_loc)

            inputs_jit = inputs
            img_real_jit = img_real

            off1 = random.randint(-lim_0, lim_0)
            off2 = random.randint(-lim_1, lim_1)
            inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

            flip = random.random() > 0.5
            if flip and self.do_flip:
                inputs_jit = torch.flip(inputs_jit, dims=(3,))

            optimizer.zero_grad()
            img_real_jit.requires_grad = True

            # -- SYNTHETIC FORWARD & ATTRIBUTION --
            syn_out, attention_syn = self._forward_and_attr(inputs_jit)

            # compute real targets once at iteration 1 and keep them.
            if iteration == 1:
                # -- REAL FORWARD & ATTRIBUTION --
                real_out, attention_real = self._forward_and_attr(img_real_jit)

                importance_neuron = []
                for att_id in range(len(attention_real)):
                    imp_neuron = attention_real[att_id].amax((2, 3))
                    imp_neuron = imp_neuron - imp_neuron.amin(1, keepdim=True)
                    imp_neuron = imp_neuron / (imp_neuron.amax(1, keepdim=True) + 1e-6)
                    importance_neuron.append(imp_neuron)

                    attention_real[att_id] = attention_real[att_id] - attention_real[att_id].amin((2, 3), keepdim=True)
                    attention_real[att_id] = attention_real[att_id] / (attention_real[att_id].amax((2, 3), keepdim=True) + 1e-6)

                attention_conv1 = attention_real[3] * real_out["conv1"]
                attention_layer1 = attention_real[2] * real_out["layer1"]
                attention_layer2 = attention_real[1] * real_out["layer2"]
                attention_layer3 = attention_real[0] * real_out["layer3"]

            for att_id in range(len(attention_syn)):
                attention_syn[att_id] = attention_syn[att_id] - attention_syn[att_id].amin((2, 3), keepdim=True)
                attention_syn[att_id] = attention_syn[att_id] / (attention_syn[att_id].amax((2, 3), keepdim=True) + 1e-6)

            attention_syn_conv1 = attention_syn[3] * syn_out["conv1"]
            attention_syn_layer1 = attention_syn[2] * syn_out["layer1"]
            attention_syn_layer2 = attention_syn[1] * syn_out["layer2"]
            attention_syn_layer3 = attention_syn[0] * syn_out["layer3"]

            loss_conv1 = sort_matching(input=attention_syn_conv1, target=attention_conv1) * importance_neuron[3].unsqueeze(-1)
            loss_layer1 = sort_matching(input=attention_syn_layer1, target=attention_layer1) * importance_neuron[2].unsqueeze(-1)
            loss_layer2 = sort_matching(input=attention_syn_layer2, target=attention_layer2) * importance_neuron[1].unsqueeze(-1)
            loss_layer3 = sort_matching(input=attention_syn_layer3, target=attention_layer3) * importance_neuron[0].unsqueeze(-1)

            loss_conv1 = loss_conv1.mean()
            loss_layer1 = loss_layer1.mean()
            loss_layer2 = loss_layer2.mean()
            loss_layer3 = loss_layer3.mean()

            loss_add = (
                self.layer_weights[0] * loss_conv1
                + self.layer_weights[1] * loss_layer1
                + self.layer_weights[2] * loss_layer2
                + self.layer_weights[3] * loss_layer3
            )

            # Graph ON:
            loss_graph = torch.tensor(0.0, device=inputs.device)
            graph_losses = []

            if self.graph_lambda > 0.0:
                num_refs = min(self.graph_num_refs, attention_layer3.size(0))
                ref_indices = random.sample(range(attention_layer3.size(0)), k=num_refs)

                syn_spatial_map = attention_syn_layer3[0:1].abs().mean(1, keepdim=True)
                syn_comp_mask = _best_component_by_activation(
                    syn_spatial_map
                )

                syn_graph = _build_graph_from_featmap_importance(
                    attention_syn_layer3[0:1],
                    spatial_mask=syn_comp_mask,
                    K_parts=self.graph_K
                )

                best_loss_val = float('inf')
                best_real_pts_raw = None
                best_ref_img_idx = None

                for r_idx in ref_indices:
                    ref_spatial_map = attention_layer3[r_idx:r_idx + 1].abs().mean(1, keepdim=True)
                    ref_mask = _best_component_by_activation(
                        ref_spatial_map
                    )

                    ref_graph = _build_graph_from_featmap_importance(
                        attention_layer3[r_idx:r_idx + 1],
                        spatial_mask=ref_mask,
                        K_parts=self.graph_K
                    )

                    Lg = graph_matching_loss(
                        syn_graph,
                        ref_graph
                    )
                    graph_losses.append(Lg)

                    if Lg.item() < best_loss_val:
                        best_loss_val = Lg.item()
                        best_real_pts_raw = ref_graph["nodes_p_raw"]
                        best_ref_img_idx = r_idx

                if len(graph_losses) > 0:
                    loss_graph = softmin_over_refs(torch.stack(graph_losses))

                if iteration % print_every == 0 and best_real_pts_raw is not None:
                    visualize_points(
                        iter_idx=iteration,
                        syn_img=inputs_jit[0].detach(),
                        ref_img=img_real_jit[best_ref_img_idx].detach(),
                        syn_pts=syn_graph["nodes_p_raw"],
                        ref_pts=best_real_pts_raw,
                        save_dir=os.path.join(self.folder_name, "points_debug")
                    )

            loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)
            loss_l2 = torch.norm(inputs_jit.view(1, -1), dim=1).mean()

            loss = (
                self.var_scale_l2 * loss_var_l2
                + self.var_scale_l1 * loss_var_l1
                + self.l2_scale * loss_l2
                + self.feat_dist * loss_add
                + self.graph_lambda * loss_graph
            )

            grad_input = torch.autograd.grad(
                loss,
                inputs,
                grad_outputs=torch.ones_like(loss),
                retain_graph=True
            )[0]
            transparency_accumulator += torch.abs(grad_input)

            if iteration % print_every == 0:
                print("------------iteration {}----------".format(iteration))
                print("total loss", loss.item())
                print("feat loss", loss_add.item())
                print("graph loss", loss_graph.item())

            wandb.log({
                "total_loss": loss.item(),
                "feat_loss": loss_add.item(),
                "graph_loss": loss_graph.item(),
                "step": iteration
            })

            save_input = inputs.data.clone()
            save_input = denormalize(save_input)

            # Optional: Log the generated image every 100 steps to watch it optimize in real-time
            if iteration % 100 == 0:
                # Assuming `img` is your optimized image tensor
                wandb.log({
                    "generated_image": wandb.Image(save_input[0], caption=f"Step {iteration}") 
                })

            loss.backward(retain_graph=True)
            optimizer.step()

            if do_clip:
                inputs.data = clip(inputs.data, use_fp16=False)

        best_inputs = inputs.data.clone()
        best_inputs = denormalize(best_inputs)
        self.save_images(best_inputs)

        name = (
            f"ch_{self.channel}_c1_{self.layer_weights[0]}_"
            f"l1_{self.layer_weights[1]}_l2_{self.layer_weights[2]}_"
            f"l3_{self.layer_weights[3]}_masked.png"
        )
        place_to_store = os.path.join(self.folder_name, name)

        save_maco(
            inputs.data.clone()[0],
            transparency_accumulator.data.clone()[0],
            percentile_image=1.0,
            percentile_alpha=98,
            filename=place_to_store
        )

    def save_images(self, images):
        for id in range(images.shape[0]):
            name = (
                f"ch_{self.channel}_c1_{self.layer_weights[0]}_"
                f"l1_{self.layer_weights[1]}_l2_{self.layer_weights[2]}_"
                f"l3_{self.layer_weights[3]}.png"
            )
            place_to_store = os.path.join(self.folder_name, name)

            image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
            pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
            pil_image.save(place_to_store)