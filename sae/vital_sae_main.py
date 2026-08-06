from __future__ import division, print_function, absolute_import, unicode_literals

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from transformers import CLIPVisionModel

from utils.utils import create_folder
from utils.vital_sae import DeepFeaturesClass

# --- YOUR SAE CLASS ---
class MatryoshkaBatchTopKSAE(nn.Module):
    def __init__(self, d_model=768, d_sae=32768, k=32):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        
        # Matryoshka Nested Fractions [1/16, 1/8, 1/4, 1/2, 1]
        self.group_fractions = [1/16, 1/8, 1/4, 1/2, 1.0]
        self.group_sizes = [int(d_sae * frac) for frac in self.group_fractions]
        
        # Weights (Initialized exactly as standard SAELens repos do)
        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae) / (d_model ** 0.5))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_model) / (d_sae ** 0.5))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        
        # Running average for thresholding during inference
        self.register_buffer("threshold_ema", torch.tensor(0.0))

    def forward(self, x):
        # 1. Pre-activations
        x_centered = x - self.b_dec
        pre_acts = (x_centered @ self.W_enc) + self.b_enc
        
        # 2. BatchTopK Thresholding
        if self.training:
            batch_size = x.shape[0]
            k_batch = self.k * batch_size
            
            # Find the dynamic threshold for the whole batch
            flat_pre_acts = pre_acts.view(-1)
            threshold = torch.topk(flat_pre_acts, k_batch).values[-1]
            
            # Update inference EMA
            self.threshold_ema = 0.99 * self.threshold_ema + 0.01 * threshold.detach()
        else:
            threshold = self.threshold_ema
            
        # 3. Apply Threshold (Sparsity)
        acts = F.relu(pre_acts)
        acts = torch.where(pre_acts >= threshold, acts, torch.zeros_like(acts))
        
        # 4. Matryoshka Reconstruction & Loss Calculation
        total_loss = 0.0
        reconstructions = {}
        
        for size in self.group_sizes:
            # Mask out latents beyond the current group
            mask = torch.zeros_like(acts)
            mask[:, :size] = 1.0
            masked_acts = acts * mask
            
            # Reconstruct and calculate MSE
            x_recon = (masked_acts @ self.W_dec) + self.b_dec
            reconstructions[size] = x_recon
            
            mse_loss = F.mse_loss(x_recon, x)
            total_loss += mse_loss

        return {
            "loss": total_loss,
            "l0_norm": (acts > 0).float().sum(dim=-1).mean(),
            "l1_norm": acts.sum(dim=-1).mean(),
            "acts": acts,
        }

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        """
        The critical function from Bussmann's / SAELens repos.
        Normalizes decoder weights AND projects gradients orthogonal to weights.
        """
        # 1. Normalize weights
        norms = torch.norm(self.W_dec, dim=1, keepdim=True)
        self.W_dec.data = self.W_dec.data / (norms + 1e-8)
        
        # 2. Project gradients (so optimizer doesn't immediately un-normalize them)
        if self.W_dec.grad is not None:
            # dot product of grad and weight
            dot = (self.W_dec.grad * self.W_dec.data).sum(dim=1, keepdim=True)
            self.W_dec.grad -= dot * self.W_dec.data


def list_of_floats(arg):
    return list(map(float, arg.split(',')))

def run(args):
    # Initialize W&B. If running a sweep, W&B will override 'args' with sweep parameters.
    wandb.init(entity="swsu-project", project="SAE-VITAL", config=args, tags=["run2"])
    config = wandb.config # Fetch hyperparameter values from the W&B config object
    
    device = torch.device('cuda' if torch.cuda.is_available() and not config.no_cuda else 'cpu')

    print("Loading CLIP Vision Model...")
    clip_model = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-base-patch32", 
        attn_implementation="eager"
    ).to(device)
    clip_model.eval()

    print("Loading Matryoshka Batch TopK SAE...")
    sae_model = MatryoshkaBatchTopKSAE(d_model=768, d_sae=32768, k=32).to(device)
    
    if config.sae_checkpoint and os.path.exists(config.sae_checkpoint):
        print(f"==> Resuming SAE from checkpoint: {config.sae_checkpoint}")
        sae_model.load_state_dict(torch.load(config.sae_checkpoint, map_location=device))
    elif config.sae_checkpoint:
        print(f"WARNING: SAE checkpoint {config.sae_checkpoint} not found!")
    
    sae_model.eval()
    create_folder(config.folder_name)

    # Swap all instances of 'args' with 'config'
    parameters = {
        "clip_model": clip_model,
        "sae_model": sae_model,
        "resolution": config.resolution,
        "do_flip": config.do_flip,
        "jitter": config.jitter,
        "num_real_img": config.num_real_img,
        "epochs": config.epochs,
        "channel": config.channel,
        "layer": config.layer,
        "arch_name": config.arch_name,
        "topk_json": config.topk_json,
        "graph_lambda": config.graph_lambda,
        "graph_K": config.graph_K
    }

    # If layer_weights comes in as a string from the sweep, convert it back to a list
    if isinstance(config.layer_weights, str):
        layer_weights = list_of_floats(config.layer_weights)
    else:
        layer_weights = config.layer_weights

    coefficients = {
        "tv_l1": config.tv_l1, "tv_l2": config.tv_l2, "l2": config.l2,
        "lr": config.lr, "feat_dist": config.feat_dist,
        "layer_weights": layer_weights,
        "graph_num_refs_per_step": config.graph_num_refs_per_step
    }

    DeepFeatureEngine = DeepFeaturesClass(
        parameters=parameters,
        coefficients=coefficients,
        folder_name=config.folder_name
    )

    DeepFeatureEngine.get_images()
    
    # Close the wandb run when finished
    wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--gpuid', type=str, default='0')
    parser.add_argument('--sae_checkpoint', type=str, default='sae/checkpoints/sae_epoch_10.pt')
    parser.add_argument('--arch_name', default='clip_vit', type=str)
    parser.add_argument('--layer', type=str, default='pooler_output')
    parser.add_argument('--channel', type=int, default=12)
    parser.add_argument('--topk_json', type=str, default='/BS/feature_viz/work/code/GraphMatchingFV/sae/top_activating_images.json')
    parser.add_argument('--layer_weights', type=list_of_floats, default="0.1, 1, 1, 1, 1")
    parser.add_argument('--feat_dist', type=float, default=50.0)
    parser.add_argument('--num_real_img', type=int, default=50)
    parser.add_argument('--resolution', type=int, default=224)
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--jitter', default=30, type=int)
    parser.add_argument('--folder_name', type=str, default='images')
    parser.add_argument('--do_flip', action='store_true')
    parser.add_argument('--tv_l1', type=float, default=0.0)
    parser.add_argument('--tv_l2', type=float, default=0.0003)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--l2', type=float, default=0.0003)
    parser.add_argument('--graph_lambda', type=float, default=0.0)
    parser.add_argument('--graph_K', type=int, default=10)
    parser.add_argument('--graph_num_refs_per_step', type=int, default=8)

    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    torch.backends.cudnn.benchmark = True
    run(args)


if __name__ == '__main__':
    main()