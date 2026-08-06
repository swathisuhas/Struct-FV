import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from transformers import CLIPVisionModel
import tqdm

# ==========================================
# 1. The Matryoshka BatchTopK SAE
# ==========================================
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

# ==========================================
# 2. The Training Loop
# ==========================================
def train(imagenet_path, epochs=10, batch_size=1024, lr=1e-3, max_grad_norm=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing training on {device}...")

    # Load Base Model (CLIP)
    clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_model.eval()
    
    # ImageNet Dataloader
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], 
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    dataset = datasets.ImageFolder(root=imagenet_path, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8,      # High worker count keeps GPU fed
        pin_memory=True,    # Speeds up CPU to GPU transfer
        drop_last=True      # Prevents small batches from breaking BatchTopK logic
    )

    # Initialize SAE
    d_model = clip_model.config.hidden_size # 768
    sae = MatryoshkaBatchTopKSAE(d_model=d_model, d_sae=32768, k=32).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))

    # -------------------------------------------------------------
    # CHECK FOR EXISTING CHECKPOINTS TO RESUME
    # -------------------------------------------------------------
    start_epoch = 0
    checkpoint_dir = "checkpoints"
    
    if os.path.exists(checkpoint_dir):
        # Find all files named sae_epoch_X.pt and extract the X values
        ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("sae_epoch_") and f.endswith(".pt")]
        if ckpt_files:
            # Sort them by epoch number to find the latest one
            epochs_saved = [int(f.split("_")[2].split(".")[0]) for f in ckpt_files]
            latest_epoch = max(epochs_saved)
            
            checkpoint_path = os.path.join(checkpoint_dir, f"sae_epoch_{latest_epoch}.pt")
            print(f" Found existing checkpoint: {checkpoint_path}")
            print(f"Loading weights and resuming from Epoch {latest_epoch + 1}...")
            
            # Load the saved weights into the SAE
            sae.load_state_dict(torch.load(checkpoint_path, map_location=device))
            
            # Update our loop starting point
            start_epoch = latest_epoch
    # -------------------------------------------------------------

    # Training
    for epoch in range(start_epoch, epochs):
        sae.train()
        pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, _ in pbar:
            images = images.to(device)
            
            # 1. Extract Embeddings (No Gradients needed for CLIP)
            with torch.no_grad():
                clip_outputs = clip_model(pixel_values=images)
                embeddings = clip_outputs.pooler_output
            
            # 2. Forward pass through SAE
            optimizer.zero_grad()
            outputs = sae(embeddings)
            loss = outputs["loss"]
            
            # 3. Backprop
            loss.backward()
            
            # 4. Critical Optimization Steps from Bussmann Repo
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_grad_norm)
            sae.make_decoder_weights_and_grad_unit_norm()
            
            optimizer.step()
            
            # Update Progress Bar
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "L0": f"{outputs['l0_norm'].item():.2f}"
            })
            
        # Save checkpoint after each epoch
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(sae.state_dict(), f"checkpoints/sae_epoch_{epoch+1}.pt")
        print(f"Saved checkpoint: checkpoints/sae_epoch_{epoch+1}.pt")

# ==========================================
# 3. Execution
# ==========================================
if __name__ == "__main__":
    # UPDATE THIS PATH to wherever ImageNet is stored on your Slurm cluster
    # e.g., "/data/datasets/imagenet" or "/local/scratch/imagenet"
    IMAGENET_PATH = "/BS/Databases23/imagenet/original/train/" 
    
    train(
        imagenet_path=IMAGENET_PATH, 
        epochs=10, 
        batch_size=1024, # Drop to 512 if you hit CUDA Out of Memory errors
        lr=1e-3
    )