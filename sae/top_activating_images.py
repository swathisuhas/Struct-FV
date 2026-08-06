import torch
import torch.nn as nn
import json
import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as Tr
from transformers import CLIPVisionModel
from tqdm import tqdm

# ==========================================
# 1. MATRYOSHKA SAE ARCHITECTURE
# ==========================================
class MatryoshkaBatchTopKSAE(nn.Module):
    def __init__(self, d_model=768, d_sae=32768, k=32):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        # Weights are initialized as parameters
        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae) / (d_model ** 0.5))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_model) / (d_sae ** 0.5))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.register_buffer("threshold_ema", torch.tensor(0.0))

    def forward(self, x):
        # We only need the pre-activations for the scanner
        x_centered = x - self.b_dec
        return (x_centered @ self.W_enc) + self.b_enc

# ==========================================
# 2. SCANNER LOGIC
# ==========================================
def scan_all_neurons(
    imagenet_path, 
    checkpoint_path, 
    target_neurons, 
    top_k=50, 
    batch_size=1024
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Models
    clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    sae = MatryoshkaBatchTopKSAE(d_model=768, d_sae=32768, k=32).to(device)
    
    # Load weights
    if os.path.exists(checkpoint_path):
        sae.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights.")
    sae.eval()

    # 2. Preprocessing
    preprocess = Tr.Compose([
        Tr.Resize(256),
        Tr.CenterCrop(224),
        Tr.ToTensor(),
        Tr.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                     std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    dataset = ImageFolder(root=imagenet_path, transform=preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Structure: {neuron_id: [(score, path), ...]}
    stats = {n: [] for n in target_neurons}

    print(f"Scanning {len(dataset)} images for {len(target_neurons)} neurons...")
    
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(loader)):
            images = images.to(device)
            
            # CLIP Embeddings
            embeds = clip(pixel_values=images).pooler_output
            
            # SAE Pre-activations
            pre_acts = sae(embeds)
            
            # Batch offsets
            batch_start = i * batch_size
            
            for n in target_neurons:
                # Extract scores for this specific neuron
                scores = pre_acts[:, n]
                for b_idx, score in enumerate(scores):
                    path = dataset.samples[batch_start + b_idx][0]
                    stats[n].append((score.item(), path))
                
                # Keep memory clean by pruning periodically
                if len(stats[n]) > 2000:
                    stats[n].sort(key=lambda x: x[0], reverse=True)
                    stats[n] = stats[n][:top_k]

    # --- NEW APPENDING LOGIC ---
    json_path = "/BS/feature_viz/work/code/GraphMatchingFV/sae/top_activating_images.json"
    
    # 1. Load existing data if the file already exists
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                output = json.load(f)
        except json.JSONDecodeError:
            output = {} # Handle empty or corrupted file gracefully
    else:
        output = {}

    # 2. Update the dictionary with the new neurons
    for n in target_neurons:
        stats[n].sort(key=lambda x: x[0], reverse=True)
        # Force keys to be strings so JSON formatting matches perfectly
        output[str(n)] = [path for score, path in stats[n][:top_k]]
        
    # 3. Write everything back to the file
    with open(json_path, "w") as f:
        json.dump(output, f, indent=4)
        
    print(f"Done! Updated 'top_activating_images.json' with {len(target_neurons)} neurons.")

if __name__ == "__main__":
    scan_all_neurons(
        imagenet_path="/BS/Databases23/imagenet/original/train/",
        checkpoint_path="/BS/feature_viz/work/code/GraphMatchingFV/sae/checkpoints/sae_epoch_10.pt",
        target_neurons=[100], # Adjust this range as needed
        top_k=50
    )