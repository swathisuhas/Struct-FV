import os
from typing import Literal
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from transformers import CLIPVisionModel

# ==========================================
# 1. YOUR CUSTOM MATRYOSHKA SAE
# ==========================================
class MatryoshkaBatchTopKSAE(torch.nn.Module):
    def __init__(self, d_model=768, d_sae=32768, k=32):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.group_fractions = [1/16, 1/8, 1/4, 1/2, 1.0]
        self.group_sizes = [int(d_sae * frac) for frac in self.group_fractions]
        
        self.W_enc = torch.nn.Parameter(torch.randn(d_model, d_sae) / (d_model ** 0.5))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.W_dec = torch.nn.Parameter(torch.randn(d_sae, d_model) / (d_sae ** 0.5))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_model))
        self.register_buffer("threshold_ema", torch.tensor(0.0))

    # (Forward pass omitted here because MACO uses the wrapper below to bypass TopK)

# ==========================================
# 2. CLIP + SAE WRAPPER FOR MACO
# ==========================================
class CLIP_SAE_Wrapper(torch.nn.Module):
    def __init__(self, clip_model, sae_model):
        super().__init__()
        self.clip_model = clip_model
        self.sae_model = sae_model
        
    def forward(self, x):
        # 1. Get embeddings from CLIP
        clip_outputs = self.clip_model(pixel_values=x)
        embeddings = clip_outputs.pooler_output
        
        # 2. Bypass Top-K: Calculate raw pre-activations
        x_centered = embeddings - self.sae_model.b_dec
        pre_acts = (x_centered @ self.sae_model.W_enc) + self.sae_model.b_enc
        
        return pre_acts

# ==========================================
# 3. OFFICIAL MACO TOOLKIT
# ==========================================
class NormalRandomResizedCrop:
    def __init__(self, output_size=(224, 224), average_crop_size=0.25, center_std=0.15, delta_std=0.05, min_crop=0.05, max_crop=1.0):
        self.output_size = output_size
        self.average_crop_size = average_crop_size
        self.center_std = center_std
        self.delta_std = delta_std
        self.min_crop = min_crop
        self.max_crop = max_crop

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = images.shape
        device = images.device
        target_h, target_w = self.output_size

        center_x = 0.5 + torch.randn(B, device=device) * self.center_std
        center_y = 0.5 + torch.randn(B, device=device) * self.center_std
        delta = self.average_crop_size + torch.randn(B, device=device) * self.delta_std
        delta = delta.clamp(self.min_crop, self.max_crop)

        theta = torch.zeros(B, 2, 3, device=device)
        theta[:, 0, 0] = delta
        theta[:, 1, 1] = delta
        theta[:, 0, 2] = 2 * center_x - 1
        theta[:, 1, 2] = 2 * center_y - 1

        grid = torch.nn.functional.affine_grid(theta, size=(B, C, target_h, target_w), align_corners=True)
        return torch.nn.functional.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

def recorrelate_colors(images: torch.Tensor) -> torch.Tensor:
    imagenet_color_correlation = torch.tensor([
        [0.56282854, 0.58447580, 0.58447580],
        [0.19482528, 0.00000000, -0.19482528],
        [0.04329450, -0.10823626, 0.06494176],
    ], dtype=images.dtype, device=images.device)

    images_permuted = images.permute(1, 2, 0)
    images_flat = images_permuted.reshape(-1, 3)
    images_flat = torch.matmul(images_flat, imagenet_color_correlation)
    images_permuted = images_flat.reshape(images_permuted.shape)
    return images_permuted.permute(2, 0, 1)

def fourier_to_image(unnormalized_average_magnitude: torch.Tensor, phase: torch.Tensor, norm: Literal["backward", "forward", "ortho"] = "backward", eps: float = 1e-5) -> torch.Tensor:
    phase = phase - phase.mean()
    phase = phase / (phase.std() + eps)
    spectrum = torch.polar(unnormalized_average_magnitude, phase)
    image = torch.real(torch.fft.ifft2(spectrum, norm=norm))
    image = image - image.mean()
    image = image / (image.std() + eps)
    image = recorrelate_colors(image)
    return torch.nn.functional.sigmoid(image)

def normalize_alpha(alpha: torch.Tensor, percentile: float = 80.0) -> torch.Tensor:
    alpha_mean = torch.mean(alpha, dim=0, keepdim=True)
    alpha_clamped = torch.clamp(alpha_mean, max=torch.quantile(alpha_mean, percentile / 100.0))
    return alpha_clamped / (alpha_clamped.max() + 1e-8)

def run_maco(model, device, unnormalized_average_magnitude, normalize_transform, target_logit_idx=0, num_steps=256, num_crops=32, average_crop_size=-1.0, noise_std=-1.0, learning_rate=1.0, model_input_shape=(224, 224)):
    model = model.to(device)
    unnormalized_average_magnitude = unnormalized_average_magnitude.to(device)

    crop_transforms = [NormalRandomResizedCrop(model_input_shape, average_crop_size=b) for b in torch.linspace(0.5, 0.05, steps=num_steps, dtype=torch.float32)]
    get_crop_transform = lambda step: crop_transforms[step]

    noise_stds = torch.logspace(0, -4, steps=num_steps, dtype=torch.float32)
    get_noise_std = lambda i: noise_stds[i]

    phase = (2 * torch.pi * torch.rand(unnormalized_average_magnitude.shape, device=device)) - torch.pi
    phase.requires_grad = True
    alpha = torch.zeros(unnormalized_average_magnitude.shape, device=device)

    optimizer = torch.optim.NAdam([phase], lr=learning_rate)

    for i in tqdm(range(num_steps), desc=f"Optimizing Latent {target_logit_idx}"):
        optimizer.zero_grad()

        x_n = fourier_to_image(unnormalized_average_magnitude, phase)[None, :, :, :]
        normalized_x_n = normalize_transform(x_n)
        normalized_x_n.requires_grad_(True)
        normalized_x_n.retain_grad()

        cropped_x_n = get_crop_transform(i)(normalized_x_n.repeat(num_crops, 1, 1, 1))

        noise_std_val = get_noise_std(i)
        cropped_x_n += torch.randn_like(cropped_x_n) * noise_std_val
        cropped_x_n += torch.rand_like(cropped_x_n) * noise_std_val - (noise_std_val / 2.0)

        logits = model(cropped_x_n)
        target_logit = logits[:, target_logit_idx]

        loss = -target_logit.mean()
        loss.backward(retain_graph=True)

        grad_x_n = normalized_x_n.grad.squeeze(0)
        alpha += grad_x_n.abs()

        optimizer.step()

    return phase.cpu().detach(), alpha.cpu().detach()


# ==========================================
# 4. EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load CLIP and your trained SAE
    print("Loading models...")
    clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_model.eval()
    
    sae = MatryoshkaBatchTopKSAE(d_model=768, d_sae=32768, k=32).to(device)
    
    # MUST MATCH your actual saved checkpoint from the training run
    checkpoint_path = "sae/checkpoints/sae_epoch_10.pt" 
    sae.load_state_dict(torch.load(checkpoint_path, map_location=device))
    sae.eval()
    
    wrapper_model = CLIP_SAE_Wrapper(clip_model, sae).to(device)
    wrapper_model.eval()

    # 2. Setup standard CLIP Normalization
    clip_normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073], 
        std=[0.26862954, 0.26130258, 0.27577711]
    )

    # 3. Create a Natural Magnitude Prior
    print("Generating natural magnitude prior...")
    # Grab ANY real image from your dataset to steal its frequency distribution
    # Example: Grab the first dog image you can find in ImageNet
    dummy_image_path = "/BS/Databases23/imagenet/original/train/n02085620/n02085620_10074.JPEG" 
    
    try:
        img_pil = Image.open(dummy_image_path).convert("RGB")
    except FileNotFoundError:
        print("Warning: Real image not found. Falling back to noise (Visualizations may look static-heavy).")
        img_pil = Image.fromarray(torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8).numpy())
        
    to_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = to_tensor(img_pil).to(device)
    
    # Compute the fixed magnitude spectrum
    unnormalized_avg_magnitude = torch.abs(torch.fft.fft2(img_tensor, norm="backward"))

    # 4. Setup output directory and target latents
    os.makedirs("sae/maco_outputs", exist_ok=True)
    
    # Pick a few distinct neurons to test (e.g., 2 foundational, 2 semantic, 1 deep)
    # latents_to_visualize = [0, 5, 250, 1024, 15000]
    latents_to_visualize = [100]

    # 5. Run the Optimization Loop
    for target_latent in latents_to_visualize:
        print(f"\n--- Visualizing Latent {target_latent} ---")
        
        phase, alpha = run_maco(
            model=wrapper_model,
            device=device,
            unnormalized_average_magnitude=unnormalized_avg_magnitude,
            normalize_transform=clip_normalize,
            target_logit_idx=target_latent,
            num_steps=256,
            num_crops=32,
            learning_rate=1.0,
            model_input_shape=(224, 224)
        )

        # 6. Reconstruct the image and mask
        final_image_tensor = fourier_to_image(unnormalized_avg_magnitude.cpu(), phase, norm="backward")
        alpha_normalized = normalize_alpha(alpha)
        
        # Save standard image
        out_img = transforms.ToPILImage()(final_image_tensor)
        out_img.save(f"sae/maco_outputs/latent_{target_latent:05d}.png")
        
        # Save RGBA image (Alpha mask applied to remove the background)
        rgba_image = torch.cat([final_image_tensor, alpha_normalized], dim=0)
        out_rgba = transforms.ToPILImage()(rgba_image)
        out_rgba.save(f"sae/maco_outputs/latent_{target_latent:05d}_masked.png")
        
    print("\nVisualization complete! Check the 'maco_outputs' folder.")