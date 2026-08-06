import json
import os
from PIL import Image
import torchvision.utils as vutils
import torchvision.transforms as Tr
import torch

def save_neuron_mosaic(json_path, save_dir, target_neurons=None, num_images=16):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    os.makedirs(save_dir, exist_ok=True)
    
    transform = Tr.Compose([
        Tr.Resize(224),
        Tr.CenterCrop(224),
        Tr.ToTensor()
    ])

    # If specific neurons are provided, convert to strings. Otherwise, process all.
    keys_to_process = [str(n) for n in target_neurons] if target_neurons else data.keys()

    for neuron_id in keys_to_process:
        if neuron_id not in data:
            print(f"Skipping Neuron {neuron_id}: Not found in JSON.")
            continue
            
        print(f"Generating mosaic for Neuron {neuron_id}...")
        paths = data[neuron_id]
        images = []
        
        for path in paths[:num_images]:
            try:
                img = Image.open(path).convert('RGB')
                images.append(transform(img))
            except Exception as e:
                print(f"Skipping {path}: {e}")
        
        if images:
            # Create a 4x4 grid of images
            grid = vutils.make_grid(torch.stack(images), nrow=4, padding=2)
            vutils.save_image(grid, os.path.join(save_dir, f"neuron_{neuron_id}_mosaic.png"))

if __name__ == "__main__":
    # Define exactly which neuron indexes you want to visualize here
    indexes_to_visualize = [100] 
    
    # Point this to your full path
    save_neuron_mosaic(
        json_path="/BS/feature_viz/work/code/GraphMatchingFV/sae/neuron_references.json",
        save_dir="/BS/feature_viz/work/code/GraphMatchingFV/sae/visualizations/",
        target_neurons=indexes_to_visualize
    )