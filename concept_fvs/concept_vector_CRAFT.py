import numpy as np
import torch
from tqdm import tqdm
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from PIL import Image
import torchvision.models as models
from math import ceil
import torchvision.utils as vutils
import argparse
from craft.craft_torch import Craft, torch_to_numpy

parser = argparse.ArgumentParser()
parser.add_argument('--num_concepts', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--class_number', type=int, default=330)
parser.add_argument('--arch', type=str, default='resnet50')

args = parser.parse_args()

IMAGENET_DIR = "/BS/Databases23/imagenet/original/train/" # change this to the path of your imagenet dataset

transform = transforms.Compose([
 transforms.Resize(256),
 transforms.CenterCrop(224),
 transforms.ToTensor(),
 transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225])
])

def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds

class Dataset_ImageNet_Real(data.Dataset):
    def __init__(
        self,
        preprocess,
        class_id,
        data_dir=IMAGENET_DIR,
        split='train'):
        
        self.preprocess = preprocess
        self.split = split

        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        split_dir = train_dir if (self.split.find('train') != -1) else val_dir

        self.data = datasets.ImageFolder(split_dir, transform=None)
        self.labels = [item[1] for item in self.data.imgs]

        label2ind = buildLabelIndex(self.data.targets)
        all_indices = []
        for label, img_indices in label2ind.items():
            if label in class_id:
                all_indices += img_indices[:300]

        self.data.imgs = [self.data.imgs[idx] for idx in  all_indices]
        self.data.samples = [self.data.samples[idx] for idx in  all_indices]
        self.data.targets = [self.data.targets[idx] for idx in  all_indices]
        self.labels = [self.labels[idx] for idx in  all_indices]

    def __getitem__(self, index):
        img, label = self.data[index]

        return self.preprocess(img), label

    def __len__(self):
        return len(self.data)
    

def patchify(inputs, patch_size=64):
    """
    Extracts overlapping patches from the input tensor.

    Args:
        inputs (torch.Tensor): A 4D tensor of shape (batch_size, channels, height, width).
        patch_size (int): The size of each square patch to extract.

    Returns:
        torch.Tensor: A 4D tensor of shape (num_patches, channels, patch_size, patch_size),
                      where num_patches is the total number of patches extracted.
    """
    # Define the stride for overlapping patches (80% overlap)
    strides = int(patch_size * 0.80)

    # Extract patches using unfold and reshape them into the desired format
    patches = torch.nn.functional.unfold(inputs, kernel_size=patch_size, stride=strides)
    patches = patches.transpose(1, 2).contiguous().view(-1, inputs.shape[1], patch_size, patch_size)
    return patches
    
def get_features_patches(model, dataloader, layer):
    """
    Extracts patches and their corresponding features from a given model and dataloader.

    Args:
        model (torch.nn.Module): The neural network model to extract features from.
        dataloader (torch.utils.data.DataLoader): The dataloader providing input images and labels.
        layer (int): The layer index from which features are extracted.

    Returns:
        torch.Tensor: A tensor containing all extracted patches of shape (num_patches, channels, patch_size, patch_size).
        torch.Tensor: A tensor containing all extracted features of shape (num_patches, feature_dim).
    """
    all_patches = []
    all_features = []

    with torch.no_grad():
        for (images, target) in tqdm(dataloader, disable=False):
            
            images = images.cuda()
            target = target.cuda()

            # Extract patches from the input images
            patches = patchify(inputs=images,
                               patch_size=64)
            
            nb_batch_patch = 50
            nb_batchs = ceil(len(patches) / nb_batch_patch)

            start_ids = [b*nb_batch_patch for b in range(nb_batchs)]

            for j in start_ids:                

                x = torch.tensor(patches[j:j+nb_batch_patch])

                # Resize patches to match the input size of the model
                x_resized = torch.nn.functional.interpolate(x, size=224, mode='bilinear', align_corners=False)

                # Extract features from the specified layer
                patch_outputs = model(x_resized)[layer].mean((2,3)).cpu()
                all_patches.append(x.cpu())
                all_features.append(patch_outputs)

    # Concatenate all patches and features
    all_patches = torch.cat(all_patches, 0)
    all_features = torch.cat(all_features, 0)

    return all_patches, all_features

if __name__ == '__main__':


    model = models.__dict__[args.arch](pretrained=True).eval()
    model = model.cuda()

    g = torch.nn.Sequential(*(list(model.children())[:8])) # input to penultimate
    h = lambda x: model.fc(torch.mean(x, (2, 3))) # penultimate layer to logits

    class_path = sorted(os.listdir(IMAGENET_DIR))[args.class_number]
    class_files = os.listdir(os.path.join(IMAGENET_DIR, class_path))

    images_preprocessed = []
    all_files = []
    for i in range(300):
        img = Image.open(os.path.join(IMAGENET_DIR, class_path, class_files[i]))
        num_channel = len(img.split())
        if num_channel != 3:
            continue
        all_files.append(os.path.join(IMAGENET_DIR, class_path, class_files[i]))
        img_p = transform(img)
        images_preprocessed.append(img_p)

    images_preprocessed = torch.stack(images_preprocessed, 0)
    images_preprocessed.shape


    craft = Craft(input_to_latent = g,
                latent_to_logit = h,
                number_of_concepts = 10,
                patch_size = 64,
                batch_size = args.batch_size,
                device = 'cuda')

    # now we can start fit the concept using our rabbit images
    # CRAFT will (1) create the patches, (2) find the concept
    # and (3) return the crops (crops), the embedding of the crops (crops_u), and the concept bank (w)
    crops, crops_u, w = craft.fit(images_preprocessed)
    crops_torch = crops.clone()

    num_crops = crops_torch.shape[0]
    num_patches = int(num_crops /images_preprocessed.shape[0])
    all_files = [ele for ele in all_files for i in range(num_patches)]

    crops = np.moveaxis(torch_to_numpy(crops), 1, -1)

    importances = craft.estimate_importance(images_preprocessed, class_id=args.class_number) # 330 is the rabbit class id in imagenet
    images_u = craft.transform(images_preprocessed)

    most_important_concepts = np.argsort(importances)[::-1][:args.num_concepts]

    folder = f'class_{args.class_number}/'
    os.makedirs(folder, exist_ok=True)

    with open(folder + 'direction.npy', 'wb') as f:

        np.save(f, w[most_important_concepts])

    
    nb_crops = 50
    id_c = 0

    for c_id in most_important_concepts:
    
        subfolder = folder + f'concept_{id_c}/'
        os.makedirs(subfolder, exist_ok=True)
        id_c += 1

        best_crops_ids = np.argsort(crops_u[:, c_id])[::-1][:nb_crops]
        best_crops = crops[best_crops_ids]
        best_samples = list(map(all_files.__getitem__, best_crops_ids))

        txt_filename = subfolder + 'files.txt'

        vutils.save_image(crops_torch[list(best_crops_ids)][:9], subfolder + f'patches.png', normalize=True, scale_each=True, nrow=int(3))

        with open(txt_filename, 'w') as f:
            for i in range(50):
                f.write(f"{best_samples[i]}\n") 
