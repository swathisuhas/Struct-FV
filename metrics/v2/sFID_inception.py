import numpy as np
import torch
from tqdm import tqdm
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import re
from PIL import Image
import torchvision.models as models
from scipy import linalg
from torchvision.models.feature_extraction import create_feature_extractor
from torch.nn.functional import adaptive_avg_pool2d
import argparse

_IMAGENET_DATASET_DIR = "/BS/Databases23/imagenet/original/"

parser = argparse.ArgumentParser(description='FID / sFID Computation (InceptionV3)')
parser.add_argument('--dataset_dir', type=str,
                    default='/BS/feature_viz/work/results/convnext_graphMatching/convnext/',
                    help='synthetic dataset directory')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
args = parser.parse_args()


# -------------------- Transforms --------------------
# Inception-V3 expects 299x299
transform_syn = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_real = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# -------------------- Datasets --------------------
def extract_number_after_cls(input_string):
    match = re.search(r'cls(\d+)', input_string)
    return int(match.group(1)) if match else None

def list_all_file_paths(directory):
    file_paths, labels = [], []
    for root, _, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            file_paths.append(full_path)
            labels.append(extract_number_after_cls(full_path))
    return file_paths, labels

class Dataset_ImageNet(data.Dataset):
    def __init__(self, preprocess, data_dir=_IMAGENET_DATASET_DIR, split='train'):
        self.preprocess = preprocess
        split_dir = os.path.join(data_dir, 'train' if 'train' in split else 'val')
        self.data = datasets.ImageFolder(split_dir, transform=None)

    def __getitem__(self, index):
        img, label = self.data[index]
        return self.preprocess(img), label

    def __len__(self):
        return len(self.data)

class Dataset_Syn(data.Dataset):
    def __init__(self, preprocess, data_dir):
        self.preprocess = preprocess
        self.data, self.labels = list_all_file_paths(data_dir)

    def __getitem__(self, index):
        img = Image.open(self.data[index]).convert("RGB")
        return self.preprocess(img), self.labels[index]

    def __len__(self):
        return len(self.data)


# -------------------- Feature extraction --------------------
def _extract_features_from_batch(feats, structured):
    """
    feats: [B, C, H, W]
    structured=True  -> [B*H*W, C] (sFID)
    structured=False -> [B, C]     (FID)
    """
    if structured:
        B, C, H, W = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(B * H * W, C)
    else:
        feats = adaptive_avg_pool2d(feats, output_size=(1, 1)).squeeze(3).squeeze(2)
    return feats


def calculate_activation_statistics(dataloader, model, structured=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    sum_ = None
    sum_sq = None
    n = 0

    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            images = images.to(device)
            out = model(images)
            feats = out["Mixed_6e"]  # Inception-V3 intermediate layer

            feats = _extract_features_from_batch(feats, structured)
            feats_np = feats.cpu().numpy().astype(np.float64)

            batch_n = feats_np.shape[0]
            n += batch_n

            if sum_ is None:
                sum_ = np.zeros(feats_np.shape[1], dtype=np.float64)
                sum_sq = np.zeros((feats_np.shape[1], feats_np.shape[1]), dtype=np.float64)

            sum_ += feats_np.sum(axis=0)
            sum_sq += feats_np.T @ feats_np

    mu = sum_ / n
    sigma = (sum_sq - n * np.outer(mu, mu)) / (n - 1)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


# -------------------- Main --------------------
if __name__ == "__main__":
    dataset_ImageNet = Dataset_ImageNet(preprocess=transform_real, split='train')
    dataset_syn = Dataset_Syn(preprocess=transform_syn, data_dir=args.dataset_dir)

    print("Number of synthetic images found:", len(dataset_syn))

    dloader_ImageNet = data.DataLoader(
        dataset_ImageNet, batch_size=args.batch_size,
        num_workers=4, shuffle=False
    )

    dloader_syn = data.DataLoader(
        dataset_syn, batch_size=args.batch_size,
        num_workers=4, shuffle=False
    )

    # -------------------- Inception-V3 --------------------

    weights = models.Inception_V3_Weights.IMAGENET1K_V1

    inception = models.inception_v3(
        weights=weights,
        aux_logits=True,          # REQUIRED by pretrained weights
        transform_input=False
    ).eval().cuda()

    # Extract Mixed_6e (TF equivalent: mixed_6/conv)
    return_nodes = {
        "Mixed_6e": "Mixed_6e"
    }

    model = create_feature_extractor(
        inception, return_nodes=return_nodes
    ).eval().cuda()

    # -------------------- sFID --------------------
    mu_real, sigma_real = calculate_activation_statistics(
        dloader_ImageNet, model, structured=True
    )
    mu_syn, sigma_syn = calculate_activation_statistics(
        dloader_syn, model, structured=True
    )

    sfid_value = calculate_frechet_distance(
        mu_real, sigma_real, mu_syn, sigma_syn
    )

    print("Structured FID (sFID) using Inception-V3 Mixed_6e:", sfid_value)
