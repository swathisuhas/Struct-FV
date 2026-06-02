import numpy as np
import torch
from tqdm import tqdm
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from PIL import Image
import torchvision.models as models
from scipy import linalg
from torchvision.models.feature_extraction import create_feature_extractor
from torch.nn.functional import adaptive_avg_pool2d
import argparse

_IMAGENET_DATASET_DIR = "/BS/Databases23/imagenet/original/"

parser = argparse.ArgumentParser(description='sFID Computation (InceptionV3 Mixed_6e)')
parser.add_argument(
    '--dataset_dir',
    type=str,
    default='/BS/feature_viz/work/results/ConvNext/MACO/',
    help='synthetic dataset directory'
)
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=4, help='number of dataloader workers')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -------------------- Transforms --------------------
# Inception-V3 input size: 299 x 299
transform_syn = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

transform_real = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------- Dataset helpers --------------------
def list_all_image_paths(directory):
    """
    Expected synthetic dataset structure:
        directory/
            t1/
                0.png
                1.png
                ...
                999.png
            t2/
                0.png
                1.png
                ...
                999.png
            t3/
                0.png
                1.png
                ...
                999.png

    Labels are not needed for sFID, so we only collect image paths.
    """
    file_paths = []
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

    for root, _, files in os.walk(directory):
        for file in sorted(files):
            ext = os.path.splitext(file)[1].lower()
            if ext not in valid_exts:
                continue
            file_paths.append(os.path.join(root, file))

    return file_paths


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
        self.data = list_all_image_paths(data_dir)

        if len(self.data) == 0:
            raise ValueError(f'No valid images found in {data_dir}')

    def __getitem__(self, index):
        img = Image.open(self.data[index]).convert("RGB")
        return self.preprocess(img), 0  # dummy label, unused

    def __len__(self):
        return len(self.data)


# -------------------- Feature extraction --------------------
def extract_features_from_batch(feats, structured=True):
    """
    feats: [B, C, H, W]

    structured=True  -> [B*H*W, C]   for sFID
    structured=False -> [B, C]       for regular FID-like pooled features
    """
    if structured:
        b, c, h, w = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(b * h * w, c)
    else:
        feats = adaptive_avg_pool2d(feats, output_size=(1, 1)).squeeze(3).squeeze(2)

    return feats


def calculate_activation_statistics(dataloader, model, structured=True):
    model.eval()

    sum_ = None
    sum_sq = None
    n = 0

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc='Extracting features'):
            images = images.to(device, non_blocking=True)
            out = model(images)
            feats = out["Mixed_6e"]  # Inception-V3 intermediate feature map

            feats = extract_features_from_batch(feats, structured=structured)
            feats_np = feats.cpu().numpy().astype(np.float64)

            batch_n = feats_np.shape[0]
            n += batch_n

            if sum_ is None:
                feat_dim = feats_np.shape[1]
                sum_ = np.zeros(feat_dim, dtype=np.float64)
                sum_sq = np.zeros((feat_dim, feat_dim), dtype=np.float64)

            sum_ += feats_np.sum(axis=0)
            sum_sq += feats_np.T @ feats_np

    if n < 2:
        raise ValueError('Need at least 2 feature vectors to compute covariance.')

    mu = sum_ / n
    sigma = (sum_sq - n * np.outer(mu, mu)) / (n - 1)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariances have different dimensions"

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Imaginary component in sqrtm: {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real

    return diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


# -------------------- Main --------------------
if __name__ == "__main__":
    dataset_ImageNet = Dataset_ImageNet(preprocess=transform_real, split='train')
    dataset_syn = Dataset_Syn(preprocess=transform_syn, data_dir=args.dataset_dir)

    print("Number of ImageNet images found:", len(dataset_ImageNet))
    print("Number of synthetic images found:", len(dataset_syn))

    dloader_ImageNet = data.DataLoader(
        dataset_ImageNet,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    dloader_syn = data.DataLoader(
        dataset_syn,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    weights = models.Inception_V3_Weights.IMAGENET1K_V1

    inception = models.inception_v3(
        weights=weights,
        aux_logits=True,
        transform_input=False
    ).to(device).eval()

    return_nodes = {
        "Mixed_6e": "Mixed_6e"
    }

    model = create_feature_extractor(
        inception,
        return_nodes=return_nodes
    ).to(device).eval()

    print("Calculating ImageNet statistics...")
    mu_real, sigma_real = calculate_activation_statistics(
        dloader_ImageNet,
        model,
        structured=True
    )

    print("Calculating synthetic dataset statistics...")
    mu_syn, sigma_syn = calculate_activation_statistics(
        dloader_syn,
        model,
        structured=True
    )

    sfid_value = calculate_frechet_distance(
        mu_real,
        sigma_real,
        mu_syn,
        sigma_syn
    )

    print("Structured FID (sFID) using Inception-V3 Mixed_6e:")
    print(sfid_value)