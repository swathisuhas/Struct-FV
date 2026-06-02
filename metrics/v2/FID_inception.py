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

_IMAGENET_DATASET_DIR = '/scratch/inf0/user/mparcham/ILSVRC2012/'

parser = argparse.ArgumentParser(description='Synthetic Dataset FID (InceptionV3 universal)')
parser.add_argument(
    '--dataset_dir',
    type=str,
    default='/BS/feature_viz/work/results/convnext_graphMatching/convnext',
    help='synthetic dataset directory'
)
parser.add_argument('--batch_size', type=int, default=128, help='batch size for data loader')
parser.add_argument('--num_workers', type=int, default=4, help='num dataloader workers')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# "Universal" / standard FID preprocessing uses InceptionV3 and 299x299
# Most common practice: resize -> center crop -> normalize with ImageNet mean/std.
transform_fid = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def extract_number_after_cls(input_string):
    match = re.search(r'cls(\d+)', input_string)
    return int(match.group(1)) if match else None


def list_all_file_paths(directory):
    file_paths, labels = [], []
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            file_paths.append(full_path)
            labels.append(extract_number_after_cls(full_path))
    return file_paths, labels


class Dataset_ImageNet(data.Dataset):
    def __init__(self, preprocess, data_dir=_IMAGENET_DATASET_DIR, split='train'):
        self.preprocess = preprocess
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        split_dir = train_dir if (split.find('train') != -1) else val_dir
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

    def load_image(self, index):
        return Image.open(self.data[index]).convert("RGB")

    def __getitem__(self, index):
        img = self.load_image(index)
        label = self.labels[index]
        return self.preprocess(img), label

    def __len__(self):
        return len(self.data)


def get_inception_pool3_activations(model, dataloader):
    feats = []
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            images = images.to(device, non_blocking=True)
            out = model(images)              # dict with our return node
            x = out["pool3"]                 # [N, 2048, 1, 1] in torchvision inception_v3
            if x.dim() == 4:
                x = adaptive_avg_pool2d(x, (1, 1))
                x = x.squeeze(-1).squeeze(-1)  # [N, 2048]
            feats.append(x.cpu().numpy())
    return np.concatenate(feats, axis=0)


def calculate_activation_statistics(dataloader, inception_pool3_extractor):
    act = get_inception_pool3_activations(inception_pool3_extractor, dataloader)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariances have different dimensions"

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Imaginary component in sqrtm: {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean


if __name__ == '__main__':
    # Datasets use the SAME FID transform for both real and synthetic
    dataset_ImageNet = Dataset_ImageNet(preprocess=transform_fid, split='train')
    dataset_syn = Dataset_Syn(preprocess=transform_fid, data_dir=args.dataset_dir)

    dloader_ImageNet = torch.utils.data.DataLoader(
        dataset_ImageNet,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    dloader_syn = torch.utils.data.DataLoader(
        dataset_syn,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    # Universal FID backbone: InceptionV3
    # IMPORTANT CHANGE:
    # Do NOT pass aux_logits when using pretrained/weights, torchvision enforces aux_logits=True.
    inception = models.inception_v3(
        weights=models.Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False
    ).to(device).eval()

    # Extract the last pooling layer ("pool3"): avgpool output
    # In torchvision InceptionV3, 'avgpool' is AdaptiveAvgPool2d((1,1)) right before dropout/fc.
    return_nodes = {"avgpool": "pool3"}
    inception_pool3 = create_feature_extractor(inception, return_nodes=return_nodes).to(device).eval()

    m_real, s_real = calculate_activation_statistics(dloader_ImageNet, inception_pool3)
    m_syn, s_syn = calculate_activation_statistics(dloader_syn, inception_pool3)

    fid_value = calculate_frechet_distance(m_real, s_real, m_syn, s_syn)

    print("FID score (InceptionV3 pool3, 2048-d)")
    print(fid_value)