import numpy as np
import torch
import clip
from tqdm import tqdm
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from PIL import Image
from imagenet_info import return_classes, return_templates
import torchvision.models as models
import argparse

_IMAGENET_DATASET_DIR = '/scratch/inf0/user/mparcham/ILSVRC2012/'

parser = argparse.ArgumentParser(description='Synthetic Dataset Classification Accuracy')
parser.add_argument('--arch', type=str, default='resnet50', help='model architecture')
parser.add_argument('--clip_arch', type=str, default='ViT-B/32', help='CLIP model architecture')
parser.add_argument('--dataset_dir', type=str, default='/BS/feature_viz/work/results/ConvNext/MACO', help='synthetic dataset directory')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for data loader')
args = parser.parse_args()


transform_syn = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

transform_real = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def list_all_file_paths_from_filenames(directory):
    """
    Expected structure:
        directory/
            t1/
                0.png
                1.png
                ...
                999.png
            t2/
                0.png
                ...
            t3/
                0.png
                ...

    Label is extracted from filename:
        0.png -> label 0
        999.png -> label 999
    """
    file_paths = []
    labels = []

    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

    for root, dirs, files in os.walk(directory):
        for file in sorted(files):
            ext = os.path.splitext(file)[1].lower()
            if ext not in valid_exts:
                continue

            filename_without_ext = os.path.splitext(file)[0]

            if not filename_without_ext.isdigit():
                continue

            label = int(filename_without_ext)
            full_path = os.path.join(root, file)

            file_paths.append(full_path)
            labels.append(label)

    return file_paths, labels


class Dataset_ImageNet(data.Dataset):
    def __init__(self, preprocess, data_dir=_IMAGENET_DATASET_DIR, split='train'):
        self.preprocess = preprocess
        self.split = split

        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        split_dir = train_dir if ('train' in self.split) else val_dir

        self.data = datasets.ImageFolder(split_dir, transform=None)
        self.labels = [item[1] for item in self.data.imgs]

    def __getitem__(self, index):
        img, label = self.data[index]
        return self.preprocess(img), label

    def __len__(self):
        return len(self.data)


class Dataset_Syn(data.Dataset):
    def __init__(self, preprocess, data_dir):
        self.preprocess = preprocess
        self.data, self.labels = list_all_file_paths_from_filenames(data_dir)

        if len(self.data) == 0:
            raise ValueError(f'No valid images found in {data_dir}')

    def load_image(self, index):
        image_path = self.data[index]
        return Image.open(image_path).convert("RGB")

    def __getitem__(self, index):
        img = self.load_image(index)
        label = self.labels[index]
        return self.preprocess(img), label

    def __len__(self):
        return len(self.data)


def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames, desc='Building zeroshot classifier'):
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).cuda()
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def extract_top_acc(dloader, clip_model, zeroshot_weights, extractor=None, real_clean=False):
    with torch.no_grad():
        top1, top5, n = 0.0, 0.0, 0.0

        for images, target in tqdm(dloader, desc='Evaluating'):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            if real_clean:
                out = extractor(images)
                prediction_logit = out.max(1).indices
                matching_indices = (prediction_logit == target)

                images = images[matching_indices]
                target = target[matching_indices]

                if images.size(0) == 0:
                    continue

            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ zeroshot_weights

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    if n == 0:
        print('No samples were evaluated.')
        return

    top1 = (top1 / n) * 100.0
    top5 = (top5 / n) * 100.0

    print(f'Total evaluated samples: {int(n)}')
    print(f'Top-1 accuracy: {top1:.2f}')
    print(f'Top-5 accuracy: {top5:.2f}')


if __name__ == '__main__':
    clip_model, _ = clip.load(args.clip_arch)
    clip_model = clip_model.cuda().eval()

    dataset_ImageNet = Dataset_ImageNet(preprocess=transform_real, split='train')
    dataset_syn = Dataset_Syn(preprocess=transform_syn, data_dir=args.dataset_dir)

    batch_size = args.batch_size

    dloader_ImageNet = torch.utils.data.DataLoader(
        dataset_ImageNet,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    dloader_syn = torch.utils.data.DataLoader(
        dataset_syn,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    imagenet_classes = return_classes()
    imagenet_templates = return_templates()

    input_resolution = clip_model.visual.input_resolution
    context_length = clip_model.context_length
    vocab_size = clip_model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    print('\nLoading zeroshot classifier...')
    zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates, clip_model)

    extractor = models.__dict__[args.arch](pretrained=True).eval().cuda()

    print('\nExtract CLIP zeroshot accuracy on ImageNet')
    extract_top_acc(
        dloader=dloader_ImageNet,
        clip_model=clip_model,
        zeroshot_weights=zeroshot_weights,
        extractor=extractor,
        real_clean=True
    )

    print('\nExtract CLIP zeroshot accuracy on synthetic dataset')
    extract_top_acc(
        dloader=dloader_syn,
        clip_model=clip_model,
        zeroshot_weights=zeroshot_weights,
        extractor=None,
        real_clean=False
    )