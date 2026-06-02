import torch
from tqdm import tqdm
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
import torchvision.models as models
import argparse

parser = argparse.ArgumentParser(description='Synthetic Dataset Classification Accuracy')
parser.add_argument('--arch', type=str, default='resnet50', help='model architecture')
parser.add_argument(
    '--dataset_dir',
    type=str,
    default='/BS/feature_viz/work/results/ConvNext/MACO',
    help='dataset directory'
)
parser.add_argument('--batch_size', type=int, default=128, help='batch size for data loader')
parser.add_argument('--num_workers', type=int, default=4, help='number of dataloader workers')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_syn = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def list_all_file_paths_from_filenames(directory):
    """
    Expected structure:
        dataset_dir/
            t1/
                0.png ... 999.png
            t2/
                0.png ... 999.png
            t3/
                0.png ... 999.png

    Label is extracted from filename:
        0.png   -> label 0
        999.png -> label 999
    """
    file_paths = []
    labels = []
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

    for root, _, files in os.walk(directory):
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


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def extract_top_acc(dloader, model):
    with torch.no_grad():
        top1, top3, n = 0.0, 0.0, 0.0

        for images, target in tqdm(dloader, desc='Evaluating'):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            logits = model(images)

            acc1, acc3 = accuracy(logits, target, topk=(1, 3))
            top1 += acc1
            top3 += acc3
            n += images.size(0)

    if n == 0:
        print('No samples were evaluated.')
        return

    top1 = (top1 / n) * 100.0
    top3 = (top3 / n) * 100.0

    print(f'Total evaluated samples: {int(n)}')
    print(f'Top-1 accuracy: {top1:.2f}')
    print(f'Top-3 accuracy: {top3:.2f}')


if __name__ == '__main__':
    print('Loading synthetic dataset')
    dataset_syn = Dataset_Syn(preprocess=transform_syn, data_dir=args.dataset_dir)

    print(f'Number of synthetic images found: {len(dataset_syn)}')

    dloader_syn = torch.utils.data.DataLoader(
        dataset_syn,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    model = models.__dict__[args.arch](pretrained=True)
    model = model.eval().to(device)

    print('Synthetic Accuracy')
    extract_top_acc(dloader=dloader_syn, model=model)