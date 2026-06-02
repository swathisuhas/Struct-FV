import torch
from tqdm import tqdm
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import re
from PIL import Image
import torchvision.models as models
import argparse

parser = argparse.ArgumentParser(description='Synthetic Dataset Classification Accuracy')
parser.add_argument('--arch', type=str, default='resnet50', help='model architecture')
parser.add_argument('--dataset_dir', type=str, default='/BS/feature_viz/work/results/ConvNext/VITAL/', help='dataset directory')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for data loader')
args = parser.parse_args()

transform_syn = transforms.Compose([
 transforms.Resize(224),
 transforms.ToTensor(),
 transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225])
])

transform_real = transforms.Compose([
 transforms.Resize(256),
 transforms.CenterCrop(224),
 transforms.ToTensor(),
 transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225])
])


def extract_number_after_cls(input_string):
    # Use regex to search for 'cls' followed by digits
    match = re.search(r'cls(\d+)', input_string)
    if match:
        # Return the extracted number as an integer
        return int(match.group(1))
    else:
        return None  # Return None if no match is found
    
    
def list_full_paths(directory):
    # List to store full paths
    full_paths = []
    
    # Iterate over all files and subdirectories in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Get full path of file and add to the list
            full_paths.append(os.path.join(root, file))
    
    return full_paths


def list_all_file_paths(directory):
    # List to store full file paths
    file_paths = []
    labels = []
    
    # Iterate over all files and subdirectories in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Get the full path of the file
            full_path = os.path.join(root, file)
            file_paths.append(full_path)
            labels.append(extract_number_after_cls(full_path))
    
    return file_paths, labels


def list_all_file_paths_traditional(directory):
    # List to store full file paths
    file_paths = []
    labels = []
    
    # Iterate over all files and subdirectories in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Get the full path of the file
            full_path = os.path.join(root, file)
            file_paths.append(full_path)
            filename_without_ext = os.path.splitext(file)[0]
            labels.append(int(filename_without_ext))
    
    return file_paths, labels


class Dataset_Syn(data.Dataset):
    def __init__(
        self,
        preprocess,
        data_dir):
        
        self.preprocess = preprocess

        self.data, self.labels = list_all_file_paths(data_dir)
        
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
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
     
     
def extract_top_acc(dloader, real_clean=False):
    
    with torch.no_grad():
        top1, top3, n = 0., 0., 0.
        for i, (images, target) in enumerate(tqdm(dloader)):
            images = images.cuda()
            target = target.cuda()
            
            logits = model(images)
            
            if real_clean:
            
                prediction_logit = logits.max(1).indices

                matching_indices = (prediction_logit == target)

                images = images[matching_indices]
                target = target[matching_indices]
                logits = logits[matching_indices]

            # measure accuracy
            acc1, acc3 = accuracy(logits, target, topk=(1, 3))
            top1 += acc1
            top3 += acc3
            n += images.size(0)

    top1 = (top1 / n) * 100
    top3 = (top3 / n) * 100   
    
    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-3 accuracy: {top3:.2f}")
    
if __name__ == '__main__':
    
    # Load the synthetic dataset
    print('Loading synthetic dataset')
    
    # Create a dataset object
    dataset_syn = Dataset_Syn(preprocess=transform_syn, data_dir=args.dataset_dir)

    batch_size = args.batch_size

    dloader_syn = torch.utils.data.DataLoader(
                                dataset_syn,
                                batch_size=batch_size, # may need to reduce this depending on your GPU 
                                num_workers=4, # may need to reduce this depending on your num of CPUs and RAM
                                shuffle=False,
                                drop_last=False,
                                pin_memory=False
                                    )


    model = models.__dict__[args.arch](pretrained=True)
    model = model.eval()
    model = model.cuda()

    print('Synthetic Accuracy')

    extract_top_acc(dloader=dloader_syn)

