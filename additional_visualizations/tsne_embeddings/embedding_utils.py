import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
from PIL import Image
import torchvision.transforms as transforms
import torch
import pandas as pd
import torchvision.models as models
import random

transform_normalize = transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
 )

transform_real = transforms.Compose(
                [transforms.Resize(256),
                 transforms.CenterCrop(224),
                 transforms.PILToTensor(),
                 transforms.ConvertImageDtype(torch.float),
                 ])

transform_syn = transforms.Compose(
                [transforms.Resize(224),
                 transforms.PILToTensor(),
                 transforms.ConvertImageDtype(torch.float),
                 ])

def split_network(model, name):
    """
    This function splits the network in Feature extractor and classifier networks.
    extract_from_layer: list of layer names from which features are to be extracted.
    after_relu: list of booleans, True means use relu after the extraction layer specified in extract_from_layer list.
    """
    feat_ext = []
    key_names = list(model._modules.keys())
    extract_from = key_names.index(name)
    for i in range(extract_from+1): # Feature extractor
        feat_ext.append(model._modules[key_names[i]])
    return torch.nn.Sequential(*feat_ext)

def extract_features(img, arch, layer_name):

    model = models.__dict__[arch](pretrained=True)
    model = model.to('cuda')
    
    model = split_network(model, layer_name)
    model.eval()

    with torch.no_grad():

        img = img.to('cuda')

        features = model(img)

    return features

def calculate_cross_entropy_loss(img, targets, arch):

    model = models.__dict__[arch](pretrained=True)
    model = model.to('cuda')
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    targets = torch.tensor(targets)

    with torch.no_grad():

        img = img.to('cuda')
        targets = targets.to('cuda')

        outputs = model(img)

        loss = criterion(outputs, targets)

    return loss

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

def visualizeEmbeddingPoints(tsne, y_test_patches, colors_per_class, filename):
    
    
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.gca().set_axis_off()
    
    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:

        if label-10000<10000 and label-10000>=0:
            marker='*'
            alpha = 1
            s = 15

        elif label-10000<20000 and label-10000>=10000:
            marker='s'
            alpha = 1
            s = 20
            
        elif label-10000<30000 and label-10000>=20000:
            marker='H'
            alpha = 1
            s = 20
            
        elif label-10000<40000 and label-10000>=30000:
            marker='^'
            alpha = 1
            s = 15
        else:
            marker='o'
            alpha = 0.05
            s = 20
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(y_test_patches) if l == label]
    
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
    
        # convert the class color to matplotlib format
        color = np.array(colors_per_class[label], dtype=float) / 255
    
        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, c=color.reshape(1,-1), marker=marker, s = s, label=label, alpha=alpha, edgecolors =color.reshape(1,-1))
    
    # build a legend using the labels we set previously
    # ax.legend(loc='best')
    
    # finally, show the plot
    plt.show()
    
    # filename = os.path.join(visualize_dir, '%s.png' % name)
    
    fig.savefig(filename, dpi=300)

def obtain_num_images(class_number, num_img):

    IMAGENET_DIR = "/BS/Databases23/imagenet/original/train/"
    images_preprocessed = []
    labels = []

    for cls in class_number:

        class_path = sorted(os.listdir(IMAGENET_DIR))[cls]
        class_files = os.listdir(os.path.join(IMAGENET_DIR, class_path))
        random.shuffle(class_files)
        inc_img = 0
        labels += [cls]*num_img

        for file in class_files:
            if inc_img == num_img:
                break
            img = Image.open(os.path.join(IMAGENET_DIR, class_path, file))
            num_channel = len(img.split())
            if num_channel != 3:
                continue
            else:
                inc_img += 1
            img = transform_real(img)
            img_p = transform_normalize(img)
            images_preprocessed.append(img_p)

    images_preprocessed = torch.stack(images_preprocessed, 0)
    labels = torch.tensor(labels)
    return images_preprocessed, labels

def load_image(filename, isSyn=True, debug=False):

    img = Image.open(filename)

    if img.mode == "RGBA":
        img = img.convert('RGB')

    
    if debug:
        plt.imshow(img)
        plt.axis('off')

    if isSyn:
        img = transform_syn(img)
        img_p = transform_normalize(img)
    else:
        img = transform_real(img)
        img_p = transform_normalize(img)

    return img, img_p[None]

def feat_histogram_all(activations, channel):

    activations = activations.cpu().numpy()
    activations = activations[:, channel,:,:].reshape(activations.shape[0], -1)
    df_merged = None

    for i in range(activations.shape[0]):
        df = pd.DataFrame(activations[i])

        if df_merged is None:

            df_merged = pd.concat([df], ignore_index=True)
        else:
            df_merged = pd.concat([df_merged, df], ignore_index=True)

    # Get the frequencies of the combined histogram
    hist, bins = np.histogram(df_merged, bins=100)
    # Normalize
    hist_norm = hist / activations.shape[0]

    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2

    plt.bar(center, hist_norm, align='center', width=width)
    plt.grid()
    plt.xlabel('activations')
    plt.ylabel('freq')
        
def feat_histogram(activations, channel):

    activations = activations.cpu().numpy()
    activations = activations[:,channel,:,:].reshape(-1)

    plt.hist(activations, bins=100)

    plt.grid()
    plt.xlabel('activations')
    plt.ylabel('freq')


def efdm_loss(input, target):

    B, C, W, H = target.size(0), target.size(1), target.size(2), target.size(3)
    
    _, index_content = torch.sort(input.view(C, -1))
    inverse_index = index_content.argsort(-1)

    value_style, _ = torch.sort(target.view(B, C, -1))

    mse_loss = (input.view(C, -1) - torch.take_along_dim(value_style, inverse_index.unsqueeze(0), dim=-1))**2

    return mse_loss.mean()
