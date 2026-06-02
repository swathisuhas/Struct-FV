from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import argparse
import torch
import os
import torchvision.models as models
from utils.utils import create_folder
from utils.opt_concept_graph import DeepFeaturesClass

def list_of_ints(arg):
    return list(map(int, arg.split(',')))
def list_of_floats(arg):
    return list(map(float, arg.split(',')))

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    print("loading torchvision model for inversion with the name: {}".format(args.arch_name))
    net = models.__dict__[args.arch_name](pretrained=True)
    net = net.to(device)
    print('==> Resuming from checkpoint..')
    net.eval()

    # temporal data and generations will be stored here
    create_folder(args.folder_name)

    parameters = dict()
    parameters["resolution"] = args.resolution
    parameters["do_flip"] = args.do_flip
    parameters["jitter"] = args.jitter
    parameters["bs"] = args.bs
    parameters["num_real_img"] = args.num_real_img
    parameters["epochs"] = args.epochs
    parameters["layer"] = args.layer
    parameters["concept"] = args.concept
    parameters["method"] = args.method
    parameters['arch_name'] = args.arch_name

    coefficients = dict()
    coefficients["tv_l1"] = args.tv_l1
    coefficients["tv_l2"] = args.tv_l2
    coefficients["l2"] = args.l2
    coefficients["lr"] = args.lr
    coefficients["feat_dist"] = args.feat_dist
    coefficients["layer_weights"] = args.layer_weights
    coefficients["graph_lambda"] = args.graph_lambda
    coefficients["graph_K"] = args.graph_K
    coefficients["graph_num_refs"] = args.graph_num_refs

    DeepFeatureEngine = DeepFeaturesClass(model=net,
                                          parameters=parameters,
                                          coefficients=coefficients,
                                          exp_name=args.exp_name,
                                          folder_name=args.folder_name)

    DeepFeatureEngine.get_images(targets=args.target)


def main():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--gpuid', type=str, default='0', help='gpu id')
    parser.add_argument('--target', type=int, default=281, help='class for concept visualization')
    parser.add_argument('--arch_name', default='resnet50', type=str, help='model name from torchvision')
    parser.add_argument('--layer', type=str, default='layer4_2', help='layer of the visualized neuron')
    parser.add_argument('--layer_weights', type=list_of_floats, default="0.1, 1, 1, 1, 0", help='contribution of each layer to the sort-matching loss')
    parser.add_argument('--feat_dist', type=float, default=1.0, help='contribution of the total sort-matching loss')
    parser.add_argument('--num_real_img', type=int, default=50, help='number of reference images')
    parser.add_argument('--concept', type=int, default=1, help='concept id to visualize')

    parser.add_argument('--method', type=str, default='LRP', choices=['LRP', 'GuidedBackprop'], help='attribution method')
    
    parser.add_argument('--resolution', type=int, default=224, help='resolution of the optimized image')
    parser.add_argument('--epochs', default=2000, type=int, help='number of iterations')
    parser.add_argument('--bs', default=1, type=int, help='batch size')
    parser.add_argument('--jitter', default=30, type=int, help='jitter parameter')
    
    parser.add_argument('--exp_name', type=str, default=None, help='where to store experimental data-name of the image file to save')
    parser.add_argument('--folder_name', type=str, default='images', help='where to store experimental data-name of the folder to save')
    
    parser.add_argument('--do_flip', action='store_true', help='apply flip during model inversion')
    parser.add_argument('--tv_l1', type=float, default=0.0, help='coefficient for total variation L1 loss')
    parser.add_argument('--tv_l2', type=float, default=0.0000003, help='coefficient for total variation L2 loss')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate for optimization')
    parser.add_argument('--l2', type=float, default=0.0000003, help='l2 loss on the image')

    parser.add_argument('--graph_lambda', type=float, default=0.01, help='weight for graph matching loss')
    parser.add_argument('--graph_K', type=int, default=10, help='number of graph nodes')
    parser.add_argument('--graph_num_refs', type=int, default=8, help='number of reference images sampled per iteration')
        
    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

    torch.backends.cudnn.benchmark = True
    run(args)


if __name__ == '__main__':
    main()
