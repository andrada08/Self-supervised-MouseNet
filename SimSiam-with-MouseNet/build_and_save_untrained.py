import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import simsiam.loader
import simsiam.builder

# mousenet stuff

import sys
sys.path.append('/nfs/nhome/live/ammarica/SimSiam-with-MouseNet/simsiam')
sys.path.append('/nfs/nhome/live/ammarica/SimSiam-with-MouseNet/simsiam/mouse_cnn')

import network
from mousenet_complete_pool import MouseNetCompletePool

# save directory

#save_dir = '/nfs/gatsbystor/ammarica/SimSiam-with-MouseNet/Checkpoints/'

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', dir='/nfs/gatsbystor/ammarica/SimSiam-with-MouseNet/Checkpoints/'):
    path = os.path.join(dir, filename)
    torch.save(state, path)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

        

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
# from MouseNet
parser.add_argument('--mask', default = 3, type=int, help='if use Gaussian mask')

# for saving different checkpoints
parser.add_argument('--save_dir', default = '/nfs/gatsbystor/ammarica/SimSiam-with-MouseNet/Checkpoints/', type=str, help='save directory for checkpoints')

# added to make network wider
parser.add_argument('--scale', default = 1, type=float, help='scale network width')

# simsiam specific configs:
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')


assert torch.cuda.is_available()
args = parser.parse_args()

# create model
print("=> creating model '{}'".format(args.arch))

if args.arch == 'mouse_net':
    net = network.load_network_from_pickle('/nfs/nhome/live/ammarica/SimSiam-with-MouseNet/simsiam/network_complete_updated_number(3,64,64).pkl')
    def get_mousenet(num_classes, zero_init_residual=True):
        return MouseNetCompletePool(num_classes, this_net=net, mask=args.mask, scale=args.scale)
    model = simsiam.builder.SimSiam(
        get_mousenet,
        args.dim, args.pred_dim, args.mask)
else:
    model = simsiam.builder.SimSiam(
        models.__dict__[args.arch],
        args.dim, args.pred_dim, args.mask)


save_checkpoint({
        #'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        #'optimizer' : optimizer.state_dict(),
    }, is_best=False, filename='untrained.pth.tar', dir=args.save_dir)

print('Saved')

    
