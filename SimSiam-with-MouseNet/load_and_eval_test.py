#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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

import pandas as pd

class Args_LCN:
   def __init__(self, task):
      self.n_first_conv = 0
      self.conv_1x1 = False
      self.freeze_1x1 = False
      self.locally_connected_deviation_eps = -1 # random; 0 for convolutional init
      self.task = task
      self.input_scale = 1
      # added dataset input_size = [64, 64] and n_classes = 1000 

# mousenet stuff

import sys
sys.path.append('/nfs/nhome/live/ammarica/SimSiam-with-MouseNet/simsiam')
sys.path.append('/nfs/nhome/live/ammarica/SimSiam-with-MouseNet/simsiam/mouse_cnn')

import network
from mousenet_complete_pool import MouseNetCompletePool

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
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4096, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
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

# for dataset name
parser.add_argument('--dataset_name', default='imagenet', type=str, help='name of dataset')

# for validation
parser.add_argument('--split_seed', default=0, type=float,
                    help='train validation split seed (default: 0)')
parser.add_argument('--val_size', default=10000, type=float,
                    help='size of validation set (default: 10000)')

# for test
parser.add_argument('--test', dest='test', action='store_true',
                    help='evaluate model on test set')

# added to make network wider
parser.add_argument('--scale', default = 1, type=float, help='scale network width')

# stop at self-sup value
parser.add_argument('--self_sup_acc', default=None, type=float, help='save checkpoint at this value')
parser.add_argument('--self_sup_stop_train', default=None, help='stop training after saving checkpoint')

# locally connected stuff
parser.add_argument('--is_LCN', default=None, help='locally connected network')

# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to simsiam pretrained checkpoint')
parser.add_argument('--lars', action='store_true',
                    help='Use LARS')

# list of checkpoint paths
parser.add_argument('--list_pretrained', default='', type=str,
                    help='list of paths to pretrained checkpoints')
parser.add_argument('--save_eval_results', default='', type=str,
                    help='path to results to evaluation')

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    # create model

    all_acc1 = []

    # load from pre-trained, before DistributedDataParallel constructor
    read_df = pd.read_csv(args.list_pretrained)
    for i in range(read_df.shape[0]):
        this_pretrained = read_df.iloc[i]
        this_name = this_pretrained['name']
        this_width = this_pretrained['width']
        this_dataset = this_pretrained['dataset']
        this_model = this_name + ' ' + str(this_width) + ' width'
        this_path = this_pretrained['checkpoint']
        this_checkpoint = f'{this_path}/model_best.pth.tar'

            # create model
        print("=> creating model '{}'".format(args.arch))

        # get_number_classes
        if this_dataset == 'imagenet':
            n_classes = 1000
        if this_dataset == 'ecoset':
            n_classes = 565

        if args.arch == 'mouse_net':
            net = network.load_network_from_pickle('/nfs/nhome/live/ammarica/SimSiam-with-MouseNet/simsiam/network_complete_updated_number(3,64,64).pkl')
            model = MouseNetCompletePool(num_classes=n_classes,this_net=net, mask=args.mask, scale=this_width)
            # can do this because they have matching out and in dims      
            new_fc = model.fc[-1]
            model.fc = new_fc
        else:
            model = models.__dict__[args.arch]()

        # # freeze all layers but the last fc
        # for name, param in model.named_parameters():
        #     if name not in ['fc.weight', 'fc.bias']:
        #         param.requires_grad = False
                
        # init the fc layer - added try for mouse_net
        # try:
        #     model.fc[-1].weight.data.normal_(mean=0.0, std=0.01)
        #     model.fc[-1].bias.data.zero_()
        # except:
        #     model.fc.weight.data.normal_(mean=0.0, std=0.01)
        #     model.fc.bias.data.zero_()

        try:
            model.fc[-1].weight.data.fill_(float('nan'))
            model.fc[-1].bias.data.fill_(float('nan'))
        except:
            model.fc.weight.data.fill_(float('nan'))
            model.fc.bias.data.fill_(float('nan'))

        if os.path.isfile(this_checkpoint):
            
            print("=> loading checkpoint '{}'".format(this_model))
            checkpoint = torch.load(this_checkpoint, map_location="cpu")
            
            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']

            print('state_dict: ', list(state_dict.keys()))

            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith('module.'): # and not k.startswith('module.encoder.fc'):
                    # remove prefix
                    state_dict[k[len("module."):]] = state_dict[k]
                # delete renamed or unused k
                # del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            print(set(msg.missing_keys))

            # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(this_checkpoint))
        else:
            print("=> no checkpoint found at '{}'".format(this_checkpoint))

        # infer learning rate before changing batch size
        init_lr = args.lr * args.batch_size / 256

        if args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features)
                model.cuda()
            else:
                model = torch.nn.DataParallel(model).cuda()

        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

        # # optimize only the linear classifier
        # parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        # print(parameters)
        # assert len(parameters) == 2  # fc.weight, fc.bias
        
        parameters = model.parameters()

        optimizer = torch.optim.SGD(parameters, init_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        if args.lars:
            print("=> use LARS optimizer.")
            from apex.parallel.LARC import LARC
            optimizer = LARC(optimizer=optimizer, trust_coefficient=.001, clip=False)

        cudnn.benchmark = True

        # Data loading code
        if this_dataset == 'imagenet':
            args.data = '/tmp/roman/imagenet'
            traindir = os.path.join(args.data, 'train')
            testdir = os.path.join(args.data, 'val')
        if this_dataset == 'ecoset':
            args.data = '/tmp/andrada/ecoset'
            traindir = os.path.join(args.data, 'train')
            valdir = os.path.join(args.data, 'val')
            testdir = os.path.join(args.data, 'test')
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        transforms_train = [
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        transforms_test = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize(64),
                transforms.ToTensor(),
                normalize,
            ]

        # changed to 64 for mousenet training
        train_dataset = datasets.ImageFolder(
            traindir, transforms.Compose(transforms_train))

        if this_dataset == 'imagenet':
            train_dataset = torch.utils.data.random_split(
                train_dataset, [len(train_dataset) - args.val_size, args.val_size],
                generator=torch.Generator().manual_seed(args.split_seed))[0]

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    
        # validation set
        if this_dataset == 'imagenet':
            val_dataset = datasets.ImageFolder(traindir, transforms.Compose(transforms_test))
            
            val_dataset = torch.utils.data.random_split(
                    val_dataset, [len(val_dataset) - args.val_size, args.val_size],
                    generator=torch.Generator().manual_seed(args.split_seed))[1]
        elif this_dataset == 'ecoset':
            val_dataset = datasets.ImageFolder(valdir, transforms.Compose(transforms_test))

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=256, shuffle=False,
            num_workers=args.workers, pin_memory=True)    

        if args.evaluate:
            validate(val_loader, model, criterion, args)
            return
        
        # test stuff    
        test_dataset = datasets.ImageFolder(testdir, transforms.Compose(transforms_test))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=256, shuffle=False,
            num_workers=args.workers, pin_memory=True)  

        if args.test:
            acc1 = validate(test_loader, model, criterion, args)
            var1 = acc1.cpu().detach().numpy()
            all_acc1.append(var1)

    tmp_df = read_df    
    print(all_acc1)
    tmp_df['acc1'] = all_acc1 
    tmp_df.to_csv(args.save_eval_results)




def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best=None, is_self_sup=None, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
    if is_self_sup:
        shutil.copyfile(filename, 'model_self_sup.pth.tar')


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
