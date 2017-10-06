import argparse
import os
import shutil
import time
import importlib

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from utils import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', default='',
                    help='path to dataset')
parser.add_argument('--source', metavar='DIR', default='source',
                    help='path to source dataset')
parser.add_argument('--target', metavar='DIR', default='target',
                    help='path to source dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--model', '-m', metavar='MODEL', default='DAN',
                    choices=['DAN', 'JAN', 'AJAN', 'GRL', 'JANA', 'CAN'])
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-c', '--classes', default=None, type=int, metavar='N',
                    help='number of classes (default: 12)')
parser.add_argument('-bc', '--bottleneck', default=256, type=int, metavar='N',
                    help='width of bottleneck')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--gpu', default='0', type=str, metavar='N',
                    help='visible gpu')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--gamma', default=0.001, type=float, metavar='M',
                    help='inv gamma')
parser.add_argument('--power', default=0.75, type=float, metavar='M',
                    help='inv power')
parser.add_argument('--alpha', default=1., type=float, metavar='M',
                    help='mmd loss weight')
parser.add_argument('--beta', default=.3, type=float, metavar='M',
                    help='cross entropy weight')
parser.add_argument('--gammaC', default=1., type=float, metavar='M',
                    help='C weight')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--train-iter', default=50000, type=int,
                    metavar='N', help='')
parser.add_argument('--test-iter', default=500, type=int,
                    metavar='N', help='')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--fromcaffe', dest='fromcaffe', action='store_true',
                    help='use caffe pre-trained model')

# For CAN
parser.add_argument('--ngf', type=int, default=64,
                    help='number of gen filters in first conv layer')
parser.add_argument('--ngblocks', type=int, default=9,
                    help='number of resnet blocks in gen net')
parser.add_argument('--ngdownsampling', type=int, default=2,
                    help='number of downsampling layers in gen net')
parser.add_argument('--ndf', type=int, default=64,
                    help='number of discrim filters in first conv layer')
parser.add_argument('--ndlayers', type=int, default=2,
                    help='number of layers in discrim net')
parser.add_argument('--no_dropout', action='store_true',
                    help='no dropout for gen net')
parser.add_argument('--norm', type=str, default='instance',
                    help='instance normalization or batch normalization')
parser.add_argument('--continue_train', action='store_true',
                    help='continue training: load the latest model')
parser.add_argument('--which_epoch', type=int, default=0,
                    help='which epoch to load? set to latest to use latest '
                         'cached model')
parser.add_argument('--prefix', type=str, default='./checkpoints',
                    help='models are saved here')
parser.add_argument('--pool_size', type=int, default=50,
                    help='the size of image buffer that stores '
                         'previously generated images')
parser.add_argument('--no_lsgan', action='store_true',
                    help='do *not* use least square GAN, if false, '
                         'use vanilla GAN')
parser.add_argument('--lambda_s', type=float, default=10.0,
                    help='weight for cycle loss (A -> B -> A)')
parser.add_argument('--lambda_t', type=float, default=10.0,
                    help='weight for cycle loss (B -> A -> B)')
parser.add_argument('--lambda_idt', type=float, default=0.0,
                    help='use identity mapping. Setting identity other than 1 '
                         'has an effect of scaling the weight of the identity '
                         'mapping loss. For example, if the weight of the '
                         'identity loss should be 10 times smaller than the '
                         'weight of the reconstruction loss, please set '
                         'optidentity = 0.1')
parser.add_argument('--display_freq', type=int, default=100,
                    help='frequency of showing training results on screen')
parser.add_argument('--save_epoch_freq', type=int, default=100,
                    help='frequency of saving checkpoints at the end of epochs')


def main():
    args = parser.parse_args()
    args.source = os.path.join(args.data, args.source)
    args.target = os.path.join(args.data, args.target)
    if not os.path.exists(args.prefix):
        os.makedirs(args.prefix)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    method = importlib.import_module('models.' + args.model)

    # create model
    model = method.Net(args).cuda()
    ### print(model)
    print(args)

    if hasattr(model, 'optimizer'):
        optimizer = model.optimizer
    else:
        optimizer = torch.optim.SGD([i.copy() for i in args.SGD_param], args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)#,
        # nesterov=False)

    cudnn.benchmark = True

    if hasattr(method, 'normalize'):
        normalize = method.normalize
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    source_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.source, transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True)

    target_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.target, transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.target, transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    method.train_val(source_loader, target_loader, val_loader,
                     model, optimizer, args)

if __name__ == '__main__':
    main()
