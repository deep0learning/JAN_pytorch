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
from PIL import Image, ImageOps
import numpy as np

from utils import *
from mysgd import SGD

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--model', '-m', metavar='MODEL', default='DAN',
                    choices=['DAN', 'JAN', 'AJAN', 'GRL', 'JANA', 'CAN', 'CAN0'])
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-c', '--classes', default=None, type=int, metavar='N',
                    help='number of classes (default: 12)')
parser.add_argument('-bc', '--bottleneck', default=256, type=int, metavar='N',
                    help='width of bottleneck (default: 256)')
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

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    method = importlib.import_module('models.' + args.model)

    # create model
    model = method.Net(args).cuda()
    ### print(model)
    print(args)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD([i.copy() for i in args.SGD_param], args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)#,
    # nesterov=False)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'validation')

    traindir = '/home/dataset/office/domain_adaptation_images/amazon/images'
    valdir = '/home/dataset/office/domain_adaptation_images/webcam/images'
    
    # TODO: For debug
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    class MyScale(object):
        def __init__(self, size, interpolation=Image.BILINEAR):
            self.size = size
            self.interpolation = interpolation

        def __call__(self, img):
            if isinstance(self.size, int):
                w, h = img.size
                if (w <= h and w == self.size) or (h <= w and h == self.size):
                    return img
                if w < h:
                    ow = self.size
                    oh = int(self.size * h / w)
                    return img.resize((ow, oh), self.interpolation)
                else:
                    oh = self.size
                    ow = int(self.size * w / h)
                    return img.resize((ow, oh), self.interpolation)
            else:
                return img.resize(self.size)#, self.interpolation)

    source_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            MyScale((256, 256)),
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    target_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            MyScale((256, 256)),
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            MyScale((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=4, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    val_source_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            MyScale((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=4, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    method.train_val(source_loader, target_loader, val_loader, val_source_loader,
                     model, criterion, optimizer, args)


if __name__ == '__main__':
    main()
