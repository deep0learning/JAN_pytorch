import argparse
import pdb
import os
import shutil
import time
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import math

from losses import *
from utils import *

### Convert back-bone model
class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        # create model
        if args.fromcaffe:
            print("=> using pre-trained model from caffe '{}'".format(args.arch))
            import models.caffe_resnet as resnet
            model = resnet.__dict__[args.arch]()
            state_dict = torch.load("models/"+args.arch+".pth")
            model.load_state_dict(state_dict)
        elif args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()

        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            self.feature_dim = model.classifier[6].in_features
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        elif args.arch.startswith('densenet'):
            self.feature_dim = model.classifier.in_features
            model = nn.Sequential(*list(model.children())[:-1])
        else:
            self.feature_dim = model.fc.in_features
            model = nn.Sequential(*list(model.children())[:-1])
            
        self.origin_feature = torch.nn.DataParallel(model)
        self.model = args.model
        self.arch = args.arch

        self.fcb = nn.Linear(self.feature_dim, args.bottleneck)
        self.fcb.weight.data.normal_(0, 0.005)
        self.fcb.bias.data.fill_(0.1)
        self.fcs = nn.Linear(args.bottleneck, args.classes, bias=False)
        self.fcs.weight.data.normal_(0, 0.01)
        # self.fcs.bias.data.fill_(0.0)
        self.fct = nn.Linear(args.bottleneck, args.classes, bias=False)
        self.fct.weight.data = self.fcs.weight.data.clone()
        # self.fct.bias.data = self.fcs.bias.data.clone()

        args.SGD_param = [
            {'params': self.origin_feature.parameters(), 'lr': 1,},
            {'params': self.fcb.parameters(), 'lr': 10},
            {'params': self.fcs.parameters(), 'lr': 10},
            {'params': self.fct.parameters(), 'lr': 10},
        ]

    def forward(self, x, asym=False):
        x = self.origin_feature(x)
        if self.arch.startswith('densenet'):
            x = F.relu(x, inplace=True)
            x = F.avg_pool2d(x, kernel_size=7)
        x = x.view(x.size(0), -1)
        x = self.fcb(x)
        if asym:
            xs, xt = x.chunk(2, 0)
            ys = self.fcs(xs)
            yt = self.fct(xt)
            return ys, yt, xs, xt
        else:
            y = self.fct(x)
            return y, x


def train_val(source_loader, target_loader, val_loader, model, criterion, optimizer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    entropy_loss = AverageMeter()

    source_cycle = iter(source_loader)
    target_cycle = iter(target_loader)
    
    U = torch.autograd.Variable(torch.Tensor(args.bottleneck, args.classes)).cuda()
    Vs = torch.autograd.Variable(torch.Tensor(args.classes, args.classes), requires_grad=True).cuda()
    Vs.data.normal_(0, 0.01)
    Vt = torch.autograd.Variable(torch.Tensor(args.classes, args.classes), requires_grad=True).cuda()
    Vt.data.normal_(0, 0.01)

    end = time.time()
    model.train(True)
    for i in range(args.train_iter):
        global global_iter
        global_iter = i
        adjust_learning_rate(optimizer, i, args)
        data_time.update(time.time() - end)
        
        source_input, label = source_cycle.next()
        target_input, _ = target_cycle.next()
        if source_input.size()[0] < args.batch_size or target_input.size()[0] < args.batch_size:
            source_cycle = iter(source_loader)
            target_cycle = iter(target_loader)
            source_input, label = source_cycle.next()
            target_input, _ = target_cycle.next()
            
        label = label.cuda(async=True)
        source_var = torch.autograd.Variable(source_input)
        target_var = torch.autograd.Variable(target_input)
        label_var = torch.autograd.Variable(label)

        inputs = torch.cat([source_var, target_var], 0)
        source_output, target_output, source_feature, target_feature = model(inputs, asym=True)
        
        if i % 30 == 0:
            W = torch.cat([model.fcs.weight.data.t(), model.fct.weight.data.t()], 1)
            V = torch.cat([Vs.data, Vt.data], 1)
            u, _, _ = torch.svd(torch.mm(W, V.t()))
            U.data = u.clone()
            
        acc_loss = criterion(source_output, label_var)
        softmax = nn.Softmax()
        jmmd_loss = JMMDLoss([source_feature, softmax(source_output)], [target_feature, softmax(target_output)])
        rec_loss = (model.fcs.weight.t() - torch.mm(U, Vs)).pow(2).mean() \
                 + (model.fct.weight.t() - torch.mm(U, Vt)).pow(2).mean()

        loss = acc_loss + jmmd_loss + args.gammaC*rec_loss

        prec1, _ = accuracy(source_output.data, label, topk=(1, 5))

        losses.update(loss.data[0], args.batch_size)
        loss1 = jmmd_loss.data[0]
        loss2 = acc_loss.data[0]
        loss3 = rec_loss.data[0]
        top1.update(prec1[0], args.batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Iter: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss1:.4f}/{loss2:.4f}/{loss3:.4f}\t'
                  'Loss {loss.val:.4f}\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   i, args.train_iter, batch_time=batch_time,
                      loss=losses, top1=top1, loss1=loss1, loss2=loss2, loss3=loss3))

        if i % args.test_iter == 0 and i != 0:
            validate(val_loader, model, criterion, args)
            model.train(True)
            batch_time.reset()
            data_time.reset()
            losses.reset()
            top1.reset()



def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output, _ = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg
