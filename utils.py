import os
import csv
import torch.optim
import torch.utils.data as data
import torchvision.datasets.folder as folder
from torch.autograd import Variable
from PIL import Image
import collections
import random


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def adjust_learning_rate(optimizer, iter_num, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (1 + args.gamma * iter_num) ** (-args.power)
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * args.SGD_param[i]['lr']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class ImageList(data.Dataset):
    def __init__(self, file, transform=None, target_transform=int,
                 loader=folder.default_loader, delimiter=' '):

        with open(file) as f:
            self.imgs = list(csv.reader(f, delimiter=' '))
        self.file = file
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


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
            return img.resize(self.size)


def get_network_parameters(*nets):
    num_params = 0
    for net in nets:
        for param in net.parameters():
            num_params += param.numel()
    return num_params



def convert_python3_state_dict(state_dict):
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.encode('utf-8')] = v
    return new_state_dict


class Saver:
    def __init__(self, prefix, nets):
        self.prefix = prefix
        self.nets = nets

    def save(self, label):
        for k in self.nets:
            filename = os.path.join(self.prefix, label, 'model-%s.pth'% k)
            torch.save(self.nets[k].state_dict(), filename)

    def load(self, label):
        for k in self.nets:
            filename = os.path.join(self.prefix, label, 'model-%s.pth'% k)
            self.nets[k].load_state_dict(torch.load(filename, lambda s, l: 'cpu'))


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.stack(return_images, 0))
        return return_images


def infinite_loader(loader):
    while True:
        # Python2
        for data in loader:
            yield data
        # Python3
        # yield from loader
