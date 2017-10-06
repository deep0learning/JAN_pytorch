import time
import torch
import torch.nn as nn
import itertools
from torch.autograd import Variable
from torchvision import transforms
from . import cycle_gan
import numpy as np

from utils import *

origin_mean = Variable(torch.Tensor([0.5, 0.5, 0.5])).view(-1, 1, 1).cuda()
origin_std = Variable(torch.Tensor([0.5, 0.5, 0.5])).view(-1, 1, 1).cuda()
new_mean = Variable(torch.Tensor([0.485, 0.456, 0.406])).view(-1, 1, 1).cuda()
new_std = Variable(torch.Tensor([0.229, 0.224, 0.225])).view(-1, 1, 1).cuda()


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.optimizer = None
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        self.source_gen = cycle_gan.define_G(3, 3, args.ngf, args.ngblocks,
                                             args.norm, not args.no_dropout,
                                             args.ngdownsampling)
        self.target_gen = cycle_gan.define_G(3, 3, args.ngf, args.ngblocks,
                                             args.norm, not args.no_dropout,
                                             args.ngdownsampling)
        self.source_discrim = cycle_gan.define_D(3, args.ndf, args.ndlayers,
                                                 args.norm, args.no_lsgan)
        self.target_discrim = cycle_gan.define_D(3, args.ndf, args.ndlayers,
                                                 args.norm, args.no_lsgan)

        self.saver = Saver(args.prefix, {
            'G_s': self.source_gen,
            'G_t': self.target_gen,
            'D_s': self.source_discrim,
            'D_t': self.target_discrim
        })
        self.fake_source_pool = ImagePool(args.pool_size)
        self.fake_target_pool = ImagePool(args.pool_size)

        if args.continue_train:
            self.saver.load('iter-%05d' % args.which_epoch)


def train_val(source_loader, target_loader, val_loader, model, optimizer, args):

    batch_time = AverageMeter()
    lossg = AverageMeter()
    lossd = AverageMeter()
    top1 = AverageMeter()

    source_cycle = infinite_loader(source_loader)
    target_cycle = infinite_loader(target_loader)

    criterionGAN = cycle_gan.GANLoss(use_lsgan=not args.no_lsgan)
    criterionCycle = torch.nn.L1Loss()
    criterionIdt = torch.nn.L1Loss()

    optimizer_gen = torch.optim.Adam(
        itertools.chain(model.source_gen.parameters(),
                        model.target_gen.parameters()),
        lr=args.lr, betas=(0.5, 0.999))
    optimizer_discrim = torch.optim.Adam(
        itertools.chain(model.source_discrim.parameters(),
                        model.target_discrim.parameters()),
        lr=args.lr, betas=(0.5, 0.999))

    end = time.time()
    model.train()
    with open(os.path.join(args.prefix, 'log.txt'), 'w') as f:
        num_params =  get_network_parameters(model)
        msg = 'Total number of parameters: %d (%.2fMB)' % \
            (num_params, float(num_params) * 4 / 1024 / 1024)
        print(msg)
        f.write(msg + '\n')
        for i in range(args.train_iter):
            # adjust_learning_rate(optimizer, i, args)

            source_input, label = next(source_cycle)
            target_input, _ = next(target_cycle)

            real_source = Variable(source_input).cuda()
            real_target = Variable(target_input).cuda()
            source_label = Variable(label).cuda()

            # -------- Train GAN -------- #
            if args.lambda_idt > 0:
                idt_source = model.target_gen(real_source)
                idt_target = model.source_gen(real_target)
                loss_idt = criterionIdt(idt_source, real_source) * \
                           args.lambda_s * args.lambda_idt + \
                           criterionIdt(idt_target, real_target) * \
                           args.lambda_t * args.lambda_idt
            else:
                loss_idt = 0

            fake_target = model.source_gen(real_source)
            fake_source = model.target_gen(real_target)
            fake_target_pred = model.source_discrim(fake_target)
            fake_source_pred = model.target_discrim(fake_source)
            loss_gen = criterionGAN(fake_source_pred, True) + \
                criterionGAN(fake_target_pred, True)

            rec_target = model.source_gen(fake_source)
            rec_source = model.target_gen(fake_target)
            loss_rec = criterionCycle(rec_source, real_source) * args.lambda_s + \
               criterionCycle(rec_target, real_target) * args.lambda_t

            loss_G = loss_idt + loss_rec + loss_gen

            optimizer_gen.zero_grad()
            loss_G.backward()
            optimizer_gen.step()

            # -------- Train Discrim -------- #
            origin_fake_target = fake_target
            fake_target = model.fake_target_pool.query(fake_target).detach()
            fake_source = model.fake_source_pool.query(fake_source).detach()

            real_target_pred = model.source_discrim(real_target)
            real_source_pred = model.target_discrim(real_source)
            fake_target_pred = model.source_discrim(fake_target)
            fake_source_pred = model.target_discrim(fake_source)

            loss_D = (criterionGAN(real_source_pred, True) +
                      criterionGAN(fake_source_pred, False)) * 0.5 + \
                     (criterionGAN(real_target_pred, True) +
                      criterionGAN(fake_target_pred, False)) * 0.5
            optimizer_discrim.zero_grad()
            loss_D.backward()
            optimizer_discrim.step()

            lossg.update(loss_G.data[0], args.batch_size)
            lossd.update(loss_D.data[0], args.batch_size)
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                msg = 'Iter: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'LossG {lossg.val:.4f}/{lossg.avg:.4f}\t' \
                      'LossD {lossd.val:.4f} ({lossd.avg:.4f})\t' \
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, args.train_iter, batch_time=batch_time,
                    lossg=lossg, lossd=lossd, top1=top1)
                print(msg)
                f.write(msg + '\n')

            if i % args.display_freq == 0:
                dirname = os.path.join(args.prefix, 'iter-%05d' % i)
                print_image(real_source, dirname, 'image-real_source-%03d.jpeg')
                print_image(origin_fake_target, dirname,
                            'image-fake_target-%03d.jpeg')
                print_image(rec_source, dirname, 'image-rec_source-%03d.jpeg')

            if i % args.save_epoch_freq == 0:
                model.saver.save('iter-%05d' % i)


def print_image(input, dirname, filename):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for index, pic in enumerate(input.data.cpu()):
        pic = pic.numpy()
        pic = np.clip((pic * 0.5 + 0.5) * 255, 0.0, 255.0)
        pic = np.rint(pic).astype(np.uint8).transpose(1, 2, 0)
        Image.fromarray(pic, 'RGB').save(os.path.join(dirname, filename) % index)
