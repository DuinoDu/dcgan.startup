#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from models import DCGAN 
from dataset import create_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='mnist | cifar10 | cifar100 | lsun ')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network')

parser.add_argument('--num_iters', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

parser.add_argument('--save_folder', default='output', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--gpuID', default=0, type=int, help='Use which gpu to train model')
parser.add_argument('--tensorboard', default=0, type=int, help='Use tensorboard for loss visualization')
parser.add_argument('--resume', default="", type=str, help='resume weight pth path')
    

args = parser.parse_args()
print(args)

# args
os.environ['CUDA_VISIBLE_DIVICES'] = str(args.gpuID)
if args.tensorboard:
   from tensorboard import SummaryWriter
   writer = SummaryWriter()
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True
if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


# data
dataset = create_dataset(args.dataset, image_size=args.image_size)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=int(args.workers))

# model
net = DCGAN()
if args.resume != "":
    net.load_state_dicts(torch.load(args.resume))

# loss
net.set_criterion([nn.BCELoss(), nn.BCELoss()])
net.train()

# optimizer
params = net.parameters()
optim_G = optim.Adam(params[0], lr=args.lr, betas=(args.beta1, 0.999))
optim_D = optim.Adam(params[1], lr=args.lr, betas=(args.beta1, 0.999))
net.set_optim([optim_G, optim_D])

if args.cuda:
    net.cuda()

# summary
if args.tensorboard:
   from tensorboardX import SummaryWriter
   writer = SummaryWriter()

# train loop
for epoch in range(args.num_iters):
    for i, data in enumerate(dataloader, 0):
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        if args.cuda:
            real_cpu = real_cpu.cuda()
        inputv = Variable(real_cpu)

        fake = net(inputv)
        loss_D, loss_G = net.criterion()
        net.backward_step()

        # log 
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
              % (epoch, args.num_iters, i, len(dataloader),
                 loss_D.data[0], loss_G.data[0]))
        if args.tensorboard:
            iteration = i + epoch * len(dataloader)
            writer.add_scalar('loss_D', loss_D.data[0], iteration)
            writer.add_scalar('loss_G', loss_G.data[1], iteration)
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % args.save_folder,
                    normalize=True)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (args.save_folder, epoch),
                    normalize=True)
    # checkpoint 
    torch.save(net.state_dicts(), os.path.join(args.save_folder, 'weights_epoch_%d.pth' % epoch))

if args.tensorboard:
    write.close()
