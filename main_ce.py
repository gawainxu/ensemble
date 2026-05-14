#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 20:38:04 2024

@author: zhi
"""
from __future__ import print_function

import os
import sys
import argparse
import time
import pickle

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn as nn

from util import AverageMeter
from util import adjust_learning_rate
from util import accuracy
import torch.optim as optim
from dataUtil import osr_splits_inliers, get_train_datasets, get_test_datasets
from networks.resnet_big import SupCEResNet
from  networks.vgg import vgg16, vgg11_bn
from networks.LeNet import LeNet5

from data_test import get_training_dataloader, get_test_dataloader

import matplotlib
matplotlib.use('Agg')

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60, 120, 160',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument("--pretrained", type=int, default=1)

    # model dataset
    parser.add_argument('--model', type=str, default='vgg16', choices=["resnet18", "resnet34", "vgg16", "simCNN", "MLP", "lenet"])
    parser.add_argument("--resnet_wide", type=int, default=1, help="factor for expanding channels in wide resnet")
    parser.add_argument('--datasets', type=str, default='cifar100_marco',
                        choices=['cifar10', "tinyimgnet", 'mnist', "svhn", "cifar100_marco"], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument("--augmentation_list", type=list, default=[])
    parser.add_argument("--argmentation_n", type=int, default=2)
    parser.add_argument("--argmentation_m", type=int, default=6)
    parser.add_argument("--randaug", type=int, default=0)

    # method
    parser.add_argument("--trail", type=int, default=0, choices=[0,1,2,3,4,5,6], help="index of repeating training")
    parser.add_argument("--action", type=str, default="training_ce",
                        choices=["training_supcon", "training_ce", "trainging_linear", "testing_known", "testing_unknown", "feature_reading"])            
        
    # other setting
    parser.add_argument('--cosine', type=bool, default=False,
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument("--feat_dim", type=int, default=128)
    


    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.datasets == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = '../datasets/'
    opt.model_path = './save/CE/{}_models'.format(opt.datasets)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = opt.datasets + "_" + opt.model + "_" + str(opt.resnet_wide)
   
    opt.model_name += 'trail_{}'.format(opt.trail) + "_" + str(opt.feat_dim) + "_" + str(opt.batch_size)

    if opt.randaug == 1:
        opt.model_name += "_randaug" + str(opt.argmentation_n) + "_" + str(opt.argmentation_m)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    
    opt.num_classes = len(osr_splits_inliers[opt.datasets][opt.trail])

    return opt


def set_loader(opt):
    # construct data loader
    
    train_dataset =  get_train_datasets(opt)
    test_dataset = get_test_datasets(opt)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                                               num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                               num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler,
                                               drop_last=True)
    return train_loader, test_loader


def set_model(opt):

    if opt.datasets == "mnist":
        in_channels = 1
    else:
        in_channels = 3

    if "resnet" in opt.model:
        model = SupCEResNet(name=opt.model, in_channels=in_channels, num_classes=opt.num_classes)
    elif "vgg" in opt.model:
        model = vgg16(num_classes=opt.num_classes)
    elif "lenet" in opt.model:
        model = LeNet5(num_classes=opt.num_classes)

    criterion = torch.nn.CrossEntropyLoss()
       
    if torch.cuda.is_available():
        
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion



def load_model(opt, model=None):
    if model is None:
        model = SupCEResNet(name=opt.model, num_classes=opt.num_classes)

    ckpt = torch.load(opt.last_model_path, map_location='cpu')
    state_dict = ckpt['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v

    state_dict = new_state_dict
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()

    return model



def save_model(model=None, optimizer=None, opt=None, epoch=0, save_file=None):
    print('==> Saving...')
    state = {'opt': opt,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,}
    torch.save(state, save_file)
    del state


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_decay_epochs,
                                                     gamma=0.2)
    return optimizer, train_scheduler


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
       
        bsz = labels.shape[0]
        
        logits = model(images)
        acc1, _, _ = accuracy(logits, labels)
        loss = criterion(logits, labels)
        losses.update(loss.item())
        accs.update(acc1)


        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'acc {accs.val:.3f} ({accs.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, accs=accs))
            sys.stdout.flush()

    return losses.avg



def val(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(images)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, _, _ = accuracy(output, labels)
            top1.update(acc1, bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {top1.val:.3f} ({top1.avg:.3f})'.format(
              idx, len(val_loader), batch_time=batch_time,
               loss=losses, top1=top1))

    return top1.avg



def main():
    opt = parse_option()

    # build data loader
    train_loader, test_loader = set_loader(opt)
    """
    train_loader = get_training_dataloader(
        num_workers=4,
        batch_size=opt.batch_size,
        shuffle=True
    )
    test_loader = get_test_dataloader(
        num_workers=4,
        batch_size=opt.batch_size,
        shuffle=True
    )
    """
    print("train_loader, ", train_loader.__len__())

    # build model and criterion
    model, criterion= set_model(opt)

    # build optimizer
    optimizer, train_scheduler = set_optimizer(opt, model)
    losses = []

    # training routine
    for epoch in range(0, opt.epochs):
        #adjust_learning_rate(opt, optimizer, epoch)
        train_scheduler.step(epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        acc_val = val(test_loader, model, criterion, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        losses.append(loss)
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)
    with open(os.path.join(opt.save_folder, "loss_" + str(opt.trail)), "wb") as f:
         pickle.dump(losses, f)


if __name__ == '__main__':
    main()
