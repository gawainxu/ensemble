from __future__ import print_function

import argparse
import math
import time
import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from pre_models_dataset import ImageNet100, ImageNet_M
from torch.utils.data import DataLoader
from pre_models_training import accuracy

from networks.resnet_big import SupCEResNet
from util import AverageMeter
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument("--action", type=str, default="trainging_linear",
                        choices=["training_supcon", "trainging_linear",
                                 "testing_known", "testing_unknown", "feature_reading"])
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=["resnet18", "resnet50"])
    parser.add_argument("--output_layer", type=str, default="layer3")
    parser.add_argument("--dataset", type=str, default="imagenet-m")
    parser.add_argument("--data_path_train", type=str, default="../datasets/imagenet-M-train")
    parser.add_argument("--data_path_test", type=str, default="../datasets/imagenet-M-test2")
    parser.add_argument("--backbone_model_direct", type=str, default="/save/SupCon/imagenet-m_models/CE_imagenet-m_resnet18_lr_0.2_decay_0.0001_bsz_128_cosine")
    parser.add_argument("--backbone_model_name", type=str, default="last.pth")
    parser.add_argument("--linear_mode", type=str, default="single")

    # other setting
    parser.add_argument('--cosine', type=bool, default=False,
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument("--seed", type=int, default=0)

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.main_dir = os.getcwd()
    opt.backbone_model_direct = opt.main_dir + opt.backbone_model_direct
    opt.backbone_model_path = os.path.join(opt.backbone_model_direct, opt.backbone_model_name)
    opt.linear_model_path = os.path.join(opt.backbone_model_direct, "last_linear_"+opt.linear_mode+".pth")

    return opt


def load_model(model, path):
    ckpt = torch.load(path, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model


def set_optimizer(opt, model):
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, save_file)
    del state


def adjust_learning_rate(opt, optimizer, epoch):
    lr = opt.lr
    if opt.cosine:
        eta_min = lr * (opt.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / opt.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
        if steps > 0:
            lr = lr * (opt.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class StrippedModel_ResNet(nn.Module):
    def __init__(self, original_model, opt):
        super().__init__()
        self.model = original_model
        self.opt = opt

    def forward(self, x):
        out = F.relu(self.model.encoder.bn1(self.model.encoder.conv1(x)))
        out = self.model.encoder.layer1(out)
        if "layer1" in self.opt.output_layer:
            # TODO how to reshape, temporally same as the final layer, maybe not the best option
            out = self.model.encoder.avgpool(out)
            out = torch.flatten(out, 1)
            return out
        out = self.model.encoder.layer2(out)
        if "layer2" in self.opt.output_layer:
            out = self.model.encoder.avgpool(out)
            out = torch.flatten(out, 1)
            return out
        out = self.model.encoder.layer3(out)
        if "layer3" in self.opt.output_layer:
            out = self.model.encoder.avgpool(out)
            out = torch.flatten(out, 1)
            return out

        out = self.model.encoder.layer4(out)
        out = self.model.encoder.avgpool(out)
        out = torch.flatten(out, 1)
        return out


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, feat_dim, num_classes=10, mode="single"):
        super(LinearClassifier, self).__init__()
        self.fc1 = nn.Linear(feat_dim, 100)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(feat_dim, num_classes)
        self.fc = nn.Linear(feat_dim, num_classes)
        self.mode = mode

    def forward(self, features):
        if "single" in self.mode:
            out = self.fc(features)
        else:
            out = self.fc1(features)
            #out = self.relu(out)
            #out = self.bn(out)
            out = self.fc2(out)
        return out



def set_model(opt):
    criterion = torch.nn.CrossEntropyLoss()
    in_channels = 3

    if opt.model == "resnet18" or opt.model == "resnet50":
        model = SupCEResNet(name=opt.model, num_classes=opt.num_classes)
    model = load_model(model, opt.backbone_model_path)

    if "resnet" in opt.model:
        out_dim = {"layer1": 64, "layer2": 128, "layer3": 256, "end": 512}
        model = StrippedModel_ResNet(model, opt)
        classifier =LinearClassifier(feat_dim=out_dim[opt.output_layer], num_classes = opt.num_classes)

    return model, classifier, criterion


def load_data(opt):

    num_classes_dict = {"imagenet100": 100, "imagenet-m": 18}

    if "imagenet100" in opt.dataset:
        dataset_train = ImageNet100(opt.data_path_train, train=True)
        dataset_test = ImageNet100(opt.data_path_test, train=False)
    elif "imagenet-m" in opt.dataset:
        dataset_train = ImageNet_M(opt.data_path_train, train=True)
        dataset_test = ImageNet_M(opt.data_path_test, train=False)

    opt.num_classes = num_classes_dict[opt.dataset]
    dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=True)

    return dataloader_train, dataloader_test


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""

    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # compute loss
        with torch.no_grad():
            features = model(images)
            features = features.cuda(non_blocking=True)

        output = classifier(features)
        loss = criterion(output, labels)
        # print(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1 = accuracy(output, labels)
        top1.update(acc1[0].item(), bsz)

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
                  'Acc {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.train()

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
            output = classifier(model(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1 = accuracy(output, labels)
            top1.update(acc1[0].item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {top1.val:.3f} ({top1.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time,
                    loss=losses, top1=top1))

    return losses.avg, top1.avg


def main():
    opt = parse_option()

    # build data loader
    dataloader_train, dataloader_test = load_data(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)
    model = model.cuda()
    classifier = classifier.cuda()

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    best_acc = 0

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(dataloader_train, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}, loss:{:.2f}'.format(
            epoch, time2 - time1, acc, loss))

        loss, val_acc = validate(dataloader_test, model, classifier, criterion, opt)
        if val_acc > best_acc:
            best_acc = val_acc

    save_file = opt.backbone_model_name.replace(".pth", "_linear") + ".pth"
    save_file = os.path.join(opt.backbone_model_direct, save_file)
    save_model(classifier, optimizer, opt, save_file)
    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
