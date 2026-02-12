import argparse
import math
import time
import os
import sys
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from pre_models_dataset import ImageNet100, ImageNet_M, ImageNet50
from pre_models_dataset import iCIFAR100
from torch.utils.data import DataLoader

from networks.resnet_big import SupCEResNet
from networks.vgg import vgg16, vgg16_bn
from networks.ViT import ViT, get_b16_config_cifar, get_b16_config
from util import AverageMeter

image_size_mapping = {"cifar100": 32, "imagenet50": 224}


def parse_option():

    parser = argparse.ArgumentParser('argument for pre-trained models')
    parser.add_argument("--dataset", type=str, default="imagenet50")
    parser.add_argument("--data_path_train", type=str, default=None)
    parser.add_argument("--data_path_test", type=str, default=None)
    parser.add_argument("--model", type=str, default="vit16", choices=["resnet18", "vgg16", "vit16"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_path", type=str, default="")

    opt = parser.parse_args()

    opt.image_size = image_size_mapping[opt.dataset]

    return opt


def load_data(opt):

    num_classes_dict = {"imagenet100": 100, "imagenet50": 50, "imagenet-m": 18, "cifar100": 50}

    if "imagenet100" in opt.dataset:
        dataset_train = ImageNet100(train=True, opt=opt)
        dataset_test = ImageNet100(train=False, opt=opt)
    if "imagenet50" in opt.dataset:
        dataset_train = ImageNet50(train=True, opt=opt)
        dataset_test = ImageNet50(train=False, opt=opt)
    elif "imagenet-m" in opt.dataset:
        dataset_train = ImageNet_M(train=True, opt=opt)
        dataset_test = ImageNet_M(train=False, opt=opt)
    elif "cifar100" in opt.dataset:
        dataset_train = iCIFAR100(root="../datasets", train=True, opt=opt)
        dataset_test = iCIFAR100(root="../datasets", train=False, opt=opt)

    opt.num_classes = num_classes_dict[opt.dataset]
    dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False)

    print("dataset_train", len(dataset_train))
    print("dataset_test", len(dataset_test))

    return dataloader_train, dataloader_test


def load_model(opt):

    if "resnet" in opt.model:
        model = SupCEResNet(name=opt.model, num_classes=opt.num_classes)
    elif "vgg" in opt.model:
        model = vgg16_bn(num_classes=opt.num_classes)
    elif "vit" in opt.model:
        if "cifar" in opt.dataset:
            configs = get_b16_config_cifar()
        elif "imagenet" in opt.dataset:
            configs = get_b16_config()
        model = ViT(image_size=opt.image_size, patch_size=configs.patch_size, num_classes=opt.num_classes,
                    embedding_dim=configs.embed_dim, depth=configs.depth, heads=configs.num_heads, mlp_dim=configs.hidden_dim,
                    dim_head = configs.head_dim, dropout = configs.dropout, emb_dropout = configs.emb_dropout)
    criterion = torch.nn.CrossEntropyLoss()

    ckpt = torch.load(opt.model_path, map_location='cpu')
    state_dict = ckpt['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict
    model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion



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
        return res   # accuracy of the top-k predictions



def testing(val_loader, model, criterion, opt):
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
            acc1 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0].item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg



def main():

    opt = parse_option()
    _, dataloader_test = load_data(opt)
    model, criterion = load_model(opt)

    loss, test_acc = testing(dataloader_test, model, criterion, opt)

    print('test accuracy: {:.2f}'.format(test_acc))


if __name__ == "__main__":
    main()






