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
from util import AverageMeter


def parse_option():

    parser = argparse.ArgumentParser('argument for pre-trained models')
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--data_path_train", type=str, default="../datasets/imagenet-M-train")
    parser.add_argument("--data_path_test", type=str, default="../datasets/imagenet-M-test1")
    parser.add_argument("--model", type=str, default="vgg16", choices=["resnet18", "vgg16", "vit16"])
    parser.add_argument("--classifier_type", type=str, default="single")

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument('--lr_decay_epochs', type=str, default='60,60,40,40',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--cosine', default=False,
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')

    opt = parser.parse_args()

    opt.model_path = './save/{}_models'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'CE_{}_{}_lr_{}_decay_{}_bsz_{}'. \
        format(opt.dataset, opt.model, opt.lr, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.lr * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.lr - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.lr

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def load_data(opt):

    num_classes_dict = {"imagenet100": 100, "imagenet50": 50, "imagenet-m": 18, "cifar100": 50}

    if "imagenet100" in opt.dataset:
        dataset_train = ImageNet100(opt.data_path_train, train=True)
        dataset_test = ImageNet100(opt.data_path_test, train=False)
    if "imagenet50" in opt.dataset:
        dataset_train = ImageNet50(opt.data_path_train, train=True)
        dataset_test = ImageNet50(opt.data_path_test, train=False)
    elif "imagenet-m" in opt.dataset:
        dataset_train = ImageNet_M(opt.data_path_train, train=True)
        dataset_test = ImageNet_M(opt.data_path_test, train=False)
    elif "cifar100" in opt.dataset:
        dataset_train = iCIFAR100(root="../datasets", train=True)
        dataset_test = iCIFAR100(root="../datasets", train=False)

    opt.num_classes = num_classes_dict[opt.dataset]
    dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False)

    return dataloader_train, dataloader_test


def load_model(opt):

    if "resnet" in opt.model:
        model = SupCEResNet(name=opt.model, num_classes=opt.num_classes)
    elif "vgg" in opt.model:
        model = vgg16_bn(num_classes=opt.num_classes)

    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def set_optimizer(opt, model):
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)
    return optimizer


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


def warmup_learning_rate(opt, epoch, batch_id, total_batches, optimizer):
    if opt.warm and epoch <= opt.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (opt.warm_epochs * total_batches)
        lr = opt.warmup_from + p * (opt.warmup_to - opt.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


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


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        print("labels", labels)
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = model(images)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1 = accuracy(output, labels, topk=(1, 5))
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
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
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
    dataloader_train, dataloader_test = load_data(opt)
    model, criterion = load_model(opt)
    optimizer = set_optimizer(opt, model)

    best_acc = 0
    print("num classes", opt.num_classes)

    for epoch in range(opt.epochs):
        adjust_learning_rate(opt, optimizer, epoch)

        time1 = time.time()
        loss, train_acc = train(dataloader_train, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # evaluation
        loss, val_acc = validate(dataloader_test, model, criterion, opt)

        if val_acc > best_acc:
            best_acc = val_acc

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

        # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == "__main__":
    main()






