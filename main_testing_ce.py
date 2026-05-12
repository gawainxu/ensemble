from __future__ import print_function

import os
import sys
import argparse
import time
import math
import copy

import torch
import torch.backends.cudnn as cudnn

from util import AverageMeter
from util import accuracy
from networks.resnet_big import SupCEResNet
from dataUtil import osr_splits_inliers, get_test_datasets

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
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
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
                        choices=["resnet18", "resnet34", "simCNN", "MLP"])
    parser.add_argument('--datasets', type=str, default='cifar100_marco',
                        choices=["cifar100_marco", 'cifar10', "tinyimgnet", 'mnist', "svhn"],
                        help='dataset')
    parser.add_argument("--backbone_model_direct", type=str, default="/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_0_128_256/")
    parser.add_argument("--backbone_model_name", type=str, default="last.pth")
    parser.add_argument("--trail", type=int, default=0)
    parser.add_argument("--temp_list", type=str, default="")

    # upsampling parameters
    parser.add_argument("--upsample", type=bool, default=False)
    parser.add_argument("--portion_out", type=float, default=0.5)
    parser.add_argument("--upsample_times", type=int, default=1)
    parser.add_argument("--last_feature_path", type=str, default=None)
    parser.add_argument("--last_model_path", type=str, default=None)
    parser.add_argument("--feat_dim", type=int, default=128)

    # other setting
    parser.add_argument('--cosine', type=bool, default=False,
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument("--seed", type=int, default=0)

    opt = parser.parse_args()

    opt.num_classes = len(osr_splits_inliers[opt.datasets][opt.trail])

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.main_dir = os.getcwd()
    opt.backbone_model_direct = opt.main_dir + opt.backbone_model_direct
    opt.backbone_model_path = os.path.join(opt.backbone_model_direct, opt.backbone_model_name)

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


def set_model(opt):
    if opt.datasets == "mnist":
        in_channels = 1
    elif opt.datasets == "FUB":
        in_channels = 1
    else:
        in_channels = 3

    model = SupCEResNet(name=opt.model, in_channels=in_channels, num_classes=opt.num_classes)
    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def testing(test_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(test_loader):
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

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {top1.val:.3f} ({top1.avg:.3f})'.format(
                    idx, len(test_loader), batch_time=batch_time,
                    loss=losses, top1=top1))

    return losses.avg, top1.avg


def main():
    opt = parse_option()

    # build data loader
    test_dataset = get_test_datasets(opt)

    train_sampler = None
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler,
                                               drop_last=True)

    # build model and criterion
    model, criterion = set_model(opt)

    # training routine
    loss, acc = testing(test_loader, model, criterion, opt)
    print('Accuracy:{:.2f}, loss:{:.2f}'.format(acc, loss))



if __name__ == '__main__':
    main()
