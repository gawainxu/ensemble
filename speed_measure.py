from __future__ import print_function

import os
import sys
import argparse
import time
import math
import pickle
import random
from datetime import datetime
import time

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model, label_convert
from dataUtil import osr_splits_inliers, get_train_datasets
from networks.resnet_big import SupConResNet, SupConResNet_MultiHead_remix
from networks.resnet_multi import SupConResNet_MultiHead
from networks.simCNN import simCNN_contrastive
from networks.resnet_preact import SupConpPreactResNet
from networks.mlp import SupConMLP
from losses import SupConLoss

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
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='1000',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet_multi',
                        choices=["resnet18", "resnet_multi", "resnet34", "preactresnet18", "preactresnet34", "simCNN",
                                 "MLP"])
    parser.add_argument("--last_model_path", type=str, default=None)
    parser.add_argument('--datasets', type=str, default='cifar10',
                        choices=["cifar-10-100-10", "cifar-10-100-50", 'cifar10', "tinyimgnet", 'mnist', "svhn",
                                 "cifar100_marco", "imagenet100"], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument("--augmentation_list", type=list, default=[])
    parser.add_argument("--argmentation_n", type=int, default=1)
    parser.add_argument("--argmentation_m", type=int, default=6)
    parser.add_argument("--multiplier", type=float, default=1)

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR', "SimCLR_CE", "MoCo"], help='choose method')
    parser.add_argument("--trail", type=int, default=0, choices=[0, 1, 2, 3, 4, 5], help="index of repeating training")
    parser.add_argument("--action", type=str, default="training_supcon",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown",
                                 "feature_reading"])
    # temperature
    parser.add_argument('--temp', type=float, default=0.05, help='temperature for loss')
    parser.add_argument('--temp1', type=float, default=0.005, help='temperature for loss')
    parser.add_argument('--temp2', type=float, default=0.01, help='temperature for loss')
    parser.add_argument('--temp3', type=float, default=0.05, help='temperature for loss')
    parser.add_argument("--out_dim", type=int, default=512, help="output dimension of the resnet blocks")
    parser.add_argument("--clip", type=float, default=None, help="for gradient clipping")

    # other setting
    parser.add_argument('--cosine', type=bool, default=False,
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument("--feat_dim", type=int, default=128)

    opt = parser.parse_args()

    opt.num_classes = len(osr_splits_inliers[opt.datasets][opt.trail])

    # check if dataset is path that passed required arguments
    if opt.datasets == 'path':
        assert opt.data_folder is not None \
               and opt.mean is not None \
               and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = '../datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.datasets)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if "multi" in opt.model:
        opt.model_name = opt.datasets + "_" + opt.model + '_trail_{}'.format(opt.trail) + "_" + str(
            opt.feat_dim) + "_" + str(opt.out_dim) + "_" + str(opt.temp1) + "_" + str(opt.temp2) + "_" + str(
            opt.temp3) + "_" + str(opt.batch_size)
    else:
        opt.model_name = opt.datasets + "_" + opt.model + '_trail_{}'.format(opt.trail) + "_" + str(
            opt.feat_dim) + "_" + str(opt.temp)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def synchronize():
    torch.cuda.synchronize()

def measure(step, warmup=5, repeats=10):
    for _ in range(warmup):
        step()

    synchronize()
    start = time.perf_counter()

    for _ in range(repeats):
        step()

    synchronize()
    seconds = time.perf_counter() - start
    ms_per_batch = 1000 * seconds / repeats
    images_per_second = 256 * repeats / seconds
    return ms_per_batch, images_per_second

def set_loader(opt):
    # construct data loader

    train_dataset = get_train_datasets(opt)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler,
                                               drop_last=True)
    return train_loader


def set_model(opt):
    if opt.datasets == "mnist":
        in_channels = 1
    else:
        in_channels = 3

    if opt.model == "resnet18" or opt.model == "resnet34":
        model = SupConResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
    elif opt.model == "preactresnet18" or opt.model == "preactresnet34":
        model = SupConpPreactResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
    elif opt.model == "MLP":
        model = SupConMLP(feat_dim=opt.feat_dim)
    elif opt.model == "resnet_multi":
        if torch.cuda.device_count() > 1:
            model = SupConResNet_MultiHead_remix(output_dim=opt.out_dim, feat_dim=opt.feat_dim, in_channels=in_channels)
        else:
            model = SupConResNet_MultiHead(output_dim=opt.out_dim, feat_dim=opt.feat_dim, in_channels=in_channels)
    else:
        model = simCNN_contrastive(opt, feature_dim=opt.feat_dim, in_channels=in_channels)

    if opt.last_model_path is not None:
        model = load_model(opt, model)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        cudnn.benchmark = True

    if opt.model == "resnet_multi":
        criterion1 = SupConLoss(temperature=opt.temp1)
        criterion2 = SupConLoss(temperature=opt.temp2)
        criterion3 = SupConLoss(temperature=opt.temp3)
        criterion1 = criterion1.cuda()
        criterion2 = criterion2.cuda()
        criterion3 = criterion3.cuda()
        return model, (criterion1, criterion2, criterion3)
    else:
        criterion = SupConLoss(temperature=opt.temp)
        criterion = criterion.cuda()
        return model, (criterion, None, None)


def load_model(opt, model=None):
    if model is None:
        model = SupConResNet(name=opt.model)

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

def train_step_single(model, images, labels, criterion, optimizer):
    features = model(images)
    features1, features2 = torch.split(features, [256, 256], dim=0)
    features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)
    loss = criterion(features, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_step_multi(model, images, labels, criterion1, criterion2, criterion3, optimizer):
    features1, features2, features3 = model(images)
    features1_1, features1_2 = torch.split(features1, [256, 256], dim=0)
    features2_1, features2_2 = torch.split(features2, [256, 256], dim=0)
    features3_1, features3_2 = torch.split(features3, [256, 256], dim=0)
    features1 = torch.cat([features1_1.unsqueeze(1), features1_2.unsqueeze(1)], dim=1)
    features2 = torch.cat([features2_1.unsqueeze(1), features2_2.unsqueeze(1)], dim=1)
    features3 = torch.cat([features3_1.unsqueeze(1), features3_2.unsqueeze(1)], dim=1)
    loss1 = criterion1(features1, labels)
    loss2 = criterion2(features2, labels)
    loss3 = criterion3(features3, labels)
    loss = loss1 + loss2 + loss3
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def inference_step(model, images):
    return model(images)


def speed(train_loader, model, criterions, optimizer, opt):
    """one epoch training"""
    criterion1, criterion2, criterion3 = criterions

    for idx, (images, labels) in enumerate(train_loader):

        images1 = images[0]
        images2 = images[1]

        images = torch.cat([images1, images2], dim=0)
        labels_convert = label_convert(labels, num_classes=opt.num_classes)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        bsz = labels.shape[0]
        start_time = time.perf_counter()
        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        model.train()
        if opt.model == "resnet_multi":
            time.sleep(1)
            train_ms, train_ips = measure(train_step_multi(model, images, labels, criterion1, criterion2, criterion3, optimizer))
        else:
            time.sleep(1)
            train_ms, train_ips = measure(train_step_single(model, images, labels, criterion1, optimizer))

        model.eval()
        infer_ms, infer_ips = measure(inference_step)

        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()

    return train_ms, train_ips, infer_ms, infer_ips

def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)
    # build model and criterion
    model, criterions = set_model(opt)
    # build optimizer
    optimizer = set_optimizer(opt, model)

    train_ms, train_ips, infer_ms, infer_ips = speed(train_loader, model, criterions, optimizer, opt)
    print(f"Training:  {train_ms:.2f} ms/batch, {train_ips:.1f} images/s")
    print(f"Inference: {infer_ms:.2f} ms/batch, {infer_ips:.1f} images/s")





if __name__ == '__main__':
    main()
