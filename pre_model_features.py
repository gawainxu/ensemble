#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 15:29:30 2022

@author: zhi
"""


import argparse
import os
import pickle

import torch
import torch.backends.cudnn as cudnn
from pre_models_dataset import ImageNet100, ImageNet_M, iCIFAR100, ImageNet50
from torch.utils.data import DataLoader

from networks.resnet_big import SupCEResNet
from networks.vgg import vgg16, vgg16_bn
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--layers_to_see', type=str, default="encoder.layer1")

    parser.add_argument('--model', type=str, default='resnet18',
                        choices=["resnet18", "resnet50"])
    parser.add_argument("--dataset", type=str, default="imagenet-m")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--outliers", type=bool, default=False)
    parser.add_argument("--train_data", type=bool, default=False)

    parser.add_argument("--data_path_train", type=str, default="../datasets/imagenet-M-train")
    parser.add_argument("--data_path_test", type=str, default="../datasets/imagenet-M-test2")
    parser.add_argument("--backbone_model_direct", type=str, default="/save/SupCon/imagenet-m_models/CE_imagenet-m_resnet18_lr_0.2_decay_0.0001_bsz_128_cosine")
    parser.add_argument("--backbone_model_name", type=str, default="last.pth")
    parser.add_argument("--features_save_path", type=str, default="")

    opt = parser.parse_args()

    opt.main_dir = os.getcwd()
    opt.backbone_model_direct = opt.main_dir + opt.backbone_model_direct
    opt.backbone_model_path = os.path.join(opt.backbone_model_direct, opt.backbone_model_name)

    opt.features_name = opt.model + "_" + opt.dataset + "_" + opt.layers_to_see
    if opt.outliers:
        opt.features_name = opt.features_name + "_" + "outliers"
    else:
        opt.features_name = opt.features_name + "_" + "inliers"

    if opt.train_data:
        opt.features_name = opt.features_name + "_" + "train"
    else:
        opt.features_name = opt.features_name + "_" + "test"

    opt.features_path = os.path.join(os.path.join(opt.main_dir, "features"), opt.features_name)

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

    if opt.model == "resnet18" or opt.model == "resnet50":
        model = SupCEResNet(name=opt.model, num_classes=opt.num_classes)
    elif "vgg" in opt.model:
        model = vgg16_bn(num_classes=opt.num_classes)

    model = load_model(model, opt.backbone_model_path)
    model = model.eval()

    return model


def load_data(opt):

    num_classes_dict = {"imagenet100": 100, "imagenet50": 50, "imagenet-m": 18, "cifar100": 50}

    if "imagenet100" in opt.dataset:
        dataset_train = ImageNet100(opt.data_path_train, train=True)
        dataset_test = ImageNet100(opt.data_path_test, train=False)
    elif "imagenet50" in opt.dataset:
        dataset_train = ImageNet50(opt.data_path_train, train=True, outliers=opt.outliers)
        dataset_test = ImageNet50(opt.data_path_test, train=False, outliers=opt.outliers)
    elif "imagenet-m" in opt.dataset:
        dataset_train = ImageNet_M(opt.data_path_train, train=True)
        dataset_test = ImageNet_M(opt.data_path_test, train=False)
    elif "cifar100" in opt.dataset:
        dataset_train = iCIFAR100(root="../datasets", train=True, outliers=opt.outliers)
        dataset_test = iCIFAR100(root="../datasets", train=False, outliers=opt.outliers)

    opt.num_classes = num_classes_dict[opt.dataset]
    dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False)

    return dataloader_train, dataloader_test

def normalFeatureReading(data_loader, model, opt):
    outputs = []
    labels = []

    for i, (img, label) in enumerate(data_loader):

        print(i)
        if i > opt.break_idx:
            break

        output_encoder = model.encoder(img)
        output_encoder = torch.squeeze(output_encoder)

        outputs.append(output_encoder.detach().numpy())
        labels.append(label.numpy())

    with open(opt.save_path, "wb") as f:
        pickle.dump((outputs, labels), f)


def normalFeatureReading_hook(model, opt, data_loader):
    outputs = []
    labels = []

    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            if type(output) is tuple:
                output = output[1]
            print("hook working!!!", name, output.shape)
            activation[name] = output.detach()

        return hook

    # https://zhuanlan.zhihu.com/p/87853615
    for name, module in model.named_modules():
        print(name)
        if name == opt.layers_to_see:
            handle = module.register_forward_hook(get_activation(name))

    for i, (img, label) in enumerate(data_loader):

        with torch.no_grad():
            img = img.float().cuda()
            activation = {}
            hook_output = model(img)
            outputs.append(activation[opt.layers_to_see].detach().cpu())
            labels.append(label.numpy().item())

    with open(opt.features_path, "wb") as f:
        pickle.dump((outputs, [], labels), f)


if __name__ == "__main__":

    opt = parse_option()

    featurePaths = []
    dataloader_train, dataloader_test = load_data(opt)

    model = set_model(opt)
    print("Model loaded!!")

    if opt.train_data:
        normalFeatureReading_hook(model, opt, dataloader_train)
    else:
        normalFeatureReading_hook(model, opt, dataloader_test)