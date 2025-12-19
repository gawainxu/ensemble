#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 15:29:30 2022

@author: zhi
"""

import os
import platform
import sys
BASE_PATH = "/home/sysgen/Jiawen/causal_OSR"
sys.path.append(BASE_PATH) 

import argparse

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
import pickle
from itertools import chain

from networks.resnet_big import SupConResNet, LinearClassifier
from networks.resnet_preact import SupConpPreactResNet
from networks.simCNN import simCNN_contrastive
from networks.mlp import SupConMLP
from featureMerge import featureMerge
from dataUtil import num_inlier_classes_mapping

from torch.utils.data import DataLoader
from dataUtil import get_train_datasets, get_test_datasets, get_outlier_datasets, osr_splits_inliers, osr_splits_outliers

torch.multiprocessing.set_sharing_strategy('file_system')


breaks = {"cifar-10-100-10": {"train": 5000, "test_known":500, "test_unknown": 50, "full": 100000}, 
          "cifar-10-100-50": {"train": 5000, "test_known": 500, "test_unknown": 50, "full": 100000}, 
           'cifar10':{"train": 5000, "test_known": 500, "test_unknown": 500, "full": 100000}, 
           "tinyimgnet":{"train": 5000, "test_known": 100, "test_unknown": 20, "full": 100000}, 
           'mnist':{"train": 5000, "test_known": 500, "test_unknown": 500, "full": 100000}, 
           "svhn":{"train": 5000, "test_known": 500, "test_unknown": 500, "full": 100000}}

def parse_option():

    parser = argparse.ArgumentParser('argument for feature reading')

    parser.add_argument('--datasets', type=str, default='tinyimgnet',
                        choices=["cifar-10-100-10", "cifar-10-100-50", 'cifar10', "tinyimgnet", 'mnist', "svhn"], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--model', type=str, default="resnet18", choices=["resnet18", "resnet34", "preactresnet18", "preactresnet34", "simCNN", "MLP"])
    parser.add_argument("--model_path", type=str, default="/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_trail_0_128_0.005_256/last.pth")
    parser.add_argument("--linear_model_path", type=str, default=None)
    parser.add_argument("--trail", type=int, default=0)
    parser.add_argument("--split_train_val", type=bool, default=True)
    parser.add_argument("--action", type=str, default="feature_reading",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown", "feature_reading"])
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument("--feature_save", type=str, default="/features/")
    parser.add_argument("--layers_to_see", type=list, default=["encoder.conv1", "encoder.layer1", "encoder.layer2",
                                                               "encoder.layer3", "encoder.layer4", "encoder.avgpool", "head"])

    # temperature
    parser.add_argument('--temp', type=float, default=0.005, help='temperature for loss')
    parser.add_argument("--architecture", type=str, default="single", choices=["single", "multi"])
    parser.add_argument("--ensemble_num", type=int, default=1)
    parser.add_argument("--feat_dim", type=int, default=128)

    parser.add_argument("--if_train", type=str, default="test_known", choices=['train', 'val', 'test_known', 'test_unknown', "full"])
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')


    opt = parser.parse_args()

    opt.main_dir = os.getcwd()
    opt.model_path = opt.main_dir + opt.model_path
    opt.feature_save = opt.main_dir + opt.feature_save
    if opt.linear_model_path is not None:
        opt.linear_model_path = opt.main_dir + opt.linear_model_path

    opt.n_cls = len(osr_splits_inliers[opt.datasets][opt.trail])
    opt.n_outs = len(osr_splits_outliers[opt.datasets][opt.trail])

    opt.break_idx = breaks[opt.datasets][opt.if_train]
    if platform.system() == 'Windows':
        opt.model_name = opt.model_path.split("\\")[-2]
    elif platform.system() == 'Linux':
        opt.model_name = opt.model_path.split("/")[-2]
    opt.save_path_all = opt.feature_save + opt.model_name + "_" + opt.if_train

    opt.num_classes = num_inlier_classes_mapping[opt.datasets]

    return opt


def load_model(opt):

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
    else:
        model = simCNN_contrastive(opt,  feature_dim=opt.feat_dim, in_channels=in_channels)
    ckpt = torch.load(opt.model_path, map_location='cpu')
    state_dict = ckpt['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v

    state_dict = new_state_dict
    model = model.cpu()
    model.load_state_dict(state_dict)
    model.eval()

    return model


def normalFeatureReading_old(data_loader, model, opt):
    
    outputs_backbone = []
    outputs = []
    labels = []

    for i, (img, label) in enumerate(data_loader):
        
        print(i)
        if i > opt.break_idx:
            break

        if opt.method == "SupCon":
            output, output_encoder = model(img)[0], model.encoder(img)
        else:
            output = model.encoder(img)

        outputs.append(output.detach().numpy())
        outputs_backbone.append(output_encoder[-1].detach().numpy())

        labels.append(label.numpy())

    with open(opt.save_path, "wb") as f:
        pickle.dump((outputs, outputs_backbone, labels), f)


def normalFeatureReading(model, opt, data_loader):
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
        for l in opt.layers_to_see:
            if name == l:
                module.register_forward_hook(get_activation(name))

    for i, (img, label) in enumerate(data_loader):

        print(i)
        if i > 100:
            break
        img = img.float()
        activation = {}
        hook_output = model(img)
        if type(hook_output) is tuple:
            hook_output = hook_output[1]
        outputs.append(activation)
        labels.append(label.numpy().item())

    with open(opt.save_path, "wb") as f:
        pickle.dump((outputs, labels), f)



def set_data(opt, class_idx=None):

    if opt.if_train == "train" or opt.if_train == "full":
        datasets = get_train_datasets(opt, class_idx)
    elif opt.if_train == "test_known":
        datasets = get_test_datasets(opt, class_idx)
    elif opt.if_train == "test_unknown":
        datasets = get_outlier_datasets(opt)

    return datasets
        

if __name__ == "__main__":
    
    opt = parse_option()

    model = load_model(opt)
    print("Model loaded!!")
    
    featurePaths= []

    if opt.if_train == "train" or opt.if_train == "test_known" or opt.if_train == "full":
        for r in range(0, opt.n_cls):                 
            opt.save_path = opt.feature_save + "temp" + str(r)
            featurePaths.append(opt.save_path)
            datasets = set_data(opt, class_idx=r)
            dataloader = DataLoader(datasets, batch_size=1, shuffle=False, sampler=None, 
                                    num_workers=1)
            normalFeatureReading(model, opt, dataloader)

        featureMerge(featurePaths, opt)

    else:
         for r in range(0, opt.n_outs):                            
            opt.save_path = opt.feature_save + "/temp" + str(r)
            featurePaths.append(opt.save_path)
            datasets = set_data(opt, class_idx=r)
            dataloader = DataLoader(datasets, batch_size=1, shuffle=False, sampler=None, 
                                    num_workers=1)
            normalFeatureReading(model, opt, dataloader)

         featureMerge(featurePaths, opt)
