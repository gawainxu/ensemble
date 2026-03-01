#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 15:29:30 2022

@author: zhi
"""


import argparse
import os
import pickle
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from pre_models_dataset import ImageNet100, ImageNet_M, iCIFAR100, ImageNet50
from pre_models_dataset import imagenet50_medium_outliers, cifar_medium_outliers
from pre_models_dataset import DTD, mnist, my_mnistmed
from torch.utils.data import DataLoader, SubsetRandomSampler

from networks.resnet_big import SupCEResNet
from networks.vgg import vgg16, vgg16_bn
from networks.ViT import ViT, get_b16_config_cifar, get_b16_config
from networks.ViT_cifar import ViT_cifar

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


downsampling = {"cifar100": {"train":{"inliers": 1 , "outliers": 1}, "test":{"inliers": 1, "outliers": 1}},
                "imagenet50": {"train":{"inliers": 0.3, "outliers": 0.2}, "test":{"inliers": 1, "outliers": 1}},
                "imagenet50_medium": {"train":{"inliers": 0.1 , "outliers": 0.1}, "test":{"inliers": 0.1, "outliers": 0.1}},
                "cifar_medium": {"train":{"inliers": 0.3 , "outliers": 0.3}, "test":{"inliers": 1, "outliers": 1}},
                "imagenet50_far": {"train":{"inliers": 0.2 , "outliers": 0.2}, "test":{"inliers": 1, "outliers": 1}},
                "cifar_far": {"train":{"inliers": 0.2 , "outliers": 0.2}, "test":{"inliers": 1, "outliers": 1}},
                "medmnist_32": {"train":{"inliers": 0.2 , "outliers": 0.2}, "test":{"inliers": 1, "outliers": 1}},
                "medmnist_224": {"train":{"inliers": 0.2 , "outliers": 0.2}, "test":{"inliers": 1, "outliers": 1}}}


image_sizes = {"cifar100": 32, "imagenet50": 224,
                "imagenet50_medium": 224, "cifar_medium": 32,
                "imagenet50_far": 224, "cifar_far": 32,
                "medmnist_32": 32, "medmnist_224": 224}

num_classes_dict = {"imagenet100": 100, "imagenet50": 50,
                    "imagenet-m": 18, "cifar100": 50,
                    "imagenet50_medium": 50, "cifar_medium": 50,
                    "imagenet50_far": 50, "cifar_far": 50,
                    "medmnist_32": 50, "medmnist_224": 50}


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--layers_to_see', type=str, default="transformer.layers.5.1.net.5")

    parser.add_argument('--model', type=str, default='resnet18',
                        choices=["resnet18", "vgg16", "vit"])
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--outliers",  action="store_true", help="if the outlier data")
    parser.add_argument("--train_data",  action="store_true", help="if the training data")
    parser.add_argument("--target_class", type=int, default=-1)

    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--backbone_model_direct", type=str, default="/save/cifar100_models/CE_cifar100_vit16_lr_0.0003_decay_0.0001_bsz_128/")
    parser.add_argument("--backbone_model_name", type=str, default="last.pth")
    parser.add_argument("--features_save_path", type=str, default="")

    opt = parser.parse_args()

    opt.main_dir = os.getcwd()
    opt.backbone_model_direct = opt.main_dir + opt.backbone_model_direct
    opt.backbone_model_path = os.path.join(opt.backbone_model_direct, opt.backbone_model_name)

    opt.features_name = opt.model + "_" + opt.dataset + "_" + opt.layers_to_see
    opt.num_classes = num_classes_dict[opt.dataset]

    if opt.outliers:
        opt.features_name = opt.features_name + "_" + "outliers"
    else:
        opt.features_name = opt.features_name + "_" + "inliers"

    if opt.train_data:
        opt.features_name = opt.features_name + "_" + "train"
    else:
        opt.features_name = opt.features_name + "_" + "test"

    opt.features_path = os.path.join(os.path.join(opt.main_dir, "features"), opt.features_name)

    if opt.target_class > 0:
        opt.features_path = opt.features_path + "_" + str(opt.target_class)

    print("train_data", opt.train_data, "outliers", opt.outliers)

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
    elif "vit" in opt.model:
        if "cifar" in opt.dataset:
            configs = get_b16_config_cifar()
            #model = ViT_cifar(num_classes=opt.num_classes)
        elif "imagenet" in opt.dataset:
            configs = get_b16_config()
        opt.image_size = image_sizes[opt.dataset]
        model = ViT(image_size=opt.image_size, patch_size=configs.patch_size, num_classes=opt.num_classes,
                        embedding_dim=configs.embed_dim, depth=configs.depth, heads=configs.num_heads,
                        mlp_dim=configs.hidden_dim,
                        dim_head=configs.head_dim, dropout=configs.dropout, emb_dropout=configs.emb_dropout)

    model = load_model(model, opt.backbone_model_path)
    model = model.eval()

    return model


def load_data(opt):

    if opt.dataset =="imagenet100":
        dataset_train = ImageNet100(data_path=opt.data_path, train=True)
        dataset_test = ImageNet100(data_path=opt.data_path, train=False)
    elif opt.dataset == "imagenet50":
        dataset_train = ImageNet50(data_path=opt.data_path, train=True, outliers=opt.outliers, target_class=opt.target_class)
        dataset_test = ImageNet50(data_path=opt.data_path, train=False, outliers=opt.outliers, target_class=opt.target_class)
    elif opt.dataset == "imagenet-m":
        dataset_train = ImageNet_M(data_path=opt.data_path, train=True)
        dataset_test = ImageNet_M(data_path=opt.data_path, train=False)
    elif opt.dataset == "cifar100":
        dataset_train = iCIFAR100(data_path=opt.data_path, train=True, outliers=opt.outliers)
        dataset_test = iCIFAR100(data_path=opt.data_path, train=False, outliers=opt.outliers)
    elif opt.dataset == "imagenet50_medium":
        dataset_train = imagenet50_medium_outliers(data_path=opt.data_path)
        dataset_test = imagenet50_medium_outliers(data_path=opt.data_path)
    elif opt.dataset == "cifar_medium":
        dataset_train = cifar_medium_outliers(data_path=opt.data_path)
        dataset_test = cifar_medium_outliers(data_path=opt.data_path)
    elif opt.dataset == "imagenet50_far":
        dataset_train = DTD(data_path=opt.data_path)
        dataset_test = DTD(data_path=opt.data_path)
    elif opt.dataset == "cifar_far":
        dataset_train = mnist(data_path=opt.data_path)
        dataset_test = mnist(data_path=opt.data_path)
    elif opt.dataset == "medmnist_32":
        dataset_train = my_mnistmed(data_size=32, if_train=True)
        dataset_test = my_mnistmed(data_size=32, if_train=False)
    elif opt.dataset == "medmnist_224":
        dataset_train = my_mnistmed(data_size=224, if_train=True)
        dataset_test = my_mnistmed(data_size=224, if_train=False)

    print("dataset_train", len(dataset_train))
    print("dataset_test", len(dataset_test))

    # downssample the dataset
    if opt.outliers:
        train_ratio = downsampling[opt.dataset]["train"]["outliers"]
        test_ratio = downsampling[opt.dataset]["test"]["outliers"]
    else:
        train_ratio = downsampling[opt.dataset]["train"]["inliers"]
        test_ratio = downsampling[opt.dataset]["test"]["inliers"]

    num_keep_train = int(len(dataset_train) * train_ratio)
    indices_train = np.random.choice(len(dataset_train), num_keep_train, replace=False)
    print("num_keep_train", num_keep_train)
    print("indices_train", len(indices_train))
    num_keep_test = int(len(dataset_test) * test_ratio)
    indices_test = np.random.choice(len(dataset_test), num_keep_test, replace=False)
    print("num_keep_test", num_keep_test)
    print("indices_test", len(indices_test))

    sampler_train = SubsetRandomSampler(indices_train)
    sampler_test = SubsetRandomSampler(indices_test)

    dataloader_train = DataLoader(dataset_train, sampler=sampler_train, batch_size=opt.batch_size, shuffle=False)
    dataloader_test = DataLoader(dataset_test, sampler=sampler_test, batch_size=opt.batch_size, shuffle=False)

    return dataloader_train, dataloader_test



def normalFeatureReading_hook_class(model, opt, data_loader, target_class):
    outputs = []
    labels = []

    activation = {}
    print("data_loader", data_loader.__len__())

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

        print(i, label.item(), target_class)
        if label.item() != target_class:
            print("not in target")
            continue
        with torch.no_grad():
            if torch.cuda.is_available():
                img = img.float().cuda()
            else:
                img = img.float()
            activation = {}
            hook_output = model(img)
            # Output the full output tokens of the attention block, including the cls
            print("activation", activation.keys())
            outputs.append(activation[opt.layers_to_see].detach().cpu())
            labels.append(label.numpy().item())

    with open(opt.features_path, "wb") as f:
        pickle.dump((outputs, [], labels), f)



def normalFeatureReading_hook(model, opt, data_loader):
    outputs = []
    labels = []

    activation = {}
    print("data_loader", data_loader.__len__())

    def get_activation(name):
        def hook(model, input, output):
            if type(output) is tuple:
                output = output[1]
            print("hook working!!!", name, output.shape)
            activation[name] = output.detach()
            print("activation", activation.keys())

        return hook

    # https://zhuanlan.zhihu.com/p/87853615
    for name, module in model.named_modules():
        print(name)
        if name == opt.layers_to_see:
            handle = module.register_forward_hook(get_activation(name))

    for i, (img, label) in enumerate(data_loader):

        print(i)
        with torch.no_grad():
            if torch.cuda.is_available():
                img = img.float().cuda()
            else:
                img = img.float()
            activation = {}
            hook_output = model(img)
            # Output the full output tokens of the attention block, including the cls
            outputs.append(activation[opt.layers_to_see].detach().cpu())
            labels.append(label.numpy().item())

    with open(opt.features_path, "wb") as f:
        pickle.dump((outputs, [], labels), f)


if __name__ == "__main__":

    opt = parse_option()

    model = set_model(opt)
    print("Model loaded!!")

    featurePaths = []
    dataloader_train, dataloader_test = load_data(opt)

    if opt.train_data:
        normalFeatureReading_hook(model, opt, dataloader_train)
    else:
        normalFeatureReading_hook(model, opt, dataloader_test)


"""
ViT layers


to_patch_embedding
to_patch_embedding.0
to_patch_embedding.1
to_patch_embedding.2
to_patch_embedding.3
dropout
transformer
transformer.norm
transformer.layers
transformer.layers.0
transformer.layers.0.0
transformer.layers.0.0.norm
transformer.layers.0.0.attend
transformer.layers.0.0.dropout
transformer.layers.0.0.to_qkv
transformer.layers.0.0.to_out
transformer.layers.0.0.to_out.0
transformer.layers.0.0.to_out.1
transformer.layers.0.1
transformer.layers.0.1.net
transformer.layers.0.1.net.0
transformer.layers.0.1.net.1
transformer.layers.0.1.net.2
transformer.layers.0.1.net.3
transformer.layers.0.1.net.4
transformer.layers.0.1.net.5
transformer.layers.1
transformer.layers.1.0
transformer.layers.1.0.norm
transformer.layers.1.0.attend
transformer.layers.1.0.dropout
transformer.layers.1.0.to_qkv
transformer.layers.1.0.to_out
transformer.layers.1.0.to_out.0
transformer.layers.1.0.to_out.1
transformer.layers.1.1
transformer.layers.1.1.net
transformer.layers.1.1.net.0
transformer.layers.1.1.net.1
transformer.layers.1.1.net.2
transformer.layers.1.1.net.3
transformer.layers.1.1.net.4
transformer.layers.1.1.net.5
transformer.layers.2
transformer.layers.2.0
transformer.layers.2.0.norm
transformer.layers.2.0.attend
transformer.layers.2.0.dropout
transformer.layers.2.0.to_qkv
transformer.layers.2.0.to_out
transformer.layers.2.0.to_out.0
transformer.layers.2.0.to_out.1
transformer.layers.2.1
transformer.layers.2.1.net
transformer.layers.2.1.net.0
transformer.layers.2.1.net.1
transformer.layers.2.1.net.2
transformer.layers.2.1.net.3
transformer.layers.2.1.net.4
transformer.layers.2.1.net.5
transformer.layers.3
transformer.layers.3.0
transformer.layers.3.0.norm
transformer.layers.3.0.attend
transformer.layers.3.0.dropout
transformer.layers.3.0.to_qkv
transformer.layers.3.0.to_out
transformer.layers.3.0.to_out.0
transformer.layers.3.0.to_out.1
transformer.layers.3.1
transformer.layers.3.1.net
transformer.layers.3.1.net.0
transformer.layers.3.1.net.1
transformer.layers.3.1.net.2
transformer.layers.3.1.net.3
transformer.layers.3.1.net.4
transformer.layers.3.1.net.5
transformer.layers.4
transformer.layers.4.0
transformer.layers.4.0.norm
transformer.layers.4.0.attend
transformer.layers.4.0.dropout
transformer.layers.4.0.to_qkv
transformer.layers.4.0.to_out
transformer.layers.4.0.to_out.0
transformer.layers.4.0.to_out.1
transformer.layers.4.1
transformer.layers.4.1.net
transformer.layers.4.1.net.0
transformer.layers.4.1.net.1
transformer.layers.4.1.net.2
transformer.layers.4.1.net.3
transformer.layers.4.1.net.4
transformer.layers.4.1.net.5
transformer.layers.5
transformer.layers.5.0
transformer.layers.5.0.norm
transformer.layers.5.0.attend
transformer.layers.5.0.dropout
transformer.layers.5.0.to_qkv
transformer.layers.5.0.to_out
transformer.layers.5.0.to_out.0
transformer.layers.5.0.to_out.1
transformer.layers.5.1
transformer.layers.5.1.net
transformer.layers.5.1.net.0
transformer.layers.5.1.net.1
transformer.layers.5.1.net.2
transformer.layers.5.1.net.3
transformer.layers.5.1.net.4
transformer.layers.5.1.net.5
to_latent
mlp_head


vgg layers

features
features.0
features.1
features.2
features.3
features.4
features.5
features.6
features.7
features.8
features.9
features.10
features.11
features.12
features.13
features.14
features.15
features.16
features.17
features.18
features.19
features.20
features.21
features.22
features.23
features.24
features.25
features.26
features.27
features.28
features.29
features.30
features.31
features.32
features.33
features.34
features.35
features.36
features.37
features.38
features.39
features.40
features.41
features.42
features.43
avgpool
classifier
classifier.0
classifier.1
classifier.2
classifier.3
classifier.4
classifier.5
classifier.6

resnet layers

encoder
encoder.conv1
encoder.bn1
encoder.layer1
encoder.layer1.0
encoder.layer1.0.conv1
encoder.layer1.0.bn1
encoder.layer1.0.conv2
encoder.layer1.0.bn2
encoder.layer1.0.shortcut
encoder.layer1.1
encoder.layer1.1.conv1
encoder.layer1.1.bn1
encoder.layer1.1.conv2
encoder.layer1.1.bn2
encoder.layer1.1.shortcut
encoder.layer2
encoder.layer2.0
encoder.layer2.0.conv1
encoder.layer2.0.bn1
encoder.layer2.0.conv2
encoder.layer2.0.bn2
encoder.layer2.0.shortcut
encoder.layer2.0.shortcut.0
encoder.layer2.0.shortcut.1
encoder.layer2.1
encoder.layer2.1.conv1
encoder.layer2.1.bn1
encoder.layer2.1.conv2
encoder.layer2.1.bn2
encoder.layer2.1.shortcut
encoder.layer3
encoder.layer3.0
encoder.layer3.0.conv1
encoder.layer3.0.bn1
encoder.layer3.0.conv2
encoder.layer3.0.bn2
encoder.layer3.0.shortcut
encoder.layer3.0.shortcut.0
encoder.layer3.0.shortcut.1
encoder.layer3.1
encoder.layer3.1.conv1
encoder.layer3.1.bn1
encoder.layer3.1.conv2
encoder.layer3.1.bn2
encoder.layer3.1.shortcut
encoder.layer4
encoder.layer4.0
encoder.layer4.0.conv1
encoder.layer4.0.bn1
encoder.layer4.0.conv2
encoder.layer4.0.bn2
encoder.layer4.0.shortcut
encoder.layer4.0.shortcut.0
encoder.layer4.0.shortcut.1
encoder.layer4.1
encoder.layer4.1.conv1
encoder.layer4.1.bn1
encoder.layer4.1.conv2
encoder.layer4.1.bn2
encoder.layer4.1.shortcut
encoder.avgpool
fc


"""