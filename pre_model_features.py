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
from pre_models_dataset import DTD, mnist
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
                "imagenet50": {"train":{"inliers": 0.3 , "outliers": 0.3}, "test":{"inliers": 1, "outliers": 1}},
                "imagenet50_medium": {"train":{"inliers": 0.3 , "outliers": 0.3}, "test":{"inliers": 1, "outliers": 1}},
                "cifar_medium": {"train":{"inliers": 0.3 , "outliers": 0.3}, "test":{"inliers": 1, "outliers": 1}},
                "imagenet50_far": {"train":{"inliers": 1 , "outliers": 1}, "test":{"inliers": 1, "outliers": 1}},
                "cifar_far": {"train":{"inliers": 0.2 , "outliers": 0.2}, "test":{"inliers": 1, "outliers": 1}}}


image_sizes = {"cifar100": 32, "imagenet50": 224,
                "imagenet50_medium": 224, "cifar_medium": 32,
                "imagenet50_far": 224, "cifar_far": 32}


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--layers_to_see', type=str, default="transformer.layers.5.1.net.5")

    parser.add_argument('--model', type=str, default='vgg16',
                        choices=["resnet18", "vgg16", "vit"])
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--outliers",  action="store_true", help="if the outlier data")
    parser.add_argument("--train_data",  action="store_true", help="if the training data")

    parser.add_argument("--data_path_train", type=str, default=None)
    parser.add_argument("--data_path_test", type=str, default=None)
    parser.add_argument("--backbone_model_direct", type=str, default="/save/cifar100_models/CE_cifar100_vit16_lr_0.0003_decay_0.0001_bsz_128/")
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

    num_classes_dict = {"imagenet100": 100, "imagenet50": 50, "imagenet-m": 18, "cifar100": 50,
                        "imagenet50_medium": 50, "cifar_medium": 50,
                        "imagenet50_far": 50, "cifar_far": 50,}

    if "imagenet100" in opt.dataset:
        dataset_train = ImageNet100(train=True)
        dataset_test = ImageNet100(train=False)
    elif "imagenet50" in opt.dataset:
        dataset_train = ImageNet50(train=True, outliers=opt.outliers)
        dataset_test = ImageNet50(train=False, outliers=opt.outliers)
    elif "imagenet-m" in opt.dataset:
        dataset_train = ImageNet_M(train=True)
        dataset_test = ImageNet_M(train=False)
    elif "cifar100" in opt.dataset:
        dataset_train = iCIFAR100(root="../datasets", train=True, outliers=opt.outliers)
        dataset_test = iCIFAR100(root="../datasets", train=False, outliers=opt.outliers)
    elif "imagenet50_medium" in opt.dataset:
        dataset_train = imagenet50_medium_outliers()
        dataset_test = imagenet50_medium_outliers()
    elif "cifar_medium" in opt.dataset:
        dataset_train = cifar_medium_outliers()
        dataset_test = cifar_medium_outliers()
    elif "imagenet50_far" in opt.dataset:
        dataset_train = DTD()
        dataset_test = DTD()
    elif "cifar_far" in opt.dataset:
        dataset_train = mnist()
        dataset_test = mnist()

    print("dataset_train", len(dataset_train))
    print("dataset_test", len(dataset_test))

    opt.num_classes = num_classes_dict[opt.dataset]

    # downssample the dataset
    if opt.outliers:
        train_ratio = downsampling[opt.dataset]["train"]["outliers"]
        test_ratio = downsampling[opt.dataset]["test"]["outliers"]
    else:
        train_ratio = downsampling[opt.dataset]["train"]["inliers"]
        test_ratio = downsampling[opt.dataset]["test"]["inliers"]

    num_keep_train = int(len(dataset_train) * train_ratio)
    indices_train = np.random.choice(len(dataset_train), num_keep_train, replace=False)
    num_keep_test = int(len(dataset_test) * test_ratio)
    indices_test = np.random.choice(len(dataset_test), num_keep_test, replace=False)

    sampler_train = SubsetRandomSampler(indices_train)
    sampler_test = SubsetRandomSampler(indices_test)

    dataloader_train = DataLoader(dataset_train, sampler=sampler_train, batch_size=opt.batch_size, shuffle=False)
    dataloader_test = DataLoader(dataset_test, sampler=sampler_test, batch_size=opt.batch_size, shuffle=False)

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

    featurePaths = []
    dataloader_train, dataloader_test = load_data(opt)

    model = set_model(opt)
    print("Model loaded!!")

    if opt.train_data:
        print("Training Data")
        normalFeatureReading_hook(model, opt, dataloader_train)
    else:
        print("Testing Data")
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

"""