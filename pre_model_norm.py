#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 15:29:30 2022

@author: zhi
"""

import os
import sys

BASE_PATH = "/home/sysgen/Jiawen/SupContrast-master"
sys.path.append(BASE_PATH)

import argparse

import torch
import numpy as np
import pickle
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA

from util import accuracy_plain, AUROC, down_sampling


torch.multiprocessing.set_sharing_strategy('file_system')


def parse_option():
    parser = argparse.ArgumentParser('argument for feature reading')

    parser.add_argument("--num_classes", type=int, default=50)
    parser.add_argument("--mode", type=str, default="pca", choices=["pca", "pooling"])

    parser.add_argument("--exemplar_features_path", type=str,
                        default="/features/resnet18_cifar100_encoder.layer4_inliers_train")
    parser.add_argument("--testing_known_features_path", type=str,
                        default="/features/resnet18_cifar100_encoder.layer4_inliers_test")
    parser.add_argument("--testing_unknown_features_path", type=str,
                        default="/features/resnet18_cifar100_encoder.layer4_outliers_test")

    parser.add_argument("--exemplar_features_path1", type=str, default=None)
    parser.add_argument("--testing_known_features_path1", type=str, default=None)
    parser.add_argument("--testing_unknown_features_path1", type=str, default=None)

    parser.add_argument("--exemplar_features_path2", type=str, default=None)
    parser.add_argument("--testing_known_features_path2", type=str, default=None)
    parser.add_argument("--testing_unknown_features_path2", type=str, default=None)

    opt = parser.parse_args()
    opt.main_dir = os.getcwd()

    opt.exemplar_features_path = opt.main_dir + opt.exemplar_features_path
    opt.testing_known_features_path = opt.main_dir + opt.testing_known_features_path
    opt.testing_unknown_features_path = opt.main_dir + opt.testing_unknown_features_path

    if opt.exemplar_features_path1 is not None:
        opt.exemplar_features_path1 = opt.main_dir + opt.exemplar_features_path1
    if opt.testing_known_features_path1 is not None:
        opt.testing_known_features_path1 = opt.main_dir + opt.testing_known_features_path1
    if opt.testing_unknown_features_path1 is not None:
        opt.testing_unknown_features_path1 = opt.main_dir + opt.testing_unknown_features_path1

    if opt.exemplar_features_path2 is not None:
        opt.exemplar_features_path2 = opt.main_dir + opt.exemplar_features_path2
    if opt.testing_known_features_path2 is not None:
        opt.testing_known_features_path2 = opt.main_dir + opt.testing_known_features_path2
    if opt.testing_unknown_features_path2 is not None:
        opt.testing_unknown_features_path2 = opt.main_dir + opt.testing_unknown_features_path2

    return opt


def load_model(model, linear_model=None, path=None):
    ckpt = torch.load(path, map_location='cpu')
    state_dict = ckpt['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict
    model.load_state_dict(state_dict)

    if linear_model is not None:
        state_dict_linear = ckpt['linear']
        new_state_dict = {}
        for k, v in state_dict_linear.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        linear_model.load_state_dict(state_dict)

        return model, linear_model

    return model


def sort_features(features_list, labels_list, opt):
    features_len = len(features_list)
    sorted_features = [[] for _ in range(opt.num_classes)]
    for i in range(features_len):
        f, l = features_list[i], labels_list[i]
        sorted_features[l].append(f.numpy())

    return sorted_features


def reshape_features(ori_features):
    ori_features = np.squeeze(np.concatenate(ori_features))
    ori_features = ori_features.reshape(ori_features.shape[0], -1)

    return ori_features


def feature_classifier(opt):
    with open(opt.exemplar_features_path, "rb") as f:
        features_exemplar, _, labels_examplar = pickle.load(f)

    if opt.exemplar_features_path1 is not None:
        with open(opt.exemplar_features_path1, "rb") as f:
            features_exemplar_head1, _, labels_examplar1 = pickle.load(f)

    if opt.exemplar_features_path2 is not None:
        with open(opt.exemplar_features_path2, "rb") as f:
            features_exemplar_head2, _, labels_examplar2 = pickle.load(f)


    if opt.testing_known_features_path is not None:
        with open(opt.testing_known_features_path, "rb") as f:
            features_testing_known_head, _, labels_testing_known = pickle.load(f)

    if opt.testing_known_features_path1 is not None:
        with open(opt.testing_known_features_path1, "rb") as f:
            features_testing_known_head1, _, labels_testing_known1 = pickle.load(f)


    if opt.testing_known_features_path2 is not None:
        with open(opt.testing_known_features_path2, "rb") as f:
            features_testing_known_head2, _, labels_testing_known2 = pickle.load(f)


    with open(opt.testing_unknown_features_path, "rb") as f:
        features_testing_unknown_head, _, labels_testing_unknown = pickle.load(f)

    if opt.testing_unknown_features_path1 is not None:
        with open(opt.testing_unknown_features_path1, "rb") as f:
            features_testing_unknown_head1, _, labels_testing_unknown1 = pickle.load(f)

    if opt.testing_unknown_features_path2 is not None:
        with open(opt.testing_unknown_features_path2, "rb") as f:
            features_testing_unknown_head2, _, labels_testing_unknown2 = pickle.load(f)

    features_testing_known_head = reshape_features(features_testing_known_head)
    features_testing_unknown_head = reshape_features(features_testing_unknown_head)
    prediction_logits_known_dis_in = np.linalg.norm(features_testing_known_head, axis=1)
    prediction_logits_unknown_dis_in = np.linalg.norm(features_testing_unknown_head, axis=1)

    # Process results AUROC and OSCR
    # for AUROC, convert labels to binary labels, assume inliers are positive
    labels_binary_known = [1 if i < 100 else 0 for i in labels_testing_known]
    labels_binary_unknown = [1 if i < 100 else 0 for i in labels_testing_unknown]
    labels_binary = np.array(labels_binary_known + labels_binary_unknown)

    probs_binary_dis = np.concatenate((prediction_logits_known_dis_in, prediction_logits_unknown_dis_in), axis=0)
    # print("probs_binary", probs_binary_dis)

    auroc = AUROC(labels_binary, probs_binary_dis)
    print("Dis AUROC is: ", auroc)

    return auroc


if __name__ == "__main__":
    opt = parse_option()

    # models, linear_model = set_model(opt)
    # print("Model loaded!!")

    auroc = feature_classifier(opt)  # oscr, acc_known


