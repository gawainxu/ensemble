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
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import pickle
import copy
from itertools import chain
from scipy.spatial.distance import mahalanobis

from util import accuracy, AverageMeter, accuracy_plain, AUROC, OSCR, down_sampling
from distance_utils import sortFeatures
from dataUtil import get_test_datasets, get_outlier_datasets

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_option():
    parser = argparse.ArgumentParser('argument for feature reading')

    parser.add_argument('--datasets', type=str, default='tinyimgnet',
                        choices=["cifar-10-100-10", "cifar-10-100-50", 'cifar10', "tinyimgnet", 'mnist', "svhn"],
                        help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument("--ensembles", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=20)

    parser.add_argument("--exemplar_features_path", type=str, default="/features/tinyimgnet_resnet_multi_trail_0_128_1024_1.0_0.5_0.05_256_train")
    parser.add_argument("--testing_known_features_path", type=str, default="/features/tinyimgnet_resnet_multi_trail_0_128_1024_1.0_0.5_0.05_256_test_known")
    parser.add_argument("--testing_unknown_features_path", type=str, default="/features/tinyimgnet_resnet_multi_trail_0_128_1024_1.0_0.5_0.05_256_test_unknown")

    parser.add_argument("--trail", type=int, default=0)
    parser.add_argument("--split_train_val", type=bool, default=True)
    parser.add_argument("--action", type=str, default="testing_known",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown",
                                 "feature_reading"])
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument("--temp", type=str, default=0.5)
    parser.add_argument("--lr", type=str, default=0.001)
    parser.add_argument("--training_bz", type=int, default=200)
    parser.add_argument("--mem_size", type=int, default=500)
    parser.add_argument("--if_train", type=str, default="train", choices=['train', 'val', 'test_known', 'test_unknown'])
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument("--downsampling_ratio_known", type=int, default=10)
    parser.add_argument("--downsampling_ratio_unknown", type=int, default=10)

    parser.add_argument("--K", type=int, default=20)

    parser.add_argument("--auroc_save_path", type=str, default="./plots/auroc")

    parser.add_argument("--with_outliers", type=bool, default=False)
    parser.add_argument("--downsample", type=bool, default=False)
    parser.add_argument("--last_feature_path", type=str, default=None)
    parser.add_argument("--downsample_ratio", type=float, default=0)

    opt = parser.parse_args()

    opt = parser.parse_args()
    opt.main_dir = os.getcwd()

    opt.exemplar_features_path = opt.main_dir + opt.exemplar_features_path
    opt.testing_known_features_path = opt.main_dir + opt.testing_known_features_path
    opt.testing_unknown_features_path = opt.main_dir + opt.testing_unknown_features_path
    opt.auroc_save_path = opt.main_dir + opt.auroc_save_path

    return opt


def KNN_logits(testing_features, sorted_exemplar_features):
    testing_similarity_logits = []

    for idx, testing_feature in enumerate(testing_features):
        # print(idx)
        similarity_logits = []
        for training_features_c in sorted_exemplar_features:
            training_features_c = np.array(training_features_c, dtype=float)
            similarities = np.matmul(training_features_c, testing_feature) / np.linalg.norm(training_features_c,
                                                                                            axis=1) / np.linalg.norm(
                testing_feature)
            ind = np.argsort(similarities)[-opt.K:]
            top_k_similarities = similarities[ind]
            similarity_logits.append(np.sum(top_k_similarities))

        testing_similarity_logits.append(similarity_logits)

    testing_similarity_logits = np.array(testing_similarity_logits)
    testing_similarity_logits = np.divide(testing_similarity_logits.T,
                                          np.sum(testing_similarity_logits, axis=1)).T  # normalization

    return testing_similarity_logits


def distances(stats, test_features, mode="mahalanobis"):
    dis_logits_out = []
    dis_logits_in = []
    dis_preds = []
    for features in test_features:
        diss = []
        for i, (mu, var) in enumerate(stats):
            # mu, var = stats[0]                             ##### delete
            if mode == "mahalanobis":
                features_normalized = features - mu
                # dis =  np.matmul(features_normalized, np.linalg.inv(var))
                # dis = np.matmul(dis, np.swapaxes(features_normalized, 0, 1))
                # dis = dis[0][0]
                dis = mahalanobis(features, mu, np.linalg.pinv(var))
            else:
                features = np.squeeze(np.array(features))
                dis = features - mu
                dis = np.sum(np.abs(dis))

            diss.append(dis)

        dis_logits_out.append(np.min(np.array(diss)) / np.sum(np.array(
            diss)))  # !!!!!!!!!!!!!!!!!! minus here !!!!!!!!!!!! to entsprechen 0 for outliers and 1 for inliers, unknown logits, flip for known logits
        dis_logits_in.append(-np.min(np.array(diss)) / np.sum(np.array(diss)))
        dis_preds.append(np.argmin(np.array(diss)))

    return dis_logits_in, dis_logits_out, dis_preds


def KNN_classifier(testing_features, testing_labels, sorted_training_features):
    print("Begin KNN Classifier!")
    testing_similarity_logits = KNN_logits(testing_features, sorted_training_features)
    prediction_logits, predictions = np.amax(testing_similarity_logits, axis=1), np.argmax(testing_similarity_logits,
                                                                                           axis=1)

    acc = accuracy_plain(predictions, testing_labels)
    print("KNN Accuracy is: ", acc)

    return prediction_logits, predictions, acc


def distance_classifier(testing_features, testing_labels, sorted_training_features):
    stats = feature_stats(sorted_training_features)
    dis_logits_in, dis_logits_out, dis_preds = distances(stats, testing_features)

    acc = accuracy_plain(dis_preds, testing_labels)
    print("Distance Accuracy is: ", acc)

    return dis_logits_in, dis_logits_out, dis_preds, acc


def feature_stats(inlier_features):
    stats = []
    for features in inlier_features:
        features = np.squeeze(np.array(features))
        mu = np.mean(features, axis=0)
        var = np.cov(features.astype(float), rowvar=False)

        stats.append((mu, var))

    return stats


def cat_head_features(head_features):
    new_head_features = []
    for hf in head_features:
        hf = [torch.squeeze(i) for i in hf]
        hf = torch.cat(hf, dim=0)
        new_head_features.append(hf.detach().numpy())

    head_features = np.array(new_head_features)
    return head_features


def feature_classifier(opt):
    with open(opt.exemplar_features_path, "rb") as f:
        features_exemplar_heads, _, labels_examplar = pickle.load(f)
        features_exemplar_backbone = cat_head_features(features_exemplar_heads)
        sorted_features_examplar_backbone = sortFeatures(features_exemplar_backbone, labels_examplar, opt)

    if opt.testing_known_features_path is not None:
        with open(opt.testing_known_features_path, "rb") as f:
            features_testing_known_backbone, _, labels_testing_known = pickle.load(f)
            features_testing_known_backbone = cat_head_features(features_testing_known_backbone)
            labels_testing_known = np.squeeze(np.array(labels_testing_known))

    features_testing_known_backbone, labels_testing_known = down_sampling(features_testing_known_backbone,
                                                                          labels_testing_known,
                                                                          opt.downsampling_ratio_known)
    prediction_logits_known, predictions_known, acc_known = KNN_classifier(features_testing_known_backbone,
                                                                           labels_testing_known,
                                                                           sorted_features_examplar_backbone)
    prediction_logits_known_dis_in, prediction_logits_known_dis_out, predictions_known_dis, acc_known_dis = distance_classifier(
        features_testing_known_backbone, labels_testing_known, sorted_features_examplar_backbone)


    with open(opt.testing_unknown_features_path, "rb") as f:
        features_testing_unknown_backbone, _, labels_testing_unknown = pickle.load(f)
        features_testing_unknown_backbone = cat_head_features(features_testing_unknown_backbone)
        labels_testing_unknown = np.squeeze(np.array(labels_testing_unknown))
        print("features_testing_unknown_backbone", features_testing_unknown_backbone.shape)

    features_testing_unknown_backbone, labels_testing_unknown = down_sampling(features_testing_unknown_backbone,
                                                                              labels_testing_unknown,
                                                                              opt.downsampling_ratio_unknown)
    prediction_logits_unknown, predictions_unknown, _ = KNN_classifier(features_testing_unknown_backbone,
                                                                       labels_testing_unknown,
                                                                       sorted_features_examplar_backbone)
    prediction_logits_unknown_dis_in, prediction_logits_unknown_dis_out, predictions_unknown_dis, acc_unknown_dis = distance_classifier(
        features_testing_unknown_backbone, labels_testing_unknown, sorted_features_examplar_backbone)

    # Process results AUROC and OSCR
    # for AUROC, convert labels to binary labels, assume inliers are positive
    labels_binary_known = [1 if i < 100 else 0 for i in labels_testing_known]
    labels_binary_unknown = [1 if i < 100 else 0 for i in labels_testing_unknown]
    labels_binary = np.array(labels_binary_known + labels_binary_unknown)
    # print("labels_binary", labels_binary)

    probs_binary = np.concatenate((prediction_logits_known, prediction_logits_unknown), axis=0)

    auroc = AUROC(labels_binary, probs_binary, opt)
    print("AUROC is: ", auroc)

    probs_binary_dis = np.concatenate((prediction_logits_known_dis_in, prediction_logits_unknown_dis_in), axis=0)
    # print("probs_binary", probs_binary_dis)

    auroc = AUROC(labels_binary, probs_binary_dis, opt)
    print("Dis AUROC is: ", auroc)

    # OSCR
    oscr = OSCR(np.array(prediction_logits_known_dis_out), np.array(prediction_logits_unknown_dis_out),
                predictions_known, labels_testing_known)
    print("OSCR is: ", oscr)

    return auroc


if __name__ == "__main__":
    opt = parse_option()

    auroc = feature_classifier(opt)  # oscr, acc_known
