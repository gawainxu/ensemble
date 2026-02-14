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


def feature_stats(inlier_features):
    stats = []
    for features in inlier_features:
        features = np.squeeze(np.array(features))
        mu = np.mean(features, axis=0)
        var = np.cov(features.astype(float), rowvar=False)

        stats.append((mu, var))

    return stats


def distances(stats, test_features, mode="pca", pca=None):
    dis_logits_out = []
    dis_logits_in = []
    dis_preds = []
    for features in test_features:
        if "pca" in mode:
            features = features.numpy()
            features = features.reshape(1, -1)
            features = pca.transform(features)
            features = np.squeeze(features)
        else:
            gap = torch.nn.AdaptiveAvgPool2d((1,1))
            features = gap(features).numpy()
            features = features.view()
        diss = []
        for i, (mu, var) in enumerate(stats):
            norm_features = features / np.linalg.norm(features)
            norm_mu = mu / np.linalg.norm(mu)
            dis = np.dot(norm_features, norm_mu)
            diss.append(dis)

        dis_logits_out.append(np.max(np.array(diss)) / np.sum(np.array(diss)))   # actually useless
        dis_logits_in.append(np.max(np.array(diss)))
        dis_preds.append(np.argmax(np.array(diss)))

    return dis_logits_in, dis_logits_out, dis_preds


def distance_classifier(testing_features, testing_labels, sorted_training_features, mode, pca=None):
    stats = feature_stats(sorted_training_features)
    dis_logits_in, dis_logits_out, dis_preds = distances(stats, testing_features, mode, pca)

    acc = accuracy_plain(dis_preds, testing_labels)
    print("Distance Accuracy is: ", acc)

    return dis_logits_in, dis_logits_out, dis_preds, acc


def sort_features(features_list, labels_list, opt):
    features_len = len(features_list)
    sorted_features = [[] for _ in range(opt.num_classes)]
    for i in range(features_len):
        f, l = features_list[i], labels_list[i]
        sorted_features[l].append(f.numpy())

    return sorted_features


def dimension_reduction_pca(sorted_features):

    features_bundle = [np.concatenate(sf) for sf in sorted_features]
    features_bundle = np.squeeze(np.concatenate(features_bundle))
    features_bundle = features_bundle.reshape(features_bundle.shape[0], -1)
    pca = PCA(n_components=374, whiten=True, svd_solver='randomized')
    pca.fit(features_bundle)

    sorted_features = [pca.transform(np.concatenate(sf).reshape(len(sf), -1)) for sf in sorted_features]

    return pca, sorted_features


def dimension_reduction_pooling(sorted_features):

    sorted_features_new = []
    for sf in sorted_features:
        gap = torch.nn.AdaptiveAvgPool2d((1,1))
        sf = [gap(f) for f in sf]
        sorted_features_new.append(sf)

    return sorted_features_new


def feature_classifier(opt):
    with open(opt.exemplar_features_path, "rb") as f:
        features_exemplar, _, labels_examplar = pickle.load(f)
    if "vit" in opt.exemplar_features_path:
        features_exemplar = [feat[:, 0] for feat in features_exemplar]

    if opt.exemplar_features_path1 is not None:
        with open(opt.exemplar_features_path1, "rb") as f:
            features_exemplar1, _, labels_examplar1 = pickle.load(f)
        if "vit" in opt.exemplar_features_path1:
            features_exemplar1 = [feat[:, 0] for feat in features_exemplar1]

    if opt.exemplar_features_path2 is not None:
        with open(opt.exemplar_features_path2, "rb") as f:
            features_exemplar2, _, labels_examplar2 = pickle.load(f)
        if "vit" in opt.exemplar_features_path2:
            features_exemplar2 = [feat[:, 0] for feat in features_exemplar2]

    sorted_features_exemplar = sort_features(features_exemplar, labels_examplar, opt)
    if "pca" in opt.mode:
        pca, sorted_features_exemplar = dimension_reduction_pca(sorted_features_exemplar)
    elif "pooling" in opt.mode:
        sorted_features_exemplar = dimension_reduction_pooling(sorted_features_exemplar)


    if opt.testing_known_features_path is not None:
        with open(opt.testing_known_features_path, "rb") as f:
            features_testing_known, _, labels_testing_known = pickle.load(f)
        if "vit" in opt.testing_known_features_path:
            features_testing_known = [feat[:, 0] for feat in features_testing_known]

    if opt.testing_known_features_path1 is not None:
        with open(opt.testing_known_features_path1, "rb") as f:
            features_testing_known1, _, labels_testing_known1 = pickle.load(f)
        if "vit" in opt.testing_known_features_path1:
            features_testing_known1 = [feat[:, 0] for feat in features_testing_known1]


    if opt.testing_known_features_path2 is not None:
        with open(opt.testing_known_features_path2, "rb") as f:
            features_testing_known2, _, labels_testing_known2 = pickle.load(f)
        if "vit" in opt.testing_known_features_path2:
            features_testing_known2 = [feat[:, 0] for feat in features_testing_known2]

    if "pca" in opt.mode:
        prediction_logits_known_dis_in, prediction_logits_known_dis_out, predictions_known_dis, acc_known_dis = distance_classifier(
        features_testing_known, labels_testing_known, sorted_features_exemplar, mode=opt.mode, pca=pca)

    with open(opt.testing_unknown_features_path, "rb") as f:
        features_testing_unknown, _, labels_testing_unknown = pickle.load(f)
    if "vit" in opt.testing_unknown_features_path:
        features_testing_unknown = [feat[:, 0] for feat in features_testing_unknown]

    if opt.testing_unknown_features_path1 is not None:
        with open(opt.testing_unknown_features_path1, "rb") as f:
            features_testing_unknown1, _, labels_testing_unknown1 = pickle.load(f)
        if "vit" in opt.testing_unknown_features_path1:
            features_testing_unknown1 = [feat[:, 0] for feat in features_testing_unknown1]

    if opt.testing_unknown_features_path2 is not None:
        with open(opt.testing_unknown_features_path2, "rb") as f:
            features_testing_unknown2, _, labels_testing_unknown2 = pickle.load(f)
        if "vit" in opt.testing_unknown_features_path2:
            features_testing_unknown2 = [feat[:, 0] for feat in features_testing_unknown2]

    if "pca" in opt.mode:
        prediction_logits_unknown_dis_in, prediction_logits_unknown_dis_out, predictions_unknown_dis, acc_unknown_dis = distance_classifier(
        features_testing_unknown, labels_testing_unknown, sorted_features_exemplar, mode=opt.mode, pca=pca)

    distance_predictions = np.concatenate((predictions_known_dis, predictions_unknown_dis), axis=0)
    labels_testing = np.concatenate((labels_testing_known, labels_testing_unknown), axis=0)


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


