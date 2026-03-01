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
from sklearn.covariance import LedoitWolf

from util import accuracy_plain, AUROC, down_sampling


torch.multiprocessing.set_sharing_strategy('file_system')


def parse_option():
    parser = argparse.ArgumentParser('argument for feature reading')

    parser.add_argument("--num_classes", type=int, default=50)
    parser.add_argument("--mode", type=str, default="pca", choices=["pca", "pooling", "none"])
    parser.add_argument("--split_mode", action="store_true")

    parser.add_argument("--exemplar_features_path", type=str,
                        default="/features/resnet18_imagenet50_encoder.avgpool_inliers_train")
    parser.add_argument("--testing_known_features_path", type=str,
                        default="/features/resnet18_imagenet50_encoder.avgpool_inliers_test")
    parser.add_argument("--testing_unknown_features_path", type=str,
                        default="/features/resnet18_imagenet50_encoder.avgpool_outliers_test")

    parser.add_argument("--exemplar_features_path1", type=str, default=None)
    parser.add_argument("--testing_known_features_path1", type=str, default=None)
    parser.add_argument("--testing_unknown_features_path1", type=str, default=None)

    parser.add_argument("--exemplar_features_path2", type=str, default=None)
    parser.add_argument("--testing_known_features_path2", type=str, default=None)
    parser.add_argument("--testing_unknown_features_path2", type=str, default=None)

    parser.add_argument("--exemplar_features_path3", type=str, default=None)
    parser.add_argument("--testing_known_features_path3", type=str, default=None)
    parser.add_argument("--testing_unknown_features_path3", type=str, default=None)

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

    if opt.exemplar_features_path3 is not None:
        opt.exemplar_features_path3 = opt.main_dir + opt.exemplar_features_path3
    if opt.testing_known_features_path3 is not None:
        opt.testing_known_features_path3 = opt.main_dir + opt.testing_known_features_path3
    if opt.testing_unknown_features_path3 is not None:
        opt.testing_unknown_features_path3 = opt.main_dir + opt.testing_unknown_features_path3

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
        #var = np.cov(features.astype(float), rowvar=False)
        lw = LedoitWolf().fit(features)

        stats.append((mu, lw.covariance_))

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
        elif "polling" in mode:
            gap = torch.nn.AdaptiveAvgPool2d((1,1))
            features = gap(features).numpy()
            features = features.view()
        else:
            print("test_features", features.shape)
            features = features.numpy()
            features = features.reshape(1, -1)
            features = np.squeeze(features)
        diss = []
        for i, (mu, var) in enumerate(stats):
            #print("features", features.shape, "mu", mu.shape, "var", var.shape)
            dis = mahalanobis(features, mu, np.linalg.inv(var))
            diss.append(dis)

        dis_logits_out.append(np.min(np.array(diss)) / np.sum(np.array(diss)))  # !!!!!!!!!!!!!!!!!! minus here !!!!!!!!!!!! to entsprechen 0 for outliers and 1 for inliers, unknown logits, flip for known logits
        dis_logits_in.append(-np.min(np.array(diss)))
        dis_preds.append(np.argmin(np.array(diss)))

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
    sorted_features_exemplar = sort_features(features_exemplar, labels_examplar, opt)

    if opt.exemplar_features_path1 is not None:
        with open(opt.exemplar_features_path1, "rb") as f:
            features_exemplar1, _, labels_examplar1 = pickle.load(f)
        if "vit" in opt.exemplar_features_path1:
            features_exemplar1 = [feat[:, 0] for feat in features_exemplar1]
        sorted_features_exemplar1 = sort_features(features_exemplar1, labels_examplar1, opt)

    if opt.exemplar_features_path2 is not None:
        with open(opt.exemplar_features_path2, "rb") as f:
            features_exemplar2, _, labels_examplar2 = pickle.load(f)
        if "vit" in opt.exemplar_features_path2:
            features_exemplar2 = [feat[:, 0] for feat in features_exemplar2]
        sorted_features_exemplar2 = sort_features(features_exemplar2, labels_examplar2, opt)

    if opt.exemplar_features_path3 is not None:
        with open(opt.exemplar_features_path3, "rb") as f:
            features_exemplar3, _, labels_examplar3 = pickle.load(f)
        if "vit" in opt.exemplar_features_path3:
            features_exemplar3 = [feat[:, 0] for feat in features_exemplar3]
        sorted_features_exemplar3 = sort_features(features_exemplar3, labels_examplar, opt)


    if "pca" in opt.mode:
        pca, sorted_features_exemplar = dimension_reduction_pca(sorted_features_exemplar)
        if opt.exemplar_features_path1 is not None:
            pca1, sorted_features_exemplar1 = dimension_reduction_pca(sorted_features_exemplar1)
        if opt.exemplar_features_path2 is not None:
            pca2, sorted_features_exemplar2 = dimension_reduction_pca(sorted_features_exemplar2)
        if opt.exemplar_features_path3 is not None:
            pca3, sorted_features_exemplar3 = dimension_reduction_pca(sorted_features_exemplar3)
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

    if opt.testing_known_features_path3 is not None:
        with open(opt.testing_known_features_path3, "rb") as f:
            features_testing_known3, _, labels_testing_known3 = pickle.load(f)
        if "vit" in opt.testing_known_features_path3:
            features_testing_known3 = [feat[:, 0] for feat in features_testing_known3]

    if "pca" in opt.mode:
        prediction_logits_known_dis_in, prediction_logits_known_dis_out, predictions_known_dis, acc_known_dis = distance_classifier(
        features_testing_known, labels_testing_known, sorted_features_exemplar, mode=opt.mode, pca=pca)

        if opt.testing_known_features_path1 is not None:
            prediction_logits_known_dis_in1, prediction_logits_known_dis_out1, predictions_known_dis1, acc_known_dis1 = distance_classifier(
                features_testing_known1, labels_testing_known1, sorted_features_exemplar1, mode=opt.mode,pca=pca1)

        if opt.testing_known_features_path2 is not None:
            prediction_logits_known_dis_in2, prediction_logits_known_dis_out2, predictions_known_dis2, acc_known_dis2 = distance_classifier(
                features_testing_known2, labels_testing_known2, sorted_features_exemplar2, mode=opt.mode, pca=pca2)

        if opt.testing_known_features_path3 is not None:
            prediction_logits_known_dis_in3, prediction_logits_known_dis_out3, predictions_known_dis3, acc_known_dis3 = distance_classifier(
                features_testing_known3, labels_testing_known3, sorted_features_exemplar3, mode=opt.mode, pca=pca3)
    else:
        prediction_logits_known_dis_in, prediction_logits_known_dis_out, predictions_known_dis, acc_known_dis = distance_classifier(
            features_testing_known, labels_testing_known, sorted_features_exemplar, mode=opt.mode)

        if opt.testing_known_features_path1 is not None:
            prediction_logits_known_dis_in1, prediction_logits_known_dis_out1, predictions_known_dis1, acc_known_dis1 = distance_classifier(
                features_testing_known1, labels_testing_known1, sorted_features_exemplar1, mode=opt.mode)

        if opt.testing_known_features_path2 is not None:
            prediction_logits_known_dis_in2, prediction_logits_known_dis_out2, predictions_known_dis2, acc_known_dis2 = distance_classifier(
                features_testing_known2, labels_testing_known2, sorted_features_exemplar2, mode=opt.mode)

        if opt.testing_known_features_path3 is not None:
            prediction_logits_known_dis_in3, prediction_logits_known_dis_out3, predictions_known_dis3, acc_known_dis3 = distance_classifier(
                features_testing_known3, labels_testing_known3, sorted_features_exemplar3, mode=opt.mode)


    with open(opt.testing_unknown_features_path, "rb") as f:
        features_testing_unknown, _, labels_testing_unknown = pickle.load(f)
    if "vit" in opt.testing_unknown_features_path:
        features_testing_unknown = [feat[:, 0] for feat in features_testing_unknown]
        print("features_testing_unknown", features_testing_unknown[0].shape)

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

    if opt.testing_unknown_features_path3 is not None:
        with open(opt.testing_unknown_features_path3, "rb") as f:
            features_testing_unknown3, _, labels_testing_unknown3 = pickle.load(f)
        if "vit" in opt.testing_unknown_features_path3:
            features_testing_unknown3 = [feat[:, 0] for feat in features_testing_unknown3]

    if "pca" in opt.mode:
        prediction_logits_unknown_dis_in, prediction_logits_unknown_dis_out, predictions_unknown_dis, acc_unknown_dis = distance_classifier(
        features_testing_unknown, labels_testing_unknown, sorted_features_exemplar, mode=opt.mode, pca=pca)

        if opt.testing_unknown_features_path1 is not None:
            prediction_logits_unknown_dis_in1, prediction_logits_unknown_dis_out1, predictions_unknown_dis1, acc_unknown_dis1 = distance_classifier(
                features_testing_unknown1, labels_testing_unknown1, sorted_features_exemplar1, mode=opt.mode, pca=pca1)

        if opt.testing_unknown_features_path2 is not None:
            prediction_logits_unknown_dis_in2, prediction_logits_unknown_dis_out2, predictions_unknown_dis2, acc_unknown_dis2 = distance_classifier(
                features_testing_unknown2, labels_testing_unknown2, sorted_features_exemplar2, mode=opt.mode, pca=pca2)

        if opt.testing_unknown_features_path3 is not None:
            prediction_logits_unknown_dis_in3, prediction_logits_unknown_dis_out3, predictions_unknown_dis3, acc_unknown_dis3 = distance_classifier(
                features_testing_unknown3, labels_testing_unknown3, sorted_features_exemplar3, mode=opt.mode, pca=pca3)

    else:
        prediction_logits_unknown_dis_in, prediction_logits_unknown_dis_out, predictions_unknown_dis, acc_unknown_dis = distance_classifier(
            features_testing_unknown, labels_testing_unknown, sorted_features_exemplar, mode=opt.mode)

        if opt.testing_unknown_features_path1 is not None:
            prediction_logits_unknown_dis_in1, prediction_logits_unknown_dis_out1, predictions_unknown_dis1, acc_unknown_dis1 = distance_classifier(
                features_testing_unknown1, labels_testing_unknown1, sorted_features_exemplar1, mode=opt.mode)

        if opt.testing_unknown_features_path2 is not None:
            prediction_logits_unknown_dis_in2, prediction_logits_unknown_dis_out2, predictions_unknown_dis2, acc_unknown_dis2 = distance_classifier(
                features_testing_unknown2, labels_testing_unknown2, sorted_features_exemplar2, mode=opt.mode)

        if opt.testing_unknown_features_path3 is not None:
            prediction_logits_unknown_dis_in3, prediction_logits_unknown_dis_out3, predictions_unknown_dis3, acc_unknown_dis3 = distance_classifier(
                features_testing_unknown3, labels_testing_unknown3, sorted_features_exemplar3, mode=opt.mode)


    #distance_predictions = np.concatenate((predictions_known_dis, predictions_unknown_dis), axis=0)
    #labels_testing = np.concatenate((labels_testing_known, labels_testing_unknown), axis=0)

    # Process results AUROC and OSCR
    # for AUROC, convert labels to binary labels, assume inliers are positive
    labels_binary_known = [1 if i < 100 else 0 for i in labels_testing_known]
    labels_binary_unknown = [1 if i < 100 else 0 for i in labels_testing_unknown]
    labels_binary = np.array(labels_binary_known + labels_binary_unknown)

    if opt.testing_unknown_features_path1 is not None:
        norm_acc = acc_known_dis / (acc_known_dis + acc_known_dis1 + acc_known_dis2 + acc_known_dis3)
        norm_acc1 = acc_known_dis1 / (acc_known_dis + acc_known_dis1 + acc_known_dis2 + acc_known_dis3)
        norm_acc2 = acc_known_dis2 / (acc_known_dis + acc_known_dis1 + acc_known_dis2 + acc_known_dis3)
        norm_acc3 = acc_known_dis3 / (acc_known_dis + acc_known_dis1 + acc_known_dis2 + acc_known_dis3)
        prediction_logits_known_dis_in = norm_acc * prediction_logits_known_dis_in + norm_acc1 * prediction_logits_known_dis_in1 + norm_acc2 * prediction_logits_known_dis_in2 + norm_acc3 * prediction_logits_known_dis_in3
        prediction_logits_unknown_dis_in = norm_acc * prediction_logits_unknown_dis_in + norm_acc1 * prediction_logits_unknown_dis_in1 + norm_acc2 * prediction_logits_unknown_dis_in2 + norm_acc3 * prediction_logits_unknown_dis_in3

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


