import pickle
import argparse
import numpy as np
import torch

from util import CKA, linear_CKA


def parse_option():

    parser = argparse.ArgumentParser('argument for feature analysis')
    
    parser.add_argument("--feature_path1", type=str,
                        default="./features/cifar10_resnet18_trail_0_128_0.05_256_test_known")
    parser.add_argument("--feature_path2", type=str,
                        default="./features/cifar10_resnet18_trail_0_128_0.01_256_test_known")
    parser.add_argument("--layers_to_see", type=list, default=["encoder.layer1", "encoder.layer2", "encoder.layer3",  "encoder.layer4", "encoder.avgpool",
                                                               "head"])   #"encoder.conv1", "encoder.layer1", "encoder.layer2",
                                                               #"encoder.layer3", "encoder.layer4", "encoder.avgpool", "head"
    parser.add_argument("--channel_wise", type=bool, default=False)
    opt = parser.parse_args()

    return opt


def sort_features(features, opt):

    sorted_features = dict()
    for k in opt.layers_to_see:
        sorted_features[k] = []
    features_len = len(features)

    for i in range(features_len):
        for k in opt.layers_to_see:
            sorted_features[k].append(features[i][k].numpy())

    for k in sorted_features.keys():
        sorted_features[k] = np.concatenate(sorted_features[k])

    return sorted_features


def analysis(sorted_features1, sorted_features2, opt):

    for k in sorted_features1.keys():
        if k not in opt.layers_to_see:
            continue
        f1 = sorted_features1[k]
        f2 = sorted_features2[k]
        if opt.channel_wise and "layer" in k:
            num_channels = f1.shape[1]
            cka = []
            for i in range(num_channels):
                f1_i = f1[:,i,:,:]
                f2_i = f2[:,i,:,:]
                f1_i = np.reshape(f1_i, (f1_i.shape[0], -1))
                f2_i = np.reshape(f2_i, (f2_i.shape[0], -1))
                cka_i = CKA(f1_i, f2_i)
                cka.append(cka_i)
            print(k, sum(cka)/len(cka))
        else:
            f1 = np.reshape(f1, (f1.shape[0], -1))
            f2 = np.reshape(f2, (f2.shape[0], -1))
            f1 = f1 - f1.mean(axis=0, keepdims=True)
            f2 = f2 - f2.mean(axis=0, keepdims=True)
            cka = linear_CKA(f1, f2)
            print(k, cka)


def main():
    
    opt = parse_option()

    with open(opt.feature_path1, "rb") as f:
        features1, labels1 = pickle.load(f)
    with open(opt.feature_path2, "rb") as f:
        features2, labels2 = pickle.load(f)

    sorted_features1 = sort_features(features1, opt)
    sorted_features2 = sort_features(features2, opt)

    analysis(sorted_features1, sorted_features2, opt)



if __name__ == '__main__':
    main()
