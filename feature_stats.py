import torch

import os
import pickle
import numpy as np
import argparse

from distance_utils import EuclideanDistance, EuclideanStat, sortFeatures
from manipulate_features import find_dominant
import matplotlib.pyplot as plt


def arg_parse():

    parser = argparse.ArgumentParser('argument for statistics')
    parser.add_argument("--inlier_feature_paths", type=list, default=["/features/cifar10_resnet18_trail_0_128_1.0_test_known",
                                                                      "/features/cifar10_resnet18_trail_0_128_0.5_test_known",
                                                                      "/features/cifar10_resnet18_trail_0_128_0.1_test_known",
                                                                      "/features/cifar10_resnet18_trail_0_128_0.05_test_known",
                                                                      "/features/cifar10_resnet18_trail_0_128_0.01_test_known",
                                                                      "/features/cifar10_resnet18_trail_0_128_0.005_test_known",
                                                                      "/features/tinyimgnet_resnet18_trail_0_128_1.0_test_known",
                                                                      "/features/tinyimgnet_resnet18_trail_0_128_0.5_test_known",
                                                                      "/features/tinyimgnet_resnet18_trail_0_128_0.1_test_known",
                                                                      "/features/tinyimgnet_resnet18_trail_0_128_0.05_test_known",
                                                                      "/features/tinyimgnet_resnet18_trail_0_128_0.01_test_known",
                                                                      "/features/tinyimgnet_resnet18_trail_0_128_0.005_test_known"])
   
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--reduced_len", type=int, default=0)

    opt = parser.parse_args()
    opt.main_dir = os.getcwd()

    return opt


def interClassSeperation(means):
    
    """
    It is to compute the distances 
    """
    distances = []
    for i in range(0, len(means)):
        for j in range(i+1, len(means)):
            distances.append(EuclideanDistance(means[i], means[j]))
            
    return np.min(np.array(distances))



def intraClassSeperation(sortedFeatures, classCenters):
    
    num_instances  = 0
    sum_distances = 0
    for features, center in zip(sortedFeatures, classCenters):
        num_instances += len(features)
        for feature in features:
            sum_distances += EuclideanDistance(feature, center)
            
    return sum_distances / num_instances


def gram(features):
    # features: nxd
    features = np.array(features)
    gram_matrix = np.matmul(features, features.T)

    return gram_matrix



if __name__ == "__main__":
        
    opt = arg_parse()
    temperatures = ["Cifar10 1.0", "Cifar10 0.5", "Cifar10 0.1", "Cifar10 0.05", "Cifar10 0.01", "Cifar10 0.005",
                    "Tinyimgnet 1.0", "Tinyimgnet 0.5", "Tinyimgnet 0.1", "Tinyimgnet 0.05", "Tinyimgnet 0.01", "Tinyimgnet 0.005"]
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(14, 4))
    
    for i, (inlier_feature_path, ax) in enumerate(zip(opt.inlier_feature_paths, axes.flatten())):
        inlier_feature_path = opt.main_dir + inlier_feature_path
        with open(inlier_feature_path, "rb") as f:
            featuresInliers, _, labelsInliers = pickle.load(f)
            
        if i < 6:
            opt.num_classes = 6
        else:
            opt.num_classes = 20
        sortedFeatures = sortFeatures(featuresInliers, labelsInliers, opt)
        gram_matrices = []
  
        for c in range(opt.num_classes):
            features_c = sortedFeatures[c]
            gram_matrix_c = gram(features_c)
            gram_matrices.append(gram_matrix_c)
            
        gram_matrices = np.array(gram_matrices)
        #gram_matrices = np.mean(gram_matrices, axis=0)
        print("Mean of gram matrix is", np.mean(gram_matrices))
        im = ax.imshow(gram_matrices[3], cmap='viridis') 
        ax.set_title(temperatures[int(i)], fontsize=10)
        # Remove axis ticks for a cleaner look
        ax.axis('off')
    
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=12)
    
    #plt.savefig("./plots/gram.pdf", bbox_inches='tight', pad_inches=0)
        
    """
    Mean of gram matrix is 0.9191464
    Mean of gram matrix is 0.92738026
    Mean of gram matrix is 0.9641541
    Mean of gram matrix is 0.98164225
    Mean of gram matrix is 0.99598855
    Mean of gram matrix is 0.9982147
    Mean of gram matrix is 0.64396924
    Mean of gram matrix is 0.60834914
    Mean of gram matrix is 0.7555859
    Mean of gram matrix is 0.8891498
    Mean of gram matrix is 0.9786245
    Mean of gram matrix is 0.98934424
    """
        
    """
    if opt.reduced_len > 0:
        featuresInliers, reduced_idx = find_dominant(featuresInliers, opt.reduced_len)

    means = EuclideanStat(sortedFeatures)    
    
    print("Intra Seperation: ", intraClassSeperation(sortedFeatures, means))
    print("Inter Seperation: ", interClassSeperation(means))
    print("Ratio: ", intraClassSeperation(sortedFeatures, means)/interClassSeperation(means))
    """

