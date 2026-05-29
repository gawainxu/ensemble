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
                                                                      "/features/tinyimgnet_resnet18_trail_0_128_1.0_test_known",
                                                                      "/features/tinyimgnet_resnet18_trail_0_128_0.5_test_known",
                                                                      "/features/tinyimgnet_resnet18_trail_0_128_0.1_test_known",
                                                                      "/features/tinyimgnet_resnet18_trail_0_128_0.05_test_known",
                                                                      "/features/tinyimgnet_resnet18_trail_0_128_0.01_test_known",])
   
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
    temperatures = ["Cifar10 1.0", "Cifar10 0.5", "Cifar10 0.1", "Cifar10 0.05", "Cifar10 0.01", 
                    "Tinyimgnet 1.0", "Tinyimgnet 0.5", "Tinyimgnet 0.1", "Tinyimgnet 0.05", "Tinyimgnet 0.01"]
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(14, 4))
    
    for i, (inlier_feature_path, ax) in enumerate(zip(opt.inlier_feature_paths, axes.flatten())):
        inlier_feature_path = opt.main_dir + inlier_feature_path
        with open(inlier_feature_path, "rb") as f:
            featuresInliers, _, labelsInliers = pickle.load(f)
            
        if i < 5:
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
        gram_matrices = np.mean(gram_matrices, axis=0)
        print("Mean of gram matrix is", np.mean(gram_matrices))
        #im = ax.imshow(gram_matrices[3], cmap='viridis') 
        #ax.set_title(temperatures[int(i)], fontsize=10)
        # Remove axis ticks for a cleaner look
        #ax.axis('off')
    
    #cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, fraction=0.046, pad=0.02)
    #cbar.ax.tick_params(labelsize=12)
    
    #plt.savefig("./plots/gram.pdf", bbox_inches='tight', pad_inches=0)
    
    cifar=[0.9191464, 0.92738026, 0.9641541, 0.98164225, 0.99598855]
    tinyimagenet=[0.64396924, 0.60834914, 0.7555859, 0.8891498, 0.9786245]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color_left = 'tab:red'
    ax1.set_xlabel('Temperatures', fontsize=18)
    ax1.set_ylabel('Cifar10', color=color_left, fontsize=20)
    ax1.plot(cifar, color=color_left, linewidth=2, label='Cifar10')
    ax1.tick_params(axis='y', labelcolor=color_left)  # Matches tick text color to line
    
    ax2 = ax1.twinx()  
    color_right = 'tab:blue'
    ax2.set_ylabel('TinyImageNet', color=color_right, fontsize=20)
    ax2.plot(tinyimagenet, color=color_right, linewidth=2,  label='TinyImageNet')
    ax2.tick_params(axis='y', labelcolor=color_right) # Matches tick text color to line

    ax1.set_xlim(0, 4)
    tick_positions = np.arange(0, 5, 1) 
    ax1.set_xticks(tick_positions)
    tick_labels = ["1.0", "0.5", "0.1", "0.05", "0.01"] 
    ax1.set_xticklabels(tick_labels)     


# Style the actual tick marks (make them a bit longer/thicker)
    ax1.tick_params(axis='x', direction='out', length=5, width=1.5, colors='black')

# 4. Combine legends from both axes into a single box
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=18)


    plt.title("Mean of the Gram Matrices vs. Temperature", fontsize=24)
    plt.tight_layout()
    plt.savefig("./plots/gram_mean.pdf")
    plt.show()


 
