import math
import os
import argparse
import numpy as np
import torch
import pickle
from dataUtil import continual_splits


def parse_option():

    parser = argparse.ArgumentParser('argument for feature comparision')

    parser.add_argument('--datasets', type=str, default='voc',
                        choices=["cifar100", 'cifar10', "tinyimgnet", 'mnist', "svhn", "voc"], help='dataset')
    parser.add_argument("--feature_path1", type=str, default="/features/voc_resnet18_task_0_128_32_mask_0_0")
    parser.add_argument("--feature_path2", type=str, default="/features/voc_resnet18_task_1_128_32_mask_0_0")
    parser.add_argument("--task_id", type=int, default=0, help="Task ID for comparing")
    
    opt = parser.parse_args()
    opt.main_dir = os.getcwd()
    opt.feature_path1 = opt.main_dir + opt.feature_path1
    opt.feature_path2 = opt.main_dir + opt.feature_path2
    
    opt.num_classes = opt.task_id * continual_splits[opt.datasets]
    
    return opt



def hsic(matrix_x: np.ndarray, matrix_y: np.ndarray) -> float:
    n = matrix_x.shape[0]
    matrix_h = np.identity(n) - (1.0 / n) * np.ones((n, n))

    x_times_h = np.matmul(matrix_x, matrix_h)
    y_times_h = np.matmul(matrix_y, matrix_h)

    return 1.0 / ((n - 1) ** 2) * np.trace(np.matmul(x_times_h, y_times_h))


def linear_cka(matrix_x: np.ndarray, matrix_y: np.ndarray) -> float:
    
    if matrix_x.ndim > 2:
        matrix_x = matrix_x.reshape([matrix_x.shape[0], -1])
    
    if matrix_y.ndim > 2:
        matrix_y = matrix_y.reshape([matrix_y.shape[0], -1])
    
    # First center the columns
    matrix_x = matrix_x - np.mean(matrix_x, 0)
    matrix_y = matrix_y - np.mean(matrix_y, 0)

    matrix_x = np.matmul(matrix_x, matrix_x.T)
    matrix_y = np.matmul(matrix_y, matrix_y.T)

    matrix_h = hsic(matrix_x=matrix_x, matrix_y=matrix_y)
    matrix_x = np.sqrt(hsic(matrix_x=matrix_x, matrix_y=matrix_x))
    matrix_y = np.sqrt(hsic(matrix_x=matrix_y, matrix_y=matrix_y))
    return matrix_h / (matrix_x * matrix_y)


class TorchCKA(object):
    def __init__(self, device):
        self.device = device

    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        identity = torch.eye(n, device=self.device)
        H = identity - unit / n
        return torch.matmul(torch.matmul(H, K), H)

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= -0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)
    
    

def sort_features(opt, features, labels):
   
    sorted_features = []
    min_num_samples = 1300

    for i in range(opt.num_classes):
        features_i = []
        for f, l in zip(features, labels):

            if l == + opt.num_classes * opt.data_task_id:    # i: #
                features_i.append(f)
        if len(features_i) < min_num_samples:
            min_num_samples = len(features_i)
        sorted_features.append(features_i)
        
    sorted_features = [features[:min_num_samples] for features in sorted_features]
    sorted_features = np.array(sorted_features)
    print("sorted_features", sorted_features.shape)
    return sorted_features


if __name__ == "__main__":
    
    opt = parse_option()
    
    with open(opt.feature_path1, "rb") as f:
        features1, labels1 = pickle.load(f)
        
    with open(opt.feature_path2, "rb") as f:
        features2, labels2 = pickle.load(f)
        
    sorted_features1 = sort_features(opt, features1, labels1)
    sorted_features2 = sort_features(opt, features2, labels2)
    
    cka = []
    
    for i in range(opt.num_classes):
        print(i)
        cka_i = linear_cka(sorted_features1[i], sorted_features2[i])
        cka.append(cka_i)
        
    print(opt.feature_path1, "mean cka", sum(cka)/len(cka))
        
    
    
    
    
