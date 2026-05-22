import argparse
import pickle
import os
import numpy as np


def parse_option():
    parser = argparse.ArgumentParser('argument for pred disagreement')

    parser.add_argument("--backbone_model_direct1", type=str, default="./save/SupCon/cifar10_resnet18_trail_0_128_1.0/")
    parser.add_argument("--backbone_model_direct2", type=str, default="./save/SupCon/cifar10_resnet18_trail_0_128_0.5/")
    parser.add_argument("--file_name", type=str, default="pred_out")

    opt = parser.parse_args()

    opt.file1 = os.path.join(opt.backbone_model_direct1, opt.file_name)
    opt.file2 = os.path.join(opt.backbone_model_direct2, opt.file_name)

    return opt



if __name__ == "__main__":

    opt = parse_option()

    with open(opt.file1, "rb") as f:
        pred1 = pickle.load(f)
    pred1 = pred1.flatten()

    with open(opt.file2, "rb") as f:
        pred2 = pickle.load(f)
    pred2 = pred2.flatten()

    matching_positions = np.sum(pred1 == pred2)
    print("matching_positions", matching_positions, "total samples", len(pred1), "agreement", matching_positions*1./len(pred1))




