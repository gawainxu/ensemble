import argparse
import pickle
import os
import numpy as np

"""
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
"""


import matplotlib.pyplot as plt
import numpy as np

# 1. Create a random 10x10 matrix
agreement1 = np.array([[0.2196,	0.2286,	0.2685,	0.1221,	0.1822,	0.2206],
             [0, 0.2323, 0.3176, 0.1343, 0.1661, 0.2015],
             [0, 0,	0.3557,	0.205, 0.2268, 0.2687],
             [0, 0,	0, 0.1534, 0.2212, 0.2685],
             [0, 0,	0, 0, 0.2259, 0.3287],
             [0, 0,	0, 0, 0, 0.3277]])


agreement2 = np.array([[0.13, 0.1201, 0.0795, 0.1, 0.0845, 0.0896],
             [0, 0.1932, 0.1381, 0.1543, 0.1243, 0.129],
             [0, 0,	0.1509,	0.1343,	0.1318,	0.1297],
             [0, 0,	0, 0.1484, 0.1371, 0.1179],
             [0, 0,	0, 0, 0.1496, 0.1313],
             [0, 0, 0, 0, 0, 0.1556]])


acc1 = np.array([[0.96,	0.97,	0.97,	0.97,	0.97,	0.97],
[0,	0.96,	0.97,	0.97,	0.97,	0.97],
[0,	0,	0.97,	0.97,	0.97,	0.97],
[0,	0,	0,	0.97,	0.97,	0.97],
[0,	0,	0,	0,	0.97,	0.97],
[0,	0,	0,	0,	0,	0.97]])


acc2 = np.array([[0.44,	0.69,	0.76,	0.78,	0.77,	0.77],
[0,	0.71,	0.75,	0.78,	0.77,	0.77,],
[0,	0,	0.75,	0.78,	0.77,	0.77],
[0,	0,	0,	0.76,	0.76,	0.78],
[0,	0,	0,	0,	0.76,	0.79],
[0,	0,	0,	0,	0,	0.75]])


x_labels = ['1.0', '0.5', '0.1', '0.05', "0.01", "0.005"]
y_labels = ['1.0', '0.5', '0.1', '0.05', "0.01", "0.005"]


# 3. Create the upper triangular mask
mask = np.triu(np.ones_like(agreement1, dtype=bool))
masked_data1 = np.ma.array(acc1, mask=~mask)
masked_data2 = np.ma.array(acc2, mask=~mask)

# 4. Set up side-by-side subplots (1 row, 2 columns)
# constrained_layout=True manages the spacing automatically
fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

# 5. Set up the custom colormap with a clean white background for masked cells
current_cmap = plt.cm.get_cmap('YlOrBr').copy()
current_cmap.set_bad(color='white')

# 6. Plot Matrix 1
im1 = axes[0].imshow(masked_data1, cmap=current_cmap, origin='upper', vmin=acc1.min(), vmax=acc1.max())
axes[0].set_title("Cifar10", fontsize=26)

axes[0].set_xticks(range(len(x_labels)))              # Set positions first
axes[0].set_xticklabels(x_labels, rotation=45, ha='right') # Set text + formatting
axes[0].set_yticks(range(len(y_labels)))              # Set positions first
axes[0].set_yticklabels(y_labels, fontsize=12)        # Set text + formatting

# 7. Plot Matrix 2 (CRITICAL: pass the exact same vmin and vmax)
im2 = axes[1].imshow(masked_data2, cmap=current_cmap, origin='upper', vmin=acc2.min(), vmax=acc2.max())
axes[1].set_title("TInyImageNet", fontsize=26)

axes[1].set_xticks(range(len(x_labels)))              # Set positions first
axes[1].set_xticklabels(x_labels, rotation=45, ha='right') # Set text + formatting
axes[1].set_yticks(range(len(y_labels)))              # Set positions first
axes[1].set_yticklabels(y_labels, fontsize=12)        # Set text + formatting

# 8. Create one single colorbar attached to the entire figure
# Passing 'im2' (or im1) works perfectly now since they share the same scale mapping
fig.colorbar(im2, ax=axes, shrink=0.7)

plt.savefig("./plots/pred.pdf")
plt.show()



