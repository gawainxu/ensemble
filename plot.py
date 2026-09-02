# visualize the feature ensemble results
#https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot

import matplotlib.pyplot as plt
import numpy as np

"""
values = {"ACCURACY": np.array([90.1, 92.2, 93.1]),
          "AUROC": np.array([69.1, 78.1, 87.3])} 
settings = (r"$\tau$=0.05", r"$\tau$=0.01", r"$\tau$=0.005")

x = np.arange(len(settings))  # the label locations
width = 0.25  
fig, ax1 = plt.subplots()
bottom = np.zeros(3)

metric, v = values.popitem()
p1 = ax1.bar(x, v, width, label=metric, color="blue")
bottom += v
ax1.bar_label(p1, padding=3)
ax1.set_ylabel('AUROC (%)')
ax1.set_ylim(60, 100)

ax2 = ax1.twinx()
metric, v = values.popitem()
p2 = ax2.bar(x+width, v, width, label=metric, color="red")
ax2.bar_label(p2, padding=3)
ax2.set_ylabel('Accuracy (%)')
ax2.set_ylim(70, 100)


#fig.legend(bbox_to_anchor=(1.2, 1))
fig.legend(loc=7)
fig.tight_layout()
fig.subplots_adjust(right=0.75)

ax1.set_title('Values of Inlier Accuracy and Outlier AUROC', fontsize=16)
ax1.set_xticks(x+width/2, settings)
plt.savefig("./test.pdf")
plt.show()

"""

"""
acc_inliers = [0.762, 0.836, 0.767, 0.804, 0.774, 0.748, 0.74]
acc_outliers = [0.32, 0.33, 0.34, 0.32, 0.41, 0.45, 0.45]

auroc_knn = [0.7373, 0.7724, 0.7337, 0.7677, 0.7837, 0.7887, 0.7801]
auroc_dis = [0.6591, 0.6658, 0.6799, 0.667, 0.6873, 0.7021, 0.7188]

settings = ("G1", "G2", "G3", "G4", "G1&G2", "G1&G2&G3", "G1&G2&G3&G4")
x = np.arange(len(settings))
width = 0.35  

# 1. Increase figsize width (from default to 12)
fig, ax1 = plt.subplots(figsize=(9, 6))

# Primary Axis (Accuracy)
p1 = ax1.bar(x, acc_outliers, width, label="Open Set Accuracy", color="green")
p3 = ax1.plot(acc_inliers, "-*", color="blue", label="In Set Accuracy", linewidth=3, markersize=8)
ax1.bar_label(p1, padding=40, label_type='center', fontsize=10, fontweight='bold',)
ax1.set_ylabel('Accuracy (%)',  fontsize=26)
ax1.set_ylim(0, 0.9)

# Secondary Axis (AUROC)
ax2 = ax1.twinx()
p2 = ax2.bar(x + width, auroc_dis, width, label="OSR AUROC", color="orange")
ax2.bar_label(p2, padding=80, label_type='center', fontsize=10, fontweight='bold',)
ax2.set_ylabel('OSR AUROC (%)', fontsize=26)
ax2.set_ylim(0, 0.9)


for i, val in enumerate(acc_inliers):
    ax1.annotate(
        text=f'{val:.3f}',            # Formats to 3 decimal places
        xy=(i, val),                  # Position of the data point
        xytext=(0, 8),                # Moves text 8 points directly above the point
        textcoords="offset points",   
        ha='center',                  # Centers text horizontally over the point
        va='bottom',                  
        fontsize=10,                  
        fontweight='bold',
        color='blue'                  # Color matches the line plot
    )

# Formatting
#ax1.set_title('Linear Probe on Open Sets and OSR Performance', fontsize=30)
ax1.set_xticks(x + width / 2)
ax1.set_xticklabels(settings, rotation=9, fontsize=14)

# 2. Legend: Horizontal (ncol=2) and placed at the bottom center
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.15), ncol=3, frameon=False,  fontsize=16)

# 3. Adjust bottom margin to make room for the legend
plt.subplots_adjust(bottom=0.2)
plt.savefig("./plots/cifar_marco.pdf")
plt.show()
"""


single_cifar = [76.48, 75.48, 80.58, 80.24, 83.78, 84.86]
multi_cifar_mean = [73.91, 77.44, 80.79, 81.64, 81.28, 80.64]
multi_cifar_max = [77.5, 80.6, 83.87, 85.84, 84.37, 84.29]

single_acc_cifar = [94.35, 94.35, 94.35, 94.02, 95.02, 94.35]
multi_acc_cifar_mean = [94.43, 94.72, 94.19, 94.09, 94.35, 93.97]
multi_acc_cifar_max = [96.01, 92.69, 93.02, 93.02, 94.68, 94.35]

single_tiny = [78.52, 76.4, 75.57, 68.9, 76, 69.57]
multi_tiny_mean = [77.46, 77.47, 78.32, 78.9, 80.21, 78.99]
multi_tiny_max = [87.8, 86, 85.14, 86.38, 83.67, 86]

single_acc_tiny = [76, 68, 71, 72, 75, 76]
multi_acc_tiny_mean = [71.38, 73.5, 75.11, 74.89, 76.5, 76.29]
multi_acc_tiny_max = [73, 77, 77, 75, 79, 79]

settings = ("1.0", "0.5", "0.1", "0.05", "0.01", "0.005")
group_gap = 0.72          # smaller = closer bar groups
x = np.arange(len(settings)) * group_gap
width = 0.106

offsets = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]) * width

# 1. Increase figsize width (from default to 12)
fig, ax1 = plt.subplots(figsize=(15, 4))

# Primary Axis (Accuracy)
p1 = ax1.bar(x+ offsets[0], single_cifar, width, label="AUROC Single", color="green")
ax1.bar_label(p1, padding=40, label_type='center', fontsize=10, )
ax1.set_ylabel('AUROC (%)',  fontsize=26)
ax1.set_ylim(0, 110)

# Secondary Axis (AUROC)
p2 = ax1.bar(x + offsets[1], multi_cifar_mean, width, label="AUROC Multi Mean", color="orange")
ax1.bar_label(p2, padding=80, label_type='center', fontsize=10, )

p3 = ax1.bar(x + offsets[2], multi_cifar_max, width, label="AUROC Multi Max", color="skyblue")
ax1.bar_label(p3, padding=80, label_type='center', fontsize=10, fontweight='bold',)


ax2 = ax1.twinx()
p4 = ax2.bar(x + offsets[3], single_acc_cifar, width, label="Acc Single", color="green", hatch="\\")
ax2.bar_label(p4, padding=80, label_type='center', fontsize=10,)
ax2.set_ylabel('Accuracy (%)', fontsize=26)
ax2.set_ylim(0, 110)


p5 = ax1.bar(x + offsets[4], multi_acc_cifar_mean, width, label="Acc Multi Mean", color="orange", hatch="\\")
ax1.bar_label(p5, padding=80, label_type='center', fontsize=10,)

p6 = ax1.bar(x + offsets[5], multi_acc_cifar_max, width, label="Acc Multi Max", color="skyblue", hatch="\\")
ax1.bar_label(p6, padding=80, label_type='center', fontsize=10,)


# Formatting
ax1.set_xticks(x + width / 2)
ax1.set_xticklabels(settings, rotation=0, fontsize=18)
# Reduce horizontal empty space inside the axes
left_edge = x[0] + offsets[0] - width / 2
right_edge = x[-1] + offsets[-1] + width / 2

ax1.set_xlim(left_edge - 0.02, right_edge + 0.02)
ax1.margins(x=0)

# Leave empty space at the bottom for legend
plt.subplots_adjust(bottom=0.28)

# Put legend in the reserved bottom space
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

handles = handles1 + handles2
labels = labels1 + labels2

order = [0, 5, 1, 3, 2, 4]

fig.legend(
    [handles[i] for i in order],
    [labels[i] for i in order],
    loc="lower center",
    bbox_to_anchor=(0.5, 0.02),
    ncol=3,
    frameon=False,
    fontsize=14
)

plt.savefig(
    "./plots/cifar_acc_single_multi.pdf",
    bbox_inches="tight",
    pad_inches=0.03
)

plt.show()