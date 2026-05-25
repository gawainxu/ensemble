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
# 1. Create dummy data with vastly different scales
cifar=[0.9191464, 0.92738026, 0.9641541, 0.98164225, 0.99598855, 0.9982147,]
tinyimagenet=[0.64396924, 0.60834914, 0.7555859, 0.8891498, 0.9786245, 0.98934424]

# 2. Set up the primary plot (Left Y-Axis)
fig, ax1 = plt.subplots(figsize=(8, 5))

color_left = 'tab:red'
ax1.set_xlabel('Temperatures', fontsize=16)
ax1.set_ylabel('Cifar10', color=color_left, fontsize=16)
ax1.plot(cifar, color=color_left, linewidth=2, label='Cifar10')
ax1.tick_params(axis='y', labelcolor=color_left)  # Matches tick text color to line

# 3. Create the twin axis (Right Y-Axis)
ax2 = ax1.twinx()  

color_right = 'tab:blue'
ax2.set_ylabel('TinyImageNet', color=color_right, fontsize=16)
ax2.plot(tinyimagenet, color=color_right, linewidth=2,  label='TinyImageNet')
ax2.tick_params(axis='y', labelcolor=color_right) # Matches tick text color to line


ax1.set_xlim(0, 5)
tick_positions = np.arange(0, 6, 1) 
ax1.set_xticks(tick_positions)
tick_labels = ["1.0", "0.5", "0.1", "0.05", "0.01", "0.005"] 
ax1.set_xticklabels(tick_labels)     


# Style the actual tick marks (make them a bit longer/thicker)
ax1.tick_params(axis='x', direction='out', length=6, width=1.5, colors='black')

# 4. Combine legends from both axes into a single box
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)


plt.title("Mean of the Gram Matrices vs. Temperature", fontsize=20)
plt.tight_layout()
plt.savefig("./plots/gram_mean.pdf")
plt.show()
"""
