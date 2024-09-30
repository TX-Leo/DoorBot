'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-08-17 02:57:08
Version: v1
File: 
Brief: 
'''

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

# Data preparation
types = ["Crossbar", "Lever", "Doorknob", "Cabinets"]
method1 = [7 / 25, 7 / 25, 13 / 25, 23 / 25]
method2 = [8 / 25, 13 / 25, 19 / 25, 23 / 25]
method3 = [16 / 25, 22 / 25, 23 / 25, 25 / 25]
method4 = [19 / 25, 23 / 25, 23 / 25, 25 / 25]

# Set bar width and positions
barWidth = 0.2
r1 = np.arange(len(method1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

# Create chart and set resolution
fig, ax = plt.subplots(dpi=300)  # Set DPI to 300 for a high-resolution image

# Draw bars
rects1 = ax.bar(r1, method1, color="#C9EFBE", width=barWidth, label="VLM+VLM")
rects2 = ax.bar(r2, method2, color="#9BDCFC", width=barWidth, label="SM+VLM")
rects3 = ax.bar(r3, method3, color="#F0CFEA", width=barWidth, label="VLM+GUM")
rects4 = ax.bar(
    r4, method4, color="#CAC8EF", width=barWidth, label="SM+GUM(Ours)"
)

# Add data labels and prevent overlapping
for rects in [rects1, rects2, rects3, rects4]:
    for rect in rects:
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            height,
            f"{int(height*100)}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )  # Adjust font size

# Set chart title and labels
plt.xlabel("Type", fontweight="bold", fontsize=13)
plt.ylabel("Success Rate", fontweight="bold", fontsize=13)
plt.xticks([r + barWidth for r in range(len(method1))], types, fontsize=12)

# Convert y-axis to percentage
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

handles, labels = ax.get_legend_handles_labels()

legend = ax.legend(handles, labels, fontsize=10)
for text in legend.get_texts():
    if text.get_text() == "SM+GUM(Ours)":
        text.set_fontweight('bold')

# Adjust layout and remove whitespace
plt.tight_layout()

# Save high-resolution image
plt.savefig("exp1_vis.png", bbox_inches="tight")

# Display the chart
plt.show()
