#!/usr/bin/env python3
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OVERLEAF_DIR = "./plots"
os.makedirs(OVERLEAF_DIR, exist_ok=True)

custom_lines = [Line2D([0], [0], color='k', linestyle='-', lw=2), Line2D([0], [0], color='k', linestyle='--', lw=2.5)]
labels = ['Connectivity', 'Relaxed Rigidity']

fig = plt.figure(figsize=(3.5, 1.2))
ax_leg = fig.add_subplot(111)
ax_leg.axis('off')
legend = ax_leg.legend(custom_lines, labels, loc='center', fontsize=16, frameon=True)
fig.canvas.draw()

try:
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    bbox_padded = matplotlib.transforms.Bbox([[bbox.x0 - 0.05, bbox.y0 - 0.05], [bbox.x1 + 0.05, bbox.y1 + 0.05]])
except:
    bbox_padded = 'tight'

fig.savefig(os.path.join(OVERLEAF_DIR, "Legend_Only.jpg"), dpi=300, bbox_inches=bbox_padded, format='jpg')
print(f"-> Standalone legend written successfully to: {OVERLEAF_DIR}/Legend_Only.jpg")
plt.close(fig)