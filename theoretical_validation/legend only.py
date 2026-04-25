import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Ensure output directory exists
OVERLEAF_DIR = r"C:\Users\lautr\Downloads\Overleaf_Final_Figures"
os.makedirs(OVERLEAF_DIR, exist_ok=True)

# 1. Define the exact lines and updated labels (No p_c or p_r)
custom_lines = [
    Line2D([0], [0], color='k', linestyle='-', lw=2),
    Line2D([0],[0], color='k', linestyle='--', lw=2.5)
]
labels =['Connectivity', 'Relaxed Rigidity']

# 2. Create a new, blank figure
fig = plt.figure(figsize=(3.5, 1.2)) # Slightly smaller width since text is shorter

# 3. Add the legend directly to the center of this blank figure
legend = fig.legend(custom_lines, labels, loc='center', fontsize=16, frameon=True)

# Remove the default axes so it's completely blank behind the legend
plt.gca().set_axis_off()

# 4. Save ONLY the legend box as a .jpg
out_path = os.path.join(OVERLEAF_DIR, "Legend_Only.jpg")

# bbox_inches='tight' crops out all the blank white space around the legend box
plt.savefig(out_path, dpi=300, bbox_inches='tight', format='jpg')
print(f"-> SUCCESS! Saved standalone legend to {out_path}")

# Explicitly display the legend box on your screen
plt.show()