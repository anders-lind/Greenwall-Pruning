import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# CUSTOMIZABLE FONT CONFIGURATION
# =====================================================================
FONT_CONFIG = {
    'title_size': 16*2,       # Main plot title
    'label_size': 13*2,       # X and Y axis labels (e.g., "Input x")
    'tick_size': 11*2,        # Numbers on the axis rulers
    'legend_size': 11*2,      # Legend box text
    'font_weight': 'normal'   # Weight for titles and labels ('normal' or 'bold')
}

# 1. Load Data
input_file = '/home/anders/workspace/masters_thesis/Greenwall-Pruning/test_results/Gripper offset test/z_offset_data copy.csv'  # Replace with your actual filename
df = pd.read_csv(input_file, skipinitialspace=True)
df.columns = df.columns.str.strip()

# 2. Calculate the True X Deviation
df['x_diff'] = df['real_x'] - df['x']

# 3. Setup the Plot
plt.figure(figsize=(9, 6))

# Plot the raw data points
plt.scatter(df['x'], df['x_diff'], color='crimson', alpha=0.6, edgecolors='k', s=100, label='Measurements')

# 4. Generate and plot your custom function
x_space = np.linspace(df['x'].min(), df['x'].max(), 200)
x_space = np.linspace(0.2, 0.8, 200)
offset_x_func = 0.1*x_space - 0.05

plt.plot(x_space, offset_x_func, color='black', linestyle='-', linewidth=2.5)

# 5. Formatting and Aesthetics using the FONT_CONFIG
plt.title('Δx = 0.1x - 0.05', 
          fontsize=FONT_CONFIG['title_size'], 
          fontweight=FONT_CONFIG['font_weight'], 
          pad=15)

plt.xlim(0.2,0.8)

plt.xlabel('Leaf x', fontsize=FONT_CONFIG['label_size'], fontweight=FONT_CONFIG['font_weight'])
plt.ylabel('Δx ', fontsize=FONT_CONFIG['label_size'], fontweight=FONT_CONFIG['font_weight'])

# Adjust the size of the axis tick numbers (e.g., 0.1, 0.2, 0.3)
plt.tick_params(axis='both', labelsize=FONT_CONFIG['tick_size'])

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left', fontsize=FONT_CONFIG['legend_size'])

# Base origin reference line
plt.axhline(0, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()