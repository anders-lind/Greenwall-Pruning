import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Data
input_file = '/home/anders/workspace/masters_thesis/Greenwall-Pruning/test_results/Gripper offset test/z_offset_data.csv'  # Replace with your actual filename
df = pd.read_csv(input_file, skipinitialspace=True)
df.columns = df.columns.str.strip()

# 2. Calculate the True Deviations (Errors)
df['y_err'] = df['real_y'] - df['y']
df['x_err'] = df['real_x'] - df['x']

# 3. Create the 2x3 Subplot Grid
# sharex='col' forces columns to match horizontal scales
# sharey='row' forces rows to match vertical error scales
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10), sharex='col', sharey='row')

# Configuration for columns
input_cols = ['x', 'y', 'gripper_z']
input_labels = ['Input x', 'Input y', 'Gripper z']

# --- Row 1: Y-Axis Error (real_y - y) ---
for col_idx in range(3):
    ax = axes[0, col_idx]
    current_x = input_cols[col_idx]
    
    ax.scatter(df[current_x], df['y_err'], color='royalblue', alpha=0.6, edgecolors='k', s=35)
    ax.set_title(f'Y Error vs {input_labels[col_idx]}', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Only label the leftmost Y-axis since they share scales
    if col_idx == 0:
        ax.set_ylabel('Y Error (real_y - y)', fontsize=12, fontweight='bold')

# --- Row 2: X-Axis Error (real_x - x) ---
for col_idx in range(3):
    ax = axes[1, col_idx]
    current_x = input_cols[col_idx]
    
    ax.scatter(df[current_x], df['x_err'], color='crimson', alpha=0.6, edgecolors='k', s=35)
    ax.set_title(f'X Error vs {input_labels[col_idx]}', fontsize=12)
    ax.set_xlabel(input_labels[col_idx], fontsize=11)  # Bottom row gets X labels
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Only label the leftmost Y-axis since they share scales
    if col_idx == 0:
        ax.set_ylabel('X Error (real_x - x)', fontsize=12, fontweight='bold')

# 4. Layout formatting
plt.suptitle('Kinematic Error Matrix: Spatial Deviations vs Command States', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()