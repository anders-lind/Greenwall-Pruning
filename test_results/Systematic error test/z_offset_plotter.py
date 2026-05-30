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
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10), sharex='col', sharey='row')

# Configuration for columns
input_cols = ['x', 'y', 'gripper_z']
input_labels = ['Input x', 'Input y', 'Gripper z']

# --- Row 1: Y-Axis Error (real_y - y) ---
for col_idx in range(3):
    ax = axes[0, col_idx]
    current_x = input_cols[col_idx]
    
    ax.scatter(df[current_x], df['y_err'], color='royalblue', alpha=0.5, edgecolors='k', s=35, label='Data')
    ax.set_title(f'Y Error vs {input_labels[col_idx]}', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    if col_idx == 0:
        ax.set_ylabel('Y Error (real_y - y)', fontsize=12, fontweight='bold')

# --- Row 2: X-Axis Error (real_x - x) ---
for col_idx in range(3):
    ax = axes[1, col_idx]
    current_x = input_cols[col_idx]
    
    ax.scatter(df[current_x], df['x_err'], color='crimson', alpha=0.5, edgecolors='k', s=35, label='Data')
    ax.set_title(f'X Error vs {input_labels[col_idx]}', fontsize=12)
    ax.set_xlabel(input_labels[col_idx], fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    if col_idx == 0:
        ax.set_ylabel('X Error (real_x - x)', fontsize=12, fontweight='bold')

# --- 4. Overlay Your Custom Functions ---

# Plot 1: dy vs y (Top Row, Middle Column)
ax_dy_y = axes[0, 1]
y_space = np.linspace(df['y'].min(), df['y'].max(), 200)
dy_y_func = 0.12-0.0546 + (0.1 * np.abs(y_space - 0.5))
ax_dy_y.plot(y_space, dy_y_func, color='black', linestyle='-', linewidth=2.5, 
             label='dy = 0.12-0.0546 + 0.1*|y-0.5|')
ax_dy_y.legend(loc='upper right')

# Plot 2: dy vs z (Top Row, Right Column)
ax_dy_z = axes[0, 2]
z_space = np.linspace(df['gripper_z'].min(), df['gripper_z'].max(), 200)
dy_z_func = -0.0546 + (0.4081 * z_space)
ax_dy_z.plot(z_space, dy_z_func, color='black', linestyle='-', linewidth=2.5, 
             label='dy = -0.0546 + 0.4081*z')
ax_dy_z.legend(loc='lower right')

# Plot 3: dx vs x (Bottom Row, Left Column)
ax_dx_x = axes[1, 0]
x_space = np.linspace(df['x'].min(), df['x'].max(), 200)
dx_x_func = 0.1 * (x_space - 0.5)
ax_dx_x.plot(x_space, dx_x_func, color='black', linestyle='-', linewidth=2.5, 
             label='dx = 0.1*(x - 0.5)')
ax_dx_x.legend(loc='upper left')


# 5. Layout formatting
plt.suptitle('Kinematic Error Matrix with Custom Function Overlays', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()