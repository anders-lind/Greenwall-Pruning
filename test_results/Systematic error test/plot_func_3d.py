import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# =====================================================================
# CUSTOMIZABLE FONT CONFIGURATION
# =====================================================================
FONT_CONFIG = {
    'title_size': 16*2,       # Main plot title
    'label_size': 11*2,       # X, Y, and Z axis labels
    'tick_size': 9*2,        # Numbers on the axis axes
    'legend_size': 11*2,      # Legend box text
    'cbar_size': 11*2,        # Color bar label size
    'font_weight': 'normal'   # Weight for labels ('normal' or 'bold')
}

# 1. Load Data
df = pd.read_csv('/home/anders/workspace/masters_thesis/Greenwall-Pruning/test_results/Gripper offset test/z_offset_data.csv', skipinitialspace=True)
df.columns = df.columns.str.strip()

# 2. Setup Features (Ignoring X)
df['y_diff'] = df['real_y'] - df['y']
df['y_abs'] = np.abs(df['y'] - 0.5)

# 3. Fit Model
X = df[['y_abs', 'gripper_z']]
y_target = df['y_diff']
model = LinearRegression()
model.fit(X, y_target)

# 4. Create 3D Surface Data
y_range = np.linspace(df['y'].min(), df['y'].max(), 50)
z_range = np.linspace(df['gripper_z'].min(), df['gripper_z'].max(), 50)
Y_grid, Z_grid = np.meshgrid(y_range, z_range)

Y_abs_grid = np.abs(Y_grid - 0.5)

flat_y_abs = Y_abs_grid.flatten()
flat_z = Z_grid.flatten()
X_plot = np.column_stack((flat_y_abs, flat_z))
diff_pred = model.predict(X_plot).reshape(Y_grid.shape)

# 5. Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(Y_grid, Z_grid, diff_pred, cmap='viridis', alpha=0.7, antialiased=True)

# Plot the actual measurements
ax.scatter(df['y'], df['gripper_z'], df['y_diff'], color='black', s=100, label='Measurements')

# Labels and Titles with FONT_CONFIG
ax.set_title('             Δy = -0.05 + (0.1 * abs(y - 0.5)) + (0.41 * z)', 
             fontsize=FONT_CONFIG['title_size'], 
             fontweight=FONT_CONFIG['font_weight'], 
             pad=20)

ax.set_xlabel('Leaf y', fontsize=FONT_CONFIG['label_size'], fontweight=FONT_CONFIG['font_weight'], labelpad=10)
ax.set_ylabel('Leaf z', fontsize=FONT_CONFIG['label_size'], fontweight=FONT_CONFIG['font_weight'], labelpad=10)
ax.set_zlabel('Δy', fontsize=FONT_CONFIG['label_size'], fontweight=FONT_CONFIG['font_weight'], labelpad=10)

# Adjust the size of the axis tick numbers for all three 3D axes
ax.tick_params(axis='both', labelsize=FONT_CONFIG['tick_size'])

# Add a color bar and configure its font sizes
cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
cbar.set_label('ΔY', fontsize=FONT_CONFIG['cbar_size'], fontweight=FONT_CONFIG['font_weight'])
cbar.ax.tick_params(labelsize=FONT_CONFIG['tick_size'])

# Adjust the viewing angle to see the "V" crease clearly
ax.view_init(elev=25, azim=45)

plt.legend(loc='upper left', fontsize=FONT_CONFIG['legend_size'])
plt.tight_layout()
plt.show()

# Export final equation for your thesis
print(f"Learned Equation: Δy = {model.intercept_:.4f} + ({model.coef_[0]:.4f} * |y - 0.5|) + ({model.coef_[1]:.4f} * z)")