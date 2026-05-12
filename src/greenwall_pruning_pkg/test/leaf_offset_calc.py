import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# 1. Load Data
df = pd.read_csv('/home/anders/workspace/masters_thesis/Greenwall-Pruning/src/greenwall_pruning_pkg/test/leaf_offset.csv', skipinitialspace=True)
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
# Define the range for the grid based on your data limits
y_range = np.linspace(df['y'].min(), df['y'].max(), 50)
z_range = np.linspace(df['gripper_z'].min(), df['gripper_z'].max(), 50)
Y_grid, Z_grid = np.meshgrid(y_range, z_range)

# Transform the Y grid into the absolute centered values for the model
Y_abs_grid = np.abs(Y_grid - 0.5)

# Flatten and predict
flat_y_abs = Y_abs_grid.flatten()
flat_z = Z_grid.flatten()
X_plot = np.column_stack((flat_y_abs, flat_z))
diff_pred = model.predict(X_plot).reshape(Y_grid.shape)

# 5. Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
# 'coolwarm' cmap is great for seeing where delta is positive vs negative
surf = ax.plot_surface(Y_grid, Z_grid, diff_pred, cmap='viridis', alpha=0.7, antialiased=True)

# Plot the actual measurements as black dots
ax.scatter(df['y'], df['gripper_z'], df['y_diff'], color='black', s=40, label='Measurements')

# Labels
ax.set_xlabel('Input y')
ax.set_ylabel('Gripper Z')
ax.set_zlabel('Delta Y (real_y - y)')
ax.set_title('3D Geometric Offset Map\nΔy = f(|y - 0.5|, z)')

# Add a color bar
fig.colorbar(surf, shrink=0.5, aspect=10, label='Predicted Offset')

# Adjust the viewing angle to see the "V" crease clearly
ax.view_init(elev=25, azim=45)

plt.legend()
plt.show()

# Export final equation for your thesis
print(f"Learned Equation: Δy = {model.intercept_:.4f} + ({model.coef_[0]:.4f} * |y - 0.5|) + ({model.coef_[1]:.4f} * z)")