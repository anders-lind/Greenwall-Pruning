import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load your dataset (Replace 'your_data.csv' with your actual filename)
df = pd.read_csv('combined_results/experiment_results.csv')

# 2. Select decoupled, continuous kinematic and perception variables
# We drop u_seed and v_seed because they are perfectly collinear with dx and dy
kinematic_features = ['dx', 'dy', 'dz', 'robot_x', 'robot_y', 'verif_dist']

# 3. Setup aesthetic style
sns.set_theme(style="ticks")

# 4. Generate the scatter matrix
# Green (#2ecc71) represents Success (1), Red (#e74c3c) represents Failure (0)
g = sns.pairplot(
    df, 
    vars=kinematic_features, 
    hue='pruning_success', 
    palette={1: '#2ecc71', 0: '#e74c3c'}, 
    diag_kind='kde',
    plot_kws={'alpha': 0.7, 's': 40, 'edgecolor': 'k', 'linewidth': 0.5},
    diag_kws={'fill': True, 'common_norm': False}
)

# 5. Add titles and clean up presentation
plt.suptitle(
    "CDPR Matrix Scatter Plot: Checking for Multivariate Normality (MVN)", 
    y=1.02, 
    fontsize=14, 
    fontweight='bold'
)

# Save the high-resolution visualization
plt.savefig('scatter_matrix.png', bbox_inches='tight', dpi=150)
plt.show()