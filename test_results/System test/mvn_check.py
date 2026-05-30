import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 1. Load your dataset
df = pd.read_csv('combined_results/experiment_results.csv')

# 2. Select the independent continuous columns (p = 6 dimensions)
features = ['dx', 'dy', 'dz', 'robot_x', 'robot_y', 'verif_dist']

def calculate_mahalanobis_and_quantiles(X):
    # Calculate group centroid and covariance matrix
    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    inv_cov = np.linalg.inv(cov)
    
    # Calculate D^2 for each row
    diff = X - mean
    md2 = np.sum(diff @ inv_cov * diff, axis=1)
    sorted_md2 = np.sort(md2)
    n = len(sorted_md2)
    
    # Calculate theoretical Chi-squared quantiles (p degrees of freedom)
    p_dims = X.shape[1]
    probs = (np.arange(1, n + 1) - 0.5) / n  # Standard i-0.5 adjustment
    chi2_quantiles = stats.chi2.ppf(probs, df=p_dims)
    
    return chi2_quantiles, sorted_md2

# 3. Separate your data vectors
g1_data = df[df['pruning_success'] == 1][features].values
g0_data = df[df['pruning_success'] == 0][features].values

# 4. Generate the side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot for Group 1: Successes
chi2_q1, md2_1 = calculate_mahalanobis_and_quantiles(g1_data)
axes[0].scatter(chi2_q1, md2_1, color='#2ecc71', alpha=0.8, edgecolors='k', label='Observed')
max_val1 = max(max(chi2_q1), max(md2_1))
axes[0].plot([0, max_val1], [0, max_val1], color='black', linestyle='--', label='Theoretical MVN Line')
axes[0].set_title(f"Group 1: Success (n={len(g1_data)}, p={g1_data.shape[1]})", fontweight='bold')
axes[0].set_xlabel("Theoretical $\chi^2$ Quantiles")
axes[0].set_ylabel("Observed Squared Mahalanobis Distance ($D^2$)")
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend()

# Plot for Group 0: Failures
chi2_q0, md2_0 = calculate_mahalanobis_and_quantiles(g0_data)
axes[1].scatter(chi2_q0, md2_0, color='#e74c3c', alpha=0.8, edgecolors='k', label='Observed')
max_val0 = max(max(chi2_q0), max(md2_0))
axes[1].plot([0, max_val0], [0, max_val0], color='black', linestyle='--', label='Theoretical MVN Line')
axes[1].set_title(f"Group 0: Failure (n={len(g0_data)}, p={g0_data.shape[1]})", fontweight='bold')
axes[1].set_xlabel("Theoretical $\chi^2$ Quantiles")
axes[1].set_ylabel("Observed Squared Mahalanobis Distance ($D^2$)")
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend()

plt.suptitle("Chi-Squared Q-Q Plot for Multivariate Normality (MVN)", y=1.02, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('chi2_qq_plot.png', bbox_inches='tight', dpi=150)
plt.show()