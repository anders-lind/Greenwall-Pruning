import pandas as pd
import numpy as np
import scipy.stats as stats

# =====================================================================
# 1. DEFINE GROUPING CRITERIA HERE
# =====================================================================

def compute_criteria(row):
    """
    Uncomment the block you want to test, or write your own custom logic.
    Each function must return a single continuous value per row.
    """
    # Distance from Homing Position
    # dx_home = row['robot_x'] - 0.52
    # dy_home = row['robot_y'] - 0.545
    # return np.sqrt(dx_home**2 + dy_home**2)
# 
    # Distance from optical center
    # dx = row['dx'] 
    # dy = row['dy']
    # dz = row['dz']
    # return np.sqrt(dx**2 + dy**2)# + dz**2)

    # Image distance from center
    # u_center, v_center = 320, 240
    # return np.sqrt((row['u_seed'] - u_center)**2 + (row['v_seed'] - v_center)**2)

    # i'th percentile mahalanobis color (i=15)
    return row['verif_dist']

    # dx_home = row['robot_x'] - 0.52
    # dy_home = row['robot_y'] - 0.545
    # dz_home = 0
    # dx = row['dx'] 
    # dy = row['dy']
    # dz = row['dz']
    # leaf_pos = np.array([dx_home,dy_home])#,dz_home])
    # leaf_delta_pos = np.array([dx,dy])#,dz])
    # return np.linalg.norm(leaf_pos + leaf_delta_pos)


# =====================================================================
# 2. DATA PROCESSING & STATISTICAL TESTING
# =====================================================================
df = pd.read_csv('combined_results/experiment_results.csv')

# Calculate the continuous metric for every row based on your function above
df['metric'] = df.apply(compute_criteria, axis=1)

# Find the median to split the data into two equal-sized groups
median_split = df['metric'].median()

# Insert this right after you calculate median_split in your script:
print("==================================================")
print(f"               MEDIAN SPLIT VALUE                ")
print("==================================================")
print(f"The calculated median boundary is: {median_split:.4f}")
print("==================================================")

# Assign rows to Group 0 (Below/Equal Median) or Group 1 (Above Median)
df['group'] = np.where(df['metric'] <= median_split, 'Low/Near Group', 'High/Far Group')

# Build the 2x2 Contingency Table (Cross-tabulation)
contingency_table = pd.crosstab(df['group'], df['pruning_success'])

print("==================================================")
print("             2x2 CONTINGENCY TABLE                ")
print("==================================================")
print(contingency_table)
print("==================================================\n")

# Calculate success rates for printing
for group_name in contingency_table.index:
    fails = contingency_table.loc[group_name, 0]
    successes = contingency_table.loc[group_name, 1]
    total = fails + successes
    rate = (successes / total) * 100
    print(f"{group_name} Success Rate: {rate:.1f}% ({successes}/{total})")

print("\n==================================================")
print("             STATISTICAL TEST RESULTS             ")
print("==================================================")

# Check if Chi-Squared assumptions are met (Expected values >= 5)
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency_table)

if np.any(expected < 5):
    print("⚠️ Warning: Expected cell counts are below 5.")
    print("Switching to Fisher's Exact Test (more accurate for small samples)...")
    odds_ratio, p_value = stats.fisher_exact(contingency_table)
    print(f"Fisher's Exact Test p-value: {p_value:.4f}")
else:
    # Using Yates' correction for continuity since it's a 2x2 table
    chi2_corr, p_corr, _, _ = stats.chi2_contingency(contingency_table, correction=True)
    print(f"Chi-Squared Statistic (with correction): {chi2_corr:.4f}")
    print(f"Chi-Squared p-value: {p_corr:.4f}")
    p_value = p_corr

# Conclusion output
print("--------------------------------------------------")
if p_value < 0.05:
    print("Result: SIGNIFICANT (p < 0.05)")
    print("Conclusion: The criteria heavily impacts pruning success.")
    print("The two groups have a statistically different success rate.")
else:
    print("Result: NOT SIGNIFICANT (p >= 0.05)")
    print("Conclusion: Pruning success is independent of this criteria.")
    print("The difference in success rates could purely be due to random chance.")
print("==================================================")