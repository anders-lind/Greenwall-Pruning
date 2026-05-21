import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

# =====================================================================
# CONFIGURATION: Optimized for Fisher's Exact Test & N = 32
NUM_GROUPS = 2    # Fixed at 2 groups (Shallow vs Deep) for Fisher's Test
BIN_WIDTH = 0.05  # Standardized width for every single histogram bar
FONT_SIZE_MAIN = 24
FONT_SIZE_SMALL = 22
FONT_SIZE_TICKS = 14  # <--- NEW: Set your custom X and Y tick font size here
# =====================================================================

# 1. Load the dataset
df = pd.read_csv("/home/anders/workspace/masters_thesis/Greenwall-Pruning/test_results/Gripper test/test.csv")
df.columns = df.columns.str.strip()  # Clean up trailing spaces

# Separate the continuous components for bin logic
all_depths = df["depth"]
all_successes = df[df["success"] == 1]["depth"]
all_failures = df[df["success"] == 0]["depth"]

# 2. Slice data into 2 equal-sized quantile groups
group_labels = ["Group 1 (Shallow)", "Group 2 (Deep)"]
df["depth_group"] = pd.qcut(df["depth"], q=NUM_GROUPS, labels=group_labels)

# 3. Calculate Global Bin Edges (Locks alignment across all rows)
min_val = all_depths.min()
max_val = all_depths.max()
bin_edges = np.arange(min_val, max_val + BIN_WIDTH, BIN_WIDTH)
num_bins = len(bin_edges) - 1

print(f"--- SAMPLE SIZE & BIN SETUP ---")
print(f"Total Sample Size (N): {len(df)} (24 Successes, 8 Failures)")
print(f"Custom Bin Width: {BIN_WIDTH}")
print(f"Total Number of Bins Generated: {num_bins}")
print("-" * 32 + "\n")

# 4. Calculate Group Success Rates
summary = (
    df.groupby("depth_group", observed=False)["success"]
    .agg(Total_Attempts="count", Successes="sum", Success_Rate="mean")
    .reset_index()
)

print(f"--- SUCCESS RATES BY DEPTH GROUPS ---")
for index, row in summary.iterrows():
    print(
        f"{row['depth_group']}: {row['Success_Rate']:.1%} success rate ({int(row['Successes'])}/{int(row['Total_Attempts'])} points)"
    )
print("-" * 45)

# 5. Statistical Test (Fisher's Exact Test)
df["failures"] = 1 - df["success"]
contingency_table = df.groupby("depth_group", observed=False)[["success", "failures"]].sum()

print("\n--- STATISTICAL TEST RESULTS ---")
print("Contingency Table Analyzed:")
print(contingency_table)
print("")

odds_ratio, p_value = stats.fisher_exact(contingency_table)

print(f"Fisher's Exact Test p-value: {p_value:.4f}")
if p_value < 0.05:
    print("Result: Statistically Significant (p < 0.05).")
    print("The proportion of successes is significantly different between the two depth groups!")
else:
    print("Result: Not Statistically Significant (p >= 0.05).")
    print("Fisher's test cannot prove a significant difference between these two groups.")


# 6. Dynamic Grid Plotting (Total Rows = NUM_GROUPS + 1 for Full Dataset Row)
TOTAL_ROWS = NUM_GROUPS + 1
# Expanded figsize width slightly from 16 to 18 to accommodate larger axis font ticks comfortably
fig, axes = plt.subplots(
    nrows=TOTAL_ROWS, ncols=3, figsize=(18, 3.8 * TOTAL_ROWS), sharex=True, sharey=True
)

# -----------------------------------------------------------------
# ROW 1: FULL DATASET (The Master Row at the top)
# -----------------------------------------------------------------
axes[0, 0].hist(all_depths, bins=bin_edges, color="skyblue", edgecolor="black", alpha=0.8)

# Bolds 'FULL DATASET' via mathtext while leaving 'Frequency' unbolded
axes[0, 0].set_ylabel(r"$\mathbf{Full\ dataset}$" + "\nFrequency", fontsize=FONT_SIZE_SMALL, color="black")

axes[0, 0].set_title("All Data Points", fontsize=FONT_SIZE_MAIN, pad=10, fontweight='bold')
axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)

axes[0, 1].hist(all_successes, bins=bin_edges, color="salmon", edgecolor="black", alpha=0.8)
axes[0, 1].set_title("Only Successes", fontsize=FONT_SIZE_MAIN, pad=10, fontweight='bold')
axes[0, 1].grid(axis="y", linestyle="--", alpha=0.5)

axes[0, 2].hist(all_failures, bins=bin_edges, color="lightgray", edgecolor="black", alpha=0.8)
axes[0, 2].set_title("Only Failures", fontsize=FONT_SIZE_MAIN, pad=10, fontweight='bold')
axes[0, 2].grid(axis="y", linestyle="--", alpha=0.5)

# -----------------------------------------------------------------
# ROWS 2 & 3: CATEGORIZED GROUPS
# -----------------------------------------------------------------
# We define specific bold math expressions for the group names to handle the labels cleanly
math_labels = [r"$\mathbf{Closest\ leaves}$", r"$\mathbf{Farthest\ leaves}$"]

for idx, group_name in enumerate(group_labels):
    # Shift plot row index down by 1 because Row 0 is occupied by the Full Dataset
    plot_row = idx + 1 
    
    group_data = df[df["depth_group"] == group_name]
    group_all = group_data["depth"]
    group_success = group_data[group_data["success"] == 1]["depth"]
    group_failure = group_data[group_data["success"] == 0]["depth"]
    
    # Column 1: All Data for this specific group
    axes[plot_row, 0].hist(group_all, bins=bin_edges, color="skyblue", edgecolor="black", alpha=0.8)
    
    # Combined label: Bolded Group Title + normal weight Frequency string
    axes[plot_row, 0].set_ylabel(math_labels[idx] + "\nFrequency", fontsize=FONT_SIZE_SMALL)
    
    axes[plot_row, 0].grid(axis="y", linestyle="--", alpha=0.5)
    
    # Column 2: Only Successes for this group
    axes[plot_row, 1].hist(group_success, bins=bin_edges, color="salmon", edgecolor="black", alpha=0.8)
    axes[plot_row, 1].grid(axis="y", linestyle="--", alpha=0.5)
    
    # Column 3: Only Failures for this group
    axes[plot_row, 2].hist(group_failure, bins=bin_edges, color="lightgray", edgecolor="black", alpha=0.8)
    axes[plot_row, 2].grid(axis="y", linestyle="--", alpha=0.5)

# Add X-axis labels exclusively to the bottom row for a clean look
axes[TOTAL_ROWS - 1, 0].set_xlabel("Depth", fontsize=FONT_SIZE_MAIN)
axes[TOTAL_ROWS - 1, 1].set_xlabel("Depth", fontsize=FONT_SIZE_MAIN)
axes[TOTAL_ROWS - 1, 2].set_xlabel("Depth", fontsize=FONT_SIZE_MAIN)

# -----------------------------------------------------------------
# UPDATE TICK FONT SIZES
# -----------------------------------------------------------------
# Iterates through every individual subplot window to dynamically change the tick labels
for row in axes:
    for ax in row:
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)

# Optimize spacing
plt.tight_layout()

# Render the layout window
plt.show()