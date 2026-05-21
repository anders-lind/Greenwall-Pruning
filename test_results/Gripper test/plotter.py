import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# CONFIGURATION: Set your custom bin width here
BIN_WIDTH = 0.04  # 0.02 0.04
# ==========================================

# 1. Load the dataset
df = pd.read_csv("/home/anders/workspace/masters_thesis/Greenwall-Pruning/test_results/Gripper test/test.csv")
df.columns = df.columns.str.strip()  # Clean up column names

# Separate the data
all_depths = df["depth"]
successes = df[df["success"] == 1]["depth"]
failures = df[df["success"] == 0]["depth"]

# 2. Mathematically align the bins for all 3 plots
# Find the absolute min and max bounds across the entire dataset
min_val = all_depths.min()
max_val = all_depths.max()

# Generate sequential bin boundaries exactly 'BIN_WIDTH' apart
bin_edges = np.arange(min_val, max_val + BIN_WIDTH, BIN_WIDTH)
num_bins = len(bin_edges) - 1

# Print the final amount of bins to the console
print(f"--- BIN SETUP ---")
print(f"Custom Bin Width: {BIN_WIDTH}")
print(f"Total Number of Bins Generated: {num_bins}")
print("-" * 17)

# 3. Create the plots with locked X and Y limits
# 'sharex=True' and 'sharey=True' force all subplots to use the exact same axis limits
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

# Plot 1: All Depths
axes[0].hist(all_depths, bins=bin_edges, color="skyblue", edgecolor="black")
axes[0].set_title("Distribution of All Depths")
axes[0].set_xlabel("Depth")
axes[0].set_ylabel("Frequency")
axes[0].grid(axis="y", linestyle="--", alpha=0.7)

# Plot 2: Depths of Successes
axes[1].hist(successes, bins=bin_edges, color="salmon", edgecolor="black")
axes[1].set_title("Depths of Successes (Success = 1)")
axes[1].set_xlabel("Depth")
axes[1].grid(axis="y", linestyle="--", alpha=0.7)

# Plot 3: Depths of Failures
axes[2].hist(failures, bins=bin_edges, color="lightgray", edgecolor="black")
axes[2].set_title("Depths of Failures (Success = 0)")
axes[2].set_xlabel("Depth")
axes[2].grid(axis="y", linestyle="--", alpha=0.7)

# Tighten the visual layout
plt.tight_layout()

# 4. Display the plots
plt.show()