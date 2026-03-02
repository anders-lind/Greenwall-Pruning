import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# --- STEP 1: CONFIGURATION (Adjust per class) ---
# This allows you to fine-tune the sensitivity and cleaning for each condition
class_configs = {
    "yellow": {
        "threshold": 5.0,       # Sensitivity: Lower = stricter
        "kernel_size": (5, 5),  # Morphological cleaning
        "cmap": "YlOrBr"         # Visualization color map
    },
    "brown": {
        "threshold": 4.0,       # Brown often needs a wider threshold
        "kernel_size": (5, 5), # Larger kernel for "crunchy" textures
        "cmap": "copper"
    },
    "glossy": {
        "threshold": 10.0,        # Strict: catches only bright reflections
        "kernel_size": (5, 5),   # Small kernel for pinpoint highlights
        "cmap": "Blues_r"
    }
}

# --- STEP 2: LOAD DATA & STATS ---
stats_path = "perception_stats.npy"
if not os.path.exists(stats_path):
    print(f"Error: {stats_path} not found. Please run your training script first.")
    exit()

# Load the dictionary of means and inverse covariances
training_stats = np.load(stats_path, allow_pickle=True).item()

# Target test image
test_file = "training_data/1_Color.png"
img_test = cv2.imread(test_file)
if img_test is None:
    print(f"Error: Could not load {test_file}")
    exit()

img_test_rgb = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
rows, cols, _ = img_test.shape

# Prepare pixels for calculation (reshape to N x 3)
pixels_test = np.reshape(img_test, (-1, 3)).astype(float)

# --- STEP 3: PERFORM 3 DISTINCT SEGMENTATIONS ---
results = {}

for cls, config in class_configs.items():
    # 1. Get stats for this specific class
    mu = training_stats[cls]["mean"]
    inv_cov = training_stats[cls]["inv_cov"]
    
    # 2. Calculate Mahalanobis Distance
    diff = pixels_test - mu
    # Quadratic form: (x-mu)^T * Sigma^-1 * (x-mu)
    dist_flat = np.sum(diff * (diff @ inv_cov), axis=1)
    dist_img = np.reshape(dist_flat, (rows, cols))
    
    # 3. Apply individual Threshold
    binary_mask = (dist_img < config["threshold"]).astype(np.uint8) * 255
    
    # 4. Apply individual Morphological Ops
    kernel = np.ones(config["kernel_size"], np.uint8)
    morphed = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel)
    
    results[cls] = {
        "mask": morphed,
        "dist_map": dist_img
    }

# --- STEP 4: VISUALIZATION ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

# Plot 1: Original Image
axes[0].imshow(img_test_rgb)
axes[0].set_title(f"Original Image: {os.path.basename(test_file)}")
axes[0].axis('off')

# Plot 2, 3, 4: Individual Class Masks
for i, cls in enumerate(class_configs.keys(), start=1):
    conf = class_configs[cls]
    mask = results[cls]["mask"]
    
    # We display the binary mask but use a color map to reflect the class type
    axes[i].imshow(mask, cmap='gray')
    axes[i].set_title(f"Class: {cls.upper()}\n(Threshold: {conf['threshold']}, Kernel: {conf['kernel_size']})")
    axes[i].axis('off')

plt.tight_layout()
plt.show()