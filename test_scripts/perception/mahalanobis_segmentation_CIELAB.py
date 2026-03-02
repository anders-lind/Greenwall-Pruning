import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Configuration for LAB space
# Note: Thresholds might need re-tuning because Lab values have different scales than BGR
class_configs = {
    "yellow": {
        "threshold": 5.0,       # Sensitivity: Lower = stricter
        "kernel_size": (5, 5),  # Morphological cleaning
        "cmap": "YlOrBr"         # Visualization color map
    },
    "brown": {
        "threshold": 3.0,       # Brown often needs a wider threshold
        "kernel_size": (5, 5), # Larger kernel for "crunchy" textures
        "cmap": "copper"
    },
    "glossy": {
        "threshold": 10.0,        # Strict: catches only bright reflections
        "kernel_size": (5, 5),   # Small kernel for pinpoint highlights
        "cmap": "Blues_r"
    }
}

# Load Stats
training_stats = np.load("perception_stats_cielab.npy", allow_pickle=True).item()

# Load Test Image
test_file = "training_data/1_Color.png"
img_bgr = cv2.imread(test_file)
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(float)
rows, cols, _ = img_lab.shape
pixels_lab = img_lab.reshape(-1, 3)

plt.figure(figsize=(16, 12))
plt.subplot(2, 2, 1)
plt.title("Original BGR")
plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
plt.axis('off')

for i, (cls, config) in enumerate(class_configs.items(), start=2):
    mu = training_stats[cls]["mean"]
    inv_cov = training_stats[cls]["inv_cov"]
    
    diff = pixels_lab - mu
    dist = np.sum(diff * (diff @ inv_cov), axis=1).reshape(rows, cols)
    
    mask = (dist < config["threshold"]).astype(np.uint8) * 255
    kernel = np.ones(config["kernel_size"], np.uint8)
    morphed = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    
    plt.subplot(2, 2, i)
    plt.imshow(morphed, cmap='gray')
    plt.title(f"LAB Segmentation: {cls.upper()}")
    plt.axis('off')

plt.tight_layout()
plt.show()