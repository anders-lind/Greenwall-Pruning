import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
IMAGE_PATH = "training_dataset/31.png"  # Path to a single representative image
STATS_PATH = "perception_stats_cielab.npy"

# Three thresholds for comparison (Linear Mahalanobis Distance D)
THRESHOLDS = [1.0, 2.0, 3.0] 

def plot_mahalanobis_direct_output():
    # 1. Load Stats and Image
    if not os.path.exists(IMAGE_PATH) or not os.path.exists(STATS_PATH):
        print("Error: Ensure your image and .npy files exist at the specified paths.")
        return
    
    mahal_data = np.load(STATS_PATH, allow_pickle=True).item()
    mu = mahal_data["yellow"]["mean"]
    inv_cov = mahal_data["yellow"]["inv_cov"]

    img_bgr = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    rows, cols, _ = img_bgr.shape

    # 2. Calculate Mahalanobis Distance Map
    pixels = img_lab.reshape(-1, 3).astype(float)
    diff = pixels - mu
    # Calculate D^2 first, then sqrt for linear distance D
    dist_sq = np.sum(diff * (diff @ inv_cov), axis=1).reshape(rows, cols)
    dist_linear = np.sqrt(dist_sq)

    # 3. Plotting (1x4 Grid)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # [Plot 1] Original Input
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original RGB Image", fontsize=16, fontweight='bold')
    axes[0].axis('off')

    # [Plots 2, 3, 4] Direct Segmentation Outputs
    for i, thresh in enumerate(THRESHOLDS):
        # Create binary mask (Pixels closer than threshold)
        mask = (dist_linear < thresh).astype(np.uint8) * 255
        
        # Visualize the mask directly (Black and White)
        # Or use a color-mapped version for better visibility:
        axes[i+1].imshow(mask, cmap='gray')
        axes[i+1].set_title(f"Output Mask ($D < {thresh}$)", fontsize=16, fontweight='bold')
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.savefig('figures/mahalanobis_raw_visualization.png', dpi=300)
    plt.savefig('figures/mahalanobis_raw_visualization.pdf')
    print("Plot saved as mahalanobis_raw_visualization.png")
    plt.show()

if __name__ == "__main__":
    plot_mahalanobis_direct_output()