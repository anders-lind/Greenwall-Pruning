import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Define input paths based on your absolute repository layout
base_dir = os.path.expanduser("~/Thesis/Experiments/perception/training_dataset")
orig_dir = os.path.join(base_dir, "originals")
mask_dir = os.path.join(base_dir, "masks")

# Get all original image paths
orig_images = sorted(glob.glob(os.path.join(orig_dir, "*.png")))

# We want to display 3 distinct examples
examples_to_show = []

for orig_path in orig_images:
    filename = os.path.basename(orig_path)
    img_id = os.path.splitext(filename)[0] # e.g., "100"
    
    # Check both possible naming variants in the masks folder
    annotated_name = f"{img_id}_annotated.png"
    yellow_name = f"{img_id}_yellow.png"
    
    mask_path = None
    if os.path.exists(os.path.join(mask_dir, annotated_name)):
        mask_path = os.path.join(mask_dir, annotated_name)
    elif os.path.exists(os.path.join(mask_dir, yellow_name)):
        mask_path = os.path.join(mask_dir, yellow_name)
        
    if mask_path:
        examples_to_show.append((orig_path, mask_path))
    
    # Stop once we have found 3 valid pairs
    if len(examples_to_show) == 3:
        break

if len(examples_to_show) < 3:
    print(f"Warning: Only found {len(examples_to_show)} matching image-mask pairs.")

# Using layout="constrained" handles the subplots bounding box much cleaner
fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")

for i, (orig_p, mask_p) in enumerate(examples_to_show):
    # Read images (OpenCV reads as BGR)
    img_bgr = cv2.imread(orig_p)
    mask_bgr = cv2.imread(mask_p)
    
    # Convert original to RGB for matplotlib
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Isolate pure red pixels [B=0, G=0, R=255] for the binary mask
    binary_mask = (mask_bgr[:, :, 2] > 200) & (mask_bgr[:, :, 1] < 50) & (mask_bgr[:, :, 0] < 50)
    binary_mask_img = np.where(binary_mask, 255, 0).astype(np.uint8)
    
    # Row 0: Original images (aspect="auto" allows the axes boxes to touch seamlessly)
    axes[0, i].imshow(img_rgb, aspect="auto")
    axes[0, i].get_xaxis().set_ticks([])
    axes[0, i].get_yaxis().set_ticks([])
    
    # Row 1: Corresponding binary masks
    axes[1, i].imshow(binary_mask_img, cmap="gray", aspect="auto")
    axes[1, i].get_xaxis().set_ticks([])
    axes[1, i].get_yaxis().set_ticks([])

# Standard text labels with larger font size and adjusted padding
axes[0, 0].set_ylabel("Raw Color Image", fontsize=16, fontweight='normal', labelpad=15)
axes[1, 0].set_ylabel("Segmentation Mask", fontsize=16, fontweight='normal', labelpad=15)

# Hide tick marks entirely and eliminate borders between the rows
for ax in axes.flat:
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# This physically crunches the remaining spaces together perfectly
fig.get_layout_engine().set(hspace=0.0, wspace=0.05)

# Save paths tracking the current working directory where script is ran
current_run_dir = os.getcwd()
save_path_png = os.path.join(current_run_dir, "figures/dataset_examples.png")
save_path_pdf = os.path.join(current_run_dir, "figures/dataset_examples.pdf")

# Save files (bbox_inches='tight' prevents labels from getting cropped out)
fig.savefig(save_path_png, dpi=300, bbox_inches="tight")
fig.savefig(save_path_pdf, bbox_inches="tight")

print(f"Saved to execution directory: {save_path_png}")
print(f"Saved to execution directory: {save_path_pdf}")

plt.show()