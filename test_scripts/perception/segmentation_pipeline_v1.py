import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import os
import time
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- STEP 1: CONFIGURATION ---
class_configs = {
    "yellow": {"threshold": 5.0, "kernel_size": (5, 5)},
    "brown":  {"threshold": 4.0, "kernel_size": (5, 5)},
    "verification_threshold": 25.0 
}

# --- STEP 2: INITIALIZATION ---
device = "cuda" if torch.cuda.is_available() else "cpu"
path_to_sam2 = os.path.expanduser("~/Thesis/sam2") # Alex
# path_to_sam2 = os.path.expanduser("~/workspace/masters_thesis/sam2") # Anders
stats_path = "perception_stats_cielab.npy"
sam2_checkpoint = os.path.join(path_to_sam2, "checkpoints/sam2.1_hiera_base_plus.pt")
model_cfg = "sam2_hiera_b+.yaml"

training_stats = np.load(stats_path, allow_pickle=True).item()

# Load SAM 2 Predictor
sam2_model = build_sam2(model_cfg, ckpt_path=None, device=device)
sd = torch.load(sam2_checkpoint, map_location=device, weights_only=True)["model"]
sam2_model.load_state_dict(sd, strict=False)
predictor = SAM2ImagePredictor(sam2_model)

# --- STEP 3: IMAGE LOADING ---
test_file = "training_data/2_Color.png"
img_bgr = cv2.imread(test_file)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(float)
rows, cols, _ = img_lab.shape
pixels_lab = img_lab.reshape(-1, 3) 

# --- STEP 4: PERCEPTION PIPELINE (Independent Streams) ---
class_data = {}
combined_morphed = np.zeros((rows, cols), dtype=np.uint8)

for cls in ["yellow", "brown"]:
    mu, inv_cov = training_stats[cls]["mean"], training_stats[cls]["inv_cov"]
    
    # 1. Mahalanobis Distance & Raw Mask
    diff = pixels_lab - mu
    dist_img = np.sum(diff * (diff @ inv_cov), axis=1).reshape(rows, cols)
    binary = (dist_img < class_configs[cls]["threshold"]).astype(np.uint8) * 255
    
    # 2. Morphological Operations
    kernel = np.ones(class_configs[cls]["kernel_size"], np.uint8)
    morphed = cv2.morphologyEx(cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    
    class_data[cls] = {"binary": binary, "morphed": morphed}
    combined_morphed = cv2.bitwise_or(combined_morphed, morphed)

# --- STEP 5: AREA-WEIGHTED SEEDING & SAM 2 ---

# 1. Find all separate "blobs" in the combined mask
contours, _ = cv2.findContours(combined_morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

best_mask = None
seed_point = None

if contours:
    # 2. CRITERIA: Select the largest contour by Area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 3. Create a mask of ONLY this largest blob
    # This prevents the Distance Transform from being distracted by smaller spots
    single_leaf_mask = np.zeros_like(combined_morphed)
    cv2.drawContours(single_leaf_mask, [largest_contour], -1, 255, -1)
    
    # 4. Find the "Deepest Point" of the largest blob
    dist_trans = cv2.distanceTransform(single_leaf_mask, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(dist_trans)
    seed_point = np.array([[max_loc[0], max_loc[1]]])
    
    # 5. SAM 2 Inference using the Area-Validated Seed
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        predictor.set_image(img_rgb)
        masks, scores, _ = predictor.predict(
            point_coords=seed_point,
            point_labels=np.array([1]), 
            multimask_output=True,
        )
    best_mask = masks[np.argmax(scores)]
else:
    print("No necrotic clusters detected above the morphological threshold.")

# --- STEP 6: SPECTRAL VERIFICATION ---
is_verified = False
verification_dist = 999.9
verified_class = "None"

if best_mask is not None:
    # 1. Extract the actual pixels from the original Lab image using the SAM 2 mask
    # We use the boolean version of the mask to index the image
    leaf_pixels_lab = img_lab[best_mask.astype(bool)]
    
    # 2. Calculate the global mean color of the segmented instance
    actual_mean_color = np.mean(leaf_pixels_lab, axis=0)
    
    # 3. Compare this mean against our trained class distributions
    min_dist = 999.9
    for cls in ["yellow", "brown"]:
        mu = training_stats[cls]["mean"]
        inv_cov = training_stats[cls]["inv_cov"]
        
        # Calculate Mahalanobis distance from the leaf mean to the class mean
        diff_v = actual_mean_color - mu
        dist_v = diff_v.T @ inv_cov @ diff_v # Quadratic form
        
        if dist_v < min_dist:
            min_dist = dist_v
            verified_class = cls
    
    verification_dist = min_dist
    
    # 4. Final Safety Check
    if verification_dist < class_configs["verification_threshold"]:
        is_verified = True
        print(f"VERIFICATION SUCCESS: Segmented leaf matches '{verified_class}' distribution.")
        print(f"Mean Mahalanobis Distance: {verification_dist:.2f}")
    else:
        print(f"VERIFICATION FAILED: Segmented object is spectraly inconsistent (Dist: {verification_dist:.2f})")
else:
    print("VERIFICATION SKIPPED: No mask was generated by SAM 2.")

# --- STEP 6: 8-PANEL PLOTTING ---
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()
fontsize = 20

# Panel 1: Input
axes[0].imshow(img_rgb)
axes[0].set_title("1. Input Image", fontsize=fontsize)

# Panel 2: Yellow Raw Mask
axes[1].imshow(class_data["yellow"]["binary"], cmap='gray')
axes[1].set_title("2. Yellow Mahal. Mask", fontsize=fontsize)

# Panel 3: Brown Raw Mask
axes[2].imshow(class_data["brown"]["binary"], cmap='gray')
axes[2].set_title("3. Brown Mahal. Mask", fontsize=fontsize)

# Panel 4: Yellow Morph Mask
axes[3].imshow(class_data["yellow"]["morphed"], cmap='gray')
axes[3].set_title("4. Yellow Morph Ops", fontsize=fontsize)

# Panel 5: Brown Morph Mask
axes[4].imshow(class_data["brown"]["morphed"], cmap='gray')
axes[4].set_title("5. Brown Morph Ops", fontsize=fontsize)

# Panel 6: Combined Cleaned
axes[5].imshow(combined_morphed, cmap='gray')
axes[5].scatter(max_loc[0], max_loc[1], c='red', marker='x', s=200)
axes[5].set_title("6. Combined Cleaned Mask", fontsize=fontsize)

# Panel 7: Distance Transform
axes[6].imshow(dist_trans, cmap='magma')
axes[6].set_title("7. Distance Transform", fontsize=fontsize)

# Panel 8: SAM2 Output
final_view = cv2.bitwise_and(img_rgb, img_rgb, mask=best_mask.astype(np.uint8))
axes[7].imshow(final_view)
axes[7].set_title("8. SAM2 Prediction", fontsize=fontsize)

for ax in axes: ax.axis('off')
plt.tight_layout()
plt.show()