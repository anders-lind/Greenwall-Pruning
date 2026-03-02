import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- STEP 1: CONFIGURATION ---
# Tuning parameters for your two target classes
class_configs = {
    "yellow": {"threshold": 5.0, "kernel_size": (5, 5)},
    "brown":  {"threshold": 4.0, "kernel_size": (5, 5)}
}

# --- STEP 2: LOAD DATA & STATS (LAB Space) ---
stats_path = "perception_stats_cielab.npy"
training_stats = np.load(stats_path, allow_pickle=True).item()

test_file = "training_data/5_Color.png"
img_bgr = cv2.imread(test_file)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(float)
rows, cols, _ = img_lab.shape
pixels_lab = img_lab.reshape(-1, 3)

# --- STEP 3: FIND THE LARGEST "ROTTEN" CLUSTER ---
combined_rotten_mask = np.zeros((rows, cols), dtype=np.uint8)

for cls in ["yellow", "brown"]:
    mu = training_stats[cls]["mean"]
    inv_cov = training_stats[cls]["inv_cov"]
    
    diff = pixels_lab - mu
    dist = np.sum(diff * (diff @ inv_cov), axis=1).reshape(rows, cols)
    
    # Binary mask for this specific class
    mask = (dist < class_configs[cls]["threshold"]).astype(np.uint8) * 255
    
    # Morphological cleaning
    kernel = np.ones(class_configs[cls]["kernel_size"], np.uint8)
    morphed = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    
    # Add to the combined rotten mask
    combined_rotten_mask = cv2.bitwise_or(combined_rotten_mask, morphed)

# Find all clusters in the combined mask
contours, _ = cv2.findContours(combined_rotten_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

seed_point = None
if contours:
    # Identify the largest continuous blob
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        seed_point = np.array([[cX, cY]])
        print(f"Seed point found at: ({cX}, {cY}) based on the largest {len(largest_contour)} pixel cluster.")

# --- STEP 4: REFINEMENT WITH SAM 2 ---
best_mask = None
if seed_point is not None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    home = os.path.expanduser("~")
    sam2_checkpoint = os.path.join(home, "/home/alex/Thesis/sam2/checkpoints/sam2.1_hiera_small.pt")
    model_cfg = "sam2_hiera_s.yaml"

    # Build and load weights (strict=False)
    sam2_model = build_sam2(model_cfg, ckpt_path=None, device=device)
    sd = torch.load(sam2_checkpoint, map_location=device, weights_only=True)["model"]
    sam2_model.load_state_dict(sd, strict=False)
    
    predictor = SAM2ImagePredictor(sam2_model)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        predictor.set_image(img_rgb)
        masks, scores, _ = predictor.predict(
            point_coords=seed_point,
            point_labels=np.array([1]), 
            multimask_output=True,
        )
    best_mask = masks[np.argmax(scores)]

# --- STEP 5: VISUALIZATION ---
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.title("1. Original + Seed Point")
plt.imshow(img_rgb)
if seed_point is not None:
    plt.scatter(seed_point[0,0], seed_point[0,1], color='red', marker='x', s=200, lw=3)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("2. Combined Mahalanobis (Yellow+Brown)")
plt.imshow(combined_rotten_mask, cmap='gray')
plt.axis('off')

if best_mask is not None:
    plt.subplot(2, 2, 3)
    plt.title("3. SAM 2 Refined Mask")
    plt.imshow(best_mask, cmap='viridis')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("4. Final Pruning Target")
    final_view = cv2.bitwise_and(img_rgb, img_rgb, mask=best_mask.astype(np.uint8))
    plt.imshow(final_view)
    plt.axis('off')

plt.tight_layout()
plt.show()