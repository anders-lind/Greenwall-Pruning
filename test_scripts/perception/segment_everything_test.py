import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import os
import time

# SAM 2 Imports
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# --- STEP 1: TRAINING (Mahalanobis) ---
# Training on image 4 where you have the ground truth annotation
img_train = cv2.imread("Greenwall-Pruning/test_scripts/perception/training_data/4_Color.png")
img_annot = cv2.imread("Greenwall-Pruning/test_scripts/perception/training_data/4_seg_yellow.png")

if img_train is None or img_annot is None:
    print("Error: Training images not found. Check your paths!")
    exit()

# Extract rotten pixels (Assuming they are marked in Red/Yellow in your mask)
mask_train = cv2.inRange(img_annot, (0, 0, 254), (1, 1, 256))
pixels_train = np.reshape(img_train, (-1, 3))
annot_pix_values = pixels_train[np.reshape(mask_train, (-1)) == 255]

# Calculate Mean and Inverse Covariance
mean = np.average(annot_pix_values, axis=0)
inv_cov = np.linalg.inv(np.cov(annot_pix_values.transpose()) + np.eye(3) * 1e-6)

# --- STEP 2: LOAD TEST IMAGE ---
test_file = "Greenwall-Pruning/test_scripts/perception/training_data/1_Color.png"
img_test = cv2.imread(test_file)
if img_test is None:
    print(f"Error: Could not load {test_file}")
    exit()
img_test_rgb = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

# --- STEP 3: INITIALIZE SAM 2 (Non-Strict Loading) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")
home = os.path.expanduser("~")

# Paths - Using the Large model as you requested
sam2_checkpoint = os.path.join(home, "Thesis/sam2/checkpoints/sam2.1_hiera_small.pt")
model_cfg = "sam2_hiera_s.yaml" 

print(f"Loading SAM 2.1 Large model architecture...")

# 1. Build skeleton (ckpt_path=None avoids the initial crash)
sam2_model = build_sam2(model_cfg, ckpt_path=None, device=device, apply_postprocessing=True)

# 2. Manual weight injection with strict=False to ignore video-tracking keys
print("Injecting weights (ignoring video keys)...")
sd = torch.load(sam2_checkpoint, map_location=device, weights_only=True)["model"]
sam2_model.load_state_dict(sd, strict=False)

# 3. Setup the Mask Generator
# Note: Lowered points_per_batch to 32 to fit Large model in 4GB VRAM
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=32,
    points_per_batch=32, 
    min_mask_region_area=500
)

# --- STEP 4: GENERATE AND SCORE ---
print(f"Processing {test_file}...")
start_time = time.time()

with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    all_masks = mask_generator.generate(img_test_rgb)

end_time = time.time()
print(f"Inference Complete! Found {len(all_masks)} objects in {end_time - start_time:.2f} seconds.")

score_map = np.zeros(img_test.shape[:2])
proposal_overlay = np.zeros_like(img_test_rgb)
candidates = []

for m in all_masks:
    seg = m['segmentation']
    area = np.sum(seg)
    
    # Filter out background/wall (objects > 30% of image)
    if area > (img_test.shape[0] * img_test.shape[1] * 0.3):
        continue

    # Score the mask using Mahalanobis distance
    pixels = img_test[seg]
    diff = pixels - mean
    dist = np.mean(np.sum(diff * (diff @ inv_cov), axis=1))
    
    candidates.append({'mask': seg, 'score': dist, 'bbox': m['bbox']})
    score_map[seg] = dist
    
    # Visualization color
    proposal_overlay[seg] = np.random.randint(0, 255, (3,))

# Identify the winner
if not candidates:
    print("No leaf candidates identified.")
    exit()

candidates.sort(key=lambda x: x['score'])
winner = candidates[0]

# --- STEP 5: FINAL VISUALIZATION ---
plt.figure(figsize=(18, 6))

# Panel 1: Target
plt.subplot(1, 3, 1)
plt.title(f"Target Selection\nMahal. Dist: {winner['score']:.2f}")
plt.imshow(img_test_rgb)
x, y, w, h = winner['bbox']
plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='lime', facecolor='none', lw=3))
plt.axis('off')

# Panel 2: All Proposals
plt.subplot(1, 3, 2)
plt.title(f"SAM 2 Proposals ({len(all_masks)})")
plt.imshow(cv2.addWeighted(img_test_rgb, 0.4, proposal_overlay, 0.6, 0))
plt.axis('off')

# Panel 3: Classification Heatmap
plt.subplot(1, 3, 3)
plt.title("Decision Heatmap\n(Red = Most Rotten)")
im = plt.imshow(score_map, cmap='jet_r', vmax=80)
plt.colorbar(im, shrink=0.6)
plt.axis('off')

plt.tight_layout()
print("Opening visualization window...")
plt.show()