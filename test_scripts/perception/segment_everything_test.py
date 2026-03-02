import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# --- STEP 1: TRAINING ---
img_train = cv2.imread("4_Color.png")
img_annot = cv2.imread("4_seg_yellow.png")
mask_train = cv2.inRange(img_annot, (0, 0, 254), (1, 1, 256))
pixels_train = np.reshape(img_train, (-1, 3))
annot_pix_values = pixels_train[np.reshape(mask_train, (-1)) == 255]
mean = np.average(annot_pix_values, axis=0)
inv_cov = np.linalg.inv(np.cov(annot_pix_values.transpose()) + np.eye(3) * 1e-6)

# --- STEP 2: LOAD TEST IMAGE ---
test_file = "1_Color.png"
img_test = cv2.imread(test_file)
img_test_rgb = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

# --- STEP 3: INITIALIZE SAM ON GPU ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")

sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam.to(device=device)

# We adjust 'points_per_side' to 32 now that we have GPU power!
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32, 
    min_mask_region_area=500 # Filter out small noise
)

# --- STEP 4: GENERATE AND SCORE ---
all_masks = mask_generator.generate(img_test_rgb)

score_map = np.zeros(img_test.shape[:2])
proposal_overlay = np.zeros_like(img_test_rgb)
candidates = []

for m in all_masks:
    seg = m['segmentation']
    
    # ROBOTICS LOGIC: Ignore masks that are way too big (likely the wall)
    # If a mask takes up more than 30% of the image, skip it
    if np.sum(seg) > (img_test.shape[0] * img_test.shape[1] * 0.3):
        continue

    # Score the mask
    pixels = img_test[seg]
    diff = pixels - mean
    dist = np.mean(np.sum(diff * (diff @ inv_cov), axis=1))
    
    candidates.append({'mask': seg, 'score': dist, 'bbox': m['bbox']})
    score_map[seg] = dist
    proposal_overlay[seg] = np.random.randint(0, 255, (3,))

# Find the winner
candidates.sort(key=lambda x: x['score'])
winner = candidates[0]

# --- STEP 5: FINAL PLOT ---
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.title(f"Target: {test_file}")
plt.imshow(img_test_rgb)
x, y, w, h = winner['bbox']
plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='lime', facecolor='none', lw=3))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title(f"GPU Proposals ({len(all_masks)})")
plt.imshow(cv2.addWeighted(img_test_rgb, 0.4, proposal_overlay, 0.6, 0))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Rottenness Heatmap")
plt.imshow(score_map, cmap='jet_r', vmax=80)
plt.colorbar(shrink=0.6)
plt.axis('off')

plt.tight_layout()
plt.show()