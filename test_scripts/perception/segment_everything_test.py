import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import os
import time

# SAM 2 Imports
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# --- STEP 1: TRAINING (Mahalanobis - Setup Cost) ---
img_train = cv2.imread("training_data/4_Color.png")
img_annot = cv2.imread("training_data/4_seg_yellow.png")

if img_train is None or img_annot is None:
    print("Error: Training images not found. Check your paths!")
    exit()

mask_train = cv2.inRange(img_annot, (0, 0, 254), (1, 1, 256))
pixels_train = np.reshape(img_train, (-1, 3))
annot_pix_values = pixels_train[np.reshape(mask_train, (-1)) == 255]
mean = np.average(annot_pix_values, axis=0)
inv_cov = np.linalg.inv(np.cov(annot_pix_values.transpose()) + np.eye(3) * 1e-6)

# --- STEP 2: LOAD TEST IMAGE ---
test_file = "training_data/1_Color.png"
img_test = cv2.imread(test_file)
if img_test is None:
    print(f"Error: Could not load {test_file}")
    exit()
img_test_rgb = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

# --- STEP 3: INITIALIZE SAM 2 (Setup Cost) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")
home = os.path.expanduser("~")

sam2_checkpoint = os.path.join(home, "Thesis/sam2/checkpoints/sam2.1_hiera_base_plus.pt")
model_cfg = "sam2_hiera_b+.yaml" 

print(f"Loading SAM 2.1 Base Plus...")
sam2_model = build_sam2(model_cfg, ckpt_path=None, device=device, apply_postprocessing=True)
sd = torch.load(sam2_checkpoint, map_location=device, weights_only=True)["model"]
sam2_model.load_state_dict(sd, strict=False)

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=32,
    points_per_batch=32, 
    min_mask_region_area=500
)

# --- STEP 4: TIMED GENERATION & SCORING ---
print(f"Starting Benchmark for {test_file}...")

if device == "cuda":
    torch.cuda.synchronize()
start_total = time.perf_counter()

# --- Part A: Raw SAM 2 Inference ---
start_inf = time.perf_counter()
with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    all_masks = mask_generator.generate(img_test_rgb)

if device == "cuda":
    torch.cuda.synchronize()
end_inf = time.perf_counter()

# --- Part B: Post-Processing (Mahalanobis Scoring) ---
start_post = time.perf_counter()

score_map = np.zeros(img_test.shape[:2])
proposal_overlay = np.zeros_like(img_test_rgb)
candidates = []

for m in all_masks:
    seg = m['segmentation']
    area = np.sum(seg)
    if area > (img_test.shape[0] * img_test.shape[1] * 0.3):
        continue

    pixels = img_test[seg]
    diff = pixels - mean
    dist = np.mean(np.sum(diff * (diff @ inv_cov), axis=1))
    
    candidates.append({'mask': seg, 'score': dist, 'bbox': m['bbox']})
    score_map[seg] = dist
    proposal_overlay[seg] = np.random.randint(0, 255, (3,))

if candidates:
    candidates.sort(key=lambda x: x['score'])
    winner = candidates[0]

end_post = time.perf_counter()
end_total = time.perf_counter()

# --- Part C: Performance Metrics ---
t_inf = (end_inf - start_inf) * 1000  # Raw GPU Inference
t_post = (end_post - start_post) * 1000 # Scoring logic
t_total = (end_total - start_total) * 1000
fps = 1000 / t_total

print(f"\n{'='*30}")
print(f"SEGMENT-ALL PERFORMANCE")
print(f"{'='*30}")
print(f"SAM 2 Raw Inference:    {t_inf:.2f} ms")
print(f"Mahalanobis Scoring:    {t_post:.2f} ms")
print(f"Total Pipeline Latency: {t_total:.2f} ms")
print(f"Operational FPS:        {fps:.2f}")
print(f"Total Objects Found:    {len(all_masks)}")
print(f"{'='*30}\n")

# --- STEP 5: VISUALIZATION ---
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.title(f"Target Selection\nTotal Time: {t_total/1000:.2f}s")
plt.imshow(img_test_rgb)
if candidates:
    x, y, w, h = winner['bbox']
    plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='lime', facecolor='none', lw=3))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title(f"SAM 2 Proposals ({len(all_masks)})")
plt.imshow(cv2.addWeighted(img_test_rgb, 0.4, proposal_overlay, 0.6, 0))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Decision Heatmap")
im = plt.imshow(score_map, cmap='jet_r', vmax=80)
plt.colorbar(im, shrink=0.6)
plt.axis('off')

plt.tight_layout()
plt.show()