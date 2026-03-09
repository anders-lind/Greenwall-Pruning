import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import os
import time
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- STEP 1: CONFIGURATION & MODEL SETUP (Outside Timing) ---
class_configs = {
    "yellow": {"threshold": 5.0, "kernel_size": (5, 5)},
    "brown":  {"threshold": 4.0, "kernel_size": (5, 5)}
}

device = "cuda" if torch.cuda.is_available() else "cpu"
home = os.path.expanduser("~")
sam2_checkpoint = os.path.join(home, "Thesis/sam2/checkpoints/sam2.1_hiera_base_plus.pt")
model_cfg = "sam2_hiera_b+.yaml"

# Load SAM 2 once (Not timed)
print("Initializing SAM 2.1 Base Plus...")
sam2_model = build_sam2(model_cfg, ckpt_path=None, device=device)
sd = torch.load(sam2_checkpoint, map_location=device, weights_only=True)["model"]
sam2_model.load_state_dict(sd, strict=False)
predictor = SAM2ImagePredictor(sam2_model)

# --- STEP 2: LOAD DATA & STATS ---
stats_path = "perception_stats_cielab.npy"
training_stats = np.load(stats_path, allow_pickle=True).item()

test_file = "training_data/2_Color.png"
img_bgr = cv2.imread(test_file)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(float)
rows, cols, _ = img_lab.shape
pixels_lab = img_lab.reshape(-1, 3)

# --- STEP 3: THE TIMED PIPELINE ---
print(f"Benchmarking Pipeline on {device}...")

if device == "cuda":
    torch.cuda.synchronize()
start_total = time.perf_counter()

# --- Part A: Mahalanobis & Clustering (Pre-processing) ---
start_pre = time.perf_counter()
combined_rotten_mask = np.zeros((rows, cols), dtype=np.uint8)

for cls in ["yellow", "brown"]:
    mu = training_stats[cls]["mean"]
    inv_cov = training_stats[cls]["inv_cov"]
    diff = pixels_lab - mu
    dist = np.sum(diff * (diff @ inv_cov), axis=1).reshape(rows, cols)
    
    mask = (dist < class_configs[cls]["threshold"]).astype(np.uint8) * 255
    kernel = np.ones(class_configs[cls]["kernel_size"], np.uint8)
    morphed = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    combined_rotten_mask = cv2.bitwise_or(combined_rotten_mask, morphed)

contours, _ = cv2.findContours(combined_rotten_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

seed_point = None
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        seed_point = np.array([[cX, cY]])

end_pre = time.perf_counter()

# --- Part B: SAM 2 Raw Inference ---
best_mask = None
start_inf = time.perf_counter()

if seed_point is not None:
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        predictor.set_image(img_rgb)
        masks, scores, _ = predictor.predict(
            point_coords=seed_point,
            point_labels=np.array([1]), 
            multimask_output=True,
        )
    best_mask = masks[np.argmax(scores)]

if device == "cuda":
    torch.cuda.synchronize()
end_inf = time.perf_counter()

# --- Part C: Final Totals ---
end_total = time.perf_counter()

t_pre = (end_pre - start_pre) * 1000  # Convert to ms
t_inf = (end_inf - start_inf) * 1000  # Convert to ms
t_total = (end_total - start_total) * 1000
fps = 1000 / t_total

print(f"\n{'='*30}")
print(f"PERFORMANCE BREAKDOWN")
print(f"{'='*30}")
print(f"Pre-processing (Mahal): {t_pre:.2f} ms")
print(f"SAM 2 Raw Inference:    {t_inf:.2f} ms")
print(f"Total Pipeline Latency: {t_total:.2f} ms")
print(f"Operational FPS:        {fps:.2f}")
print(f"{'='*30}\n")

# --- STEP 4: VISUALIZATION ---
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.title(f"Targeting: {t_total:.1f}ms total")
plt.imshow(img_rgb)
if seed_point is not None:
    plt.scatter(seed_point[0,0], seed_point[0,1], color='red', marker='x', s=200, lw=3)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("Combined Rotten Mask")
plt.imshow(combined_rotten_mask, cmap='gray')
plt.axis('off')

if best_mask is not None:
    plt.subplot(2, 2, 3)
    plt.title("Refined Mask")
    plt.imshow(best_mask, cmap='viridis')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("Final Pruning Result")
    plt.imshow(cv2.bitwise_and(img_rgb, img_rgb, mask=best_mask.astype(np.uint8)))
    plt.axis('off')

plt.tight_layout()
plt.show()