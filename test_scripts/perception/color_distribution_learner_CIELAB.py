import numpy as np
import cv2
import os
import glob

data_path = "training_data"
classes = ["yellow", "brown", "glossy"]
pixel_data = {cls: [] for cls in classes}

color_images = glob.glob(os.path.join(data_path, "*_Color.png"))
color_images.sort()

for color_path in color_images:
    prefix = os.path.basename(color_path).split('_')[0]
    img_bgr = cv2.imread(color_path)
    if img_bgr is None: continue
    
    # CONVERT TO LAB
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    
    for cls in classes:
        mask_path = os.path.join(data_path, f"{prefix}_seg_{cls}.png")
        if os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path)
            mask = cv2.inRange(mask_img, (0, 0, 250), (10, 10, 255))
            
            valid_pixels = img_lab[mask == 255]
            if len(valid_pixels) > 0:
                pixel_data[cls].append(valid_pixels)

# Statistics calculation
stats_lab = {}
for cls in classes:
    if len(pixel_data[cls]) > 0:
        all_pixels = np.vstack(pixel_data[cls]).astype(float)
        stats_lab[cls] = {
            "mean": np.mean(all_pixels, axis=0),
            "inv_cov": np.linalg.inv(np.cov(all_pixels.T) + np.eye(3) * 1e-6)
        }

np.save("perception_stats_cielab.npy", stats_lab)
print(stats_lab)
print("Stats saved in CIELAB space.")