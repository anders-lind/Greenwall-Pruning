import numpy as np
import cv2
import os
import glob

# --- CONFIGURATION ---
data_path = "training_data"
# Define the classes we are looking for based on your file suffixes
classes = ["yellow", "brown", "glossy"]

# Dictionary to hold the raw pixel values for each class
pixel_data = {cls: [] for cls in classes}

# --- STEP 1: DATA COLLECTION ---
# We look for all images ending in _Color.png to find the "base" images
color_images = glob.glob(os.path.join(data_path, "*_Color.png"))
color_images.sort()

print(f"Found {len(color_images)} training sets. Starting pixel extraction...")

for color_path in color_images:
    # Extract the prefix (e.g., '1', '2', etc.)
    prefix = os.path.basename(color_path).split('_')[0]
    img = cv2.imread(color_path)
    
    if img is None: continue
    
    # Check for each class mask associated with this image prefix
    for cls in classes:
        mask_path = os.path.join(data_path, f"{prefix}_seg_{cls}.png")
        
        if os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path)
            # Find pixels where the mask is Red (assuming your labeling style)
            # (0, 0, 254) to (1, 1, 256) catches the red channel
            mask = cv2.inRange(mask_img, (0, 0, 250), (10, 10, 255))
            
            # Extract pixels from the color image using the mask
            valid_pixels = img[mask == 255]
            if len(valid_pixels) > 0:
                pixel_data[cls].append(valid_pixels)
                print(f"  [Image {prefix}] Added {len(valid_pixels)} pixels to class: {cls}")

# --- STEP 2: STATISTICAL CALCULATION ---
stats = {}

print("\n--- Final Statistics ---")
for cls in classes:
    if len(pixel_data[cls]) > 0:
        # Stack all collected pixel arrays into one giant list
        all_pixels = np.vstack(pixel_data[cls])
        
        # Calculate Mean
        mean = np.mean(all_pixels, axis=0)
        
        # Calculate Covariance (and its inverse for Mahalanobis)
        cov = np.cov(all_pixels.transpose())
        inv_cov = np.linalg.inv(cov + np.eye(3) * 1e-6)
        
        stats[cls] = {
            "mean": mean,
            "cov": cov,
            "inv_cov": inv_cov,
            "count": len(all_pixels)
        }
        
        print(f"Class: {cls.upper()}")
        print(f"  Pixel Count: {len(all_pixels)}")
        print(f"  Mean (BGR): {mean.round(2)}")
        print(f"  Covariance:\n {cov.round(3)}")
    else:
        print(f"Class: {cls.upper()} - No data found.")

# --- STEP 3: SAVE FOR LATER ---
# You can save this as a .npy file so your SAM 2 script can just load it
np.save("perception_stats.npy", stats)
print("\nStats saved to perception_stats.npy")