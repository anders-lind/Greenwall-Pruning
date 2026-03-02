import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# --- STEP 1: AGGREGATE ALL ANNOTATED DATA ---
image_indices = [1, 2, 3, 4, 5, 6]
annotation_types = ['yellow']#, 'yellow', 'glossy']

all_rotten_pixels = []

for idx in image_indices:
    img_path = f"{idx}_Color.png"
    if not os.path.exists(img_path):
        continue
        
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for annot_type in annotation_types:
        annot_path = f"{idx}_seg_{annot_type}.png"
        
        if os.path.exists(annot_path):
            annot_img = cv2.imread(annot_path)
            # Find the annotated (red/yellow) pixels
            # Adjusting mask to be slightly more lenient if needed
            mask = cv2.inRange(annot_img, (0, 0, 250), (10, 10, 255)) 
            
            # Extract pixels
            pixels = img_rgb[mask == 255]
            if len(pixels) > 0:
                all_rotten_pixels.append(pixels)

# Combine all lists into one large numpy array
training_data = np.vstack(all_rotten_pixels)

# --- STEP 2: CALCULATE STATISTICS ---
mean_vec = np.mean(training_data, axis=0)
cov_mat = np.cov(training_data.T)

# --- STEP 3: VISUALIZE HISTOGRAMS ---
colors = ('r', 'g', 'b')
plt.figure(figsize=(12, 6))
plt.title("Color Channel Distributions (Rotten Samples)")
plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Frequency")

for i, col in enumerate(colors):
    hist = np.histogram(training_data[:, i], bins=256, range=(0, 256))[0]
    plt.plot(hist, color=col, label=f'{col.upper()} channel')
    plt.fill_between(range(256), hist, color=col, alpha=0.2)

plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Total training pixels collected: {training_data.shape[0]}")