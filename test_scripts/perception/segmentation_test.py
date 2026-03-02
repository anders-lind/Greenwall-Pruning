import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor

# --- STEP 1: TRAINING (Mahalanobis) ---
img_train = cv2.imread("4_Color.png")
img_annot = cv2.imread("4_seg_yellow.png")

lower_limit = (0, 0, 254)
upper_limit = (1, 1, 256)
mask_train = cv2.inRange(img_annot, lower_limit, upper_limit)

pixels_train = np.reshape(img_train, (-1, 3))
mask_pixels_flat = np.reshape(mask_train, (-1))
annot_pix_values = pixels_train[mask_pixels_flat == 255, ]

mean = np.average(annot_pix_values, axis=0)
cov = np.cov(annot_pix_values.transpose())
inv_cov = np.linalg.inv(cov + np.eye(3) * 1e-6)

# --- STEP 2: PROCESSING TEST IMAGE ---
test_file = "2_Color.png"
img_test = cv2.imread(test_file)
img_test_rgb = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
rows, cols, _ = img_test.shape

pixels_test = np.reshape(img_test, (-1, 3))
diff = pixels_test - mean
mahalanobis_dist = np.sum(diff * (diff @ inv_cov), axis=1)
dist_img = np.reshape(mahalanobis_dist, (rows, cols))

# --- STEP 3: FINDING THE SEED POINT ---
threshold_val = 10
raw_mask = (dist_img < threshold_val).astype(np.uint8) * 255

# Clean the mask to find solid blobs
kernel = np.ones((10, 10), np.uint8)
mask_morphed = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
mask_morphed = cv2.morphologyEx(mask_morphed, cv2.MORPH_CLOSE, kernel)

# Find contours of the rotten areas
contours, _ = cv2.findContours(mask_morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) > 0:
    # 1. Find the largest blob (by area)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 2. Calculate the Centroid (Center of Mass)
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        # Fallback to the first point if area is 0
        cX, cY = largest_contour[0][0]
    
    seed_point = np.array([[cX, cY]])
else:
    print("No rotten leaves detected.")
    seed_point = None

# --- STEP 4: REFINEMENT WITH SAM ---
if seed_point is not None:
    sam_checkpoint = "sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device.upper()}")

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(img_test_rgb)

    # Prompt SAM with the centroid of the largest blob
    masks, scores, _ = predictor.predict(
        point_coords=seed_point,
        point_labels=np.array([1]), 
        multimask_output=True,
    )

    # Use the highest scoring mask
    best_mask = masks[np.argmax(scores)]
    final_segmented = cv2.bitwise_and(img_test, img_test, mask=best_mask.astype(np.uint8)*255)
    final_segmented_rgb = cv2.cvtColor(final_segmented, cv2.COLOR_BGR2RGB)

# --- STEP 5: VISUALIZATION ---
plt.figure(figsize=(20, 5))

# Panel 1: Original
plt.subplot(1, 4, 1)
plt.title(f"1. Original: {test_file}")
plt.imshow(img_test_rgb)
if 'seed_point' in locals() and seed_point is not None:
    plt.scatter(seed_point[0,0], seed_point[0,1], color='red', marker='x', s=100)
plt.axis('off')

# Panel 2: Mahalanobis Result
plt.subplot(1, 4, 2)
plt.title("2. Mahalanobis Mask")
plt.imshow(mask_morphed, cmap='gray')
plt.axis('off')

# Panel 3 & 4: SAM Results (Only if detected)
if 'best_mask' in locals():
    plt.subplot(1, 4, 3)
    plt.title("3. SAM Refined Mask")
    plt.imshow(best_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("4. Final Segmented Leaf")
    plt.imshow(final_segmented_rgb)
    plt.axis('off')
else:
    plt.subplot(1, 4, 3)
    plt.text(0.5, 0.5, 'No Detection\nTry increasing\nthreshold_val', 
             ha='center', va='center', fontsize=12, color='red')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.title("4. Final Result")
    plt.imshow(np.zeros_like(img_test_rgb)) # Black image
    plt.axis('off')

plt.tight_layout()
plt.show()