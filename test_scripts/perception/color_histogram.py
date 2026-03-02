import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_histograms(image_path):
    # Load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("Error: Image not found.")
        return

    # Convert color spaces
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # --- 1. RGB Histogram ---
    colors = ('r', 'g', 'b')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        axs[0].plot(hist, color=col, lw=2)
    axs[0].set_title('RGB Histogram')
    axs[0].set_xlim([0, 256])
    axs[0].set_xlabel('Pixel Intensity')

    # --- 2. CIELAB Histogram ---
    # We focus on 'a' (Green-Red) and 'b' (Blue-Yellow) channels
    # a: 0 = Green, 255 = Red | b: 0 = Blue, 255 = Yellow (in OpenCV scaling)
    lab_labels = ('L*', 'a* (Green-Red)', 'b* (Blue-Yellow)')
    lab_colors = ('black', 'magenta', 'orange')
    for i, col in enumerate(lab_colors):
        hist = cv2.calcHist([img_lab], [i], None, [256], [0, 256])
        axs[1].plot(hist, color=col, lw=2)
    axs[1].set_title('CIELAB Histogram')
    axs[1].set_xlim([0, 256])
    axs[1].legend(lab_labels)

    plt.tight_layout()
    plt.show()

# Change this to your image path
plot_histograms("/home/alex/4_Color.png")