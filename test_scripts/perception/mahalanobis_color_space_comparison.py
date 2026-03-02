"""Compare how Mahalanobis distance behaves in different color spaces.

This script reads a training image together with an annotation mask to extract
pixels belonging to a target class (e.g. yellow leaves).  It computes a mean
vector and covariance matrix for the annotated pixels in several colour spaces
(RGB, CIELAB, HSV, YCrCb, ...).  A separate test image is then scored using the
same sets of statistics and a range of thresholds is applied to compute the
fraction of pixels that fall below each threshold.

For each threshold we also compute the overlap between the sets of pixels
selected by the different colour spaces, giving an indication of how the choice
of representation affects which pixels are considered "close" to the model.

Usage is simple: edit the hard‑coded filenames or (later) adapt to take CLI
parameters.  The output is printed to the console and a few diagnostic
histograms/plots are drawn.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from itertools import combinations
import sys


def convert_space(img, space: str) -> np.ndarray:
    """Convert a BGR image to the desired color space.

    Supported spaces: 'BGR' (identity), 'RGB', 'LAB', 'HSV', 'YCrCb'.
    """
    if space == 'BGR':
        return img.copy()
    if space == 'RGB':
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if space == 'LAB':
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    if space == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if space == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    raise ValueError(f"Unsupported space '{space}'")


def mask_from_annotation(annot_img: np.ndarray, name: str = "annotation", tol: int = 8) -> np.ndarray:
    """Create a binary mask from an annotation image.

    This tries to detect red annotation marks (pure red in RGB: (255,0,0),
    which corresponds to BGR (0,0,255) in OpenCV). If red pixels are
    present (optionally within a tolerance), those pixels are treated as the
    annotated class. If no red is detected the function falls back to the
    previous heuristic (non-black pixels) and emits a warning because that
    often means the annotation image contains original image pixels.
    """
    if annot_img is None:
        return None

    h, w = annot_img.shape[:2]
    # exact BGR red
    red_exact = (annot_img[:, :, 0] == 0) & (annot_img[:, :, 1] == 0) & (annot_img[:, :, 2] == 255)
    if red_exact.sum() > 0 and red_exact.sum() < h * w:
        return red_exact

    # tolerant match (in case of minor compression/antialiasing)
    diff = np.abs(annot_img.astype(int) - np.array([0, 0, 255], dtype=int))
    red_tol = np.all(diff <= tol, axis=2)
    if red_tol.sum() > 0 and red_tol.sum() < h * w:
        print(f"Detected red annotations in {name} (tolerance={tol}).")
        return red_tol

    # fallback: non-black pixels
    gray = cv2.cvtColor(annot_img, cv2.COLOR_BGR2GRAY)
    fb = gray > 0
    print(f"Warning: no red annotation found in {name}; using non-black fallback (this may mark many pixels).", file=sys.stderr)
    return fb


def compute_statistics(pixels: np.ndarray, eps: float = 1e-6):
    """Return (mean, cov, inv_cov) for an array of shape (N, C)."""
    mean = np.mean(pixels, axis=0)
    cov = np.cov(pixels.T)
    inv_cov = np.linalg.inv(cov + np.eye(cov.shape[0]) * eps)
    return mean, cov, inv_cov


def mahalanobis_distance(pixels: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray):
    """Compute Mahalanobis distance of each row in ``pixels`` from ``mean``."""
    diff = pixels - mean
    # efficient quadratic form: sum(diff * (diff @ inv_cov), axis=1)
    return np.sum(diff * (diff @ inv_cov), axis=1)


def main():
    # --- configuration ----------------------------------------------------
    train_img_file = "4_Color.png"          # image used for computing mean/cov
    train_annot_file = "4_seg_yellow.png"  # annotation; nonzero pixels indicate class
    test_img_file = "2_Color.png"           # image to score

    #color_spaces = ["RGB", "LAB", "HSV", "YCrCb"]
    color_spaces = ["RGB", "LAB", "HSV"]
    thresholds = np.linspace(1, 100, 50)  # arbitrary range; adjust as needed

    # read training data --------------------------------------------------
    img_train = cv2.imread(train_img_file)
    assert img_train is not None, f"Failed to load {train_img_file}"
    annot = cv2.imread(train_annot_file)
    assert annot is not None, f"Failed to load {train_annot_file}"

    # create binary mask from annotation; prefer explicit red markers if present
    mask = mask_from_annotation(annot, name="training annotation")

    # test image + optional test annotation -------------------------------
    img_test = cv2.imread(test_img_file)
    assert img_test is not None, f"Failed to load {test_img_file}"
    h, w, _ = img_test.shape
    pixels_test_flat = img_test.reshape(-1, 3)

    # if you have a ground-truth annotation for the test image, set this
    # filename to compute false positives against the test GT.  If left as
    # None, the script will fall back to using the training annotation (old
    # behaviour).
    test_annot_file = "2_seg_yellow.png"
    test_mask = None
    if test_annot_file is not None:
        test_annot = cv2.imread(test_annot_file)
        if test_annot is not None:
            test_mask = mask_from_annotation(test_annot, name="test annotation")
        else:
            print(f"Warning: failed to load {test_annot_file}; falling back to training annotation for FP calculation")


    # compute statistics for each colour space using same annotated pixels
    stats = {}
    for space in color_spaces:
        train_conv = convert_space(img_train, space)
        pixels_train = train_conv.reshape(-1, 3)
        annotated_pixels = pixels_train[mask.flatten()]
        mean, cov, inv_cov = compute_statistics(annotated_pixels)
        stats[space] = {
            "mean": mean,
            "cov": cov,
            "inv_cov": inv_cov,
        }

    # for each space, score the test image once (distances are independent of threshold)
    distances = {}
    for space in color_spaces:
        test_conv = convert_space(img_test, space)
        dists = mahalanobis_distance(test_conv.reshape(-1, 3),
                                      stats[space]["mean"],
                                      stats[space]["inv_cov"])
        distances[space] = dists

    # analysis --------------------------------------------------------------
    # Decide which ground-truth to use for FP calculation. Prefer a test
    # annotation if available; otherwise use the training annotation (which
    # was the previous behaviour).
    if test_mask is not None:
        gt_mask_flat = test_mask.flatten()
    else:
        gt_mask_flat = mask.flatten()
    neg_mask_flat = ~gt_mask_flat

    header = ["Threshold"]
    header += [f"%{space}" for space in color_spaces]
    header += [f"FP%_{space}" for space in color_spaces]
    print(", ".join(header))

    # prepare containers for per-threshold true/false positive counts
    total = h * w
    tp_history = {space: [] for space in color_spaces}
    fp_history = {space: [] for space in color_spaces}
    perc_history = {space: [] for space in color_spaces}

    for t in thresholds:
        masks = {space: (distances[space] < t) for space in color_spaces}
        counts = {space: masks[space].sum() for space in color_spaces}
        perc = {space: counts[space] / total * 100.0 for space in color_spaces}

        # true positives: pixels inside GT and selected by mahalanobis
        tp_counts = {space: np.logical_and(masks[space], gt_mask_flat).sum()
                     for space in color_spaces}

        # false positives: pixels outside GT but selected by mahalanobis
        fp_counts = {space: np.logical_and(masks[space], neg_mask_flat).sum()
                     for space in color_spaces}

        # store history
        for space in color_spaces:
            tp_history[space].append(tp_counts[space])
            fp_history[space].append(fp_counts[space])
            perc_history[space].append(perc[space])
        row = [f"{t:.1f}"]
        row += [f"{perc[space]:.2f}" for space in color_spaces]
        row += [f"{(fp_counts[space]/total*100.0):.2f}" for space in color_spaces]
        print(", ".join(row))

    # Analyse results: find threshold with best TP/FP ratio and best precision
    print("\nSummary (best thresholds per color space):")
    for space in color_spaces:
        tps = np.array(tp_history[space], dtype=float)
        fps = np.array(fp_history[space], dtype=float)
        ths = np.array(thresholds)

        # TP/FP ratio (handle FP==0)
        ratios = np.full_like(tps, -np.inf)
        nonzero_fp = fps > 0
        ratios[nonzero_fp] = tps[nonzero_fp] / fps[nonzero_fp]
        # if FP==0 and TP>0, set ratio to +inf (ideal)
        ratios[(fps == 0) & (tps > 0)] = np.inf

        best_idx = int(np.nanargmax(ratios))
        best_t = ths[best_idx]
        best_tp = int(tps[best_idx])
        best_fp = int(fps[best_idx])
        best_ratio = ratios[best_idx]
        # precision = TP / (TP + FP)
        precisions = np.divide(tps, (tps + fps), out=np.zeros_like(tps), where=(tps+fps)>0)
        best_prec_idx = int(np.nanargmax(precisions))
        best_prec_t = ths[best_prec_idx]
        best_prec = precisions[best_prec_idx]

        ratio_str = "inf" if np.isinf(best_ratio) else f"{best_ratio:.2f}"
        print(f"- {space}: best TP/FP at t={best_t:.1f} -> TP={best_tp}, FP={best_fp}, ratio={ratio_str}, precision={best_prec:.3f} (best precision t={best_prec_t:.1f})")

    # Plot metrics vs thresholds for each colour space
    plt.figure(figsize=(14, 10))

    # TP percentage only
    ax1 = plt.subplot(2, 2, 1)
    for space in color_spaces:
        ax1.plot(thresholds, np.array(tp_history[space]) / total * 100.0, label=f"TP% {space}")
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Percent of image (%)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title('True Positive (%) vs Threshold (log-log)')
    ax1.legend(loc='best', fontsize='small')

    # FP percentage
    ax2 = plt.subplot(2, 2, 2)
    for space in color_spaces:
        ax2.plot(thresholds, np.array(fp_history[space]) / total * 100.0, label=f"FP% {space}")
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Percent of image (%)')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title('False Positive (%) vs Threshold (log-log)')
    ax2.legend(loc='best', fontsize='small')

    # TP/FP ratio (use log scale to handle large ratios)
    ax3 = plt.subplot(2, 1, 2)
    for space in color_spaces:
        tps = np.array(tp_history[space], dtype=float)
        fps = np.array(fp_history[space], dtype=float)
        ratios = np.full_like(tps, np.nan)
        nonzero_fp = fps > 0
        ratios[nonzero_fp] = tps[nonzero_fp] / fps[nonzero_fp]
        ratios[(fps == 0) & (tps > 0)] = np.nanmax(ratios[nonzero_fp]) if np.any(nonzero_fp) else 1e6
        ax3.plot(thresholds, ratios, label=space)
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('TP / FP (ratio)')
    ax3.set_yscale('log')
    ax3.set_title('TP/FP Ratio vs Threshold (log scale)')
    ax3.legend(loc='best', fontsize='small')

    plt.tight_layout()
    plt.show()

    # optional visualization ------------------------------------------------
    # we can plot histograms of distances in each space
    plt.figure(figsize=(8, 6))
    for space in color_spaces:
        plt.hist(distances[space], bins=100, alpha=0.5, label=space)
    plt.xlabel("Mahalanobis distance")
    plt.ylabel("Pixel count")
    plt.legend()
    plt.title("Distance distributions in different color spaces")
    plt.show()


if __name__ == "__main__":
    main()
