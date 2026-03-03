"""Visualize Mahalanobis decision boundaries in RGB space.

Loads a dictionary of statistics saved with `np.save(..., allow_pickle=True)`
under the name ``perception_stats.npy``.  Each entry is expected to be a
sub-dictionary containing at least ``mean`` and ``inv_cov`` (as produced by
the training code in earlier scripts).  Optionally a ``threshold`` value may
be stored; otherwise a default value will be used.

The script reads an image, converts it to RGB and scatter-plots the pixel
values in the R–G plane.  For each class in the stats file a decision boundary
is drawn on the same plot: the locus of points for which
Mahalanobis distance to the class mean equals the chosen threshold.  The B
channel of the contour is fixed at the mean blue value of the class so that
the boundary is a true slice of the 3D ellipsoid.

Usage:
    python mahalanobis_rgb_decision_boundary.py [image_file]

If no image argument is given the script defaults to ``training_data/1_Color.png``
currently used elsewhere in the repo.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D


def mahalanobis_distance_rgb(r, g, b, mean, inv_cov):
    """Return squared Mahalanobis distance of rgb vectors to mean.

    ``r`` and ``g`` are arrays of identical shape; ``b`` may be a scalar or
    array broadcastable to that shape. ``mean`` is length-3 and ``inv_cov`` is
    3x3.
    """
    # broadcast b to match r, g
    b_arr = np.full_like(r, b) if np.isscalar(b) else b
    pts = np.stack([r, g, b_arr], axis=-1)
    diff = pts - mean
    d2 = np.einsum('...i,ij,...j->...', diff, inv_cov, diff)
    return d2


def load_stats(path="perception_stats.npy"):
    stats = np.load(path, allow_pickle=True).item()
    return stats


def draw_ellipsoid(ax, mean, inv_cov, thresh, color, alpha=0.2):
    """Plot an ellipsoid corresponding to Mahalanobis distance = thresh.

    inv_cov: inverse covariance matrix
    """
    # eigen-decomposition of inv_cov
    w, v = np.linalg.eigh(inv_cov)
    # radii in eigenbasis: sqrt(thresh / lambda)
    radii = np.sqrt(thresh / w)
    # parameterise unit sphere
    u = np.linspace(0, 2*np.pi, 60)
    v_ang = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v_ang))
    y = np.outer(np.sin(u), np.sin(v_ang))
    z = np.outer(np.ones_like(u), np.cos(v_ang))
    # scale by radii
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            [x[i,j], y[i,j], z[i,j]] = radii * np.array([x[i,j], y[i,j], z[i,j]])
    # rotate and translate
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            vec = np.array([x[i,j], y[i,j], z[i,j]])
            xyz = v @ vec + mean
            x[i,j], y[i,j], z[i,j] = xyz
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, antialiased=True)


def main():
    img_file = sys.argv[1] if len(sys.argv) > 1 else "training_data/1_Color.png"
    img_bgr = cv2.imread(img_file)
    assert img_bgr is not None, f"Failed to load {img_file}"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape(-1, 3)

    stats = load_stats()

    # choose a grid resolution for R,G
    grid_n = 200
    r_vals = np.linspace(0, 255, grid_n)
    g_vals = np.linspace(0, 255, grid_n)
    R, G = np.meshgrid(r_vals, g_vals)
    # create 3D scatter
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sel = np.random.choice(pixels.shape[0], size=min(50000, pixels.shape[0]), replace=False)
    ax.scatter(pixels[sel, 0], pixels[sel, 1], pixels[sel, 2], s=1, alpha=0.3, c='k')

    # draw an ellipsoid for each class
    for cls, entry in stats.items():
        mean = np.asarray(entry["mean"])
        inv_cov = np.asarray(entry["inv_cov"])
        thresh = entry.get("threshold", 5.0)
        color = np.random.rand(3,)
        draw_ellipsoid(ax, mean, inv_cov, thresh, color=color, alpha=0.2)
        # label using a proxy point at mean
        ax.text(mean[0], mean[1], mean[2], cls, color=color)

    ax.set_xlabel('R channel')
    ax.set_ylabel('G channel')
    ax.set_zlabel('B channel')
    ax.set_title('RGB pixel cloud with Mahalanobis ellipsoids')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
