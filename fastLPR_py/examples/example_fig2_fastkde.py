"""
Code to generate Figure 2: Kernel Density Estimation (1D, 2D, and 3D Examples)

This script reproduces Figure 2 from the fastLPR paper, demonstrating:
  Row 1 - KDE Examples:
  - Panel (a): 1D KDE with multiple bandwidths
  - Panel (b): 2D KDE contour plot
  - Panel (c): 3D KDE volume rendering
  Row 2 - Bandwidth Selection:
  - Panel (d): 1D bandwidth selection via LCV
  - Panel (e): 2D bandwidth selection heatmap
  - Panel (f): 3D bandwidth selection volume rendering

The figure follows JSS publication standards with:
  - Fixed random seed for reproducibility
  - Consistent styling (fonts, colors, sizes)
  - 300 DPI resolution for publication
  - Self-contained code (no external dependencies except fastLPR)

Copyright (c) 2024 fastLPR Development Team
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import sys

# Add fastLPR to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastlpr import cv_fastkde, get_hlist

print()
print("=" * 80)
print("Figure 2: Kernel Density Estimation (1D, 2D, and 3D Examples)")
print("=" * 80)
print()

################################################################################
# Panel (a) and (d): 1D Bimodal Distribution
################################################################################

print("Generating Panel (a) and (d): 1D KDE...")

# Set random seed for reproducibility (matches paper)
np.random.seed(42)

# Generate bimodal data: two Gaussian modes
# Check for quick test mode (reduced samples for faster testing)
quick_test = os.environ.get("FASTLPR_QUICK_TEST", "0") == "1"
if quick_test:
    n1 = 500  # Reduced for quick testing (50% of original)
    n2 = 200
    print("  [QUICK TEST MODE: Using 50% sample sizes for faster testing]")
else:
    n1 = 1000  # First mode sample size
    n2 = 400  # Second mode sample size
x1 = np.random.randn(n1, 1) * 0.5  # First mode at 0, std=0.5
x2 = np.random.randn(n2, 1) * 0.7 + 3  # Second mode at 3, std=0.7
x = np.vstack([x1, x2])

print(f"  - Generated {len(x)} samples from bimodal distribution")

# Create bandwidth list (log-spaced from 0.01 to 2)
hlist = get_hlist(20, np.array([[0.01, 2]]))  # Default is logspace
print(f"  - Testing {len(hlist)} bandwidths")

# Compute KDE with automatic bandwidth selection via LCV
opt = {"verbose": False}
start_time = time.time()
kde = cv_fastkde(x, hlist, opt)
elapsed = time.time() - start_time

print(f"  - Computation time: {elapsed:.3f} seconds")
print(f"  - Selected bandwidth: h = {kde.h[0]:.4f}")

################################################################################
# Panel (b) and (e): 2D Density Estimation
################################################################################

print("\nGenerating Panel (b) and (e): 2D KDE...")

# Set random seed for reproducibility
np.random.seed(44)

# Generate 2D data with two clusters
if quick_test:
    n_cluster = 500  # Reduced for quick testing (50% of original)
else:
    n_cluster = 1000
x2d_1 = np.random.randn(n_cluster, 2) * 0.5  # First cluster at origin
x2d_2 = np.random.randn(n_cluster, 2) * 0.7 + np.array(
    [[2, 2]]
)  # Second cluster at (2,2)
x2d = np.vstack([x2d_1, x2d_2])

print(f"  - Generated {x2d.shape[0]} samples from 2D bimodal distribution")

# Create 2D bandwidth list (log-spaced grid)
hlist2d = get_hlist(
    np.array([20, 20]), np.array([[0.1, 2], [0.1, 2]])
)  # Default is logspace
print(f"  - Testing {hlist2d.shape[0]} bandwidth combinations")

# Compute 2D KDE with automatic bandwidth selection
opt2d = {
    "verbose": False,
    "N": np.array([51, 51]),  # Grid size (odd keeps origin centered)
    "xrange": np.array([[-2, 4], [-2, 4]]),  # Evaluation range
}
start_time = time.time()
kde2d = cv_fastkde(x2d, hlist2d, opt2d)
elapsed2d = time.time() - start_time

print(f"  - Computation time: {elapsed2d:.3f} seconds")
print(
    f"  - Selected bandwidth: h = [{kde2d.h[0]:.4f}, {kde2d.h[1]:.4f}]"
)

################################################################################
# Panel (c) and (f): 3D Density Estimation
################################################################################

print("\nGenerating Panel (c) and (f): 3D KDE...")

# Set random seed for reproducibility
np.random.seed(45)

# Generate 3D data with three clusters
if quick_test:
    n_cluster = 400  # Reduced for quick testing (50% of original)
else:
    n_cluster = 800
x3d_1 = np.random.randn(n_cluster, 3) * 0.3 + np.array([[-1, -1, -1]])  # Cluster 1
x3d_2 = np.random.randn(n_cluster, 3) * 0.4 + np.array([[1, 1, 0]])  # Cluster 2
x3d_3 = np.random.randn(n_cluster, 3) * 0.25 + np.array([[0, -1, 1]])  # Cluster 3
x3d = np.vstack([x3d_1, x3d_2, x3d_3])

print(f"  - Generated {x3d.shape[0]} samples from 3D trimodal distribution")

# Create 3D bandwidth list (log-spaced) - use anisotropic 10x10x10 grid
hlist3d = get_hlist(
    np.array([10, 10, 10]), np.array([[0.2, 0.6], [0.2, 0.6], [0.2, 0.6]])
)  # Default is logspace
print(f"  - Testing {hlist3d.shape[0]} bandwidth combinations")

# Compute 3D KDE with automatic bandwidth selection
opt3d = {
    "verbose": False,
    "N": np.array([31, 31, 31]),  # Grid size for evaluation
    "xrange": np.array([[-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5]]),  # Evaluation range
}
start_time = time.time()
kde3d = cv_fastkde(x3d, hlist3d, opt3d)
elapsed3d = time.time() - start_time

print(f"  - Computation time: {elapsed3d:.3f} seconds")
print(f"  - Selected bandwidth: h = {kde3d.h[0]:.4f}")

################################################################################
# Create Figure with 2x3 Layout (Row 1: KDE, Row 2: Bandwidth Selection)
################################################################################

print("\nCreating figure...")

# Create figure with publication-quality size (wider for 3 columns)
fig = plt.figure(figsize=(18, 9), facecolor="w")

# Set default font to Arial (sans-serif) for JSS style
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]

################################################################################
# ROW 1: KDE EXAMPLES
################################################################################

################################################################################
# Panel (a): 1D KDE with Multiple Bandwidths
################################################################################

ax1 = plt.subplot(2, 3, 1)

# Plot histogram (light gray, semi-transparent)
ax1.hist(
    x.flatten(),
    bins=30,
    density=True,
    color=[0.7, 0.7, 0.7],
    edgecolor="k",
    alpha=0.5,
    label="Histogram",
)

# Plot KDE curves for selected bandwidths
test_h_idx = [0, 4, 9, 14, 19]  # Show 5 different bandwidths (0-indexed in Python)
colors = plt.cm.tab10(np.linspace(0, 0.9, len(test_h_idx)))

for i, idx in enumerate(test_h_idx):
    opt_temp = {"verbose": False}
    kde_temp = cv_fastkde(x, hlist[idx : idx + 1], opt_temp)
    ax1.plot(
        kde_temp.grid[0],
        kde_temp.fhat,
        color=colors[i],
        linewidth=2,
        label=f"h={hlist[idx][0]:.3f}",
    )

# Plot selected bandwidth with thick red line
ax1.plot(
    kde.grid[0],
    kde.fhat,
    "r-",
    linewidth=3,
    label=f"h={kde.h[0]:.3f} (selected)",
)

ax1.set_xlabel("x", fontsize=14)
ax1.set_ylabel("Density", fontsize=14)
ax1.set_title("(a) 1D KDE with Bandwidth Comparison", fontsize=16, fontweight="bold")
ax1.legend(loc="upper right", fontsize=11)
ax1.grid(True)
ax1.tick_params(labelsize=12)

################################################################################
# Panel (b): 2D KDE (Contour)
################################################################################

ax2 = plt.subplot(2, 3, 2)

# Plot filled contours with jet colormap
contour = ax2.contourf(
    kde2d.grid[0], kde2d.grid[1], kde2d.fhat.T, levels=20, cmap="jet"
)
plt.colorbar(contour, ax=ax2)

# Add black contour lines on top (like MATLAB) for better delineation
ax2.contour(
    kde2d.grid[0],
    kde2d.grid[1],
    kde2d.fhat.T,
    levels=10,
    colors="k",
    linewidths=0.5,
    alpha=0.4,
)

# Overlay data points (small black dots)
ax2.plot(x2d[:, 0], x2d[:, 1], "k.", markersize=3)

ax2.set_xlabel("$x_1$", fontsize=14)
ax2.set_ylabel("$x_2$", fontsize=14)
ax2.set_title("(b) 2D KDE Density Contours", fontsize=16, fontweight="bold")
ax2.tick_params(labelsize=12)
ax2.set_aspect("equal", adjustable="box")

################################################################################
# Panel (c): 3D KDE (Volume Rendering)
################################################################################

ax3 = plt.subplot(2, 3, 3, projection="3d")

# Evaluate KDE on 3D grid
X3, Y3, Z3 = np.meshgrid(kde3d.grid[0], kde3d.grid[1], kde3d.grid[2], indexing="ij")
pdf_3d = kde3d.fhat.reshape(len(kde3d.grid[0]), len(kde3d.grid[1]), len(kde3d.grid[2]))

# Use volume rendering with transparency based on density
# Filter to show only significant density regions (lower threshold for more cloud-like appearance)
density_thresh = np.percentile(
    kde3d.fhat[kde3d.fhat > 0], 10
)  # Show more of the distribution
idx_sig = kde3d.fhat > density_thresh

# Create grid points for visualization
x_grid_3d = np.column_stack([X3.flatten(), Y3.flatten(), Z3.flatten()])
x_sig = x_grid_3d[idx_sig.flatten(), :]
c_data = kde3d.fhat[idx_sig]

# Normalize density for coloring and transparency
c_norm = (c_data - c_data.min()) / (c_data.max() - c_data.min() + 1e-10)

# Map density to RGBA colors with per-point transparency
# Use 'viridis' colormap (better than 'jet' for scientific visualization)
cmap = cm.get_cmap("viridis")
colors_rgb = cmap(c_norm)  # Shape: (N, 4) with RGBA

# Set alpha channel based on density (higher density = more opaque)
alpha_data = 0.02 + 0.5 * c_norm  # Map to [0.02, 0.52] for cloud-like appearance
colors_rgb[:, 3] = alpha_data  # Set alpha channel

# Plot volume as scatter3 with per-point RGBA colors
scatter = ax3.scatter(
    x_sig[:, 0],
    x_sig[:, 1],
    x_sig[:, 2],
    s=20,  # Smaller markers for smoother appearance
    c=colors_rgb,  # Use RGBA colors directly
    marker="o",
)

ax3.set_xlabel("$x_1$", fontsize=14)
ax3.set_ylabel("$x_2$", fontsize=14)
ax3.set_zlabel("$x_3$", fontsize=14)
ax3.set_title("(c) 3D KDE Volume Rendering", fontsize=16, fontweight="bold")
ax3.tick_params(labelsize=12)
ax3.view_init(elev=20, azim=45)
ax3.grid(True)
plt.colorbar(scatter, ax=ax3, shrink=0.5)
ax3.set_facecolor([0.95, 0.95, 0.95])

################################################################################
# ROW 2: BANDWIDTH SELECTION
################################################################################

################################################################################
# Panel (d): 1D Bandwidth Selection (LCV)
################################################################################

ax4 = plt.subplot(2, 3, 4)

if kde.lcv is not None:
    # Plot LCV scores
    ax4.plot(
        np.log10(kde.lcv["hlist"].flatten()),
        kde.lcv["lcv_m"],
        "k-o",
        linewidth=2,
        markersize=6,
        markerfacecolor="w",
        label="LCV Score",
    )

    # Mark selected bandwidth with red star (1-SE rule)
    ax4.plot(
        np.log10(kde.lcv["h1se"]),
        kde.lcv["lcv_m"][kde.lcv["id1se"]],
        "r*",
        markersize=20,
        linewidth=2,
        label="Selected (1-SE)",
    )

    ax4.set_xlabel("$\\log_{10}(h)$", fontsize=14)
    ax4.set_ylabel("LCV Score", fontsize=14)
    ax4.set_title("(d) 1D Bandwidth Selection via LCV", fontsize=16, fontweight="bold")
    ax4.legend(loc="lower left", fontsize=11)
    ax4.grid(True)
    ax4.tick_params(labelsize=12)
else:
    # Single bandwidth case
    ax4.text(
        0.5,
        0.5,
        "Single bandwidth (no selection needed)",
        ha="center",
        fontsize=14,
        transform=ax4.transAxes,
    )
    ax4.axis("off")
    ax4.set_title("(d) 1D Bandwidth Selection (LCV)", fontsize=16, fontweight="bold")

################################################################################
# Panel (e): 2D Bandwidth Selection (LCV Heatmap)
################################################################################

ax5 = plt.subplot(2, 3, 5)

if kde2d.lcv is not None:
    # Reshape LCV scores into 2D grid
    h1_unique = np.unique(kde2d.lcv["hlist"][:, 0])
    h2_unique = np.unique(kde2d.lcv["hlist"][:, 1])
    lcv_grid = kde2d.lcv["lcv_m"].reshape(len(h1_unique), len(h2_unique))

    # Plot heatmap with jet colormap
    im = ax5.imshow(
        lcv_grid.T,
        cmap="jet",
        origin="lower",
        extent=[
            np.log10(h1_unique.min()),
            np.log10(h1_unique.max()),
            np.log10(h2_unique.min()),
            np.log10(h2_unique.max()),
        ],
        aspect="auto",
    )
    plt.colorbar(im, ax=ax5)

    # Mark selected bandwidth with red star (1-SE rule)
    ax5.plot(
        np.log10(kde2d.lcv["h1se"][0]),
        np.log10(kde2d.lcv["h1se"][1]),
        "r*",
        markersize=20,
        linewidth=2,
    )

    # Add annotation
    ax5.text(
        np.log10(kde2d.lcv["h1se"][0]),
        np.log10(kde2d.lcv["h1se"][1]) + 0.1,
        "1-SE",
        color="w",
        fontsize=12,
        ha="center",
        fontweight="bold",
    )

    ax5.set_xlabel("$\\log_{10}(h_1)$", fontsize=14)
    ax5.set_ylabel("$\\log_{10}(h_2)$", fontsize=14)
    ax5.set_title("(e) 2D Bandwidth Selection via LCV", fontsize=16, fontweight="bold")
    ax5.tick_params(labelsize=12)
else:
    # Single bandwidth case
    ax5.text(
        0.5,
        0.5,
        "Single bandwidth (no selection needed)",
        ha="center",
        fontsize=14,
        transform=ax5.transAxes,
    )
    ax5.axis("off")
    ax5.set_title(
        "(e) 2D Bandwidth Selection (Heatmap)", fontsize=16, fontweight="bold"
    )

################################################################################
# Panel (f): 3D Bandwidth Selection (Volume Rendering)
################################################################################

ax6 = plt.subplot(2, 3, 6, projection="3d")

if kde3d.lcv is not None:
    # Get LCV scores and bandwidth list
    lcv_scores = kde3d.lcv["lcv_m"]
    h_list = kde3d.lcv["hlist"]

    # For 3D anisotropic bandwidth, reshape LCV scores to 3D grid
    # h_list is [1000×3] for 10×10×10 grid
    n_h = int(round(h_list.shape[0] ** (1 / 3)))  # Should be 10
    LCV_3d = lcv_scores.reshape(n_h, n_h, n_h)

    # Get unique bandwidth values for each dimension
    h1_values = np.unique(h_list[:, 0])
    h2_values = np.unique(h_list[:, 1])
    h3_values = np.unique(h_list[:, 2])

    # Convert to log10 scale for consistency with panels (d) and (e)
    log10_h1 = np.log10(h1_values)
    log10_h2 = np.log10(h2_values)
    log10_h3 = np.log10(h3_values)

    # Create 3D grid with log10 bandwidth values
    H1, H2, H3 = np.meshgrid(log10_h1, log10_h2, log10_h3, indexing="ij")

    # Flatten for scatter3
    h1_flat = H1.flatten()
    h2_flat = H2.flatten()
    h3_flat = H3.flatten()
    lcv_flat = LCV_3d.flatten()

    # Normalize LCV for coloring and transparency
    lcv_norm = (lcv_flat - lcv_flat.min()) / (lcv_flat.max() - lcv_flat.min() + 1e-10)

    # Map LCV to RGBA colors with per-point transparency
    cmap_lcv = cm.get_cmap("viridis")
    colors_lcv = cmap_lcv(lcv_norm)  # Shape: (N, 4) with RGBA

    # Set alpha channel based on LCV (higher LCV = more visible)
    alpha_lcv = 0.1 + 0.7 * lcv_norm  # Map to [0.1, 0.8]
    colors_lcv[:, 3] = alpha_lcv  # Set alpha channel

    # Plot volume as scatter3 with per-point RGBA colors and size variation
    marker_size = 15 + 40 * lcv_norm  # Larger markers for higher LCV
    scatter = ax6.scatter(
        h1_flat,
        h2_flat,
        h3_flat,
        s=marker_size,
        c=colors_lcv,  # Use RGBA colors directly
        marker="o",  # Circular markers (not hexagons)
    )

    # Mark selected bandwidth with large red star
    h_selected = kde3d.lcv["h1se"]
    ax6.plot(
        [np.log10(h_selected[0])],
        [np.log10(h_selected[1])],
        [np.log10(h_selected[2])],
        "r*",
        markersize=25,
        linewidth=4,
    )

    # Set axis labels (linear scale showing log10 values, consistent with panels d and e)
    ax6.set_xlabel("$\\log_{10}(h_1)$", fontsize=14)
    ax6.set_ylabel("$\\log_{10}(h_2)$", fontsize=14)
    ax6.set_zlabel("$\\log_{10}(h_3)$", fontsize=14)
    ax6.set_title("(f) 3D Bandwidth Selection via LCV", fontsize=16, fontweight="bold")
    ax6.tick_params(labelsize=12)
    ax6.view_init(elev=30, azim=45)
    ax6.grid(True)
    cb = plt.colorbar(scatter, ax=ax6, shrink=0.5)
    cb.set_label("LCV Score", fontsize=12)
    ax6.set_facecolor([0.95, 0.95, 0.95])
else:
    # Single bandwidth case
    ax6.text(
        0.5,
        0.5,
        "Single bandwidth (no selection needed)",
        ha="center",
        fontsize=14,
        transform=ax6.transAxes,
    )
    ax6.axis("off")
    ax6.set_title("(f) 3D Bandwidth Selection", fontsize=16, fontweight="bold")

################################################################################
# Add Main Title
################################################################################

fig.suptitle(
    "Kernel Density Estimation with Automatic Bandwidth Selection (1-SE Rule)",
    fontsize=18,
    fontweight="bold",
)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.96])

################################################################################
# Save Figure for Publication
################################################################################

print("\nSaving figure...")

# Create output directory if it doesn't exist
fig_dir = os.path.join(os.path.dirname(__file__), "..", "fig", "reproduced")
os.makedirs(fig_dir, exist_ok=True)

# Save as PNG (300 DPI for publication)
png_path = os.path.join(fig_dir, "fig2_fastkde_python.png")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
print(f"  - Saved PNG: {png_path}")

# Save as PDF for vector graphics
pdf_path = os.path.join(fig_dir, "fig2_fastkde_python.pdf")
plt.savefig(pdf_path, bbox_inches="tight")
print(f"  - Saved PDF: {pdf_path}")

################################################################################
# Summary
################################################################################

print()
print("=" * 80)
print("Figure 2 Generation Complete!")
print("=" * 80)
print()

print("Summary:")
print(f"  - 1D KDE: {len(x)} samples, selected h = {kde.h[0]:.4f}")
print(
    f"  - 2D KDE: {x2d.shape[0]} samples, selected h = [{kde2d.h[0]:.4f}, {kde2d.h[1]:.4f}]"
)
print(f"  - 3D KDE: {x3d.shape[0]} samples, selected h = {kde3d.h[0]:.4f}")
print(f"  - Figure saved to: {fig_dir}")
print()

# Optionally show the figure (set FASTLPR_SHOW_PLOT=1 to display)
if os.environ.get("FASTLPR_SHOW_PLOT", "0") == "1":
    plt.show()
else:
    plt.close(fig)  # Close figure to free memory
