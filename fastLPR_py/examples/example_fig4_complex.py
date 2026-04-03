"""
Code to generate Figure 4: Complex-Valued Regression (log(z) function)

This script reproduces Figure 4 from the fastLPR paper, demonstrating:
  - Panel (a): Real part of log(z)
  - Panel (b): Imaginary part of log(z)
  - Panel (c): Magnitude |log(z)|
  - Panel (d): Angle of log(z)

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
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import sys

# Add parent directory to path to import fastlpr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastlpr import cv_fastlpr, fastlpr_predict, get_hlist

print()
print("=" * 80)
print("Figure 4: Complex-Valued Regression (log(z) function)")
print("=" * 80)
print()

################################################################################
# Generate Complex-Valued Data
################################################################################

print("Generating complex-valued data...")

# Set random seed for reproducibility
np.random.seed(42)

# Grid resolution
n = 100

# Create uniform grid in complex plane
# Real part: [0.1, 2], Imaginary part: [-2, 2]
x1 = np.linspace(0.1, 2, n)
x2 = np.linspace(-2, 2, n)
X1, X2 = np.meshgrid(x1, x2, indexing="ij")

# Convert to complex numbers: z = x1 + i*x2
# For fastLPR, we need to pass x as 2D array [real, imag]
x_complex = X1.ravel() + 1j * X2.ravel()
x = np.column_stack([X1.ravel(), X2.ravel()])

# Compute complex logarithm: log(z)
y = np.log(x_complex)

# Add complex-valued noise
yr = (
    y
    + 0.1 * np.std(y.real) * np.random.randn(len(y))
    + 0.1 * 1j * np.std(y.imag) * np.random.randn(len(y))
)

print(f"  - Generated {len(x)} samples")
print(f"  - Real(z) range: [{x_complex.real.min():.2f}, {x_complex.real.max():.2f}]")
print(f"  - Imag(z) range: [{x_complex.imag.min():.2f}, {x_complex.imag.max():.2f}]")

################################################################################
# Perform Complex-Valued Regression
################################################################################

print("\nPerforming complex-valued regression...")

# Bandwidth selection
# For 2D: dh=[20, 20], hrange=[[0.1, 1.0], [0.1, 1.0]]
hlist = get_hlist([5, 5], [[0.1, 1.0], [0.1, 1.0]])
opt = {
    "order": 1,  # Local linear regression
    "dstd": 3,  # 1-SE rule multiplier = 3 (prefer smoother fits)
    "verbose": False,
}

start_time = time.time()
regs = cv_fastlpr(x, yr, hlist, opt)
elapsed_regression = time.time() - start_time

print(f"  - Regression computation time: {elapsed_regression:.3f} seconds")
h_selected = regs.h
print(f"  - Selected bandwidth: h = [{h_selected[0]:.4f}, {h_selected[1]:.4f}]")

# Get fitted surface from regs (for surface plots)
print("\nPreparing fitted surface for visualization...")
start_predict = time.time()
# regs.yhat is 1D flattened grid values, reshape to 2D
# regs.grid is tuple of 1D coordinate arrays, convert to meshgrids
grid_shape = tuple(len(g) for g in regs.grid)
X1_grid, X2_grid = np.meshgrid(regs.grid[0], regs.grid[1], indexing="ij")
yhat_grid = regs.yhat.reshape(grid_shape)
elapsed_predict = time.time() - start_predict
print(f"  - Surface preparation time: {elapsed_predict:.3f} seconds")
print(f"  - Grid shape: {grid_shape}")

################################################################################
# Create Figure with 2x2 Layout
################################################################################

print("\nCreating figure...")
start_plot = time.time()

# Create figure with publication-quality size
fig = plt.figure(figsize=(14, 10), facecolor="w")

# Color for scatter plots (dark red from jet colormap)
jet_cmap = plt.cm.jet
cl = jet_cmap(0.9)

# View angle for all subplots (elevation=30, azimuth=-60)
view_elev = 30
view_azim = -60

################################################################################
# Panel (a): Real Part of log(z)
################################################################################

# Use compact layout with balanced spacing
ax1 = fig.add_subplot(2, 2, 1, projection="3d")
ax1.set_position([0.07, 0.56, 0.40, 0.38])

# Plot fitted surface FIRST (using grid-based fitted values)
yhat_real_grid = yhat_grid.real
surf1 = ax1.plot_surface(
    X1_grid,
    X2_grid,
    yhat_real_grid,
    cmap="jet",
    edgecolor="none",
    alpha=0.9,
    shade=True,
)

# Plot noisy data as scatter (plot all points to match MATLAB)
ax1.scatter(
    x_complex.real,
    x_complex.imag,
    yr.real,
    s=6,
    c=[cl],
    marker=".",
    alpha=0.3,
)

# Add legend manually
from matplotlib.lines import Line2D

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=cl,
        markersize=8,
        alpha=0.6,
        label="Noisy data",
    ),
    Line2D([0], [0], color="blue", lw=4, label="Fitted surface"),
]
ax1.legend(handles=legend_elements, loc="upper right", fontsize=10)

# Labels and styling
ax1.set_xlabel("Real(z)", fontsize=14)
ax1.set_ylabel("Imag(z)", fontsize=14)
ax1.set_zlabel("Real(log(z))", fontsize=14)
ax1.set_title("(a) Real(log(z))", fontsize=16, fontweight="bold")
ax1.view_init(elev=view_elev, azim=view_azim)
ax1.set_zlim([-4, 4])
ax1.set_zticks([-4, 0, 4])
ax1.tick_params(labelsize=12)

################################################################################
# Panel (b): Imaginary Part of log(z)
################################################################################

# Use compact layout with balanced spacing
ax2 = fig.add_subplot(2, 2, 2, projection="3d")
ax2.set_position([0.55, 0.56, 0.40, 0.38])

# Plot fitted surface FIRST (using grid-based fitted values)
yhat_imag_grid = yhat_grid.imag
surf2 = ax2.plot_surface(
    X1_grid,
    X2_grid,
    yhat_imag_grid,
    cmap="jet",
    edgecolor="none",
    alpha=0.9,
    shade=True,
)

# Plot noisy data as scatter (plot all points to match MATLAB)
ax2.scatter(
    x_complex.real,
    x_complex.imag,
    yr.imag,
    s=6,
    c=[cl],
    marker=".",
    alpha=0.3,
)

# Add legend
ax2.legend(handles=legend_elements, loc="upper right", fontsize=10)

# Labels and styling
ax2.set_xlabel("Real(z)", fontsize=14)
ax2.set_ylabel("Imag(z)", fontsize=14)
ax2.set_zlabel("Imag(log(z))", fontsize=14)
ax2.set_title("(b) Imag(log(z))", fontsize=16, fontweight="bold")
ax2.view_init(elev=view_elev, azim=view_azim)
ax2.set_zlim([-4, 4])
ax2.set_zticks([-4, 0, 4])
ax2.tick_params(labelsize=12)

################################################################################
# Panel (c): Magnitude |log(z)|
################################################################################

# Use compact layout with balanced spacing
ax3 = fig.add_subplot(2, 2, 3, projection="3d")
ax3.set_position([0.07, 0.10, 0.40, 0.38])

# Plot fitted surface FIRST (using grid-based fitted values)
yhat_abs_grid = np.abs(yhat_grid)
surf3 = ax3.plot_surface(
    X1_grid, X2_grid, yhat_abs_grid, cmap="jet", edgecolor="none", alpha=0.7, shade=True
)

# Plot noisy data as scatter (plot all points to match MATLAB)
ax3.scatter(
    x_complex.real,
    x_complex.imag,
    np.abs(yr),
    s=8,
    c=[cl],
    marker=".",
    alpha=0.4,
)

# Add legend
ax3.legend(handles=legend_elements, loc="upper right", fontsize=10)

# Labels and styling
ax3.set_xlabel("Real(z)", fontsize=14)
ax3.set_ylabel("Imag(z)", fontsize=14)
ax3.set_zlabel("|log(z)|", fontsize=14)
ax3.set_title("(c) |log(z)|", fontsize=16, fontweight="bold")
ax3.view_init(elev=view_elev, azim=view_azim)
ax3.set_zlim([0, 4])
ax3.set_zticks([0, 2, 4])
ax3.tick_params(labelsize=12)

################################################################################
# Panel (d): Angle of log(z)
################################################################################

# Use compact layout with balanced spacing
ax4 = fig.add_subplot(2, 2, 4, projection="3d")
ax4.set_position([0.55, 0.10, 0.40, 0.38])

# Plot fitted surface FIRST (using grid-based fitted values)
yhat_angle_grid = np.angle(yhat_grid)
surf4 = ax4.plot_surface(
    X1_grid,
    X2_grid,
    yhat_angle_grid,
    cmap="jet",
    edgecolor="none",
    alpha=0.7,
    shade=True,
)

# Plot noisy data as scatter (plot all points to match MATLAB)
ax4.scatter(
    x_complex.real,
    x_complex.imag,
    np.angle(yr),
    s=8,
    c=[cl],
    marker=".",
    alpha=0.4,
)

# Add legend
ax4.legend(handles=legend_elements, loc="upper right", fontsize=10)

# Labels and styling
ax4.set_xlabel("Real(z)", fontsize=14)
ax4.set_ylabel("Imag(z)", fontsize=14)
ax4.set_zlabel("angle(log(z))", fontsize=14)
ax4.set_title("(d) angle(log(z))", fontsize=16, fontweight="bold")
ax4.view_init(elev=view_elev, azim=view_azim)
ax4.set_zlim([-np.pi, np.pi])
ax4.set_zticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
ax4.set_zticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
ax4.tick_params(labelsize=12)

elapsed_plot = time.time() - start_plot
print(f"  - Plotting time: {elapsed_plot:.3f} seconds")

################################################################################
# Save Figure for Publication
################################################################################

print("\nSaving figure...")
start_save = time.time()

# Create output directory if it doesn't exist
fig_dir = os.path.join(os.path.dirname(__file__), "..", "fig", "reproduced")
os.makedirs(fig_dir, exist_ok=True)

# Save as PNG (300 DPI for publication)
png_path = os.path.join(fig_dir, "fig4_complex_python.png")
fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="w")
print(f"  - Saved PNG: {os.path.abspath(png_path)}")

# Save as PDF (vector graphics)
pdf_path = os.path.join(fig_dir, "fig4_complex_python.pdf")
fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="w")
print(f"  - Saved PDF: {os.path.abspath(pdf_path)}")

elapsed_save = time.time() - start_save
print(f"  - Saving time: {elapsed_save:.3f} seconds")

################################################################################
# Summary
################################################################################

print()
print("=" * 80)
print("Figure 4 Generation Complete!")
print("=" * 80)
print()

print("Timing Summary:")
print(
    f"  - Regression computation: {elapsed_regression:.3f} seconds ({elapsed_regression/60:.1f} minutes)"
)
print(f"  - Prediction: {elapsed_predict:.3f} seconds")
print(f"  - Plotting: {elapsed_plot:.3f} seconds")
print(f"  - Saving: {elapsed_save:.3f} seconds")
total_time = elapsed_regression + elapsed_predict + elapsed_plot + elapsed_save
print(f"  - TOTAL: {total_time:.3f} seconds ({total_time/60:.1f} minutes)")
print()

print("Summary:")
print(f"  - Complex data: {len(x)} samples")
print("  - Function: log(z) where z = x1 + i*x2")
print(f"  - Selected bandwidth: h = [{h_selected[0]:.4f}, {h_selected[1]:.4f}]")
print("  - Four components visualized:")
print("    * Panel (a): Real part")
print("    * Panel (b): Imaginary part")
print("    * Panel (c): Magnitude")
print("    * Panel (d): Angle")
print(f"  - Figure saved to: {os.path.abspath(fig_dir)}")
print()

print("Key observations:")
print("  - Real part shows log(|z|) behavior")
print("  - Imaginary part shows arg(z) with branch cut")
print("  - Magnitude shows |log(z)| (always non-negative)")
print("  - Angle shows phase of log(z)")
print()

# Optionally show the figure (set FASTLPR_SHOW_PLOT=1 to display)
if os.environ.get("FASTLPR_SHOW_PLOT", "0") == "1":
    plt.show()
else:
    plt.close(fig)  # Close figure to free memory
