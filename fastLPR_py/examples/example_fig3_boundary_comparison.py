"""
Code to generate Figure 3: Boundary Comparison (NW vs LL vs LQ Regression)

This script reproduces Figure 3 from the fastLPR paper, demonstrating:
  - Comparison of three local polynomial regression methods:
    * NW (Nadaraya-Watson, order 0 - local constant)
    * LL (Local Linear, order 1)
    * LQ (Local Quadratic, order 2)
  - Oscillating test function with noise
  - Comparison of boundary behavior

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
from scipy.special import j1  # Bessel function of the first kind
import time
import os
import sys

# Add parent directory to path to import fastlpr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastlpr import cv_fastlpr, get_hlist

print()
print("=" * 80)
print("Figure 3: Boundary Comparison (NW vs LL vs LQ Regression)")
print("=" * 80)
print()

################################################################################
# Generate Test Data (same as MATLAB version)
################################################################################

print("Generating test data...")

# Set random seed for reproducibility (same as MATLAB)
np.random.seed(0)

# Sample size (same as MATLAB)
n = 500

# Generate scattered data in [0, 20] (same as MATLAB)
x = np.random.rand(n, 1) * 20

# True function: Bessel function of the first kind (same as MATLAB)
y_true = j1(x.ravel())

# Add Gaussian noise (same as MATLAB)
y = y_true + 0.4 * np.std(y_true) * np.random.randn(len(y_true))

print(f"  - Generated {n} samples")
print(f"  - x range: [{x.min():.1f}, {x.max():.1f}]")
print(f"  - Function: Bessel J1(x)")

################################################################################
# Fit Three Regression Models
################################################################################

print("\nFitting regression models...")

# Create evaluation grid
x_grid = np.linspace(0, 20, 500).reshape(-1, 1)

# Use automatic bandwidth selection with appropriate range
# Generate bandwidth candidates (focus on larger range for smoother fits)
hlist = get_hlist(50, [0.01, 2.0])
print("  - Using automatic bandwidth selection (GCV)")
print(
    f"  - Bandwidth range: [{hlist.min():.3f}, {hlist.max():.3f}] in normalized scale"
)

# Options for regression
opt = {"N": 500, "verbose": False, "dstd": 1}  # Grid size  # Use standardization (required for NUFFT)

# Order 0: Nadaraya-Watson (local constant)
print("  - Fitting NW (order 0) with auto bandwidth...")
opt["order"] = 0
start_time = time.time()
reg_nw = cv_fastlpr(x, y, hlist, opt)
t_nw = time.time() - start_time
y_nw = reg_nw.fpp_yhat(x_grid)
h_nw = reg_nw.h[0]

# Order 1: Local Linear
print("  - Fitting LL (order 1) with auto bandwidth...")
opt["order"] = 1
start_time = time.time()
reg_ll = cv_fastlpr(x, y, hlist, opt)
t_ll = time.time() - start_time
y_ll = reg_ll.fpp_yhat(x_grid)
h_ll = reg_ll.h[0]

# Order 2: Local Quadratic
print("  - Fitting LQ (order 2) with auto bandwidth...")
opt["order"] = 2
start_time = time.time()
reg_lq = cv_fastlpr(x, y, hlist, opt)
t_lq = time.time() - start_time
y_lq = reg_lq.fpp_yhat(x_grid)
h_lq = reg_lq.h[0]

print(f"  - Computation times: NW={t_nw:.3f}s, LL={t_ll:.3f}s, LQ={t_lq:.3f}s")
print(f"  - Selected bandwidths: NW={h_nw:.3f}, LL={h_ll:.3f}, LQ={h_lq:.3f}")

################################################################################
# Create Main Figure
################################################################################

print("\nCreating figure...")

# Create figure with publication-quality size
fig = plt.figure(figsize=(12, 7), facecolor="w")

# Main plot
ax_main = plt.axes([0.08, 0.12, 0.88, 0.82])
ax_main.set_box_aspect(None)

# Plot scattered data points (larger black dots) - now visible in legend
ax_main.plot(x, y, "k.", markersize=6, label="Noisy data", zorder=1)

# Plot regression curves with distinct colors and thicker lines
ax_main.plot(
    x_grid, y_nw, color=[0, 0.7, 0], linewidth=3, label="NW regression", zorder=2
)  # Green
ax_main.plot(x_grid, y_ll, "r-", linewidth=3, label="LL regression", zorder=3)  # Red
ax_main.plot(x_grid, y_lq, "b-", linewidth=3, label="LQ regression", zorder=4)  # Blue

# Set axis limits (same as MATLAB)
ax_main.set_xlim([0, 20])
ax_main.set_ylim([-0.6, 1.0])
ax_main.set_yticks(np.arange(-0.6, 1.1, 0.4))
ax_main.tick_params(labelsize=14)
ax_main.grid(False)

# Add legend at top
ax_main.legend(
    loc="upper center",
    fontsize=14,
    frameon=True,
    facecolor="w",
    edgecolor="k",
    ncol=4,
)

################################################################################
# Save Figure for Publication
################################################################################

print("\nSaving figure...")

# Create output directory if it doesn't exist
fig_dir = os.path.join(os.path.dirname(__file__), "..", "fig", "reproduced")
os.makedirs(fig_dir, exist_ok=True)

# Save as PNG (300 DPI for publication)
png_path = os.path.join(fig_dir, "fig3_boundary_comparison_python.png")
fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="w")
print(f"  - Saved PNG: {os.path.abspath(png_path)}")

# Save as PDF (vector graphics)
pdf_path = os.path.join(fig_dir, "fig3_boundary_comparison_python.pdf")
fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="w")
print(f"  - Saved PDF: {os.path.abspath(pdf_path)}")

################################################################################
# Summary
################################################################################

print()
print("=" * 80)
print("Figure 3 Generation Complete!")
print("=" * 80)
print()

print("Summary:")
print(f"  - Data: {n} samples with oscillating function")
print(f"  - Bandwidths (auto-selected): NW={h_nw:.3f}, LL={h_ll:.3f}, LQ={h_lq:.3f}")
print("  - Methods compared: NW (order 0), LL (order 1), LQ (order 2)")
print(f"  - Figure saved to: {os.path.abspath(fig_dir)}")
print()

print("Key observations:")
print("  - NW (green): Smoother but higher bias at boundaries")
print("  - LL (red): Reduces boundary bias compared to NW")
print("  - LQ (blue): Best fit in high curvature regions")
print()

# Optionally show the figure (set FASTLPR_SHOW_PLOT=1 to display)
if os.environ.get("FASTLPR_SHOW_PLOT", "0") == "1":
    plt.show()
else:
    plt.close(fig)  # Close figure to free memory
