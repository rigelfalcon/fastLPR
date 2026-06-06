"""
Code to generate Figure 5: Heteroscedastic Regression (1D and 2D)

This script reproduces Figure 5 from the fastLPR paper, demonstrating:
  - Panel (a): 1D mean estimation with intervals (CI and PI)
  - Panel (b): 1D variance estimation (log-scale)
  - Panel (c): 2D mean estimation with intervals (CI and PI)
  - Panel (d): 2D variance estimation (log-scale)

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

from fastlpr import cv_fastlpr, get_hlist
from fastlpr.confidence import fastlpr_interval, fastlpr_plot_interval

print()
print("=" * 80)
print("Figure 5: Heteroscedastic Regression (1D and 2D)")
print("=" * 80)
print()

################################################################################
# Panel (a) and (b): 1D Heteroscedastic Regression
################################################################################

print("Generating 1D heteroscedastic data...")

# Set random seed for reproducibility
np.random.seed(42)

# Generate 1D data
n1d = 10000
x1d = 2 * 2 * (np.random.rand(n1d, 1) - 0.5)  # Uniform in [-2, 2]

# Define ground truth functions
fun_mu_1d = lambda x: x**3  # Cubic mean function
fun_sigma_1d = lambda x: 1 + 4 * np.exp(-(x**2))  # Variance depends on x

# Generate data
y1d = fun_mu_1d(x1d).ravel()
s1d = fun_sigma_1d(x1d).ravel() * 0.5 * np.var(y1d)
yr1d = y1d + np.random.randn(len(y1d)) * np.sqrt(s1d)

print(f"  - Generated {n1d} 1D samples")

# Estimate conditional mean
print("  - Estimating 1D mean...")
opt1d = {"order": 1, "num_dof_samples": 10, "verbose": False}
hlist1d = get_hlist(20, [0.01, 1.0])
start_time = time.time()
regs_mu_1d = cv_fastlpr(x1d, yr1d, hlist1d, opt1d)
time_mu_1d = time.time() - start_time
yp1d = regs_mu_1d.yhat
print(f"    Computation time: {time_mu_1d:.3f} seconds")

# Estimate conditional variance
print("  - Estimating 1D variance...")
sr1d = (yr1d - yp1d) ** 2
opt1d["y_type_out"] = "variance"
opt1d["num_dof_samples"] = 10
start_time = time.time()
regs_sigma_1d = cv_fastlpr(x1d, sr1d, hlist1d, opt1d)
time_sigma_1d = time.time() - start_time
print(f"    Computation time: {time_sigma_1d:.3f} seconds")

# Compute intervals (CI and PI)
ci1d = fastlpr_interval(regs_mu_1d, regs_sigma_1d, 0.05, type="confidence")
pi1d = fastlpr_interval(regs_mu_1d, regs_sigma_1d, 0.05, type="prediction")

################################################################################
# Panel (c) and (d): 2D Heteroscedastic Regression
################################################################################

print("\nGenerating 2D heteroscedastic data...")

# Set random seed
np.random.seed(44)

# Generate 2D data
n2d = 1200
x2d = 2 * 2 * (np.random.rand(n2d, 2) - 0.5)  # Uniform in [-2, 2] x [-2, 2]

# Define ground truth functions
fun_mu_2d = lambda x1, x2: x1**3 + x2**3
fun_sigma_2d = lambda x1, x2: 1 + 4 * np.exp(-(x1**2 + x2**2))

# Generate data
y2d = fun_mu_2d(x2d[:, 0], x2d[:, 1])
s2d = fun_sigma_2d(x2d[:, 0], x2d[:, 1]) * 1 * np.std(y2d)
yr2d = y2d + np.sqrt(s2d) * np.random.randn(len(y2d))

print(f"  - Generated {n2d} 2D samples")

# Estimate conditional mean
print("  - Estimating 2D mean...")
opt2d = {"order": 1, "dstd": 1, "verbose": False}
hlist2d = get_hlist([10, 10], [[0.01, 1.0], [0.01, 1.0]])
start_time = time.time()
regs_mu_2d = cv_fastlpr(x2d, yr2d, hlist2d, opt2d)
time_mu_2d = time.time() - start_time
yp2d = regs_mu_2d.yhat
print(f"    Computation time: {time_mu_2d:.3f} seconds")

# Estimate conditional variance
print("  - Estimating 2D variance...")
sr2d = (yr2d - yp2d) ** 2
opt2d["y_type_out"] = "variance"
opt2d["dstd"] = 1
start_time = time.time()
regs_sigma_2d = cv_fastlpr(x2d, sr2d, hlist2d, opt2d)
time_sigma_2d = time.time() - start_time
print(f"    Computation time: {time_sigma_2d:.3f} seconds")

# Compute intervals (CI and PI)
ci2d = fastlpr_interval(regs_mu_2d, regs_sigma_2d, 0.05, type="confidence")
pi2d = fastlpr_interval(regs_mu_2d, regs_sigma_2d, 0.05, type="prediction")

################################################################################
# Create Figure with 2x2 Layout
################################################################################

print("\nCreating figure...")

# Create figure with publication-quality size
fig = plt.figure(figsize=(14, 10), facecolor="w")

################################################################################
# Panel a): 1D Mean with Confidence Intervals
################################################################################

ax1 = fig.add_subplot(2, 2, 1)
ax1.set_position([0.08, 0.55, 0.40, 0.38])

# Plot data (DARK blue for y+ε, green circles for y)
ax1.plot(x1d, yr1d, ".", color=[0, 0, 0.8], markersize=4, label=r"$y + \epsilon$")
ax1.plot(
    x1d,
    y1d,
    "o",
    color=[0, 0.7, 0],
    markersize=4,
    markeredgewidth=0.8,
    fillstyle="none",
    label=r"$y$",
)

# Plot PI first (light blue, wider)
fastlpr_plot_interval(pi1d, grid=regs_mu_1d.grid, ax=ax1, color=[0.5, 0.7, 1], alpha=0.15, label="PI")
# Plot CI second (green, narrower)
fastlpr_plot_interval(ci1d, grid=regs_mu_1d.grid, ax=ax1, color="g", alpha=0.2, label="CI")

# Plot estimated mean (yellow)
x_grid = regs_mu_1d.grid[0]
y_mean = regs_mu_1d.fpp_yhat(x_grid[:, None])
ax1.plot(x_grid, y_mean, linewidth=2.5, color=[1, 0.8, 0], label=r"$\hat{m}$")

# Plot variance distribution (magenta, shifted down to floor)
# Bell opens UPWARD (high variance = peak) - intuitive visualization
x_eval = np.linspace(-2, 2, 200)
sigma_dist = fun_sigma_1d(x_eval) * 0.5 * np.var(y1d)
y_offset = -28
pdf_scale = 10
ax1.plot(
    x_eval,
    y_offset + pdf_scale * sigma_dist / np.max(sigma_dist),
    "m-",
    linewidth=2,
    label=r"$\sigma^2$",
)

# Labels and styling
ax1.set_xlim([-2.2, 2.2])
ax1.set_ylim([-30, 20])
ax1.set_xticks(np.arange(-2, 3, 1))
ax1.set_yticks(np.arange(-30, 21, 10))
ax1.tick_params(labelsize=12)
ax1.set_title("(a) 1D Mean with CI and PI", fontsize=13, fontweight="normal")
ax1.legend(loc="lower right", fontsize=11)
ax1.grid(True, alpha=0.3)

################################################################################
# Panel b): 1D Variance Estimation (Log-scale)
################################################################################

ax2 = fig.add_subplot(2, 2, 2)
ax2.set_position([0.56, 0.55, 0.40, 0.38])

# Compute log-transformed values
log_sigma_true = 2 * np.log(np.sqrt(s1d))
# Add minimum threshold to avoid log(0) causing -inf values at boundaries
residuals_abs = np.abs(yr1d - yp1d)
residuals_abs_clipped = np.maximum(residuals_abs, 1e-10)  # Minimum threshold
log_residuals = 2 * np.log(residuals_abs_clipped)

# Plot scatter points
ax2.plot(
    x1d,
    log_sigma_true,
    "s",
    color="m",
    markersize=3,
    markeredgewidth=0.5,
    # Removed fillstyle="none" to match MATLAB (filled squares are more visible than hollow)
    label=r"$2\log(\sigma)$",
)
ax2.plot(
    x1d,
    log_residuals,
    ".",
    color=[0, 0.7, 0],
    markersize=4,
    label=r"$2\log(|y-\hat{y}|)$",
)

# Plot estimated log-variance (grey line)
# FIX: Use MATLAB range [-2.2, 2.2] with linear extrapolation (matching MATLAB behavior)
# MATLAB uses griddedInterpolant with Method='spline', ExtrapolationMethod='linear'
# Python scipy cubic extrapolation causes wild oscillations outside data bounds
x_eval = np.linspace(-2.2, 2.2, 200)[:, None]

# Apply linear extrapolation for points outside data range [-2.0, 2.0]
# Data was generated with x in [-2, 2], so extrapolation is needed at boundaries
sp1d_eval = np.zeros(len(x_eval))
data_min, data_max = -2.0, 2.0

# Separate inside and outside points
inside_mask = (x_eval.ravel() >= data_min) & (x_eval.ravel() <= data_max)

# Interpolate inside points normally
if np.any(inside_mask):
    sp1d_eval[inside_mask] = regs_sigma_1d.fpp_yhat(x_eval[inside_mask])

# Linear extrapolation for outside points (matching MATLAB ExtrapolationMethod='linear')
# Get boundary values and gradients
left_boundary_pts = np.array([[data_min], [data_min + 0.05]])
right_boundary_pts = np.array([[data_max - 0.05], [data_max]])
left_vals = regs_sigma_1d.fpp_yhat(left_boundary_pts)
right_vals = regs_sigma_1d.fpp_yhat(right_boundary_pts)

# Left extrapolation
left_gradient = (left_vals[1] - left_vals[0]) / 0.05
left_mask = x_eval.ravel() < data_min
sp1d_eval[left_mask] = left_vals[0] + left_gradient * (
    x_eval.ravel()[left_mask] - data_min
)

# Right extrapolation
right_gradient = (right_vals[1] - right_vals[0]) / 0.05
right_mask = x_eval.ravel() > data_max
sp1d_eval[right_mask] = right_vals[1] + right_gradient * (
    x_eval.ravel()[right_mask] - data_max
)

# Clip variance estimates to avoid negative values or NaN in log
sp1d_eval_clipped = np.maximum(sp1d_eval, 1e-10)
log_sigma_est = 2 * np.log(np.sqrt(sp1d_eval_clipped))
ax2.plot(
    x_eval,
    log_sigma_est,
    "-",
    color=[0.5, 0.5, 0.5],
    linewidth=2.5,
    label=r"$2\log(\sqrt{\hat{\Sigma}})$",
)

# Labels and styling
ax2.set_xlim([-2.2, 2.2])
ax2.set_ylim([-20, 10])
ax2.set_xticks(np.arange(-2, 3, 1))
ax2.set_yticks(np.arange(-20, 11, 10))
ax2.tick_params(labelsize=12)
ax2.set_title(
    "(b) 1D Variance Estimation (Log-scale)", fontsize=13, fontweight="normal"
)
ax2.legend(loc="lower left", fontsize=11)
ax2.grid(True, alpha=0.3)

################################################################################
# Panel c): 2D Mean with Confidence Intervals
################################################################################

ax3 = fig.add_subplot(2, 2, 3, projection="3d")
ax3.set_position([0.05, 0.05, 0.43, 0.42])

# Plot data (BLUE dots for y+ε, red circles for y)
ax3.scatter(x2d[:, 0], x2d[:, 1], yr2d, s=15, c="b", alpha=0.3, label=r"$y+\epsilon$")
ax3.scatter(
    x2d[:, 0],
    x2d[:, 1],
    y2d,
    s=20,
    c="r",
    marker="o",
    alpha=0.5,
    edgecolors="r",
    linewidths=0.8,
    label=r"$y$",
)

# Plot PI surfaces first (light blue)
fastlpr_plot_interval(pi2d, grid=regs_mu_2d.grid, ax=ax3, color=[0.5, 0.7, 1], alpha=0.15, label="PI")
# Plot CI surfaces second (green)
fastlpr_plot_interval(ci2d, grid=regs_mu_2d.grid, ax=ax3, color="g", alpha=0.4, label="CI")

# Plot estimated mean surface with black grid lines
G1, G2 = np.meshgrid(regs_mu_2d.grid[0], regs_mu_2d.grid[1], indexing="ij")
grid_points = np.column_stack([G1.ravel(), G2.ravel()])
y_mean_2d = regs_mu_2d.fpp_yhat(grid_points).reshape(G1.shape)
surf = ax3.plot_surface(
    G1,
    G2,
    y_mean_2d,
    cmap="jet",
    edgecolor="k",
    linewidth=0.1,
    alpha=0.9,
    label=r"$\hat{m}$",
)

# Shift variance for visualization (on the floor)
s2d_low = s2d - np.min(s2d) - 30
ax3.scatter(
    x2d[:, 0],
    x2d[:, 1],
    s2d_low,
    s=15,
    c="m",
    marker="s",
    linewidths=0.5,
    label=r"$\sigma^2$",
)

# Labels and styling
ax3.view_init(elev=5, azim=-27)
ax3.set_xlim([-2, 2])
ax3.set_ylim([-2, 2])
ax3.set_xticks(np.arange(-2, 3, 2))
ax3.set_yticks(np.arange(-2, 3, 2))
ax3.tick_params(labelsize=12)
ax3.set_title("(c) 2D Mean with CI and PI", fontsize=13, fontweight="normal")
# Skip legend due to matplotlib 3.7+ bug with Poly3DCollection
# ax3.legend(loc="lower left", fontsize=10)

# Add colorbar with padding to avoid overlap
fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=5, pad=0.1)

################################################################################
# Panel d): 2D Variance Estimation (Log-scale)
################################################################################

ax4 = fig.add_subplot(2, 2, 4, projection="3d")
ax4.set_position([0.52, 0.05, 0.43, 0.42])

# Compute log-transformed values
log_sigma_true_2d = 2 * np.log(np.sqrt(s2d))
# Add minimum threshold to avoid log(0) causing -inf values
residuals_abs_2d = np.abs(yr2d - yp2d)
residuals_abs_clipped_2d = np.maximum(residuals_abs_2d, 1e-10)  # Minimum threshold
log_residuals_2d = 2 * np.log(residuals_abs_clipped_2d)

# Plot scatter points
ax4.scatter(
    x2d[:, 0],
    x2d[:, 1],
    log_sigma_true_2d,
    s=15,
    c="m",
    marker="s",
    linewidths=0.5,
    label=r"$2\log(\sigma)$",
)
ax4.scatter(
    x2d[:, 0],
    x2d[:, 1],
    log_residuals_2d,
    s=10,
    c=[0, 0.7, 0],
    marker=".",
    label=r"$2\log(|y-\hat{y}|)$",
)

# Plot estimated log-variance surface with BLACK grid lines
sp2d_eval = regs_sigma_2d.fpp_yhat(grid_points).reshape(G1.shape)
# Clip variance estimates to avoid negative values or NaN in log
sp2d_eval_clipped = np.maximum(sp2d_eval, 1e-10)
log_sigma_est_2d = 2 * np.log(np.sqrt(sp2d_eval_clipped))
ax4.plot_surface(
    G1,
    G2,
    log_sigma_est_2d,
    cmap="viridis",
    edgecolor="k",
    linewidth=0.1,
    alpha=0.9,
    label=r"$2\log(\sqrt{\hat{\Sigma}})$",
)

# Labels and styling
ax4.view_init(elev=5, azim=-27)
ax4.set_xlim([-2, 2])
ax4.set_ylim([-2, 2])
ax4.set_xticks(np.arange(-2, 3, 2))
ax4.set_yticks(np.arange(-2, 3, 2))
ax4.tick_params(labelsize=12)
ax4.set_title(
    "(d) 2D Variance Estimation (Log-scale)", fontsize=13, fontweight="normal"
)
# Skip legend due to matplotlib 3.7+ bug with Poly3DCollection
# ax4.legend(loc="lower left", fontsize=10)

# NO colorbar for panel d)

################################################################################
# Save Figure for Publication
################################################################################

print("\nSaving figure...")

# Create output directory if it doesn't exist
fig_dir = os.path.join(os.path.dirname(__file__), "..", "fig", "reproduced")
os.makedirs(fig_dir, exist_ok=True)

# Save as PNG (300 DPI for publication)
png_path = os.path.join(fig_dir, "fig5_heteroscedasticity_python.png")
fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="w")
print(f"  - Saved PNG: {os.path.abspath(png_path)}")

# Save as PDF (vector graphics)
pdf_path = os.path.join(fig_dir, "fig5_heteroscedasticity_python.pdf")
fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="w")
print(f"  - Saved PDF: {os.path.abspath(pdf_path)}")

################################################################################
# Summary
################################################################################

print()
print("=" * 80)
print("Figure 5 Generation Complete!")
print("=" * 80)
print()

print("Summary:")
print(f"  - 1D data: {n1d} samples")
print(f"  - 2D data: {n2d} samples")
print("  - Demonstrates heteroscedastic regression:")
print("    * Panel (a): 1D mean with intervals (CI and PI)")
print("    * Panel (b): 1D variance estimation (log-scale)")
print("    * Panel (c): 2D mean with intervals (CI and PI)")
print("    * Panel (d): 2D variance estimation (log-scale)")
print(f"  - Figure saved to: {os.path.abspath(fig_dir)}")
print()

print("Key observations:")
print("  - Variance depends on predictor values (heteroscedastic)")
print("  - Confidence intervals adapt to local variance")
print("  - Log-scale visualization helps identify variance patterns")
print()

# Optionally show the figure (set FASTLPR_SHOW_PLOT=1 to display)
if os.environ.get("FASTLPR_SHOW_PLOT", "0") == "1":
    plt.show()
else:
    plt.close(fig)  # Close figure to free memory
