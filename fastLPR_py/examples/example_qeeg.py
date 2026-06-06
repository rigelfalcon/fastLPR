"""
Code to generate Figure 6: Real-World Applications (qEEG and MRI)

This script reproduces Figure 6 from the fastLPR paper, demonstrating:
  - Panel (a): 2D qEEG log-spectrum regression (age × frequency)
  - Panel (b): 3D MRI T1 brain image regression (spatial coordinates)

The figure follows JSS publication standards with:
  - Fixed random seed for reproducibility
  - Consistent styling (fonts, colors, sizes)
  - 300 DPI resolution for publication
  - Self-contained code (no external dependencies except fastLPR)

Copyright (c) 2024 fastLPR Development Team
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import time
import os
import sys

# Add parent directory to path to import fastlpr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastlpr import cv_fastlpr, get_hlist

print()
print("=" * 80)
print("Figure 6: Real-World Applications (qEEG and MRI)")
print("=" * 80)

# Check for quick test mode (reduced samples for faster testing)
quick_test = os.environ.get("FASTLPR_QUICK_TEST", "0") == "1"
if quick_test:
    print("\n[QUICK TEST MODE: Using reduced sample sizes for faster testing]")

################################################################################
# Panel (a): 2D qEEG Log-Spectrum Regression
################################################################################

print("\nPanel (a): 2D qEEG Log-Spectrum Regression...")

# Load qEEG data
data_file = os.path.join(os.path.dirname(__file__), "..", "data", "data_qeeg.csv")
if not os.path.isfile(data_file):
    raise FileNotFoundError(f"qEEG data file not found: {data_file}")

data_table = pd.read_csv(data_file)

# Subsample for quick testing
if quick_test:
    # Use 10% of data for quick testing
    np.random.seed(42)
    sample_indices = np.random.choice(
        len(data_table), size=len(data_table) // 10, replace=False
    )
    data_table = data_table.iloc[sample_indices].reset_index(drop=True)

print(f"  - Loaded {len(data_table)} qEEG samples")

# Prepare data: X = [age (log10), frequency], y = log-spectrum
X_qeeg = data_table[["age", "freq"]].values


# Parse complex numbers from string format "real+imag*i"
def parse_complex(s):
    """Parse complex number from MATLAB-style string format."""
    if isinstance(s, (int, float, complex)):
        return complex(s)
    # Replace 'i' with 'j' for Python complex number format
    s = str(s).replace("i", "j")
    return complex(s)


y_qeeg_complex = data_table["log10_10"].apply(parse_complex).values
y_qeeg = y_qeeg_complex.real  # Use real part only

# Normalize predictors (REMOVED - matching MATLAB)
# X_mean = X_qeeg.mean(axis=0)
# X_std = X_qeeg.std(axis=0)
# X_qeeg_norm = (X_qeeg - X_mean) / X_std

print(
    f"  - Age range: [{10**X_qeeg[:, 0].min():.1f}, {10**X_qeeg[:, 0].max():.1f}] years"
)
print(f"  - Frequency range: [{X_qeeg[:, 1].min():.2f}, {X_qeeg[:, 1].max():.2f}] Hz")

# Regression with bandwidth selection
opt_qeeg = {
    "order": 0,  # Local constant (Nadaraya-Watson) - matching MATLAB
    "dstd": 10,
    "kernel_type": "gaussian",
    "verbose": False,
}

hlist_qeeg = get_hlist(
    [5, 5], [[0.1, 1.0], [0.1, 2.0]]
)  # Match MATLAB: age [0.1,1], freq [0.1,2]
print(f"  - Performing regression with {len(hlist_qeeg)} bandwidth combinations...")

start_time = time.time()
regs_qeeg = cv_fastlpr(
    X_qeeg, y_qeeg, hlist_qeeg, opt_qeeg
)  # Use raw data, not normalized
time_qeeg = time.time() - start_time

print(f"  - Computation time: {time_qeeg:.2f} seconds")
print(
    f"  - Selected bandwidth: h = [{regs_qeeg.h[0]:.4f}, {regs_qeeg.h[1]:.4f}]"
)

# Create grid for visualization (in original coordinate space)
n_grid = 50
age_grid = np.linspace(X_qeeg[:, 0].min(), X_qeeg[:, 0].max(), n_grid)
freq_grid = np.linspace(X_qeeg[:, 1].min(), X_qeeg[:, 1].max(), n_grid)
Age_grid, Freq_grid = np.meshgrid(age_grid, freq_grid, indexing="ij")

# Get predictions
grid_points = np.column_stack([Age_grid.ravel(), Freq_grid.ravel()])
Y_pred_qeeg = regs_qeeg.fpp_yhat(grid_points).reshape(Age_grid.shape)

################################################################################
# Panel (b): 3D MRI T1 Brain Image Regression
################################################################################

print("\nPanel (b): 3D MRI T1 Brain Image Regression...")

# Load MRI T1 data
mri_file = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "subjectimage_T1.mat",
)
if not os.path.isfile(mri_file):
    raise FileNotFoundError(f"MRI data file not found: {mri_file}")

mri_data = sio.loadmat(mri_file)
Cube = mri_data["Cube"]
print(f"  - Loaded MRI T1 data: {Cube.shape}")

# Create scattered data from non-zero voxels
I, J, K = np.where(Cube > 0)
X_mri_full = np.column_stack([I, J, K])
y_mri_full = Cube[Cube > 0].astype(np.float32)

# Subsample to reasonable size for computational efficiency
np.random.seed(42)
if quick_test:
    n_samples = min(10000, len(y_mri_full))  # 10% for quick testing
else:
    n_samples = min(100000, len(y_mri_full))

idx_sample = np.random.permutation(len(y_mri_full))[:n_samples]
X_mri = X_mri_full[idx_sample, :]
y_mri = y_mri_full[idx_sample]

print(f"  - Total non-zero voxels: {len(y_mri_full)}")
print(f"  - Using {n_samples} scattered samples for regression")
print(
    f"  - Spatial range: [{X_mri[:, 0].min()}, {X_mri[:, 0].max()}] × "
    f"[{X_mri[:, 1].min()}, {X_mri[:, 1].max()}] × "
    f"[{X_mri[:, 2].min()}, {X_mri[:, 2].max()}]"
)

# Normalize predictors (COMMENTED OUT TO MATCH MATLAB)
# X_mri_mean = X_mri.mean(axis=0)
# X_mri_std = X_mri.std(axis=0)
# X_mri_norm = (X_mri - X_mri_mean) / X_mri_std

# Regression with bandwidth selection (use smaller grid for 3D)
opt_mri = {
    "order": 0,  # Local constant (order 2 not supported for 3D)
    "dstd": 1,
    "kernel_type": "gaussian",
    "N": [64, 64, 64],  # Grid size for FFT
    "verbose": False,
}

# Use fixed bandwidth for faster computation
hlist_mri = np.array([[0.03, 0.03, 0.03]])
print(
    f"  - Performing regression with bandwidth h = [{hlist_mri[0, 0]:.3f}, "
    f"{hlist_mri[0, 1]:.3f}, {hlist_mri[0, 2]:.3f}]..."
)

start_time = time.time()
regs_mri = cv_fastlpr(X_mri, y_mri, hlist_mri, opt_mri)  # Use X_mri not X_mri_norm
time_mri = time.time() - start_time

print(f"  - Computation time: {time_mri:.2f} seconds")
print(
    f"  - Selected bandwidth: h = [{regs_mri.h[0]:.4f}, "
    f"{regs_mri.h[1]:.4f}, {regs_mri.h[2]:.4f}]"
)

# Create grid for visualization (higher resolution for better quality)
grid_res = 50
x_range = [X_mri[:, 0].min(), X_mri[:, 0].max()]
y_range = [X_mri[:, 1].min(), X_mri[:, 1].max()]
z_range = [X_mri[:, 2].min(), X_mri[:, 2].max()]

X_grid, Y_grid, Z_grid = np.meshgrid(
    np.linspace(x_range[0], x_range[1], grid_res),
    np.linspace(y_range[0], y_range[1], grid_res),
    np.linspace(z_range[0], z_range[1], grid_res),
    indexing="ij",
)

x_grid = np.column_stack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()])

# Get predictions
y_pred_mri = regs_mri.fpp_yhat(x_grid)

################################################################################
# Create Figure with 2 Panels
################################################################################

print("\nCreating figure...")

fig = plt.figure(figsize=(14, 6), facecolor="w")

################################################################################
# Panel (a): qEEG 2D Surface
################################################################################

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.set_position([0.08, 0.15, 0.38, 0.75])

# Plot surface
surf1 = ax1.plot_surface(
    Age_grid, Freq_grid, Y_pred_qeeg, cmap="viridis", edgecolor="none", alpha=0.9
)

# Add heatmap at the bottom
z_bottom = Y_pred_qeeg.min() - 0.5
ax1.plot_surface(
    Age_grid,
    Freq_grid,
    np.ones_like(Y_pred_qeeg) * z_bottom,
    facecolors=plt.cm.viridis(
        (Y_pred_qeeg - Y_pred_qeeg.min()) / (Y_pred_qeeg.max() - Y_pred_qeeg.min())
    ),
    edgecolor="none",
    alpha=0.8,
)

ax1.set_xlabel("Age (log$_{10}$ years)", fontsize=14)
ax1.set_ylabel("Frequency (Hz)", fontsize=14)
ax1.set_zlabel("Log-spectrum", fontsize=14)
ax1.set_title("(a) 2D qEEG Regression", fontsize=16, fontweight="bold")
ax1.view_init(elev=30, azim=-37.5)
ax1.grid(True)
ax1.tick_params(labelsize=12)

# Add colorbar with padding to avoid overlap with z-axis
cb1 = fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.15)
cb1.set_label("Log-spectrum", fontsize=12)

################################################################################
# Panel (b): MRI 3D Volume (Scatter3 with Transparency)
################################################################################

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.set_position([0.56, 0.15, 0.38, 0.75])

# Reshape predictions back to 3D grid
y_pred_mri_3d = y_pred_mri.reshape(grid_res, grid_res, grid_res)

# Normalize intensity for coloring and transparency
y_norm = (y_pred_mri - y_pred_mri.min()) / (y_pred_mri.max() - y_pred_mri.min() + 1e-10)

# Filter out low values (threshold at 0.05) to show more structure
mask = y_norm > 0.05
x_grid_filtered = x_grid[mask]
y_pred_mri_filtered = y_pred_mri[mask]
y_norm_filtered = y_norm[mask]

# Map intensity to RGBA colors with per-point transparency
# This allows seeing both skull (low intensity) and brain (high intensity)
from matplotlib import cm

cmap_mri = cm.get_cmap("jet")
colors_mri = cmap_mri(y_norm_filtered)  # Shape: (N, 4) with RGBA

# Set alpha channel based on intensity (higher intensity = more opaque)
# Use a nonlinear mapping to enhance contrast between skull and brain
alpha_mri = 0.05 + 0.7 * (y_norm_filtered**0.5)  # Square root for better contrast
colors_mri[:, 3] = alpha_mri  # Set alpha channel

# Compute marker size based on intensity
marker_size = 3 + 8 * y_norm_filtered

# Plot 3D scatter with per-point RGBA colors
scatter = ax2.scatter(
    x_grid_filtered[:, 0],
    x_grid_filtered[:, 1],
    x_grid_filtered[:, 2],
    s=marker_size,
    c=colors_mri,  # Use RGBA colors directly for per-point transparency
    marker="o",
    edgecolors="none",
)

ax2.set_xlabel("x (normalized)", fontsize=14)
ax2.set_ylabel("y (normalized)", fontsize=14)
ax2.set_zlabel("z (normalized)", fontsize=14)
ax2.set_title("(b) 3D MRI T1 Regression", fontsize=16, fontweight="bold")
ax2.grid(True)
ax2.set_box_aspect([1, 1, 1])
ax2.view_init(elev=22, azim=124)
ax2.tick_params(labelsize=12)
ax2.set_facecolor([0.95, 0.95, 0.95])

# Add colorbar with padding to avoid overlap with z-axis
cb2 = fig.colorbar(scatter, ax=ax2, shrink=0.5, aspect=10, pad=0.15)
cb2.set_label("Intensity", fontsize=12)

################################################################################
# Save Figure
################################################################################

print("\nSaving figure...")

# Create output directory
fig_dir = os.path.join(os.path.dirname(__file__), "..", "fig", "reproduced")
os.makedirs(fig_dir, exist_ok=True)

# Save as PNG
png_path = os.path.join(fig_dir, "fig6_applications_python.png")
fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="w")
print(f"  - Saved PNG: {os.path.abspath(png_path)}")

# Save as PDF
pdf_path = os.path.join(fig_dir, "fig6_applications_python.pdf")
fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="w")
print(f"  - Saved PDF: {os.path.abspath(pdf_path)}")

################################################################################
# Summary
################################################################################

print()
print("=" * 80)
print("Figure 6 Generation Complete!")
print("=" * 80)
print()

print("Summary:")
print(f"  Panel (a) - qEEG 2D Regression:")
print(f"    - Samples: {len(X_qeeg)}")
print(f"    - Computation time: {time_qeeg:.2f} sec")
print(
    f"    - Selected bandwidth: h = [{regs_qeeg.h[0]:.4f}, {regs_qeeg.h[1]:.4f}]"
)
print(f"  Panel (b) - MRI 3D Regression:")
print(f"    - Samples: {len(X_mri)}")
print(f"    - Computation time: {time_mri:.2f} sec")
print(
    f"    - Selected bandwidth: h = [{regs_mri.h[0]:.4f}, "
    f"{regs_mri.h[1]:.4f}, {regs_mri.h[2]:.4f}]"
)
print(f"  Figure saved to: {os.path.abspath(fig_dir)}")
print()

# Optionally show the figure (set FASTLPR_SHOW_PLOT=1 to display)
if os.environ.get("FASTLPR_SHOW_PLOT", "0") == "1":
    plt.show()
else:
    plt.close(fig)  # Close figure to free memory
