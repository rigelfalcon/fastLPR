"""
Code to generate the qEEG figure (fig_qeeg) for the fastLPR paper.

qEEG cross-spectral normative modeling (Manuscript Section 4).
  - Data: data_qeeg_cross_only.csv (N = 66505, complex-valued response)
  - Native complex-valued local polynomial regression (order = 1)
  - GCV-based bandwidth selection with the 1-SE rule, effective DoF tracking
  - Prediction and pointwise confidence bands on a dense grid

Five-panel figure:
  (a) Raw data scatter on (age, frequency), colored by |y|
  (b) GCV bandwidth selection surface over the (h1, h2) grid, 1-SE marker
  (c) Fitted real-part surface Re(m_hat)
  (d) Fitted imaginary-part surface Im(m_hat)
  (e) 95% confidence band at the f = 10 Hz slice (real top, imaginary bottom)

Self-contained (no external dependencies except fastlpr).
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm

# Add fastLPR to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastlpr import cv_fastlpr, get_hlist, fastlpr_predict

print()
print("=" * 80)
print("qEEG Cross-Spectral Normative Modeling")
print("=" * 80)

################################################################################
# Load and explore the data
################################################################################

print("\nLoading data...")
data_file = os.path.join(
    os.path.dirname(__file__), "..", "data", "data_qeeg_cross_only.csv"
)
qeeg = pd.read_csv(data_file)
# pandas reads the complex column as strings; convert after replacing the
# R/MATLAB imaginary unit "i" with Python's "j".
x = qeeg[["age", "freq"]].values
y = qeeg["riemlogm10_1"].apply(lambda s: complex(s.replace("i", "j"))).values
print(f"  - Observations: {x.shape[0]}")
print(f"  - Real part range: [{y.real.min():.3f}, {y.real.max():.3f}]")
print(f"  - Imaginary part range: [{y.imag.min():.3f}, {y.imag.max():.3f}]")

################################################################################
# Bandwidth selection and model fitting
################################################################################

print("\nFitting complex-valued LPR (order = 1, GCV bandwidth selection)...")
hlist = get_hlist([9, 9], [[1e-3, 2], [0.05, 2]])
opt = {"order": 1, "calc_dof": True, "dstd": 1, "seed": 42, "verbose": False}

t0 = time.time()
result = cv_fastlpr(x, y, hlist, opt)
elapsed = time.time() - t0

h1se = np.asarray(result.gcv_yhat["h1se"], dtype=float).ravel()
hmin = np.asarray(result.gcv_yhat["hmin"], dtype=float).ravel()
print(f"  - Selected bandwidth (1-SE): [{h1se[0]:.4f}, {h1se[1]:.4f}]")
print(f"  - Selected bandwidth (min):  [{hmin[0]:.4f}, {hmin[1]:.4f}]")
if result.dof is not None:
    print(f"  - Effective DoF: {result.dof:.1f}")
print(f"  - Computation time: {elapsed:.1f} seconds")

################################################################################
# Prediction and confidence bands on a dense grid
################################################################################

print("\nPredicting on 100 x 100 evaluation grid...")
n_grid = 100
age_grid = np.linspace(x[:, 0].min(), x[:, 0].max(), n_grid)
freq_grid = np.linspace(x[:, 1].min(), x[:, 1].max(), n_grid)
A, F = np.meshgrid(age_grid, freq_grid, indexing="ij")
x_eval = np.column_stack([A.ravel(), F.ravel()])
pred = fastlpr_predict(result, x_eval)
re_mat = pred.real.reshape(n_grid, n_grid)
im_mat = pred.imag.reshape(n_grid, n_grid)

# Pointwise standard error via the local-polynomial expression used for the
# confidence bands: se^2 = sigma^2 * nu / (|H| * s_0), evaluated at each point.
# (See Manuscript Section 4.) s_0 lives on the internal grid; interpolate it
# onto the evaluation grid, mirroring the R fpp_s0 evaluator.
resid = y - fastlpr_predict(result, x)
sig2 = float(np.mean(np.abs(resid) ** 2))
nu = 0.079577471546  # Gaussian kernel, d = 2, order = 1
prod_h = float(np.prod(h1se))
s0_grid = np.real(np.asarray(result.s0))
s0_interp = RegularGridInterpolator(
    result.grid, s0_grid, bounds_error=False, fill_value=None
)
s0_eval = np.maximum(np.real(s0_interp(x_eval)), 1e-10)
se_eval = np.sqrt(sig2 * nu / (prod_h * s0_eval))
zval = norm.ppf(0.975)

################################################################################
# Build the 5-panel figure
################################################################################

print("\nCreating figure...")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
fig = plt.figure(figsize=(18, 10), facecolor="w")
# 2 x 3 grid: (a)(b) top, (c)(d) bottom on left/middle columns; the right
# column stacks the two CI sub-plots that together form panel (e).
gs = fig.add_gridspec(2, 3)

## Panel (a): raw scatter colored by |y|
ax_a = fig.add_subplot(gs[0, 0])
absy = np.abs(y)
rng = np.random.default_rng(0)
sub = rng.choice(x.shape[0], size=min(20000, x.shape[0]), replace=False)
sc = ax_a.scatter(x[sub, 0], x[sub, 1], c=absy[sub], s=2, cmap="viridis")
ax_a.set_xlabel(r"$\log_{10}(\mathrm{age})$", fontsize=13)
ax_a.set_ylabel("Frequency (Hz)", fontsize=13)
ax_a.set_title("(a) Raw data, colored by |y|", fontsize=15, fontweight="bold")
plt.colorbar(sc, ax=ax_a)

## Panel (b): GCV bandwidth selection surface
ax_b = fig.add_subplot(gs[0, 1])
hlist_arr = np.asarray(hlist)
h1_u = np.unique(hlist_arr[:, 0])
h2_u = np.unique(hlist_arr[:, 1])
gcv_m = np.asarray(result.gcv_yhat["gcv_m"], dtype=float).ravel()
gcv_grid = np.full((len(h1_u), len(h2_u)), np.nan)
for k in range(hlist_arr.shape[0]):
    i1 = int(np.searchsorted(h1_u, hlist_arr[k, 0]))
    i2 = int(np.searchsorted(h2_u, hlist_arr[k, 1]))
    gcv_grid[i1, i2] = gcv_m[k]
im = ax_b.imshow(
    gcv_grid.T,
    origin="lower",
    aspect="auto",
    cmap="viridis",
    extent=[
        np.log10(h1_u.min()),
        np.log10(h1_u.max()),
        np.log10(h2_u.min()),
        np.log10(h2_u.max()),
    ],
)
plt.colorbar(im, ax=ax_b)
ax_b.plot(np.log10(hmin[0]), np.log10(hmin[1]), "o", color="blue",
          markersize=10, label="GCV min")
ax_b.plot(np.log10(h1se[0]), np.log10(h1se[1]), "*", color="red",
          markersize=18, label="1-SE")
ax_b.set_xlabel(r"$\log_{10}(h_1)$", fontsize=13)
ax_b.set_ylabel(r"$\log_{10}(h_2)$", fontsize=13)
ax_b.set_title("(b) GCV bandwidth surface", fontsize=15, fontweight="bold")
ax_b.legend(loc="upper right", fontsize=10)

## Panel (c): fitted real-part surface
ax_c = fig.add_subplot(gs[1, 0])
cf = ax_c.contourf(A, F, re_mat, levels=30, cmap="viridis")
ax_c.contour(A, F, re_mat, levels=10, colors="k", linewidths=0.4)
plt.colorbar(cf, ax=ax_c)
ax_c.set_xlabel(r"$\log_{10}(\mathrm{age})$", fontsize=13)
ax_c.set_ylabel("Frequency (Hz)", fontsize=13)
ax_c.set_title(r"(c) Fitted real part $\Re\,\hat{m}$", fontsize=15,
               fontweight="bold")

## Panel (d): fitted imaginary-part surface
ax_d = fig.add_subplot(gs[1, 1])
cf = ax_d.contourf(A, F, im_mat, levels=30, cmap="viridis")
ax_d.contour(A, F, im_mat, levels=10, colors="k", linewidths=0.4)
plt.colorbar(cf, ax=ax_d)
ax_d.set_xlabel(r"$\log_{10}(\mathrm{age})$", fontsize=13)
ax_d.set_ylabel("Frequency (Hz)", fontsize=13)
ax_d.set_title(r"(d) Fitted imag part $\Im\,\hat{m}$", fontsize=15,
               fontweight="bold")

## Panel (e): 95% CI band at f = 10 Hz slice (real top, imag bottom)
jf = int(np.argmin(np.abs(freq_grid - 10.0)))
# x_eval was built with indexing="ij"; select rows at the chosen frequency.
mask = np.isclose(x_eval[:, 1], freq_grid[jf])
ag = x_eval[mask, 0]
order = np.argsort(ag)
ag = ag[order]
re_slice = pred.real[mask][order]
im_slice = pred.imag[mask][order]
se_slice = se_eval[mask][order]

ax_e1 = fig.add_subplot(gs[0, 2])
ax_e1.fill_between(ag, re_slice - zval * se_slice, re_slice + zval * se_slice,
                   color=(0.2, 0.2, 0.8), alpha=0.25)
ax_e1.plot(ag, re_slice, "k-", lw=2)
ax_e1.set_xlabel(r"$\log_{10}(\mathrm{age})$", fontsize=13)
ax_e1.set_ylabel(r"$\Re(m)$", fontsize=13)
ax_e1.set_title("(e) 95% CI at f = 10 Hz (real)", fontsize=14,
                fontweight="bold")

ax_e2 = fig.add_subplot(gs[1, 2])
ax_e2.fill_between(ag, im_slice - zval * se_slice, im_slice + zval * se_slice,
                   color=(0.8, 0.2, 0.2), alpha=0.25)
ax_e2.plot(ag, im_slice, "k-", lw=2)
ax_e2.set_xlabel(r"$\log_{10}(\mathrm{age})$", fontsize=13)
ax_e2.set_ylabel(r"$\Im(m)$", fontsize=13)
ax_e2.set_title("95% CI at f = 10 Hz (imag)", fontsize=14, fontweight="bold")

fig.suptitle(
    "qEEG Cross-Spectral Normative Modeling (fastLPR)",
    fontsize=18,
    fontweight="bold",
)
plt.tight_layout(rect=[0, 0, 1, 0.97])

################################################################################
# Save figure
################################################################################

fig_dir = os.path.join(os.path.dirname(__file__), "..", "fig", "reproduced")
os.makedirs(fig_dir, exist_ok=True)
png_path = os.path.join(fig_dir, "fig_qeeg.png")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
print(f"  - Saved PNG: {png_path}")
pdf_path = os.path.join(fig_dir, "fig_qeeg.pdf")
plt.savefig(pdf_path, bbox_inches="tight")
print(f"  - Saved PDF: {pdf_path}")
plt.close(fig)

print("\nExample completed successfully!")
