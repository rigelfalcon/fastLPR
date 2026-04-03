# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Structures and helpers for regression models in the Python port.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .core import create_grid


@dataclass
class RegressionModel:
    """Container mirroring the MATLAB `regs` structure inputs."""

    x_raw: np.ndarray
    y_raw: np.ndarray
    x_scaled: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    grid_axes_raw: List[np.ndarray]
    grid_axes_scaled: List[np.ndarray]
    bandwidths: np.ndarray
    options: Dict[str, object]


def create_regression_model(
    x: np.ndarray,
    y: np.ndarray,
    bandwidths: np.ndarray,
    options: Dict[str, object],
) -> RegressionModel:
    """
    Standardize inputs, generate evaluation grids, and collect metadata.
    """

    x_arr = np.asarray(x, dtype=float)
    # Allow complex-valued responses
    y_arr = np.asarray(y)

    if x_arr.ndim != 2:
        raise ValueError("Input x must be a 2D array.")
    # Support both single-response (N,) or (N,1) and multi-response (N, dy)
    if y_arr.ndim == 1:
        y_arr = y_arr[:, None]  # Convert to (N, 1)
    if y_arr.ndim != 2 or y_arr.shape[0] != x_arr.shape[0]:
        raise ValueError("Input y must be 1D or 2D array with same number of samples as x.")

    # Standardize data (z-score normalization)
    # MATLAB ALWAYS calls zscore(x) unconditionally
    # NOTE: The `dstd` option in MATLAB controls the 1-SE bandwidth selection rule,
    # NOT data standardization. Python should always standardize to match MATLAB.
    mean = x_arr.mean(axis=0)
    std = x_arr.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    x_scaled = (x_arr - mean) / std

    # Center data at origin
    # This is critical for symmetric padding and correct NUFFT
    x_min_before_center = x_scaled.min(axis=0)
    x_max_before_center = x_scaled.max(axis=0)
    center_offset = (x_max_before_center + x_min_before_center) / 2
    x_scaled = x_scaled - center_offset

    # Update x_min and x_max after centering
    x_min = x_scaled.min(axis=0)
    x_max = x_scaled.max(axis=0)

    # Bandwidth is specified in the NORMALIZED space (after z-scoring)
    # The user should specify h in the normalized space, not the original space
    #  h has consistent meaning across different data scales

    # Create grid in raw space (no padding)
    # MATLAB default: N = ceil(Tx^(1/dx)) where Tx = number of data points
    # This gives good balance of accuracy and speed
    n_samples, dims = x_arr.shape
    default_n = int(np.ceil(n_samples ** (1.0 / dims)))
    grid_axes_raw = create_grid(x_arr, options, default_n=default_n)

    # Scale grid to match scaled data
    # Grid is created from raw data, so we need to apply same transformation
    grid_axes_scaled = []
    for dim, axis in enumerate(grid_axes_raw):
        # Apply z-score transformation
        axis_scaled = (axis - mean[dim]) / std[dim]
        # Apply centering (same as data) - use the center_offset computed above
        axis_scaled = axis_scaled - center_offset[dim]
        grid_axes_scaled.append(axis_scaled)

    # Filter out invalid bandwidths (too small for grid) BEFORE cross-validation
    # Python was calling compute_kernel_fourier inside each nufft_regression call, but not
    # propagating the filtered bandwidths back - this caused bandwidth selection mismatch!
    # Using cached version ensures KDF is reused in subsequent cv_fastlpr calls
    from .convolution import compute_kernel_fourier_cached

    grid_shape = np.array([len(ax) for ax in grid_axes_scaled])
    order = int(options.get("order", 0))
    kernel_type = str(options.get("kernel_type", "gaussian"))

    # Compute KDF once to get filtered bandwidths
    # This removes bandwidths that are too small for the grid
    flag_power2 = bool(options.get("flag_power2", True))
    kd_ft, kdf_params = compute_kernel_fourier_cached(
        x_scaled, bandwidths, grid_shape, order=order, kernel_type=kernel_type,
        flag_power2=flag_power2
    )

    # Extract filtered bandwidths from params (from MATLAB regs.h after fastLPR_kdf)
    bandwidths_filtered = kdf_params["h"]

    return RegressionModel(
        x_raw=x_arr,
        y_raw=y_arr,
        x_scaled=x_scaled,
        mean=mean,
        std=std,
        grid_axes_raw=grid_axes_raw,
        grid_axes_scaled=grid_axes_scaled,
        bandwidths=bandwidths_filtered,  # Use filtered bandwidths (from MATLAB)
        options=options,
    )
