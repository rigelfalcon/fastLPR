# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Kernel functions for fastLPR.

Port from fastLPR/utility/core/kernel_function.m.

This module provides kernel functions for local polynomial regression and KDE.
Supports Gaussian, Epanechnikov, and custom kernels.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import numpy as np

# Use centralized backend detection
from .backend_selection import HAS_NUMBA

if HAS_NUMBA:
    import numba

SQRT_2PI = np.sqrt(2.0 * np.pi)

# Kernel type constants
KERNEL_GAUSSIAN = "gaussian"
KERNEL_EPANECHNIKOV = "epanechnikov"
KERNEL_TRICUBE = "tricube"
KERNEL_QUARTIC = "quartic"
KERNEL_TRIWEIGHT = "triweight"
KERNEL_UNIFORM = "uniform"

# Supported kernel types
SUPPORTED_KERNELS = [
    KERNEL_GAUSSIAN,
    KERNEL_EPANECHNIKOV,
    KERNEL_TRICUBE,
    KERNEL_QUARTIC,
    KERNEL_TRIWEIGHT,
    KERNEL_UNIFORM,
]


def nufftfreqs(
    N: Union[int, np.ndarray], df: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generate frequency grid for NUFFT.

    Port from fastLPR/utility/core/nufftfreqs.m.

    Creates the frequency grid used in Non-Uniform Fast Fourier Transform.
    Frequencies are centered around zero with proper spacing.

    Args:
        N: Grid size in each dimension (scalar or array)
        df: Frequency spacing in each dimension (default: ones)

    Returns:
        Frequency grid with shape (*N, dx) where dx is number of dimensions
    """
    # Convert N to array if scalar
    if isinstance(N, int):
        N = np.array([N])
    elif not isinstance(N, np.ndarray):
        N = np.array(N)

    dx = len(N)

    # Default frequency spacing
    if df is None:
        df = np.ones(dx)

    # Compute frequency range for each dimension
    # Frequencies: fix(-N/2) to N-fix(N/2)-1
    freq_ranges = []
    for i in range(dx):
        start = -N[i] // 2
        end = N[i] - N[i] // 2 - 1
        freqs = np.linspace(start, end, N[i])
        freqs = df[i] * freqs
        freq_ranges.append(freqs)

    # Create multidimensional grid
    if dx == 1:
        grid = freq_ranges[0][:, None]
    else:
        # Use meshgrid to create ND grid
        grids = np.meshgrid(*freq_ranges, indexing="ij")
        # Stack along last dimension
        grid = np.stack(grids, axis=-1)

    return grid


def compute_grid_params(N: Union[int, Sequence[int]], accuracy: int = 8) -> tuple:
    """
    Compute NUFFT grid parameters.

    Port from fastLPR/utility/core/compute_grid_params.m.

    Calculates spreading width, oversampling ratio, and Gaussian width
    for Non-Uniform FFT based on desired accuracy.

    Parameters
    ----------
    N : int or array-like
        Original grid size (scalar or vector for multi-dimensional)
    accuracy : int, default=8
        Desired accuracy in decimal digits
        - 6: ratio=1 (no oversampling)
        - 7-11: ratio=2 (2x oversampling)
        - >11: ratio=3 (3x oversampling)

    Returns
    -------
    Msp : int or ndarray
        Spreading width (number of neighbors)
    Mr : int or ndarray
        Oversampled grid size = ratio * N
    tau : float or ndarray
        Gaussian spreading parameter
    ratio : int
        Oversampling ratio (1, 2, or 3)

    Examples
    --------
    >>> Msp, Mr, tau, ratio = compute_grid_params(100, 8)
    >>> # Returns: Msp=8, Mr=200, tau=..., ratio=2

    Notes
    -----
    - Higher accuracy requires more oversampling
    - tau controls Gaussian spreading width
    - Msp is number of grid points affected by each sample
    """
    N = np.atleast_1d(np.asarray(N, dtype=int))

    # Determine oversampling ratio based on accuracy
    if accuracy <= 6:
        ratio = 1  # No oversampling
    elif 6 < accuracy <= 11:
        ratio = 2  # 2x oversampling
    else:
        ratio = 3  # 3x oversampling

    # Compute grid parameters
    Msp = accuracy * np.ones_like(N, dtype=int)  # Spreading width
    tau = (np.pi * Msp) / (N * N * ratio * (ratio - 0.5))  # Gaussian width
    Mr = ratio * N  # Oversampled grid size

    # Return scalars if input was scalar
    if N.size == 1:
        return int(Msp[0]), int(Mr[0]), float(tau[0]), ratio
    return Msp, Mr, tau, ratio


def kernel_function(
    x: np.ndarray,
    c: Optional[np.ndarray] = None,
    h: Optional[Union[float, np.ndarray]] = None,
    kernel_type: str = "gaussian",
    kernel_custom: Optional[Callable] = None,
    opt: Optional[dict] = None,
) -> np.ndarray:
    """
    Evaluate kernel function at given points.

    Port from fastLPR/utility/core/kernel_function.m.

    Computes kernel function values K((x-c)/h) for various kernel types.
    Supports both grid-based and scattered data formats.

    Parameters
    ----------
    x : ndarray
        Evaluation points, either:
        - Grid format: (N1, N2, ..., Nd, d) array
        - Scattered format: (T, d) matrix
    c : ndarray, optional
        Center point (1, d) vector, default: 0
    h : float or ndarray, optional
        Bandwidth (scalar or (dh, dx) matrix), default: 1
    kernel_type : str, default='gaussian'
        Kernel type:
        - 'gaussian' or 'gauss' or 'normal' - Gaussian kernel
        - 'epanechnikov' or 'epan' - Epanechnikov kernel
        - 'gaussian_ft' or 'normal_ft' - Gaussian in Fourier domain
        - 'polynomial' or 'poly' - Polynomial kernel
    kernel_custom : callable, optional
        Custom kernel function handle. If provided, overrides kernel_type
    opt : dict, optional
        Options structure (for polynomial kernel: opt['order'])

    Returns
    -------
    K : ndarray
        Kernel values at each point

    Notes
    -----
    Kernel Definitions:
    - Gaussian: K(u) = (1/sqrt(2pi)) * exp(-0.5 * ||u||^2)
    - Epanechnikov: K(u) = (3/4) * max(0, 1 - ||u||^2)
    - Gaussian FT: K(u) = exp(-0.5 * ||u||^2) [no normalization]
    - Polynomial: K(u) = ||u||^order

    - Automatically handles multi-dimensional data
    - Normalizes by bandwidth: u = (x-c)/h
    - Normalizes by det(h) for proper density
    - Uses eps instead of 0 for Epanechnikov to avoid division by zero
    """
    # Input validation and defaults
    if c is not None:
        x = x - c  # Center at c

    if h is None:
        h = 1.0

    if opt is None:
        opt = {}

    # Process bandwidth
    h = np.atleast_2d(np.asarray(h, dtype=float))
    dh, dx = h.shape

    # Determine data format and dimension
    d = x.ndim
    deth = np.prod(h, axis=1)  # Determinant of bandwidth matrix

    # Normalize: u = (x-c)/h
    # We need to handle multiple bandwidths by computing for each bandwidth separately
    if dh == 1:
        # Single bandwidth: simple division
        if d > dx:  # Grid format
            h_reshaped = h.T.reshape((1,) * (d - 1) + (dx,))
        else:  # Scattered format
            # Keep h as (1, dx) for broadcasting with (N, dx)
            h_reshaped = h
        x = x / h_reshaped
        deth_reshaped = deth
    else:
        # Multiple bandwidths: compute for each bandwidth and stack
        x_list = []
        for i in range(dh):
            h_i = h[i, :]
            if d > dx:  # Grid format
                h_reshaped = h_i.reshape((1,) * (d - 1) + (dx,))
            else:  # Scattered format
                h_reshaped = h_i
            x_i = x / h_reshaped
            x_list.append(x_i)
        x = np.stack(x_list, axis=-1)
        deth_reshaped = deth.reshape((1,) * d + (dh,))

    # Evaluate kernel function
    if kernel_custom is not None:
        # Use custom kernel function
        result = kernel_custom(x)
    else:
        # Use built-in kernel
        kernel_type_lower = kernel_type.lower()

        # Determine axis to sum over for ||u||^2
        # MATLAB passes d=ndims(x) to kernel functions, which then do sum(x.^2, d)
        # For scattered format (N, dx): ndims=2, sum along axis 2 (last axis) → (N, 1)
        # For grid format (N1, N2, ..., Nd, dx): ndims=dx+1, sum along axis dx+1 (last axis)
        # In Python: axis indexing is 0-based, so we use axis d-1 (last axis)
        # This is ALWAYS the last axis, regardless of format!
        sum_axis = d - 1

        if kernel_type_lower in ("epan", "epanechnikov"):
            result = _epan_kernel(x, sum_axis)
        elif kernel_type_lower in ("gauss", "gaussian", "normal"):
            result = _gaussian_kernel(x, sum_axis, dx)
        elif kernel_type_lower in ("gaussian_ft", "normal_ft"):
            result = _gaussian_kernel_ft(x, sum_axis)
        elif kernel_type_lower in ("poly", "polynomial"):
            if "order" not in opt:
                raise ValueError(
                    'Polynomial kernel requires opt["order"] to be specified'
                )
            result = _polynomial_kernel(x, sum_axis, opt["order"])
        elif kernel_type_lower in ("tricube",):
            result = _tricube_kernel(x, sum_axis)
        elif kernel_type_lower in ("quartic", "biweight"):
            result = _quartic_kernel(x, sum_axis)
        elif kernel_type_lower in ("triweight",):
            result = _triweight_kernel(x, sum_axis)
        elif kernel_type_lower in ("uniform", "rectangular"):
            result = _uniform_kernel(x, sum_axis)
        else:
            raise ValueError(
                f"Unknown kernel type: {kernel_type}. "
                f'Supported types: {", ".join(SUPPORTED_KERNELS)}'
            )

        # Normalize by det(h) for proper density
        if dh == 1:
            result = result / deth
        else:
            result = result / deth_reshaped

        result = np.squeeze(result)

    return result


def _epan_kernel(x: np.ndarray, d: int) -> np.ndarray:
    """
    Epanechnikov kernel: K(u) = (3/4) * max(0, 1 - ||u||^2).

    Compact support: K(u) = 0 for ||u|| > 1.
    Note: Use eps instead of 0 to avoid division by zero in sparse regions.

    MEMORY OPTIMIZATION: Uses in-place operations where possible.
    """
    # Compute sum of squares in-place
    result = np.sum(np.square(x), axis=d)
    # Compute 1 - ||u||^2 in-place
    np.negative(result, out=result)
    np.add(result, 1, out=result)
    # Apply 0.75 * max(eps, ...)
    np.multiply(result, 0.75, out=result)
    np.maximum(result, np.finfo(float).eps, out=result)
    return result


def _gaussian_kernel(x: np.ndarray, sum_axis: int, data_dim: int) -> np.ndarray:
    """
    Gaussian kernel: K(u) = (1/sqrt(2pi)) * exp(-0.5 * ||u||^2).

    This uses 1D normalization (1/sqrt(2pi)) for all dimensions,
    matching MATLAB's kernel_function.m.
    This is NOT the standard d-dimensional Gaussian density normalization!
    The d-dimensional normalization would be (1/sqrt(2pi))^d, but MATLAB
    uses 1D normalization for all dimensions.

    MEMORY OPTIMIZATION: Uses in-place operations to reduce temporaries.

    Parameters
    ----------
    x : ndarray
        Normalized coordinates
    sum_axis : int
        Axis to sum over for ||u||^2
    data_dim : int
        Data dimension (d) - NOT USED, kept for API compatibility
    """
    # MEMORY OPTIMIZATION: Compute in-place
    # Instead of: (1.0 / SQRT_2PI) * np.exp(-0.5 * np.sum(x**2, axis=sum_axis))
    result = np.sum(np.square(x), axis=sum_axis)
    np.multiply(result, -0.5, out=result)
    np.exp(result, out=result)
    np.multiply(result, 1.0 / SQRT_2PI, out=result)
    return result


def _gaussian_kernel_ft(x: np.ndarray, d: int) -> np.ndarray:
    """
    Gaussian kernel in Fourier domain (no 1/sqrt(2pi) normalization).

    Used for NUFFT-based convolution.

    MEMORY OPTIMIZATION: Uses in-place operations.
    """
    result = np.sum(np.square(x), axis=d)
    np.multiply(result, -0.5, out=result)
    np.exp(result, out=result)
    return result


def _polynomial_kernel(x: np.ndarray, d: int, order: int) -> np.ndarray:
    """
    Polynomial kernel: K(u) = ||u||^order.

    Used for weighted polynomial regression.
    """
    return np.sum(x**order, axis=d)


def _tricube_kernel(x: np.ndarray, d: int) -> np.ndarray:
    """
    Tricube kernel: K(u) = (70/81) * max(0, (1 - ||u||^3)^3).

    Compact support: K(u) = 0 for ||u|| > 1.
    Smoother than Epanechnikov at the boundary.
    """
    u_norm = np.sqrt(np.sum(x**2, axis=d))
    return np.maximum(
        np.finfo(float).eps, (70.0 / 81.0) * (1 - u_norm**3) ** 3 * (u_norm <= 1)
    )


def _quartic_kernel(x: np.ndarray, d: int) -> np.ndarray:
    """
    Quartic (Biweight) kernel: K(u) = (15/16) * max(0, (1 - ||u||^2)^2).

    Compact support: K(u) = 0 for ||u|| > 1.
    Also known as the biweight kernel.
    """
    u_sq = np.sum(x**2, axis=d)
    return np.maximum(
        np.finfo(float).eps, (15.0 / 16.0) * (1 - u_sq) ** 2 * (u_sq <= 1)
    )


def _triweight_kernel(x: np.ndarray, d: int) -> np.ndarray:
    """
    Triweight kernel: K(u) = (35/32) * max(0, (1 - ||u||^2)^3).

    Compact support: K(u) = 0 for ||u|| > 1.
    Higher order than quartic, smoother at the boundary.
    """
    u_sq = np.sum(x**2, axis=d)
    return np.maximum(
        np.finfo(float).eps, (35.0 / 32.0) * (1 - u_sq) ** 3 * (u_sq <= 1)
    )


def _uniform_kernel(x: np.ndarray, d: int) -> np.ndarray:
    """
    Uniform (Rectangular) kernel: K(u) = 0.5 if ||u|| <= 1, else 0.

    Compact support: K(u) = 0 for ||u|| > 1.
    Simplest kernel, but not smooth.
    """
    u_norm = np.sqrt(np.sum(x**2, axis=d))
    return 0.5 * (u_norm <= 1).astype(float)


# Numba-optimized versions for performance-critical operations
if HAS_NUMBA:

    @numba.jit(nopython=True, cache=True)
    def _gaussian_kernel_numba(x: np.ndarray) -> np.ndarray:
        """Numba-optimized Gaussian kernel for 1D case."""
        return (1.0 / SQRT_2PI) * np.exp(-0.5 * x**2)

    @numba.jit(nopython=True, cache=True)
    def _gaussian_kernel_2d_numba(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Numba-optimized Gaussian kernel for 2D case."""
        return (1.0 / (2.0 * np.pi)) * np.exp(-0.5 * (x**2 + y**2))
