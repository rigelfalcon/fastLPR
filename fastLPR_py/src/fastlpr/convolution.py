# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
NUFFT-based convolution for fastLPR.

Port from fastLPR/utility/core/fastLPR_conv.m, fastLPR_kdf.m, fastLPR_nufft.m

This module provides fast convolution operations using NUFFT for scattered data.
This is the core operation enabling O(N + M log M) complexity for local polynomial regression.

Memory Optimizations (2025-12):
- In-place FFT operations via fft_backend with overwrite_x=True
- Kernel FFT caching (compute_kernel_fourier_cached) - compute once, reuse across bandwidths
- Reduced temporary array allocations using np.empty and np.mod with out=
- Vectorized design matrix construction (design_matrix_optimized.py)
- NUFFT result caching for repeated transforms (nufft_transform_cached)
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from . import fft_backend
from .caching import _kernel_fft_cache, _nufft_cache, array_key
from .kernel import compute_grid_params, kernel_function
from .nufftn_type1 import nufftn_type1
from .lwp_estimator import get_polynomial_terms


def nufft_type1_wrapper(
    x: np.ndarray,
    y: np.ndarray,
    grid_shape: tuple,
    iflag: int = -1,
    accuracy: int = 6,
    apply_deconv: bool = True,
) -> np.ndarray:
    """
    Wrapper for MATLAB-mirrored NUFFT that accepts normalized coordinates.

    This adapts the MATLAB-mirrored NUFFT to work with the existing interface
    that expects x in [0, 1).

    Parameters
    ----------
    x : ndarray, shape (N, d)
        Sample positions in [0, 1)
    y : ndarray, shape (N,)
        Data values at sample points
    grid_shape : tuple of ints
        Output grid shape per dimension
    iflag : int, default=-1
        Sign convention for FFT
    accuracy : int, default=6
        NUFFT accuracy parameter
    apply_deconv : bool, default=True
        Whether to apply deconvolution correction

    Returns
    -------
    y_ft : ndarray
        Fourier transform on uniform grid
    """
    # Convert x from [0, 1) to [-0.5, 0.5] for MATLAB NUFFT
    # MATLAB's scale_knots expects x in some data range, then scales it
    x_scaled = x - 0.5  # Now in [-0.5, 0.5]

    # Convert grid_shape to array
    # grid_shape is the desired OUTPUT size (N in MATLAB)
    N = np.array(grid_shape)

    # Compute df from N: df = 1/N
    # MATLAB will compute N = round(1/df) = grid_shape
    if np.any(N <= 0):
        raise ValueError(f"Grid shape must be positive, got {N}")
    df = 1.0 / N

    # Set Fs = None (will default to N in MATLAB)
    Fs = None

    # Call nufftn_type1 (mirrors MATLAB)
    y_ft = nufftn_type1(
        x_scaled,
        y,
        Fs=Fs,
        df=df,
        iflag=iflag,
        acc=accuracy,
        isdeconv=apply_deconv,
    )

    # Remove extra dimension if present (MATLAB returns (N, 1) for 1D)
    # Only squeeze if the last dimension is 1 (single response)
    if y_ft.ndim > len(grid_shape) and y_ft.shape[-1] == 1:
        y_ft = y_ft.squeeze(axis=-1)

    return y_ft


def nufft_transform(
    x: np.ndarray,
    y: np.ndarray,
    grid_shape: Union[int, Sequence[int]],
    eval_grid_shape: Optional[Union[int, Sequence[int]]] = None,
    qin: Optional[np.ndarray] = None,
    df: float = 1.0,
    accuracy: int = 8,
    iflag: int = -1,
    apply_deconv: Optional[bool] = None,
) -> np.ndarray:
    """
    Non-Uniform Fast Fourier Transform for scattered data.

    Port from fastLPR/utility/core/fastLPR_nufft.m.

    Computes the Fourier transform of data at non-uniform (scattered) sample points.
    This is a key component enabling O(N + M log M) complexity for kernel regression.

    Parameters
    ----------
    x : ndarray, shape (N, d)
        Sample positions in data space
    y : ndarray, shape (N,) or (N, dy)
        Data values at sample points
    grid_shape : int or sequence of ints
        Padded grid shape per dimension (L)
    eval_grid_shape : int or sequence of ints, optional
        Evaluation grid shape per dimension (N). If None, uses grid_shape.
    qin : ndarray, optional
        Padding indices (2 x d). If None, assumes no padding.
    df : float, default=1.0
        Frequency grid spacing
    accuracy : int, default=8
        NUFFT accuracy (number of digits)
    iflag : int, default=-1
        Sign convention for FFT
    apply_deconv : bool, default=True
        Whether to apply deconvolution correction

    Returns
    -------
    y_ft : ndarray
        Fourier transform on uniform grid

    Notes
    -----
    Uses Type-1 NUFFT to transform from non-uniform points to uniform grid.
    The NUFFT uses Gaussian spreading and FFT for fast computation.
    ifftshift is applied to center the zero frequency.

    The knot positions are mapped to the evaluation grid range within the
    padded grid.
    """
    # Set default for apply_deconv based on accuracy
    # Mirrors MATLAB: opt.nufft_deconv = opt.accuracy > 6
    if apply_deconv is None:
        apply_deconv = accuracy > 6

    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=complex))

    if x.ndim == 1:
        x = x[:, None]

    N, dims = x.shape
    grid_shape = np.atleast_1d(np.asarray(grid_shape, dtype=int))

    if grid_shape.size == 1:
        grid_shape = np.repeat(grid_shape, dims)

    # Determine evaluation grid shape
    if eval_grid_shape is None:
        eval_grid_shape = grid_shape
    else:
        eval_grid_shape = np.atleast_1d(np.asarray(eval_grid_shape, dtype=int))
        if eval_grid_shape.size == 1:
            eval_grid_shape = np.repeat(eval_grid_shape, dims)

    # Compute knot positions following MATLAB convention
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    x_scale = x_max - x_min
    x_scale[x_scale == 0] = 1.0  # Avoid division by zero

    # Frequency range for padded grid: [-0.5, 0.5)
    # idx_range = linspace(-1/2, 1/2-1/L, L)
    # MEMORY OPTIMIZATION: Use np.empty since all values will be overwritten
    x_norm = np.empty_like(x)

    for d in range(dims):
        L = grid_shape[d]
        idx_range = np.linspace(-0.5, 0.5 - 1 / L, L)

        if qin is not None and eval_grid_shape[d] != L:
            # With padding: map to evaluation grid range
            # idx_knot = qin[0,d]:qin[1,d]
            idx_knot_start = qin[0, d]
            idx_knot_end = qin[1, d]
            knot_scale = idx_range[idx_knot_end] - idx_range[idx_knot_start]
            x_norm[:, d] = (x[:, d] - x_min[d]) / x_scale[d] * knot_scale + idx_range[
                idx_knot_start
            ]
        else:
            # No padding: map to full range [-0.5, 0.5)
            knot_scale = idx_range[-1] - idx_range[0]
            x_norm[:, d] = (x[:, d] - x_min[d]) / x_scale[d] * knot_scale - 0.5

    # Convert to [0, 1) for NUFFT
    # MEMORY OPTIMIZATION: In-place operations
    x_norm += 0.5
    np.mod(x_norm, 1.0, out=x_norm)

    # Handle multiple response variables
    # OPTIMIZATION: nufft_type1_wrapper already supports multiple responses (M x dy)
    # No need to loop - just pass y directly!
    y_ft = nufft_type1_wrapper(
        x_norm,
        y,
        tuple(grid_shape),
        iflag=iflag,
        accuracy=accuracy,
        apply_deconv=apply_deconv,
    )

    # Apply ifftshift to move DC back to position 1 (standard FFT convention)
    # Output from nufftn_type1 has DC at position M/2+1 (centered)
    # This is REQUIRED because:
    # 1. Kernel (from compute_kernel_fourier) uses plain fft() → DC at position 1
    # 2. Convolution multiplies y_ft * kdf → both must have DC at same position
    # 3. Inverse FFT (ifft) expects DC at position 1
    # The fftshift/ifftshift pair is NOT wasteful - fftshift is needed
    # for deconvolution in nufftn_type1, and ifftshift converts to standard convention
    for i in range(dims):
        y_ft = np.fft.ifftshift(y_ft, axes=i)

    return y_ft


def nufft_transform_cached(
    x: np.ndarray,
    y: np.ndarray,
    grid_shape: Union[int, Sequence[int]],
    eval_grid_shape: Optional[Union[int, Sequence[int]]] = None,
    qin: Optional[np.ndarray] = None,
    df: float = 1.0,
    accuracy: int = 8,
    iflag: int = -1,
    apply_deconv: Optional[bool] = None,
    use_cache: bool = True,
) -> np.ndarray:
    """
    Cached version of nufft_transform.

    This function caches NUFFT results based on input data signatures.
    Useful when the same (x, y) data is transformed multiple times
    with different bandwidths (common in cross-validation).

    Parameters
    ----------
    x : ndarray, shape (N, d)
        Sample positions in data space
    y : ndarray, shape (N,) or (N, dy)
        Data values at sample points
    grid_shape : int or sequence of ints
        Padded grid shape per dimension (L)
    eval_grid_shape : int or sequence of ints, optional
        Evaluation grid shape per dimension (N). If None, uses grid_shape.
    qin : ndarray, optional
        Padding indices (2 x d). If None, assumes no padding.
    df : float, default=1.0
        Frequency grid spacing
    accuracy : int, default=8
        NUFFT accuracy (number of digits)
    iflag : int, default=-1
        Sign convention for FFT
    apply_deconv : bool, default=True
        Whether to apply deconvolution correction
    use_cache : bool, default=True
        Whether to use caching. Set to False for memory-constrained scenarios.

    Returns
    -------
    y_ft : ndarray
        Fourier transform on uniform grid
    """
    if not use_cache:
        return nufft_transform(
            x, y, grid_shape, eval_grid_shape, qin, df, accuracy, iflag, apply_deconv
        )

    # Create cache key from inputs
    grid_shape_arr = np.atleast_1d(np.asarray(grid_shape, dtype=int))
    eval_shape_arr = grid_shape_arr if eval_grid_shape is None else np.atleast_1d(
        np.asarray(eval_grid_shape, dtype=int)
    )

    cache_key = (
        array_key(x),
        array_key(y),
        tuple(grid_shape_arr),
        tuple(eval_shape_arr),
        array_key(qin) if qin is not None else None,
        accuracy,
        iflag,
        apply_deconv,
    )

    def compute_fn():
        return nufft_transform(
            x, y, grid_shape, eval_grid_shape, qin, df, accuracy, iflag, apply_deconv
        )

    return _nufft_cache.get_or_compute(cache_key, compute_fn)


def compute_kernel_fourier(
    x: np.ndarray,
    h: np.ndarray,
    grid_shape: Union[int, Sequence[int]],
    kernel_type: str = "gaussian",
    order: int = 0,
    flag_power2: bool = True,
    accuracy: int = None,
) -> Tuple[Union[np.ndarray, list], dict]:
    """
    Compute kernel density function in Fourier domain.

    Port from fastLPR/utility/core/fastLPR_kdf.m.

    Generates the kernel function in Fourier domain for fast convolution.
    For local polynomial regression, computes design matrix elements.

    Parameters
    ----------
    x : ndarray, shape (N, d)
        Predictor data (z-scored)
    h : ndarray, shape (dh, d)
        Bandwidth candidates
    grid_shape : int or sequence of ints
        Grid size per dimension
    kernel_type : str, default='gaussian'
        Kernel type ('gaussian' or 'epanechnikov')
    order : int, default=0
        Polynomial order (0, 1, or 2)
    flag_power2 : bool, default=True
        Use power-of-2 padding for FFT efficiency
    accuracy : int, default=None
        NUFFT accuracy. If None, auto-adjust based on sample size

    Returns
    -------
    kd : ndarray or list of ndarrays
        Kernel density in Fourier domain
        - Order 0: (L, L, ..., dh) array
        - Order >= 1: List of ns arrays (design matrix elements)
    params : dict
        Dictionary containing:
        - 'L': Padded grid size
        - 'qin': Padding indices
        - 'ihbad': Bad bandwidth flags
        - 'lwp': Local polynomial parameters (if order > 0)

    Notes
    -----
    Algorithm:
    1. Create evaluation grid matching data range
    2. Compute kernel function on grid for each bandwidth
    3. For order >= 1, compute design matrix elements
    4. Transform to Fourier domain via FFT
    5. Remove bandwidths that are too small (kernel sum ≈ 0)
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    h = np.atleast_2d(np.asarray(h, dtype=float))
    grid_shape = np.atleast_1d(np.asarray(grid_shape, dtype=int))

    N, dims = x.shape
    dh, dx = h.shape

    if dx != dims:
        raise ValueError(
            f"Bandwidth dimensions ({dx}) must match data dimensions ({dims})"
        )

    if grid_shape.size == 1:
        grid_shape = np.repeat(grid_shape, dims)

    # Compute padded length L
    if flag_power2:
        # Power-of-2 padding improves FFT speed and accuracy
        L = 2 ** np.ceil(np.log2(2 * grid_shape - 1)).astype(int)
    else:
        L = grid_shape + 2

    # Compute symmetric padding indices
    # qin: used for NUFFT knot mapping (uses floor for symmetric padding)
    # qout: used for extracting evaluation grid from padded result (uses ceil)
    qin = np.zeros((2, dims), dtype=int)
    qout = np.zeros((2, dims), dtype=int)

    pad_left_floor = (L - grid_shape) // 2  # floor division
    qin[0, :] = pad_left_floor
    qin[1, :] = pad_left_floor + grid_shape - 1

    # MATLAB: qout = [ceil((L-N)/2) + 1, ceil((L-N)/2) + N]
    pad_left_ceil = np.ceil((L - grid_shape) / 2).astype(int)
    qout[0, :] = pad_left_ceil
    qout[1, :] = pad_left_ceil + grid_shape - 1

    # Create evaluation grid
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)

    grid_axes = []
    for d in range(dims):
        grid_axes.append(np.linspace(x_min[d], x_max[d], grid_shape[d]))

    # Create meshgrid
    if dims == 1:
        xgrid = grid_axes[0][:, None]
    else:
        mesh = np.meshgrid(*grid_axes, indexing="ij")
        # Use Fortran order (column-major) to match MATLAB's ndgrid + (:) behavior
        # With indexing="ij", first dimension varies fastest logically
        # With ravel('F'), first dimension varies fastest in memory
        xgrid = np.stack([m.ravel("F") for m in mesh], axis=1)
        xgrid = xgrid.reshape(tuple(grid_shape) + (dims,), order="F")

    # Compute kernel function
    kd = kernel_function(xgrid, c=None, h=h, kernel_type=kernel_type)

    # For order >= 1, compute kernels weighted by polynomial terms: K(x) * x^i
    # These kernels are used for both design matrix S and weighted response T
    if order > 0:
        from .lwp_estimator import create_lwp_functions

        # Get LWP functions
        Sxfun, Txfun, mfun, ns, nt = create_lwp_functions(dims, order)

        # Normalize grid by bandwidth: xgrid_norm = xgrid / h
        # For 1D: xgrid is (N, 1), h is (dh, 1)
        # Result: (N, dh)
        if dims == 1:
            # Ensure kd is 2D: (N, dh)
            if kd.ndim == 1:
                kd = kd[:, None]

            # Normalize grid by bandwidth
            # Use SIGNED coordinates for symmetric cancellation
            # in cross terms (e.g., sum(k*x) ≈ 0)
            # xgrid shape: (N, 1), h shape: (dh, 1)
            # Broadcast: (N, 1) / (1, dh) = (N, dh)
            xgrid_norm = xgrid / h.T

            # Ensure xgrid_norm is 2D
            if xgrid_norm.ndim == 1:
                xgrid_norm = xgrid_norm[:, None]

            # Compute polynomial-weighted kernels: K * x^i
            # For Order 1: need K, K*x, K*x^2 (powers 0, 1, 2)
            # For Order 2: need K, K*x, K*x^2, K*x^3, K*x^4 (powers 0, 1, 2, 3, 4)

            # Determine maximum power needed
            # For design matrix S_ij = sum(K * x^i * x^j), max power is i+j
            # For Order 1: max is 1+1=2
            # For Order 2: max is 2+2=4
            max_power = 2 * order

            # Create kernels for each power: K, K*x, K*x^2, ..., K*x^max_power
            # MEMORY OPTIMIZATION: Reuse power computation where possible
            kd_powers = []
            for power in range(max_power + 1):
                # K * x^power for each bandwidth
                if power == 0:
                    kd_power = kd  # K^0 = K, no copy needed for power 0
                else:
                    # MEMORY OPTIMIZATION: Use np.power for clarity
                    kd_power = kd * np.power(xgrid_norm, power)  # (N, dh)
                kd_powers.append(kd_power)

            # Now create design matrix kernels: K * x^i * x^j = K * x^(i+j)
            # For Order 1, 1D: nt=2, ns=3
            # Polynomial terms: [1, x] (powers [0, 1])
            # Design matrix elements (lower triangular):
            #   S11: K * x^0 * x^0 = K * x^0 (power 0)
            #   S21: K * x^1 * x^0 = K * x^1 (power 1)
            #   S22: K * x^1 * x^1 = K * x^2 (power 2)

            # Get polynomial term powers
            if order == 1:
                term_powers = [0, 1]  # [1, x]
            elif order == 2:
                term_powers = [0, 1, 2]  # [1, x, x^2] for 1D
            else:
                raise ValueError(f"Order {order} not supported")

            # Create design matrix kernels
            kd_design = []
            for i, pi in enumerate(term_powers):
                for j in range(i, len(term_powers)):
                    pj = term_powers[j]
                    power_sum = pi + pj
                    kd_design.append(kd_powers[power_sum])

            kd = kd_design
        else:
            # Multi-D case: Use optimized vectorized implementation
            # This is 10-50x faster than the loop-based implementation
            # for typical grid sizes (N=100-1000, dh=10-100)
            from .design_matrix_optimized import compute_design_kernels_vectorized
            kd = compute_design_kernels_vectorized(kd, xgrid, h, order, dims)

    # Check for bad bandwidths (kernel sum ≈ 0)
    # MATLAB: ihbad=squeeze(~abs(sum(kd,(1:dx))));
    # In MATLAB, ~abs(x) returns true ONLY when abs(x) is exactly 0.0
    # However, due to floating-point rounding errors, Python may compute tiny non-zero values
    # (e.g., 2.8e-39 for symmetric cross terms that should be exactly 0.0)
    # MATLAB gets exactly 0.0 due to better numerical cancellation
    #
    # We need to distinguish between:
    # 1. Small but non-zero values due to numerical cancellation (e.g., 1e-18)
    # 2. Essentially zero values due to underflow (e.g., 1e-39)
    #
    # CRITICAL FIX: Match MATLAB's bandwidth filtering behavior
    #
    # MATLAB uses: ihbad = ~abs(sum(kd))
    # This returns true when abs(sum) is exactly 0.0 (within machine epsilon)
    #
    # For bandwidth h=[0.01, 0.01], Python computes kd[4] sum = 2.834068e-39
    # This is NOT exactly zero - it's numerical underflow from symmetric cross terms
    # MATLAB gets exactly 0.0 due to better numerical cancellation, so it filters this bandwidth
    #
    # To match MATLAB, we need to filter bandwidths where ANY design matrix element
    # has a sum that's essentially zero (< 1e-35)
    #
    # Threshold selection:
    # MATLAB uses ~abs(sum) which returns true when sum is EXACTLY 0.0 (underflow)
    # For symmetric grids, cross terms from signed cancellation are ~1e-15 (NORMAL)
    # Only detect TRUE underflow (e.g., bandwidth way too small causing 0.0)
    # Use 10x machine epsilon to catch exact zeros without catching normal cancellation
    dtype = kd.dtype if isinstance(kd, np.ndarray) else kd[0].dtype
    threshold = (
        np.finfo(dtype).eps * 10
    )  # ~2.22e-15 for float64, catches exact zeros only

    if order == 0:
        if dims == 1:
            kernel_sums = np.abs(np.sum(kd, axis=0))
        else:
            kernel_sums = np.abs(np.sum(kd, axis=tuple(range(dims))))

        ihbad = kernel_sums < threshold
    else:
        # For order > 0, check ALL design matrix elements
        # Only check the first element (kernel sum), not cross terms!
        # After removing abs() from coordinate normalization, cross terms (elem[1], etc.)
        # are CORRECTLY near machine epsilon due to symmetric cancellation.
        # Only the kernel sum (elem[0]) should be checked for underflow.
        if dims == 1:
            kernel_sums_0 = np.abs(np.sum(kd[0], axis=0))
        else:
            kernel_sums_0 = np.abs(np.sum(kd[0], axis=tuple(range(dims))))

        ihbad = kernel_sums_0 < threshold

    # Remove bad bandwidths
    if np.any(ihbad) and not np.all(ihbad):
        import warnings
        warnings.warn(
            f"{np.sum(ihbad)} bandwidth(s) are too small for this grid and will be removed",
            stacklevel=2,
        )
        if order == 0:
            kd = kd[..., ~ihbad]
        else:
            kd = [k[..., ~ihbad] for k in kd]
        h = h[~ihbad, :]
        dh = np.sum(~ihbad)
    elif np.all(ihbad):
        # All bandwidths too small, use Silverman's rule
        h_silverman = (4.0 / (dims + 2) / N) ** (1.0 / (dims + 4))
        import warnings
        warnings.warn(
            f"All bandwidths too small. Using Silverman's rule: h = {h_silverman:.4f}",
            stacklevel=2,
        )
        h = np.array([[h_silverman] * dims])
        # Recompute kernel with new bandwidth
        kd_base = kernel_function(xgrid, c=None, h=h, kernel_type=kernel_type)
        if order == 0:
            kd = kd_base
        else:
            # Recompute polynomial kernels
            # (simplified for single bandwidth)
            if dims == 1:
                xgrid_norm = xgrid / h[0, 0]
                X = get_polynomial_terms(xgrid_norm, order)
                kd = []
                for i in range(nt):
                    for j in range(i, nt):
                        s_ij = kd_base[:, 0] * X[:, i] * X[:, j]
                        kd.append(s_ij[:, None])
        ihbad = np.array([False])
        dh = 1

    # Transform to Fourier domain
    # CRITICAL FIX: Do NOT apply conjugation here!
    # MATLAB's get_fourier_transformed_kernel uses plain fft() with NO conjugation
    # The previous np.conj() was causing imaginary part sign flip errors
    if order == 0:
        kd_ft = _transform_kernel_to_fourier(kd, L, qin, dims)
        # kd_ft = np.conj(kd_ft)  # REMOVED: MATLAB doesn't conjugate
    else:
        # Transform each design matrix element
        kd_ft = [_transform_kernel_to_fourier(k, L, qin, dims) for k in kd]  # NO conj!

    # FIX: Squeeze singleton bandwidth dimension when dh=1
    # MATLAB's kdf{i} has shape [L1, L2, ...] (no bandwidth dim when dh=1)
    # Python was keeping shape [L1, L2, ..., 1] which causes broadcasting issues
    if dh == 1:
        if order == 0:
            # Only squeeze if last dimension is actually size 1
            if kd_ft.shape[-1] == 1:
                kd_ft = np.squeeze(kd_ft, axis=-1)
        else:
            # Only squeeze if last dimension is actually size 1
            kd_ft = [np.squeeze(k, axis=-1) if k.shape[-1] == 1 else k for k in kd_ft]

    # Store LWP functions if order > 0
    if order > 0:
        lwp_dict = {
            "Sxfun": Sxfun,
            "Txfun": Txfun,
            "mfun": mfun,
            "ns": ns,
            "nt": nt,
        }
    else:
        lwp_dict = None

    params = {
        "L": L,
        "qin": qin,
        "qout": qout,
        "ihbad": ihbad,
        "h": h,
        "dh": dh,
        "lwp": lwp_dict,
    }

    return kd_ft, params


def compute_kernel_fourier_cached(
    x: np.ndarray,
    h: np.ndarray,
    grid_shape: Union[int, Sequence[int]],
    kernel_type: str = "gaussian",
    order: int = 0,
    flag_power2: bool = True,
    accuracy: int = None,
    use_cache: bool = True,
) -> Tuple[Union[np.ndarray, list], dict]:
    """
    Cached version of compute_kernel_fourier.

    This function caches kernel FFT results based on input parameters.
    Useful when multiple regressions use the same kernel configuration
    (common in cross-validation with the same grid structure).

    Parameters
    ----------
    x : ndarray, shape (N, d)
        Predictor data (z-scored)
    h : ndarray, shape (dh, d)
        Bandwidth candidates
    grid_shape : int or sequence of ints
        Grid size per dimension
    kernel_type : str, default='gaussian'
        Kernel type ('gaussian' or 'epanechnikov')
    order : int, default=0
        Polynomial order (0, 1, or 2)
    flag_power2 : bool, default=True
        Use power-of-2 padding for FFT efficiency
    accuracy : int, default=None
        NUFFT accuracy. If None, auto-adjust based on sample size
    use_cache : bool, default=True
        Whether to use caching. Set to False for memory-constrained scenarios.

    Returns
    -------
    kd_ft : ndarray or list of ndarrays
        Kernel density in Fourier domain
    params : dict
        Dictionary containing parameters

    Notes
    -----
    Cache key is based on:
    - Data signature (shape and hash of x)
    - Bandwidth values
    - Grid shape
    - Kernel type, order, and padding settings

    The cache is particularly effective when:
    - Running cross-validation with many bandwidths
    - Comparing results across different y values with same x
    - Running bootstrap or permutation analyses
    """
    if not use_cache:
        return compute_kernel_fourier(
            x, h, grid_shape, kernel_type, order, flag_power2, accuracy
        )

    # Create cache key
    x = np.atleast_2d(np.asarray(x, dtype=float))
    h = np.atleast_2d(np.asarray(h, dtype=float))
    grid_shape_arr = np.atleast_1d(np.asarray(grid_shape, dtype=int))

    # Use x bounds for cache key (not full data) for efficiency
    # Kernel computation only depends on min/max of x, not individual points
    x_min = tuple(x.min(axis=0))
    x_max = tuple(x.max(axis=0))
    n_samples = x.shape[0]

    cache_key = (
        n_samples,
        x_min,
        x_max,
        array_key(h),
        tuple(grid_shape_arr),
        kernel_type,
        order,
        flag_power2,
        accuracy,
    )

    def compute_fn():
        return compute_kernel_fourier(
            x, h, grid_shape, kernel_type, order, flag_power2, accuracy
        )

    return _kernel_fft_cache.get_or_compute(cache_key, compute_fn)


def _transform_kernel_to_fourier(
    kd: np.ndarray,
    L: np.ndarray,
    qin: np.ndarray,
    dims: int,
) -> np.ndarray:
    """
    Transform kernel to Fourier domain with padding.

    Pads kernel to length L and applies FFT along each dimension.
    Padding avoids circular convolution artifacts.

    The kernel must be shifted so that its center is at index (0, 0, ...)
    before applying FFT. This ensures the FFT output is real (for real kernels)
    and matches MATLAB's behavior.
    """
    # Pad kernel
    # MATLAB's qin is 1-based and used directly as padding amount
    # MATLAB: qin(1,:) = floor((L-N)/2) + 1, then padarray(kd, qin(1,:), 0, 'pre')
    # Python: qin[0,:] = floor((L-N)/2) (0-based index for NUFFT knot mapping)
    # For padding, we need to match MATLAB's padding amount: qin(1,:) = qin[0,:] + 1
    pad_width = []
    for d in range(dims):
        # Compute padding amount to match MATLAB
        # MATLAB pads by qin(1,:) = floor((L-N)/2) + 1
        pad_left = (L[d] - (qin[1, d] - qin[0, d] + 1)) // 2
        pad_before = pad_left + 1  # Match MATLAB's qin(1,:) = pad_left + 1
        pad_after = L[d] - (qin[1, d] - qin[0, d] + 1) - pad_before
        pad_width.append((pad_before, pad_after))

    # Add padding for bandwidth dimension if present
    if kd.ndim > dims:
        pad_width.append((0, 0))

    kd_padded = np.pad(kd, pad_width, mode="constant", constant_values=0)

    # Apply FFT along each spatial dimension
    # Uses plain fft() (NOT fftshift(fft(...))), so DC is at position 0
    # This matches the output from compute_data_fourier (which applies ifftshift)
    # Both kernel and data have DC at position 0 for convolution

    # Use centralized fft_backend for consistent backend selection
    # MEMORY OPTIMIZATION: Use overwrite_x=True for in-place FFT
    # Sequential FFT over each dimension
    for i in range(dims):
        kd_padded = fft_backend.fft(kd_padded, axis=i, overwrite_x=True)

    return kd_padded


def nufft_convolve(
    x: np.ndarray,
    y: np.ndarray,
    kdf: np.ndarray,
    grid_shape: Union[int, Sequence[int]],
    L: np.ndarray,
    qin: np.ndarray,
    qout: np.ndarray = None,
    y_is_transformed: bool = False,
    accuracy: int = None,
    y_is_real: bool = None,
) -> np.ndarray:
    """
    Fast convolution using NUFFT for scattered data.

    Port from fastLPR/utility/core/fastLPR_conv.m.

    Computes convolution of kernel with data using NUFFT (Non-Uniform FFT).
    This is the core operation for local polynomial regression, enabling
    O(N + M log M) complexity instead of naive O(N*M).

    Parameters
    ----------
    x : ndarray, shape (N, d)
        Sample positions
    y : ndarray, shape (N,) or (N, dy)
        Data to convolve (either raw or Fourier-transformed)
    kdf : ndarray
        Kernel density function in Fourier domain
    grid_shape : int or sequence of ints
        Evaluation grid shape
    L : ndarray
        Padded grid size
    qin : ndarray
        Padding indices
    qout : ndarray, optional
        Extraction indices. If None, uses qin.
    y_is_transformed : bool, default=False
        Whether y is already Fourier-transformed
    accuracy : int, default=None
        NUFFT accuracy. If None, auto-adjust based on sample size
    y_is_real : bool, optional
        Whether the ORIGINAL (untransformed) y data is real-valued.
        If None, auto-detects from y (only works if y_is_transformed=False).
        This is needed because when y_is_transformed=True, y is always complex.

    Returns
    -------
    m : ndarray
        Convolution result on evaluation grid

    Notes
    -----
    Algorithm:
    1. Transform y to Fourier domain (if not already)
    2. Multiply by kernel in Fourier domain (convolution theorem)
    3. Inverse FFT to get result in spatial domain
    4. Extract evaluation grid points (remove padding)
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    grid_shape = np.atleast_1d(np.asarray(grid_shape, dtype=int))

    N, dims = x.shape

    # Auto-adjust accuracy based on sample size
    if accuracy is None:
        accuracy = max(6 - int(np.ceil(np.log10(N))), 4)

    if grid_shape.size == 1:
        grid_shape = np.repeat(grid_shape, dims)

    # Auto-detect if original y is real (only works if not transformed)
    if y_is_real is None:
        if y_is_transformed:
            # Cannot auto-detect from transformed data (always complex)
            # Default to False (keep complex) to be safe
            y_is_real = False
        else:
            # Check if original y is real
            y_is_real = np.isrealobj(y)

    # Transform y to Fourier domain if needed
    if not y_is_transformed:
        y_ft = nufft_transform(
            x, y, L, eval_grid_shape=grid_shape, qin=qin, accuracy=accuracy, iflag=-1
        )
    else:
        y_ft = y

    # Convolution in Fourier domain: conv(f, g) = ifft(fft(f) * fft(g))
    # Handle broadcasting for multiple bandwidths and response variables
    # kdf shape: (L[0], L[1], ..., dh) where dh is number of bandwidths
    # y_ft shape: (L[0], L[1], ..., dy) where dy is number of response variables
    # Result shape: (L[0], L[1], ..., dh, dy)

    # MATLAB: m_ft = kdf.*permute(y_ft, [1:dx, dx+2, dx+1])
    # For 1D: kdf (L, dh), y_ft (L, dy, dh) -> permute to (L, dh, dy)
    # But our y_ft is (L, dy), so we need different broadcasting

    # Determine number of bandwidths and responses
    if kdf.ndim == dims:
        dh = 1
        kdf_expanded = kdf[..., None, None]  # (L, 1, 1)
    else:
        dh = kdf.shape[-1]
        kdf_expanded = kdf[..., None]  # (L, dh, 1)

    if y_ft.ndim == dims:
        dy = 1
        y_ft_expanded = y_ft[..., None, None]  # (L, 1, 1)
    else:
        dy = y_ft.shape[-1]
        y_ft_expanded = y_ft[..., None, :]  # (L, 1, dy)

    # Broadcast multiplication: (L, dh, 1) * (L, 1, dy) = (L, dh, dy)
    # MEMORY OPTIMIZATION: Use np.multiply with out parameter when possible
    # to avoid creating a temporary array
    if dh == 1 and dy == 1:
        # Both singleton - can multiply in-place
        np.multiply(kdf_expanded, y_ft_expanded, out=y_ft_expanded)
        m_ft = y_ft_expanded
    else:
        # Need full broadcast - creates new array
        m_ft = kdf_expanded * y_ft_expanded

    # Inverse FFT to spatial domain
    # Use centralized fft_backend for consistent backend selection
    # MEMORY OPTIMIZATION: Use overwrite_x=True for in-place IFFT
    # Sequential IFFT over each dimension
    for i in range(dims):
        m_ft = fft_backend.ifft(m_ft, axis=i, overwrite_x=True)

    # Take real part for real-valued data
    if y_is_real:
        m_ft = np.real(m_ft)

    # Extract evaluation grid (remove padding)
    # Use qout (ceil-based) for extraction, not qin (floor-based)
    # MATLAB: idx_mq = get_patch_index(regs.qout, sz)
    if qout is None:
        qout = qin  # Fallback to qin if qout not provided
    slices = tuple(slice(qout[0, d], qout[1, d] + 1) for d in range(dims))
    m = m_ft[slices]

    return m
