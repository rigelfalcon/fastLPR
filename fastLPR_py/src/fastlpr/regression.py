# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Regression pipeline for fastLPR using NUFFT acceleration.

Port from fastLPR/utility/core/fastLPR_s.m,. fastLPR_y.m, fastLPR_reg.m

This module provides the core regression functions for local polynomial regression
using NUFFT-based convolution for O(N + M log M) complexity.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .convolution import compute_kernel_fourier, compute_kernel_fourier_cached, nufft_convolve, nufft_transform
from .lwp_estimator import create_lwp_functions, get_polynomial_terms, vec2tril

# Import optimized interpolation for DOF estimation hot path
try:
    from .interp_optimized import OptimizedInterpolator, batch_linear_interp, NUMBA_AVAILABLE
    _FAST_INTERP_AVAILABLE = NUMBA_AVAILABLE
except ImportError:
    _FAST_INTERP_AVAILABLE = False


def compute_design_matrix(
    x: np.ndarray,
    h: np.ndarray,
    grid_shape: Union[int, np.ndarray],
    kernel_type: str = "gaussian",
    order: int = 0,
    accuracy: int = None,
    flag_power2: bool = True,
) -> Tuple[Union[np.ndarray, list], dict]:
    """
    Compute design matrix S = X'*W*X using NUFFT.

    Port from fastLPR/utility/core/fastLPR_s.m.

    For order 0 (Nadaraya-Watson), S is a scalar (sum of kernel weights).
    For order >= 1, S is a matrix with elements computed via convolution.

    Parameters
    ----------
    x : ndarray, shape (N, d)
        Predictor data (z-scored)
    h : ndarray, shape (dh, d)
        Bandwidth candidates
    grid_shape : int or ndarray
        Grid size per dimension
    kernel_type : str, default='gaussian'
        Kernel type
    order : int, default=0
        Polynomial order (0, 1, or 2)
    accuracy : int, default=None
        NUFFT accuracy. If None, auto-adjust based on sample size:
        accuracy = max(6 - ceil(log10(n_samples)), 1)

    Returns
    -------
    s : ndarray or list of ndarrays
        Design matrix elements
        - Order 0: scalar (sum of weights)
        - Order >= 1: list of ns elements
    params : dict
        Parameters including kernel in Fourier domain

    Notes
    -----
    Uses NUFFT-based convolution to compute design matrix elements efficiently.
    For order 0: S = sum(K(x_i - x_j))
    For order >= 1: S_ij = sum(K(x_k - x_j) * X_k^i * X_k^j)
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    h = np.atleast_2d(np.asarray(h, dtype=float))
    grid_shape = np.atleast_1d(np.asarray(grid_shape, dtype=int))

    N, dims = x.shape

    # Auto-adjust accuracy based on sample size
    if accuracy is None:
        accuracy = max(6 - int(np.ceil(np.log10(N))), 4)

    # Compute kernel in Fourier domain (with caching for efficiency)
    # Caching is particularly useful when compute_design_matrix is called multiple times
    # with the same parameters (e.g., during DOF estimation where the same KDF is needed)
    kdf, params = compute_kernel_fourier_cached(
        x, h, grid_shape, kernel_type=kernel_type, order=order,
        flag_power2=flag_power2, accuracy=accuracy
    )

    # Create vector of ones for convolution
    ones = np.ones(N, dtype=float)

    if order == 0:
        # Order 0: Nadaraya-Watson estimator
        # S = sum of kernel weights at each evaluation point
        # Add eps for numerical stability (prevents division by zero)
        s = nufft_convolve(
            x,
            ones,
            kdf,
            grid_shape,
            params["L"],
            params["qin"],
            params.get("qout", None),
            y_is_transformed=False,
            accuracy=accuracy,
            y_is_real=True,  # ones is always real
        )
        s = s + np.finfo(float).eps
    else:
        # Order >= 1: Local polynomial regression
        # Compute each element of the design matrix S via convolution
        # Get LWP functions
        Sxfun, Txfun, mfun, ns, nt = create_lwp_functions(dims, order)

        # For order >= 1, kdf is a list of kernels (one for each design matrix element)
        # Each kernel already contains K(x) * x^i * x^j
        # We just need to convolve with ones vector

        # kdf is now a list of Fourier-transformed kernels
        # kdf[i] contains FFT(K(x) * x^i * x^j) for the i-th design matrix element

        s = []
        for i in range(ns):
            # Convolve with ones vector to get S_ij at each grid point
            s_ij = nufft_convolve(
                x,
                ones,
                kdf[i],
                grid_shape,
                params["L"],
                params["qin"],
                params.get("qout", None),
                y_is_transformed=False,
                accuracy=accuracy,
                y_is_real=True,  # ones is always real
            )
            s.append(s_ij)

        # Verify we computed the right number of elements
        assert len(s) == ns, f"Expected {ns} design matrix elements, got {len(s)}"

        params["lwp"] = {
            "Sxfun": Sxfun,
            "Txfun": Txfun,
            "mfun": mfun,
            "ns": ns,
            "nt": nt,
        }

    params["kdf"] = kdf
    return s, params


def solve_regression_system(
    s: Union[np.ndarray, list],
    t: Union[np.ndarray, list],
    order: int = 0,
    regularization: float = 1e-6,
    lwp_params: Optional[dict] = None,
    grid_shape: Optional[Union[int, Sequence[int]]] = None,
) -> np.ndarray:
    """
    Solve regression system m = solve(S, T).

    Port from fastLPR/utility/core/fastLPR_reg.m.

    Computes the regression estimate using precomputed design matrix (S)
    and weighted response (T). For higher-order polynomials, applies
    adaptive regularization to prevent instability in sparse regions.

    Parameters
    ----------
    s : ndarray or list of ndarrays
        Design matrix elements
    t : ndarray or list of ndarrays
        Weighted response
    order : int, default=0
        Polynomial order
    regularization : float, default=1e-6
        Regularization parameter
    lwp_params : dict, optional
        Local polynomial parameters (for order >= 1)
    grid_shape : int or sequence of ints, optional
        Original grid shape for proper reshaping of results

    Returns
    -------
    m : ndarray
        Regression estimate

    Notes
    -----
    For order 0: m = T / S (Nadaraya-Watson)
    For order >= 1: Solve S * beta = T, return beta[0]

    Adaptive regularization is applied based on maximum diagonal element
    to ensure stability without suppressing the fit in high-signal areas.
    """
    if order == 0:
        # Order 0: Nadaraya-Watson estimator
        # m = sum(K * y) / sum(K)

        # Squeeze singleton dimensions to ensure proper broadcasting
        s = np.squeeze(s)
        if hasattr(t, "shape"):
            t = np.squeeze(t)

        # Adaptive threshold for numerical stability
        s_threshold = np.max(s) * regularization

        # Handle broadcasting for multiple responses
        # s shape: (grid_size, dh) or (grid_size,)
        # t shape: (grid_size, dh, dy) or (grid_size, dh) or (grid_size,)
        # Need to broadcast s to match t's shape
        if t.ndim > s.ndim:
            # Add extra dimensions to s for broadcasting
            s_broadcast = s
            for _ in range(t.ndim - s.ndim):
                s_broadcast = s_broadcast[..., np.newaxis]
            m = t / np.maximum(s_broadcast, s_threshold)
        else:
            m = t / np.maximum(s, s_threshold)
    else:
        # Order >= 1: Local polynomial regression
        if lwp_params is None:
            raise ValueError("lwp_params required for order >= 1")

        mfun = lwp_params["mfun"]
        nt = lwp_params["nt"]

        # Apply adaptive regularization to design matrix
        # s_reg{diag_idx} = s{diag_idx} + lambda_fixed
        # where lambda_fixed = alpha * max_diag, alpha = 1e-6

        # Find maximum diagonal element across all grid points
        max_diag = 0.0
        for k in range(nt):
            # Diagonal index in column-major lower triangular storage
            # For a matrix of size nt x nt:
            # - Column j has (nt - j) elements
            # - Diagonal (k,k) is the first element of column k
            # - Index = sum of column lengths before k = k*nt - k*(k-1)/2
            diag_idx = k * nt - k * (k - 1) // 2
            max_diag = max(max_diag, np.max(np.abs(s[diag_idx])))

        # Compute fixed regularization parameter
        # alpha = 1e-6 for consistency with MATLAB
        # The regularization parameter is ignored for order >= 1 (adaptive regularization is used instead)
        alpha = 1e-6
        lambda_fixed = alpha * max_diag

        # Apply regularization to diagonal elements
        # MEMORY OPTIMIZATION: Only copy elements that need modification
        s_reg = list(s)  # Shallow copy - don't copy arrays until needed
        for k in range(nt):
            # Diagonal index in column-major lower triangular storage
            diag_idx = k * nt - k * (k - 1) // 2
            # Only copy and modify diagonal elements (not the entire s list)
            s_reg[diag_idx] = s[diag_idx] + lambda_fixed

        # Solve using symbolic formula (VECTORIZED, like MATLAB)
        # The symbolic formula operates on entire grids at once
        # s_reg and t are lists of arrays, one for each design matrix element

        # Call mfun with regularized design matrix and weighted response
        # This is FULLY VECTORIZED and operates on entire grids at once
        m = mfun(s_reg, t)

    return m


def compute_gcv(
    y: np.ndarray,
    yhat: np.ndarray,
    dof: float,
    metric: str = "gcv",
    dof_stderr: float = 0.0,
    penalty_std: float = 0.0,
    penalty_mean: float = None,
    aggregation: str = "first",
) -> Tuple[float, dict]:
    """
    Compute GCV score for bandwidth selection.

    Port from fastLPR/utility/core/fastLPR_gcv.m. and fastLPR_dof.m

    Generalized Cross-Validation (GCV) is used to select the optimal bandwidth
    by balancing fit quality and model complexity.

    Parameters
    ----------
    y : ndarray, shape (N,) or (N, dy)
        Observed response values. For multi-response (dy > 1), GCV is computed
        for each column and aggregated according to the `aggregation` parameter.
    yhat : ndarray, shape (N,) or (N, dy)
        Fitted values. Must match shape of y.
    dof : float
        Degrees of freedom (trace of smoother matrix)
    metric : str, default='gcv'
        Metric to use: 'gcv', 'aic', or 'bic'
    dof_stderr : float, default=0.0
        Standard error of DOF estimate (from Hutchinson's method)
    penalty_std : float, default=0.0
        Standard deviation of GCV penalty (1/pdof^2) across DOF samples.
        Used for accurate GCV standard error calculation (matches MATLAB).
        If 0, stderr will be 0.
    aggregation : str, default='first'
        How to aggregate GCV across multiple response columns (dy > 1):
        - 'first': Use only first response column (default, for backward compatibility)
        - 'average': Average GCV across all response columns
        - 'sum': Sum GCV across all response columns

    Returns
    -------
    score : float
        GCV score (lower is better)
    info : dict
        Additional information (RSS, effective DOF, stderr, etc.)
        For multi-response, includes 'gcv_per_column' with per-column GCV values.

    Notes
    -----
    GCV formula:
        GCV = MSE / pdof^2
    where:
        pdof = (1/N) * tr(I-H) = 1 - dof/N  (proportion of residual DOF)
        MSE = RSS / N

    Clamping:
        pdof is clamped to minimum 0.01 to prevent penalty explosion
        This gives maximum penalty = 1/0.01^2 = 10,000

    GCV stderr:
        gcv_sd = mse * pdof_inv_sd
    where pdof_inv_sd is the standard deviation of 1/pdof^2 across DOF samples
    """
    # Allow complex-valued responses
    y = np.asarray(y)
    yhat = np.asarray(yhat)

    # Handle 1D arrays (single response) - convert to (N, 1) for uniform processing
    if y.ndim == 1:
        y = y[:, None]
    if yhat.ndim == 1:
        yhat = yhat[:, None]

    N, dy = y.shape
    if yhat.shape[0] != N:
        raise ValueError(
            f"y and yhat must have the same number of samples. Got {N} and {yhat.shape[0]}"
        )
    if yhat.shape[1] != dy:
        raise ValueError(
            f"y and yhat must have the same number of response columns. Got {dy} and {yhat.shape[1]}"
        )

    residuals = y - yhat

    # Compute RSS and MSE per response column
    # For complex, use squared magnitude
    # MEMORY OPTIMIZATION: Compute abs^2 in-place where possible
    residuals_abs = np.abs(residuals)
    np.square(residuals_abs, out=residuals_abs)  # In-place square
    rss_per_col = np.sum(residuals_abs, axis=0)  # Shape: (dy,)
    mse_per_col = rss_per_col / N  # Shape: (dy,)

    if metric == "gcv":
        # Generalized Cross-Validation (from MATLAB)
        # Compute pdof = (1/N) * tr(I-H) = 1 - dof/N
        pdof = 1.0 - dof / N

        # Clamp pdof to [0, 1] for numerical stability
        pdof = np.clip(pdof, 0.0, 1.0)

        # Clamp pdof to minimum 0.01 for GCV penalty
        # This prevents penalty explosion when pdof -> 0 (overfitting)
        pdof_for_penalty = max(pdof, 0.01)

        # Compute GCV penalty: 1/pdof^2
        # If a per-sample mean penalty was supplied (mean(1/pdof^2) across DOF
        # samples), use it to match MATLAB/R. Otherwise fall back to computing
        # the penalty from the mean DOF (1/(1-dof_mean/N)^2).
        if penalty_mean is not None:
            penalty = float(penalty_mean)
        else:
            penalty = 1.0 / (pdof_for_penalty**2)

        # GCV = MSE * penalty - compute per column
        gcv_per_col = mse_per_col * penalty  # Shape: (dy,)

        # Aggregate GCV across response columns
        if aggregation == "first":
            score = float(gcv_per_col[0])
            mse = float(mse_per_col[0])
            rss = float(rss_per_col[0])
        elif aggregation == "average":
            score = float(np.mean(gcv_per_col))
            mse = float(np.mean(mse_per_col))
            rss = float(np.mean(rss_per_col))
        elif aggregation == "sum":
            score = float(np.sum(gcv_per_col))
            mse = float(np.sum(mse_per_col))
            rss = float(np.sum(rss_per_col))
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}. Use 'first', 'average', or 'sum'.")

        # Compute GCV stderr directly from penalty std
        # For multi-response, use the aggregated MSE
        if penalty_std > 0:
            stderr = mse * penalty_std  # DIRECT calculation (from MATLAB!)
        else:
            stderr = 0.0
    elif metric == "aic":
        # Akaike Information Criterion - aggregate across columns
        if aggregation == "first":
            rss = float(rss_per_col[0])
        elif aggregation == "average":
            rss = float(np.mean(rss_per_col))
        else:
            rss = float(np.sum(rss_per_col))
        mse = rss / N
        score = N * np.log(rss / N) + 2 * dof
        stderr = 0.0
        gcv_per_col = None
        pdof = None
        pdof_for_penalty = None
        penalty = None
    elif metric == "bic":
        # Bayesian Information Criterion - aggregate across columns
        if aggregation == "first":
            rss = float(rss_per_col[0])
        elif aggregation == "average":
            rss = float(np.mean(rss_per_col))
        else:
            rss = float(np.sum(rss_per_col))
        mse = rss / N
        score = N * np.log(rss / N) + np.log(N) * dof
        stderr = 0.0
        gcv_per_col = None
        pdof = None
        pdof_for_penalty = None
        penalty = None
    else:
        raise ValueError(f"Unknown metric: {metric}")

    info = {
        "rss": rss,
        "rss_per_column": rss_per_col,
        "mse": mse,
        "mse_per_column": mse_per_col,
        "dof": dof,
        "pdof": pdof if metric == "gcv" else None,
        "pdof_for_penalty": pdof_for_penalty if metric == "gcv" else None,
        "penalty": penalty if metric == "gcv" else None,
        "gcv_per_column": gcv_per_col if metric == "gcv" else None,
        "n": N,
        "dy": dy,
        "residuals": residuals,
        "stderr": stderr,
        "aggregation": aggregation,
    }

    return score, info


def compact_results(results: dict) -> dict:
    """
    Remove intermediate variables to save memory.

    Port from fastLPR/utility/core/fastLPR_compact.m.

    Removes large intermediate arrays that are no longer needed
    after regression is complete.

    Parameters
    ----------
    results : dict
        Results dictionary

    Returns
    -------
    results : dict
        Compacted results dictionary
    """
    # Remove intermediate variables
    keys_to_remove = ["kdf", "s", "t", "y_ft", "m_ft"]

    for key in keys_to_remove:
        if key in results:
            del results[key]

    return results


class ComplexInterpolator:
    """
    Wrapper for RegularGridInterpolator that handles complex values
    and implements MATLAB-style linear extrapolation.

    .. deprecated:: 1.1.0
        Use :class:`OptimizedInterpolator` from `interp_optimized.py` instead.
        OptimizedInterpolator provides 3-7x speedup with Numba acceleration
        and automatic fallback to scipy when needed.

    Scipy's RegularGridInterpolator doesn't support complex values or
    proper linear extrapolation, so we:
    1. Interpolate real and imaginary parts separately
    2. Use linear extrapolation for out-of-bounds points (like MATLAB)

    Port from fastLPR/utility/core/fastLPR_gridinterp.m which uses:
        opt.Method = 'spline';
        opt.ExtrapolationMethod = 'linear';
    """

    def __init__(self, grid_axes, values, method="cubic"):
        """
        Initialize ComplexInterpolator.

        .. deprecated:: 1.1.0
            Use OptimizedInterpolator instead for better performance.

        Parameters
        ----------
        grid_axes : sequence of ndarrays
            Grid axes for each dimension
        values : ndarray
            Values on the grid (can be complex)
        method : str, default='cubic'
            Interpolation method for RegularGridInterpolator.
            Default is 'cubic' to match MATLAB's griddedInterpolant 'spline' method.
        """
        warnings.warn(
            "ComplexInterpolator is deprecated since v1.1.0. "
            "Use OptimizedInterpolator from interp_optimized.py instead for 3-7x speedup.",
            DeprecationWarning,
            stacklevel=2
        )
        self.grid_axes = tuple(np.asarray(ax) for ax in grid_axes)
        self.bounds = [(ax.min(), ax.max()) for ax in self.grid_axes]

        if np.iscomplexobj(values):
            self.is_complex = True
            self.interp_real = RegularGridInterpolator(
                grid_axes,
                values.real,
                method=method,
                bounds_error=False,
                fill_value=None,  # Allow extrapolation for interior calculation
            )
            self.interp_imag = RegularGridInterpolator(
                grid_axes,
                values.imag,
                method=method,
                bounds_error=False,
                fill_value=None,
            )
        else:
            self.is_complex = False
            self.interp_real = RegularGridInterpolator(
                grid_axes, values, method=method, bounds_error=False, fill_value=None
            )

    def _linear_extrapolate(self, points, interp):
        """
        Implement MATLAB-style linear extrapolation for out-of-bounds points.

        For points outside the grid, use linear extrapolation based on the
        gradient at the boundary, matching MATLAB's 'linear' ExtrapolationMethod.

        Optimized: Uses vectorized operations and batched interp() calls for performance.
        """
        points = np.atleast_2d(points)
        n_points, n_dims = points.shape

        # Build bounds arrays for vectorized operations
        bounds_lower = np.array([self.bounds[d][0] for d in range(n_dims)])
        bounds_upper = np.array([self.bounds[d][1] for d in range(n_dims)])

        # Vectorized bounds check (all dimensions at once)
        out_of_bounds = np.any(
            (points < bounds_lower) | (points > bounds_upper), axis=1
        )

        # If all points are in bounds, just use the interpolator directly
        if not np.any(out_of_bounds):
            return interp(points)

        # Initialize result array
        result = np.zeros(n_points)

        # Evaluate in-bounds points directly (batch call)
        in_bounds = ~out_of_bounds
        if np.any(in_bounds):
            result[in_bounds] = interp(points[in_bounds])

        # Handle out-of-bounds points with vectorized linear extrapolation
        if np.any(out_of_bounds):
            oob_points = points[out_of_bounds]
            n_oob = oob_points.shape[0]

            # Vectorized clipping to boundary (all points, all dims at once)
            oob_clipped = np.clip(oob_points, bounds_lower, bounds_upper)

            # Batch evaluation at boundary points
            val_at_boundary = interp(oob_clipped)

            # Compute step sizes for each dimension
            h = (bounds_upper - bounds_lower) / 100.0

            # Vectorized gradient computation
            # For each dimension, compute gradient using appropriate difference method
            gradients = np.zeros((n_oob, n_dims))

            for d in range(n_dims):  # Only loop over dimensions (typically 1-3)
                # Determine which points are at lower/upper boundary for this dim
                at_lower = oob_clipped[:, d] <= bounds_lower[d] + h[d]
                at_upper = oob_clipped[:, d] >= bounds_upper[d] - h[d]
                at_interior = ~at_lower & ~at_upper

                # Forward difference for lower boundary points
                if np.any(at_lower):
                    pt_plus = oob_clipped[at_lower].copy()
                    pt_plus[:, d] = np.minimum(pt_plus[:, d] + h[d], bounds_upper[d])
                    val_plus = interp(pt_plus)
                    gradients[at_lower, d] = (val_plus - val_at_boundary[at_lower]) / h[d]

                # Backward difference for upper boundary points
                if np.any(at_upper):
                    pt_minus = oob_clipped[at_upper].copy()
                    pt_minus[:, d] = np.maximum(pt_minus[:, d] - h[d], bounds_lower[d])
                    val_minus = interp(pt_minus)
                    gradients[at_upper, d] = (val_at_boundary[at_upper] - val_minus) / h[d]

                # Central difference for interior points (rare case)
                if np.any(at_interior):
                    pt_plus = oob_clipped[at_interior].copy()
                    pt_minus = oob_clipped[at_interior].copy()
                    pt_plus[:, d] = pt_plus[:, d] + h[d]
                    pt_minus[:, d] = pt_minus[:, d] - h[d]
                    val_plus = interp(pt_plus)
                    val_minus = interp(pt_minus)
                    gradients[at_interior, d] = (val_plus - val_minus) / (2 * h[d])

            # Vectorized linear extrapolation: f(x) = f(x_boundary) + gradient · (x - x_boundary)
            displacement = oob_points - oob_clipped
            oob_result = val_at_boundary + np.sum(gradients * displacement, axis=1)

            result[out_of_bounds] = oob_result

        return result

    def __call__(self, points):
        """Evaluate interpolator at points with MATLAB-style linear extrapolation."""
        points = np.atleast_2d(points)

        if self.is_complex:
            real_part = self._linear_extrapolate(points, self.interp_real)
            imag_part = self._linear_extrapolate(points, self.interp_imag)
            return real_part + 1j * imag_part
        else:
            return self._linear_extrapolate(points, self.interp_real)


def create_interpolator(
    grid_axes: Tuple[np.ndarray, ...],
    values: np.ndarray,
    method: str = "linear",  # Use linear for Numba acceleration (7x speedup)
):
    """
    Create interpolator for regression results.

    Port from fastLPR/utility/core/fastLPR_gridinterp.m.

    Creates an interpolator for efficient evaluation at arbitrary points.
    Handles both real and complex-valued data.

    Uses linear interpolation by default to leverage Numba acceleration.
    For the typical grid densities used in fastLPR (M=100-256 per dimension),
    linear interpolation provides sufficient accuracy with much better speed.

    Parameters
    ----------
    grid_axes : tuple of ndarrays
        Grid axes for each dimension
    values : ndarray
        Values on the grid (can be complex)
    method : str, default='linear'
        Interpolation method ('linear', 'nearest', 'cubic')
        'linear' uses Numba (fast), 'cubic' falls back to scipy (slow)

    Returns
    -------
    interpolator : OptimizedInterpolator
        Numba-accelerated interpolator object

    Notes
    -----
    **Performance vs Accuracy Tradeoff (2025-12-26):**
    - MATLAB uses 'spline' (cubic) interpolation in fastLPR_gridinterp.m
    - Python defaults to 'linear' for 3-5x speedup in CV loops
    - This may cause small accuracy differences in cross-validation tests
    - For exact MATLAB match, use: method='cubic' (but 7x slower)
    - For production use, 'linear' is recommended (accuracy loss < 1e-3 typical)
    """
    return OptimizedInterpolator(grid_axes, values, method=method)


def _compute_regression_with_cached_S(
    x: np.ndarray,
    y: np.ndarray,
    h: np.ndarray,
    grid_shape: np.ndarray,
    S: Union[np.ndarray, list],
    params: dict,
    order: int,
    regularization: float,
) -> np.ndarray:
    """
    Compute regression using pre-computed design matrix S.

    This is an optimization for DOF estimation where S is computed once
    and reused for all random samples (since S depends only on x and h, not y).

    Parameters
    ----------
    x : ndarray, shape (N, d)
        Predictor data
    y : ndarray, shape (N,) or (N, dy)
        Response data (can be multiple responses)
    h : ndarray, shape (dh, d)
        Bandwidth candidates
    grid_shape : ndarray
        Grid size per dimension
    S : ndarray or list
        Pre-computed design matrix
    params : dict
        Pre-computed parameters from design matrix computation
    order : int
        Polynomial order
    regularization : float
        Regularization parameter

    Returns
    -------
    m : ndarray
        Regression estimates on grid
    """
    # Auto-adjust accuracy based on sample size
    N = len(x)
    accuracy = max(6 - int(np.ceil(np.log10(N))), 4)

    # Step 1: Compute weighted response T = X' * W * y
    # (S is already computed and cached)
    # Check if y is real-valued (needed for proper output type)
    y_is_real = np.isrealobj(y)

    if order == 0:
        # Order 0: T = sum(K * y)
        t = nufft_convolve(
            x,
            y,
            params["kdf"],
            grid_shape,
            params["L"],
            params["qin"],
            params.get("qout", None),
            y_is_transformed=False,
            accuracy=accuracy,
            y_is_real=y_is_real,
        )
        t = np.squeeze(t)
    else:
        # Order >= 1: T_i = sum(K * X^i * y) for each polynomial term
        lwp_params = params.get("lwp", None)
        if lwp_params is None:
            raise ValueError("LWP parameters not found in cached params")

        nt = lwp_params["nt"]
        kdf = params["kdf"]

        # OPTIMIZATION: Transform y to Fourier domain ONCE (expensive part!)
        # Then reuse y_ft for all nt convolutions (cheap - just multiplication + IFFT)
        y_ft = nufft_transform(
            x,
            y,
            params["L"],
            eval_grid_shape=grid_shape,
            qin=params["qin"],
            accuracy=accuracy,
            iflag=-1,
        )

        # Now convolve with each kernel (cheap - just multiplication + IFFT)
        t = []
        for i in range(nt):
            t_i = nufft_convolve(
                x,
                y_ft,  # Use pre-transformed y_ft!
                kdf[i],
                grid_shape,
                params["L"],
                params["qin"],
                params.get("qout", None),
                y_is_transformed=True,  # y_ft is already transformed
                accuracy=accuracy,
                y_is_real=y_is_real,  # Pass original y's realness
            )
            t_i = np.squeeze(t_i)
            t.append(t_i)

        assert len(t) == nt, f"Expected {nt} weighted response elements, got {len(t)}"

    # Step 2: Solve regression system using cached S
    lwp_params = params.get("lwp", None)
    m = solve_regression_system(
        S,
        t,
        order=order,
        regularization=regularization,
        lwp_params=lwp_params,
        grid_shape=grid_shape,
    )

    return m


def estimate_dof_nufft(
    points_scaled: np.ndarray,
    points_raw: np.ndarray,
    bandwidth: np.ndarray,
    grid_shape: np.ndarray,
    grid_axes_raw: Tuple[np.ndarray, ...],
    order: int,
    regularization: float,
    num_samples: int = 10,
    random_state: int = 42,
) -> Tuple[float, float]:
    """
    Estimate degrees of freedom using NUFFT-based regression (single bandwidth).

    Port from fastLPR/utility/core/fastLPR_dof.m.

    Uses Hutchinson's method to estimate tr(H) where H is the smoother matrix.

    Algorithm:
        1. Generate random vectors: p ~ N(0,I)
        2. Compute regression on grid: mq = fastLPR_reg(regs, p)
        3. Interpolate back to data points: phat = fastLPR_gridinterp(regs, mq)
        4. Estimate trace: tr(H) ~= N * (p'*phat) / (p'*p)

    Parameters
    ----------
    points_scaled : ndarray
        Data points in scaled space (for NUFFT regression)
    points_raw : ndarray
        Data points in raw/original space (for interpolation)
    bandwidth : ndarray
        Bandwidth (1D array)
    grid_shape : ndarray
        Grid shape
    grid_axes_raw : tuple of ndarrays
        Grid axes in raw/original space (for interpolation)
    order : int
        Polynomial order
    regularization : float
        Regularization parameter
    num_samples : int
        Number of random samples for Hutchinson estimator
    random_state : int
        Random seed

    Returns
    -------
    dof : float
        Estimated degrees of freedom (tr(H))
    dof_stderr : float
        Standard error of DOF estimate
    """
    points_scaled = np.asarray(points_scaled, dtype=float)
    points_raw = np.asarray(points_raw, dtype=float)
    bandwidth = np.asarray(bandwidth, dtype=float)
    # Use MT19937 (Mersenne Twister) to match MATLAB's default RNG algorithm
    rng = np.random.Generator(np.random.MT19937(random_state))
    n = points_scaled.shape[0]

    estimates = []
    for _ in range(max(1, num_samples)):
        z = rng.standard_normal(n)

        # Compute regression on grid (from MATLAB fastLPR_reg)
        # Use scaled points for NUFFT regression
        m_grid, _ = nufft_regression(
            points_scaled,
            z,
            bandwidth.reshape(1, -1),
            grid_shape,
            order=order,
            regularization=regularization,
        )

        # Interpolate back to data points (from MATLAB fastLPR_gridinterp)
        # MATLAB creates interpolator with xlist (raw space) and evaluates at xraw (raw space)
        interpolator = create_interpolator(grid_axes_raw, m_grid)
        z_hat = interpolator(points_raw)

        # Compute tr(H) using Hutchinson's method
        z_dot_z = np.dot(z, z)
        z_dot_zhat = np.dot(z, z_hat)
        trace_H = n * (z_dot_zhat / z_dot_z)
        estimates.append(trace_H)

    estimates = np.asarray(estimates, dtype=float)
    mean_trace = float(np.mean(estimates))
    if estimates.size > 1:
        stderr = float(np.std(estimates, ddof=1) / np.sqrt(estimates.size))
    else:
        stderr = 0.0
    return mean_trace, stderr


def _interpolate_with_linear_extrapolation(
    grid_axes: Tuple[np.ndarray, ...],
    values: np.ndarray,
    points: np.ndarray,
    method: str = "cubic",
) -> np.ndarray:
    """
    Interpolate grid values at points with MATLAB-style linear extrapolation.

    This function matches MATLAB's griddedInterpolant behavior with:
    - 'spline' interpolation method (cubic in scipy)
    - 'linear' extrapolation method for out-of-bounds points

    Port from fastLPR_gridinterp.m.

    Parameters
    ----------
    grid_axes : tuple of ndarrays
        Grid axes for each spatial dimension
    values : ndarray, shape (n1, n2, ..., nd, *extra_dims)
        Values at grid points. Can have extra dimensions beyond spatial dims.
    points : ndarray, shape (n_points, n_dims)
        Query points
    method : str, default='cubic'
        Interpolation method ('linear' or 'cubic')
        'cubic' matches MATLAB's 'spline' for regular grids

    Returns
    -------
    result : ndarray, shape (n_points, *extra_dims)
        Interpolated values at query points

    Notes
    -----
    MATLAB's griddedInterpolant with 'linear' extrapolation:
    - For points inside grid bounds: uses the specified interpolation method
    - For points outside grid bounds: uses linear extrapolation based on
      gradient at the nearest boundary point

    SciPy's RegularGridInterpolator with fill_value=None uses nearest-value
    extrapolation (constant), not linear extrapolation. This function fixes
    that by implementing proper linear extrapolation.

    Performance:
    - When method='linear' and Numba is available, uses 7x faster Numba interpolation
    - Numba path uses boundary clamping (no extrapolation needed for DOF estimation)
    """
    points = np.atleast_2d(points)
    n_points, n_dims = points.shape

    # Build bounds arrays
    bounds_lower = np.array([ax.min() for ax in grid_axes])
    bounds_upper = np.array([ax.max() for ax in grid_axes])

    # Check which points are out of bounds
    out_of_bounds = np.any(
        (points < bounds_lower) | (points > bounds_upper), axis=1
    )

    # OPTIMIZATION: Use Numba-accelerated interpolation for linear method (7x speedup)
    # This is safe because DOF estimation points are typically within grid bounds
    # (they are the same data points used to create the grid)
    if method == "linear" and _FAST_INTERP_AVAILABLE and not np.any(out_of_bounds):
        # batch_linear_interp expects shape (grid_shape..., n_batches)
        # But DOF estimation may pass (grid_shape..., dh, num_samples)
        # Reshape to merge extra dimensions into one batch dimension
        grid_shape = tuple(len(ax) for ax in grid_axes)
        spatial_dims = len(grid_axes)

        if values.ndim > spatial_dims + 1:
            # Multiple trailing dimensions (e.g., dh, num_samples)
            # Reshape to (grid_shape..., dh*num_samples)
            extra_shape = values.shape[spatial_dims:]
            n_extra = int(np.prod(extra_shape))
            values_flat = values.reshape(grid_shape + (n_extra,))

            # Interpolate with flattened batches
            result_flat = batch_linear_interp(grid_axes, values_flat, points, use_numba=True)

            # Reshape result back to (n_points, dh, num_samples)
            return result_flat.reshape((n_points,) + extra_shape)
        else:
            # Standard case: (grid_shape..., n_batches) or (grid_shape...)
            return batch_linear_interp(grid_axes, values, points, use_numba=True)

    # Create scipy interpolator (fallback for cubic or out-of-bounds cases)
    interp = RegularGridInterpolator(
        grid_axes, values, method=method, bounds_error=False, fill_value=None
    )

    # If all points are in bounds, use scipy directly
    if not np.any(out_of_bounds):
        return interp(points)

    # Get result shape (depends on whether values has extra dimensions)
    test_result = interp(points[:1])  # Evaluate at one point to get shape
    if test_result.ndim > 1:
        extra_shape = test_result.shape[1:]
        result = np.zeros((n_points,) + extra_shape, dtype=values.dtype)
    else:
        extra_shape = ()
        result = np.zeros(n_points, dtype=values.dtype)

    # Evaluate in-bounds points directly
    in_bounds = ~out_of_bounds
    if np.any(in_bounds):
        result[in_bounds] = interp(points[in_bounds])

    # Handle out-of-bounds points with linear extrapolation
    if np.any(out_of_bounds):
        oob_points = points[out_of_bounds]
        n_oob = oob_points.shape[0]

        # Clip to boundary
        oob_clipped = np.clip(oob_points, bounds_lower, bounds_upper)

        # Evaluate at boundary
        val_at_boundary = interp(oob_clipped)

        # Compute step sizes for gradient
        h = (bounds_upper - bounds_lower) / 100.0
        h = np.maximum(h, 1e-10)  # Prevent division by zero

        # Compute gradient at each boundary point
        # Initialize gradient array with proper shape
        if extra_shape:
            gradient_sum = np.zeros((n_oob,) + extra_shape, dtype=values.dtype)
        else:
            gradient_sum = np.zeros(n_oob, dtype=values.dtype)

        for d in range(n_dims):
            # Determine direction of extrapolation for this dimension
            below_lower = oob_points[:, d] < bounds_lower[d]
            above_upper = oob_points[:, d] > bounds_upper[d]

            # Compute displacement for this dimension
            displacement_d = oob_points[:, d] - oob_clipped[:, d]

            if np.any(below_lower):
                # Forward difference at lower boundary
                pt_plus = oob_clipped[below_lower].copy()
                pt_plus[:, d] = np.minimum(pt_plus[:, d] + h[d], bounds_upper[d])
                val_plus = interp(pt_plus)
                grad = (val_plus - val_at_boundary[below_lower]) / h[d]
                if extra_shape:
                    gradient_sum[below_lower] += grad * displacement_d[below_lower, np.newaxis]
                else:
                    gradient_sum[below_lower] += grad * displacement_d[below_lower]

            if np.any(above_upper):
                # Backward difference at upper boundary
                pt_minus = oob_clipped[above_upper].copy()
                pt_minus[:, d] = np.maximum(pt_minus[:, d] - h[d], bounds_lower[d])
                val_minus = interp(pt_minus)
                grad = (val_at_boundary[above_upper] - val_minus) / h[d]
                if extra_shape:
                    gradient_sum[above_upper] += grad * displacement_d[above_upper, np.newaxis]
                else:
                    gradient_sum[above_upper] += grad * displacement_d[above_upper]

        # Linear extrapolation: f(x) = f(x_boundary) + gradient · (x - x_boundary)
        result[out_of_bounds] = val_at_boundary + gradient_sum

    return result


def estimate_dof_nufft_vectorized(
    points_scaled: np.ndarray,
    points_for_interp: np.ndarray,
    bandwidths: np.ndarray,
    grid_shape: np.ndarray,
    grid_axes_for_interp: Tuple[np.ndarray, ...],
    order: int,
    regularization: float,
    num_samples: int = 10,
    random_state: int = 42,
    random_matrix: np.ndarray = None,
    S_precomputed: Union[np.ndarray, list] = None,
    params_precomputed: dict = None,
    interp_method: str = "cubic",
    y_is_complex: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate degrees of freedom for multiple bandwidths (vectorized).

    Port from fastLPR/utility/core/fastLPR_dof.m. (vectorized version)

    This is the OPTIMIZED version that processes all bandwidths at once,
    matching MATLAB's vectorization strategy.

    Algorithm:
        1. Generate random vectors: p ~ N(0,I) with shape (n, num_samples)
        2. Compute design matrix S for ALL bandwidths at once (CACHED!)
        3. For each sample, compute regression for all bandwidths
        4. Interpolate back to data points: phat shape (n, dh, num_samples)
        5. Estimate trace for each bandwidth: tr(H) ~= N * (p'*phat) / (p'*p)

    Parameters
    ----------
    points_scaled : ndarray, shape (n, dims)
        Data points in scaled space (for NUFFT regression)
    points_for_interp : ndarray, shape (n, dims)
        Data points in SCALED space (for interpolation back to data points)
        Must be in the same space as grid_axes_for_interp!
    bandwidths : ndarray, shape (dh, dims)
        Multiple bandwidths to evaluate
    grid_shape : ndarray
        Grid shape
    grid_axes_for_interp : tuple of ndarrays
        Grid axes in SCALED space (for interpolation)
        Must be in the same space as points_for_interp!
    order : int
        Polynomial order
    regularization : float
        Regularization parameter
    num_samples : int
        Number of random samples for Hutchinson estimator
    random_state : int or None
        Random seed. If None and random_matrix is None, uses default seed.
    random_matrix : ndarray, shape (n, num_samples), optional
        Pre-generated random matrix for exact verification with MATLAB.
        If provided, this overrides random_state and num_samples.
    S_precomputed : ndarray or list, optional
        Pre-computed design matrix S. If provided, skips design matrix computation.
        This is a MAJOR optimization since S doesn't depend on y (random samples).
    params_precomputed : dict, optional
        Pre-computed parameters from design matrix computation.
        Required if S_precomputed is provided.
    interp_method : str, optional
        Interpolation method: 'linear' or 'cubic' (default: 'cubic')
        - 'cubic': Matches MATLAB's spline interpolation exactly
        - 'linear': Faster; sufficient for 1D but may shift 2D+ bandwidth selection

    Returns
    -------
    dof : ndarray, shape (dh,)
        Estimated degrees of freedom for each bandwidth
    dof_stderr : ndarray, shape (dh,)
        Standard error of DOF estimate for each bandwidth
    penalty_std : ndarray, shape (dh,)
        Standard deviation of GCV penalty (1/pdof^2) across DOF samples.
        Used for accurate GCV standard error in 1-SE rule (matches MATLAB fastLPR_dof.m).
    """
    points_scaled = np.asarray(points_scaled, dtype=float)
    points_for_interp = np.asarray(points_for_interp, dtype=float)
    bandwidths = np.atleast_2d(np.asarray(bandwidths, dtype=float))

    n = points_scaled.shape[0]
    dh = bandwidths.shape[0]
    grid_size = np.prod(grid_shape)

    # Generate random vectors: p ~ N(0,I) with shape (n, num_samples)
    # MATLAB: p=randn(regs.Ty,regs.opt.num_dof_sample) where Ty = number of data points
    if random_matrix is not None:
        # Use pre-generated random matrix for exact verification with MATLAB
        p = np.asarray(random_matrix, dtype=float)
        if p.shape[0] != n:
            raise ValueError(
                f"random_matrix has wrong shape: expected ({n}, *), got {p.shape}"
            )
        num_samples = p.shape[1]  # Override num_samples
    else:
        # Generate random vectors using Python RNG
        # Use MT19937 (Mersenne Twister) to match MATLAB's default RNG algorithm
        rng = np.random.Generator(np.random.MT19937(random_state))
        p = rng.standard_normal((n, num_samples))

    # OPTIMIZATION: Use cached design matrix if provided
    # Design matrix S depends only on x and h, NOT on y (random samples).
    # Skipping re-computation saves one full S build per DoF call.
    if S_precomputed is not None and params_precomputed is not None:
        # Use cached design matrix and params
        S = S_precomputed
        params = params_precomputed

        # Use the filtered bandwidths from the design matrix (params["h"])
        # Original bandwidths may have been filtered during design matrix computation
        h_filtered = params.get("h", bandwidths)
        dh = params.get("dh", h_filtered.shape[0])

        # Compute only the weighted response T for random samples
        # This is much faster than full nufft_regression
        m_all_samples = _compute_regression_with_cached_S(
            points_scaled,
            p,  # Shape: (n, num_samples)
            h_filtered,  # Use filtered bandwidths!
            grid_shape,
            S,
            params,
            order=order,
            regularization=regularization,
        )
    else:
        # Run full NUFFT regression (computes both S and T)
        # MATLAB: mq = fastLPR_reg(regs, p) where p has shape (Ty, num_samples)
        # The symbolic formulas support multiple responses via NumPy broadcasting!
        # Process all samples at once for ALL orders (not just order=0)
        m_all_samples, _ = nufft_regression(
            points_scaled,
            p,  # Shape: (n, num_samples)
            bandwidths,
            grid_shape,
            order=order,
            regularization=regularization,
        )

    # m_all_samples shape: (grid_shape..., dh, num_samples) or squeezed variants
    m_all_samples = np.atleast_1d(m_all_samples)

    # Determine expected shape and reshape to (grid_size, dh, num_samples)
    if m_all_samples.shape == tuple(grid_shape) + (dh, num_samples):
        m_grid = m_all_samples.reshape(grid_size, dh, num_samples)
    elif m_all_samples.shape == tuple(grid_shape) + (num_samples,):
        m_grid = m_all_samples.reshape(grid_size, 1, num_samples)
    elif m_all_samples.shape == (grid_size, dh, num_samples):
        m_grid = m_all_samples
    elif m_all_samples.shape == (grid_size, num_samples):
        m_grid = m_all_samples[:, np.newaxis, :]
    else:
        m_grid = m_all_samples.reshape(grid_size, dh, num_samples)

    # Reshape m_grid from (grid_size, dh, num_samples) to (grid_shape..., dh, num_samples)
    # This allows the interpolator to handle all dimensions efficiently in one call
    m_grid_shaped = m_grid.reshape(tuple(grid_shape) + (dh, num_samples))

    # Use MATLAB-compatible interpolation with linear extrapolation
    # This is critical for accurate DOF estimation when data points are near grid boundaries
    phat_all = _interpolate_with_linear_extrapolation(
        grid_axes_for_interp,
        m_grid_shaped,
        points_for_interp,
        method=interp_method,  # 'cubic' matches MATLAB's 'spline', 'linear' is faster
    )

    # Reshape to (n, num_samples, dh) to match expected shape
    # phat_all shape: (n, dh, num_samples) -> (n, num_samples, dh)
    phat = np.transpose(phat_all, (0, 2, 1))
    phat = np.real(phat)  # Take real part to match MATLAB

    # Compute tr(H) using Hutchinson's method for each bandwidth
    # MATLAB: pdof = (sum(p.*(p-phat))./sum(p.*p))
    # p shape: (n, num_samples)
    # phat shape: (n, num_samples, dh)
    # Need to broadcast p to (n, num_samples, 1) for element-wise operations
    p_expanded = p[:, :, np.newaxis]  # Shape: (n, num_samples, 1)

    # Compute dot products
    p_dot_p = np.sum(p * p, axis=0)  # Shape: (num_samples,)
    p_dot_phat = np.sum(p_expanded * phat, axis=0)  # Shape: (num_samples, dh)

    # Estimate tr(H) for each bandwidth and sample
    # MATLAB: pdof = (sum(p.*(p-phat))./sum(p.*p))
    # But we want tr(H), not pdof = tr(I-H)/n
    # tr(H) = n * (1 - pdof) = n * (p'*phat) / (p'*p)
    trace_H = n * (p_dot_phat / p_dot_p[:, np.newaxis])  # Shape: (num_samples, dh)

    # Handle NaN values (from regression failures with very small bandwidths)
    # Replace NaN with n (full DOF, no smoothing) to match MATLAB's behavior
    # MATLAB gets very small DOF (~0) for overfitting bandwidths, which corresponds to pdof~1
    # When regression fails (NaN), treat as no smoothing (DOF=n, pdof=0)
    #  GCV will be very large (bad) for these bandwidths
    trace_H = np.where(np.isnan(trace_H), n, trace_H)

    # Clamp trace_H to valid range [0, n] for numerical stability
    # tr(H) should theoretically be in [0, n], but Hutchinson estimator has variance
    # and can give values outside this range, especially for extreme bandwidths
    # Clamping prevents extreme values in GCV computation and improves numerical stability
    trace_H = np.clip(trace_H, 0.0, n)

    # Compute mean and stderr across samples for each bandwidth
    dof_mean = np.mean(trace_H, axis=0)  # Shape: (dh,)
    if num_samples > 1:
        dof_stderr = np.std(trace_H, axis=0, ddof=1) / np.sqrt(
            num_samples
        )  # Shape: (dh,)
    else:
        dof_stderr = np.zeros(dh)

    # Compute penalty std directly from DOF samples
    # This is critical for accurate GCV standard error in 1-SE rule!
    # penalty = 1 ./ pdof_for_penalty.^2; pdof_inv_sd = std(penalty, [], 1);
    if num_samples > 1:
        # Convert DOF to pdof (proportion of residual DOF)
        pdof_samples = 1 - trace_H / n  # Shape: (num_samples, dh)

        # Apply complex adjustment
        # Complex-valued data: the trace estimation sees both real and imaginary parts
        # pdof(:,:,~regs.y_isreal)=2*pdof(:,:,~regs.y_isreal)-1
        # This adjusts for the doubled DOF from real+imag parts
        if y_is_complex:
            pdof_samples = 2 * pdof_samples - 1

        # Clamp to valid range [0, 1]
        pdof_clamped = np.clip(pdof_samples, 0.0, 1.0)

        # Apply minimum threshold for penalty computation
        # Prevents penalty explosion when pdof -> 0 (overfitting)
        pdof_for_penalty = np.maximum(pdof_clamped, 0.01)

        # Compute penalty: 1/pdof^2 for all samples
        penalty_samples = 1.0 / (pdof_for_penalty**2)  # Shape: (num_samples, dh)

        # Compute std directly from samples
        # This is the EXACT standard deviation, not delta method approximation!
        penalty_std = np.std(penalty_samples, axis=0, ddof=1)  # Shape: (dh,)

        # Per-sample mean penalty: mean(1/pdof^2) across DOF samples.
        # This matches MATLAB (fastlpr_dof.m) and R (fastlpr_y.R penalty_mean),
        # which average the per-sample penalty rather than computing
        # 1/(1-dof_mean/N)^2 from the mean DOF.
        penalty_mean = np.mean(penalty_samples, axis=0)  # Shape: (dh,)

        # Also compute adjusted mean DOF for complex data
        if y_is_complex:
            # pdof_complex = 2*pdof_real - 1
            # dof_complex = n*(1-pdof_complex) = n*(1-(2*pdof_real-1)) = n*(2-2*pdof_real) = 2*n - 2*n*pdof_real
            # = 2*(n - n*pdof_real) = 2*dof_real (but clamped pdof is different!)
            pdof_mean_clamped = np.mean(pdof_clamped, axis=0)
            dof_mean = n * (1 - pdof_mean_clamped)
    else:
        penalty_std = np.zeros(dh)
        # Single sample: per-sample mean equals 1/(1-dof_mean/N)^2, which is what
        # compute_gcv computes when penalty_mean is None. Signal fallback.
        penalty_mean = None

    return dof_mean, dof_stderr, penalty_std, penalty_mean


def nufft_regression_with_precomputed_kdf(
    x: np.ndarray,
    y: np.ndarray,
    kd_ft: Union[np.ndarray, list],
    kdf_params: dict,
    grid_shape: Union[int, np.ndarray],
    order: int = 0,
    accuracy: int = None,
    regularization: float = 1e-6,
    s_precomputed: Union[np.ndarray, list, None] = None,
) -> Tuple[np.ndarray, dict]:
    """
    NUFFT regression using precomputed kernel in Fourier domain.

    This function is more efficient than nufft_regression when computing
    regression for multiple bandwidths, as it reuses the precomputed KDF.

    Mirrors MATLAB's workflow: fastLPR_s + fastLPR_y + fastLPR_reg

    Parameters
    ----------
    x : ndarray, shape (n, dims)
        Data points (scaled)
    y : ndarray, shape (n,) or (n, dy)
        Response values
    kd_ft : ndarray or list of ndarrays
        Precomputed kernel in Fourier domain
        For order 0: shape (N1, N2, ..., Nd, dh)
        For order >= 1: list of ns arrays, each shape (N1, N2, ..., Nd, dh)
    kdf_params : dict
        Parameters from compute_kernel_fourier:
        - L: Oversampling factors
        - qin: Input grid size
        - qout: Output grid size (optional)
        - h: Filtered bandwidths (dh, dims)
        - dh: Number of bandwidths
        - lwp: Local polynomial parameters (for order >= 1)
    grid_shape : int or ndarray
        Grid shape
    order : int, default=0
        Polynomial order
    accuracy : int, optional
        NUFFT accuracy parameter
    regularization : float, default=1e-6
        Regularization parameter
    s_precomputed : ndarray or list, optional
        Pre-computed design matrix S (from :func:`compute_design_matrix`).
        If provided, Step 1 (computing S via NUFFT convolutions with a ones
        vector) is skipped entirely. Shape/structure must match the internal
        layout: a single ndarray for ``order == 0``, or a list of ``ns``
        ndarrays for ``order >= 1``. Used by :func:`~fastlpr.api.cv_fastlpr`
        to share S between the DoF Hutchinson probes and the actual
        y-regression, saving ~10-15% of total wall time on 2D problems.

    Returns
    -------
    m : ndarray, shape (N1, N2, ..., Nd, dh) or (N1, N2, ..., Nd, dh, dy)
        Regression estimates at grid points for all bandwidths
    params : dict
        Parameters including kdf_params
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.asarray(y)
    grid_shape = np.atleast_1d(np.asarray(grid_shape, dtype=int))

    # Auto-adjust accuracy based on sample size
    if accuracy is None:
        n_samples = len(x)
        accuracy = max(6 - int(np.ceil(np.log10(n_samples))), 4)

    # Extract parameters
    L = kdf_params["L"]
    qin = kdf_params["qin"]
    qout = kdf_params.get("qout", None)
    dh = kdf_params["dh"]

    # Step 1: Compute design matrix S using precomputed KDF
    # For order 0: S is just the kernel sum
    # For order >= 1: S is a list of design matrix elements
    from .convolution import nufft_convolve

    if s_precomputed is not None:
        # OPTIMIZATION: Reuse S already computed by the DoF path (or by the caller).
        # S depends only on (x, h, kernel, order); it is identical to what we would
        # recompute here since both paths call nufft_convolve(x, ones, kd_ft[i], ...).
        # See compute_design_matrix() for the canonical layout.
        s = s_precomputed
    else:
        ones = np.ones(len(x))

        if order == 0:
            # Order 0: S = sum of kernel weights
            s = nufft_convolve(
                x,
                ones,
                kd_ft,
                grid_shape,
                L,
                qin,
                qout,
                y_is_transformed=False,
                accuracy=accuracy,
                y_is_real=True,
            )
        else:
            # Order >= 1: Compute each design matrix element
            lwp = kdf_params["lwp"]
            ns = lwp["ns"]

            s = []
            for i in range(ns):
                s_ij = nufft_convolve(
                    x,
                    ones,
                    kd_ft[i],
                    grid_shape,
                    L,
                    qin,
                    qout,
                    y_is_transformed=False,
                    accuracy=accuracy,
                    y_is_real=True,
                )
                s.append(s_ij)

    # Step 2: Compute weighted response T
    if order == 0:
        # Order 0: T = sum of kernel-weighted y
        t = nufft_convolve(
            x,
            y,
            kd_ft,
            grid_shape,
            L,
            qin,
            qout,
            y_is_transformed=False,
            accuracy=accuracy,
            y_is_real=np.isrealobj(y),
        )
    else:
        # Order >= 1: Compute each weighted response element
        lwp = kdf_params["lwp"]
        nt = lwp["nt"]

        # Transform y to Fourier domain ONCE
        from .convolution import nufft_transform

        y_ft = nufft_transform(
            x,
            y,
            L,
            eval_grid_shape=grid_shape,
            qin=qin,
            accuracy=accuracy,
            iflag=-1,
        )

        # Compute T elements by multiplying with KDF in Fourier domain
        t = []
        for i in range(nt):
            # Get the corresponding kernel index for this polynomial term
            # For order 1, 1D: nt=2, we need kd[0] and kd[1]
            # For order 1, 2D: nt=3, we need kd[0], kd[1], kd[2]
            t_i = nufft_convolve(
                x,
                y_ft,  # Pass Fourier-transformed y
                kd_ft[i],
                grid_shape,
                L,
                qin,
                qout,
                y_is_transformed=True,
                accuracy=accuracy,
                y_is_real=np.isrealobj(y),
            )
            t.append(t_i)

    # Step 3: Solve regression system
    lwp_params = kdf_params.get("lwp", None)
    m = solve_regression_system(
        s,
        t,
        order=order,
        regularization=regularization,
        lwp_params=lwp_params,
        grid_shape=grid_shape,
    )

    params = {
        "kdf_params": kdf_params,
        "order": order,
        "regularization": regularization,
        "s": s,  # Design matrix S (needed for s_0 extraction in CI/PI)
    }

    return m, params


def nufft_regression(
    x: np.ndarray,
    y: np.ndarray,
    h: np.ndarray,
    grid_shape: Union[int, np.ndarray],
    kernel_type: str = "gaussian",
    order: int = 0,
    accuracy: int = None,
    regularization: float = 1e-6,
) -> Tuple[np.ndarray, dict]:
    """
    Complete NUFFT-accelerated regression pipeline.

    Combines design matrix computation, convolution, and solving
    into a single function for convenience.

    Parameters
    ----------
    x : ndarray, shape (N, d)
        Predictor data
    y : ndarray, shape (N,)
        Response data
    h : ndarray, shape (dh, d)
        Bandwidth candidates
    grid_shape : int or ndarray
        Grid size per dimension
    kernel_type : str, default='gaussian'
        Kernel type
    order : int, default=0
        Polynomial order
    accuracy : int, default=None
        NUFFT accuracy. If None, auto-adjust based on sample size:
        accuracy = max(6 - ceil(log10(n_samples)), 1)
    regularization : float, default=1e-6
        Regularization parameter

    Returns
    -------
    m : ndarray
        Regression estimates on grid
    params : dict
        Parameters and intermediate results

    Notes
    -----
    This is the main regression function that combines:
    1. Compute design matrix S using NUFFT
    2. Compute weighted response T using NUFFT
    3. Solve S * beta = T for regression coefficients
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    # Allow complex-valued responses
    y = np.asarray(y)
    h = np.atleast_2d(np.asarray(h, dtype=float))
    grid_shape = np.atleast_1d(np.asarray(grid_shape, dtype=int))

    # Auto-adjust accuracy based on sample size
    if accuracy is None:
        n_samples = len(x)
        accuracy = max(6 - int(np.ceil(np.log10(n_samples))), 4)

    # Step 1: Compute design matrix S
    s, params = compute_design_matrix(
        x, h, grid_shape, kernel_type=kernel_type, order=order, accuracy=accuracy
    )

    # Check if y is real-valued (needed for proper output type)
    y_is_real = np.isrealobj(y)

    # Step 2: Compute weighted response T = X' * W * y
    if order == 0:
        # Order 0: T = sum(K * y)
        t = nufft_convolve(
            x,
            y,
            params["kdf"],
            grid_shape,
            params["L"],
            params["qin"],
            params.get("qout", None),
            y_is_transformed=False,
            accuracy=accuracy,
            y_is_real=y_is_real,
        )
        # Squeeze singleton dimensions for single bandwidth/response
        # Shape: (grid_size, dh, dy) -> squeeze to (grid_size,) if dh=1 and dy=1
        t = np.squeeze(t)
    else:
        # Order >= 1: T_i = sum(K * X^i * y) for each polynomial term
        # The first nt kernels in kdf are for the weighted response
        # kdf[0] = K (for T1 = sum(K*y))
        # kdf[1] = K*x (for T2 = sum(K*x*y))
        # etc.
        lwp_params = params.get("lwp", None)
        if lwp_params is None:
            raise ValueError("LWP parameters not found in design matrix computation")

        nt = lwp_params["nt"]
        kdf = params["kdf"]

        # OPTIMIZATION: Process all nt kernels with all dy responses in ONE call!
        # Instead of looping over nt kernels, we can transform y once and reuse it

        # Transform y to Fourier domain ONCE (this is the expensive part!)
        from .convolution import nufft_transform

        y_ft = nufft_transform(
            x,
            y,
            params["L"],
            eval_grid_shape=grid_shape,
            qin=params["qin"],
            accuracy=accuracy,
            iflag=-1,
        )
        # y_ft shape: (L, dy) or (L,) if dy=1

        # Now convolve with each kernel (cheap - just multiplication + FFT)
        t = []
        for i in range(nt):
            t_i = nufft_convolve(
                x,
                y_ft,  # Use pre-transformed y_ft!
                kdf[i],
                grid_shape,
                params["L"],
                params["qin"],
                params.get("qout", None),
                y_is_transformed=True,  # y_ft is already transformed
                accuracy=accuracy,
                y_is_real=y_is_real,  # Pass original y's realness
            )
            # Squeeze singleton dimensions for single bandwidth/response
            t_i = np.squeeze(t_i)
            t.append(t_i)

        # Verify we computed the right number of elements
        assert len(t) == nt, f"Expected {nt} weighted response elements, got {len(t)}"

    # Step 3: Solve regression system
    lwp_params = params.get("lwp", None)
    m = solve_regression_system(
        s,
        t,
        order=order,
        regularization=regularization,
        lwp_params=lwp_params,
        grid_shape=grid_shape,
    )

    # Note: Don't squeeze m here - it already has the correct grid shape from solve_regression_system
    # Squeezing would flatten multi-dimensional grids

    # Step 4: Estimate degrees of freedom using Hutchinson's method
    # This is critical for GCV bandwidth selection
    # DOF estimation is disabled here because it creates circular dependency
    # (nufft_regression calls estimate_degrees_of_freedom which calls nufft_regression)
    # DOF should be estimated separately by the caller if needed
    dof = len(x)  # Placeholder: assume full DOF (no smoothing)
    dof_stderr = 0.0

    params["s"] = s
    params["t"] = t
    params["m"] = m
    params["dof"] = dof
    params["dof_stderr"] = dof_stderr

    return m, params
