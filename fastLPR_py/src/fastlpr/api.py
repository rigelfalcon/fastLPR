# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
User-facing API for fastlpr (Python).

The functions implemented here mirror the MATLAB `cv_fastLPR` and
`cv_fastKDE` interfaces while returning Python-native data structures.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence

import numpy as np

from .bandwidth import get_hlist
from .core import (
    create_grid,
    create_interpolator,
    density_on_grid,
)
from .model import create_regression_model
from .structures import KDEOutput, RegressionOutput
from .utils import set_defaults
from .utils.helpers import _as_column_matrix, _default_hrange
from .regression import nufft_regression, compute_gcv
from .timing import timer



def _prepare_bandwidths(
    h: Optional[Sequence[Sequence[float]]],
    dims: int,
    options: Dict[str, object],
    default_range: np.ndarray,
) -> np.ndarray:
    if h is None:
        dh = options.get("dh", None)
        if dh is None:
            dh = 20 if dims == 1 else [20] * dims
        bandwidths = get_hlist(dh, default_range)
    else:
        h_arr = np.asarray(h, dtype=float)
        if h_arr.ndim == 1:
            if h_arr.size != dims:
                raise ValueError(
                    "Scalar bandwidth must match the number of dimensions."
                )
            bandwidths = h_arr.reshape(1, -1)
        else:
            if h_arr.shape[1] != dims:
                raise ValueError(
                    "Bandwidth array must have the same dimensionality as the input data."
                )
            bandwidths = h_arr.reshape(-1, dims)

    # Validate bandwidths: must be positive
    if np.any(bandwidths <= 0):
        raise ValueError(
            "Bandwidth values must be strictly positive. "
            f"Found non-positive values: {bandwidths[bandwidths <= 0]}"
        )

    return bandwidths


def _one_se_rule(
    scores: np.ndarray,
    errs: np.ndarray,
    bandwidths: np.ndarray,
    prefer_smaller: bool,
    dstd: float = 1.0,
) -> int:
    """
    Apply 1-SE rule for bandwidth selection.

    MATLAB equivalent: fastLPR_gcv.m

    Args:
        scores: CV scores for each bandwidth
        errs: Standard errors for each score
        bandwidths: Bandwidth values
        prefer_smaller: If True, minimize scores; if False, maximize scores
        dstd: Number of standard errors for threshold (default: 1.0)
              MATLAB uses dstd from opt.dstd (typically 1 for mean, 5 for variance)

    Key differences from naive implementation:
    1. Filter out NaN/Inf scores before finding minimum (MATLAB's min() ignores NaN)
    2. Only consider valid bandwidths in 1-SE rule
    3. Among candidates within 1-SE, select largest bandwidth (smoothest fit)
    4. Support configurable dstd parameter (MATLAB uses opt.dstd)
    """
    # Filter out NaN and Inf values (MATLAB's min() ignores these automatically)
    valid_mask = np.isfinite(scores)

    if not np.any(valid_mask):
        # All scores are NaN/Inf - this is a critical error
        raise ValueError(
            "All GCV scores are NaN or Inf. Regression failed for all bandwidths. "
            "Try: (1) Use larger bandwidth h, (2) Increase grid points, "
            "(3) Check for data quality issues"
        )

    # Get indices of valid scores
    valid_indices = np.where(valid_mask)[0]
    valid_scores = scores[valid_mask]
    valid_errs = errs[valid_mask]
    valid_bandwidths = bandwidths[valid_mask]

    if prefer_smaller:
        # Find minimum among valid scores
        best_valid_idx = int(np.argmin(valid_scores))
        best_idx = valid_indices[best_valid_idx]

        # Compute 1-SE threshold (MATLAB: semin = gcv_m(idmin) + dstd * gcv_sd(idmin))
        threshold = valid_scores[best_valid_idx] + dstd * valid_errs[best_valid_idx]

        # Find all valid candidates within 1-SE of minimum
        candidates_mask = valid_scores <= threshold
        candidate_indices = valid_indices[candidates_mask]

        if len(candidate_indices) == 0:
            # Fallback: if no candidates within 1-SE, just use the best
            return best_idx

        # Among candidates, select the one with largest bandwidth (sum of h)
        # This prefers smoother fits and reduces overfitting
        candidate_bandwidths = bandwidths[candidate_indices]
        bandwidth_sums = np.sum(candidate_bandwidths, axis=1)
        selected_idx = candidate_indices[np.argmax(bandwidth_sums)]
    else:
        # For maximization (e.g., likelihood)
        best_valid_idx = int(np.argmax(valid_scores))
        best_idx = valid_indices[best_valid_idx]

        # Compute 1-SE threshold
        threshold = valid_scores[best_valid_idx] - dstd * valid_errs[best_valid_idx]

        # Find all valid candidates within 1-SE of maximum
        candidates_mask = valid_scores >= threshold
        candidate_indices = valid_indices[candidates_mask]

        if len(candidate_indices) == 0:
            # Fallback: if no candidates within 1-SE, just use the best
            return best_idx

        # Among candidates, select the one with largest bandwidth
        candidate_bandwidths = bandwidths[candidate_indices]
        bandwidth_sums = np.sum(candidate_bandwidths, axis=1)
        selected_idx = candidate_indices[np.argmax(bandwidth_sums)]

    return selected_idx


def cv_fastlpr(
    x: np.ndarray,
    y: np.ndarray,
    h: Optional[Sequence[Sequence[float]]] = None,
    options: Optional[dict] = None,
) -> RegressionOutput:
    """
    Fast Local Polynomial Regression with automatic bandwidth selection via GCV.

    This is the main function for fastLPR Python port. It performs nonparametric
    regression using kernel-weighted local polynomial methods with NUFFT
    (Non-Uniform Fast Fourier Transform) acceleration. The function automatically
    selects the optimal bandwidth via Generalized Cross-Validation (GCV) when
    multiple bandwidth candidates are provided.

    Parameters
    ----------
    x : np.ndarray
        N x d matrix of predictors (N samples, d dimensions).
        Can be real or complex-valued. Each row is one observation.
    y : np.ndarray
        Response vector with N samples.
        Accepts: 1D array (N,), column vector (N, 1), or row vector (1, N).
        For multi-response regression: 2D array (N, dy) where dy > 1.
        Can be real or complex-valued. Must have same number of samples as x.

        Multi-response support (dy > 1):
        - Each column is treated as a separate response variable
        - All responses share the same bandwidth (joint estimation)
        - GCV aggregation across responses controlled by `gcv_selection` option
        - Output yhat will have shape (N, dy)
    h : Optional[Sequence[Sequence[float]]]
        Bandwidth parameter(s) (optional, default: automatic selection).
        - Scalar: same bandwidth for all dimensions
        - 1 x d vector: different bandwidth per dimension
        - k x d matrix: grid of k bandwidth combinations for GCV selection
    options : Optional[dict]
        Options dictionary with fields:
        - order: Polynomial order (default: 0)
                 0 = Nadaraya-Watson (local constant)
                 1 = Local linear regression
                 2 = Local quadratic regression
        - kernel_type: Kernel function (default: 'gaussian')
        - regularization: Regularization parameter (default: 1e-8)
        - dstd: Number of standard errors for 1-SE bandwidth selection rule (default: 10)
                The 1-SE rule selects the largest bandwidth h such that:
                GCV(h) <= GCV_min + dstd * SE(GCV_min)
                - dstd=0: Minimum GCV (no 1-SE rule)
                - dstd=1: Standard 1-SE rule (prefer smoother fits)
                - dstd=5-20: More aggressive smoothing (for variance estimation)
        - num_dof_sample: Number of random vectors for DOF estimation (default: 10)
                Uses Hutchinson's randomized trace estimation method.
                Higher values give more stable DOF estimates but slower computation.
        - y_type_out: Output type (default: 'mean')
                      'mean' = estimate conditional mean E[Y|X]
                      'variance' = estimate conditional variance Var[Y|X]
                      For variance: input should be squared residuals
        - N: Grid resolution for evaluation (default: auto)
        - xrange: Evaluation range (default: data range)
        - random_matrix: Pre-generated random matrix for DOF estimation (default: None)
                         Shape: (n, num_samples). For exact verification with MATLAB.
                         If provided, overrides dof_seed and num_dof_sample.
        - dof_seed: Random seed for DOF estimation (default: 42)
                    Set to match MATLAB's rng seed for exact cross-validation.
                    Only used if random_matrix is None.
        - gcv_selection: GCV aggregation method for multi-response (default: 'first')
                         Determines how GCV is computed when y has multiple columns:
                         - 'first': Use GCV from first response column only (default)
                         - 'average': Average GCV across all response columns
                         - 'sum': Sum GCV across all response columns
                         This is only relevant when y has shape (N, dy) with dy > 1.

    Returns
    -------
    RegressionOutput
        Structure containing regression results:
        - yhat: Fitted values at data points (samples)
        - grid: Evaluation grid points (tuple of arrays)
        - bandwidth: Selected bandwidth
        - gcv_result: GCV results (if multiple bandwidths provided)
        - interpolant: Interpolator for prediction at any point
        - metadata: Additional information

    Examples
    --------
    Basic 1D regression with local linear fit:

    >>> import numpy as np
    >>> from fastlpr import cv_fastlpr, get_hlist
    >>> np.random.seed(42)
    >>> x = np.random.rand(50, 1)
    >>> y = np.sin(2*np.pi*x[:,0]) + 0.1*np.random.randn(50)
    >>> hlist = get_hlist(5, [0.1, 0.5])
    >>> result = cv_fastlpr(x, y, h=hlist, options={'order': 1, 'calc_dof': False})
    >>> result.yhat.shape
    (50,)
    >>> isinstance(result.h, np.ndarray)
    True

    Local quadratic regression (order=2):

    >>> result2 = cv_fastlpr(x, y, h=[[0.2]], options={'order': 2, 'calc_dof': False})
    >>> len(result2.grid) == 1  # 1D data
    True

    Predict at new points using the interpolant:

    >>> x_new = np.linspace(0, 1, 10).reshape(-1, 1)
    >>> y_pred = result.fpp_yhat(x_new)
    >>> y_pred.shape
    (10,)
    """

    x_arr = _as_column_matrix(x)

    # Validate and convert y to column matrix
    # Support both single response (1D/column vector) and multi-response (N, dy)
    y_arr = np.asarray(y)

    if y_arr.ndim == 1:
        # 1D array -> column vector (N, 1)
        y_arr = y_arr[:, None]
    elif y_arr.ndim == 2:
        if y_arr.shape[0] == 1 and y_arr.shape[1] > 1:
            # Row vector (1, N) -> transpose to (N, 1)
            y_arr = y_arr.T
        # (N, 1) or (N, dy) - keep as is
    else:
        raise ValueError(
            f"Response y must be 1D or 2D array. Got {y_arr.ndim}D array."
        )

    # Get number of response variables
    dy = y_arr.shape[1]

    # Check dimension compatibility
    if y_arr.shape[0] != x_arr.shape[0]:
        raise ValueError(
            f"Response y must have same number of samples as x. "
            f"x has {x_arr.shape[0]} samples, y has {y_arr.shape[0]} samples."
        )

    # Input validation: check for NaN/Inf
    if np.any(np.isnan(x_arr)):
        raise ValueError("Input x contains NaN values. Please remove or impute NaN values before calling cv_fastlpr.")
    if np.any(np.isinf(x_arr)):
        raise ValueError("Input x contains Inf values. Please remove or handle Inf values before calling cv_fastlpr.")
    if np.any(np.isnan(y_arr)):
        raise ValueError("Response y contains NaN values. Please remove or impute NaN values before calling cv_fastlpr.")
    if np.any(np.isinf(y_arr)):
        raise ValueError("Response y contains Inf values. Please remove or handle Inf values before calling cv_fastlpr.")

    opts = set_defaults(
        options,
        order=0,
        regularization=1e-8,
        num_dof_sample=10,  # Number of DOF samples for stable DOF estimation (MATLAB: num_dof_sample)
        kernel_type="gaussian",
        y_type_out="mean",
        calc_dof=True,
        gcv_selection="first",  # How to aggregate GCV for multi-response: 'first', 'average', 'sum'
    )

    # Adaptive NUFFT accuracy based on sample size (matches MATLAB behavior)
    # Formula: accuracy = max(6 - ceil(log10(N)), 1)
    # Rationale:
    #   - accuracy parameter controls NUFFT approximation error ≈ 10^(-accuracy)
    #   - Smaller datasets need higher accuracy (fewer points to average errors)
    #   - accuracy=6: ~1e-6 error for N < 1000 (small datasets)
    #   - accuracy=5: ~1e-5 error for N < 10000 (medium datasets)
    #   - accuracy=4: ~1e-4 error for N ≥ 1e6 (very large datasets, speed priority)
    #   - Minimum accuracy=4 matches MATLAB's adaptive scheme
    n_samples = x_arr.shape[0]
    if opts.get("accuracy") is None:
        adaptive_accuracy = max(6 - int(np.ceil(np.log10(n_samples))), 4)
        opts["accuracy"] = adaptive_accuracy
    order = int(opts["order"])
    if order not in (0, 1, 2):
        raise ValueError("Polynomial order must be 0, 1, or 2.")

    dims = x_arr.shape[1]

    # Dimension limits and warnings
    if dims > 10:
        raise ValueError(
            f"dx={dims} exceeds maximum supported dimension (10). "
            f"High-dimensional regression suffers from the curse of dimensionality."
        )
    elif dims > 6:
        warnings.warn(
            f"dx={dims} is high-dimensional and may cause memory issues. "
            f"Consider dimensionality reduction techniques.",
            UserWarning
        )

    default_range = _default_hrange(x_arr)
    bandwidths = _prepare_bandwidths(h, dims, opts, default_range)

    # Handle variance estimation mode
    y_type_out = str(opts.get("y_type_out", "mean"))
    if y_type_out == "variance":
        # Save original y for normalization later
        # CRITICAL: Use ravel() to ensure 1D array - prevents broadcasting bug
        # where (N, 1) * (N,) would broadcast to (N, N) instead of (N,)
        y_raw_original = y_arr.ravel().copy()
        # Transform to log scale for numerical stability
        # Add small constant 1/Ty to avoid log(0)
        # We'll compute Ty after creating the model
        y_arr_for_model = y_arr  # Will be transformed after model creation
    else:
        y_raw_original = None
        y_arr_for_model = y_arr

    # Create model structure
    model = create_regression_model(x_arr, y_arr_for_model, bandwidths, opts)

    # Apply log transform for variance estimation
    if y_type_out == "variance":
        # Ty is the number of DATA POINTS, not grid points!
        # MATLAB: y=log(regs.y+1/regs.Ty)
        Ty = len(y_arr)  # Number of data points
        y_arr_log = np.log(y_arr + 1.0 / Ty)
        # Update model with log-transformed y
        model.y_raw = y_arr_log
    regularization = float(opts["regularization"])
    kernel_type = str(opts["kernel_type"])

    # Get grid shape from grid axes
    grid_shape = np.array([len(ax) for ax in model.grid_axes_scaled])

    # Check if we should compute DOF and GCV (calc_dof option)
    # When calc_dof=False, skip expensive DOF estimation
    calc_dof = bool(opts.get("calc_dof", True))

    # Compute KDF once for all bandwidths (from MATLAB fastLPR_create + fastLPR_kdf)
    # This is the key optimization that makes cross-validation efficient
    # Using cached version ensures same KDF is reused for DOF estimation later
    with timer(f"Computing KDF for {len(model.bandwidths)} bandwidths"):
        from .convolution import compute_kernel_fourier_cached

        flag_power2 = bool(opts.get("flag_power2", True))
        kd_ft, kdf_params = compute_kernel_fourier_cached(
            model.x_scaled,
            model.bandwidths,
            grid_shape,
            order=order,
            kernel_type=kernel_type,
            flag_power2=flag_power2,
        )

    # Update model with filtered bandwidths (some may have been removed as "too small")
    bandwidths_filtered = kdf_params["h"]
    dh_filtered = kdf_params["dh"]
    ihbad = kdf_params["ihbad"]

    num_removed = len(model.bandwidths) - dh_filtered
    if num_removed > 0:
        import warnings
        warnings.warn(
            f"{num_removed} bandwidth(s) are too small for this grid and will be removed",
            stacklevel=2,
        )

    # FAST PATH: When calc_dof=False, skip DOF and GCV computation
    if not calc_dof:
        # Compute regression directly without DOF
        with timer(f"NUFFT regression (fast path, no DOF)"):
            from .regression import nufft_regression_with_precomputed_kdf

            # Compute regression for the bandwidth(s)
            m_all, params = nufft_regression_with_precomputed_kdf(
                model.x_scaled,
                model.y_raw,
                kd_ft,
                kdf_params,
                grid_shape,
                order=order,
                accuracy=opts.get("accuracy"),
                regularization=regularization,
            )

        # Use first bandwidth (or only bandwidth) for output
        # Handle multi-response case: m_all shape is (grid_shape..., dh, dy) for multi-response
        # vs (grid_shape..., dh) for single response
        if dh_filtered == 1:
            m = m_all
        else:
            # If multiple bandwidths, use the first one
            # For multi-response (dy > 1): m_all shape is (..., dh, dy), select [..., 0, :]
            # For single response (dy = 1): m_all shape is (..., dh), select [..., 0]
            if dy > 1:
                # Multi-response: bandwidth is second-to-last axis
                m = m_all[..., 0, :]
            else:
                # Single response: bandwidth is last axis
                m = m_all[..., 0]

        selected_bandwidth = bandwidths_filtered[0]
        grid_values = m

        # Extract s_0 (zero-th kernel moment) for CI/PI construction
        # Reuse S already computed inside nufft_regression_with_precomputed_kdf
        s_fast = params["s"]
        if order == 0:
            s0_squeezed = np.real(np.squeeze(s_fast))
        else:
            s0_squeezed = np.real(np.squeeze(s_fast[0]))
        # Select bandwidth index 0 if multiple bandwidths
        if s0_squeezed.ndim > dims and dh_filtered > 1:
            s0_selected = s0_squeezed[..., 0]
        else:
            s0_selected = s0_squeezed

        # Create interpolator in raw (original) space
        # Use linear interpolation for speed in fast path (cubic is ~100x slower)
        interp_method = opts.get("interp_method", "linear")
        interpolant = create_interpolator(tuple(model.grid_axes_raw), grid_values, method=interp_method)

        # Evaluate at original data points (in raw space)
        fitted_at_samples = interpolant(model.x_raw)

        # Handle variance estimation output
        d_normalization = None
        if y_type_out == "variance":
            Ty = len(y_raw_original)
            d_normalization = 1.0 / (
                (1.0 / Ty) * np.sum(y_raw_original * np.exp(-fitted_at_samples))
            )
            fitted_at_samples = np.exp(fitted_at_samples) / d_normalization
            grid_values = np.exp(grid_values) / d_normalization
            interpolant = create_interpolator(tuple(model.grid_axes_raw), grid_values, method=interp_method)

        # Create minimal GCV result (no actual GCV computed)
        gcv_result = {
            "bandwidths": bandwidths_filtered,
            "gcv_m": np.array([np.nan]),
            "gcv_sd": np.array([np.nan]),
            "idmin": 0,
            "id1se": 0,
            "h1se": selected_bandwidth,
            "hmin": selected_bandwidth,
            "dof": np.array([np.nan]),
            "dof_stderr": np.array([np.nan]),
        }

        # Compute residuals
        # CRITICAL: Use ravel() to ensure 1D arrays - prevents broadcasting bug
        # where (N, 1) - (N,) would broadcast to (N, N) instead of (N,)
        if y_type_out == "variance":
            residuals = y_raw_original - np.asarray(fitted_at_samples).ravel()
        else:
            residuals = model.y_raw.ravel() - np.asarray(fitted_at_samples).ravel()

        metadata = {
            "options": opts,
            "order": order,
            "bandwidths_tested": bandwidths_filtered,
            "fitted_at_samples": fitted_at_samples,
            "residuals": residuals,
            "mean": model.mean,
            "std": model.std,
            "dof": np.nan,
            "dof_stderr": np.nan,
        }

        return RegressionOutput(
            yhat=fitted_at_samples,
            grid=tuple(model.grid_axes_raw),
            h=selected_bandwidth,
            gcv=gcv_result,
            fpp_yhat=interpolant,
            metadata=metadata,
            dof=np.nan,
            dof_stderr=np.nan,
            d=d_normalization,
            s0=s0_selected,
        )

    # FULL PATH: Compute DOF and GCV for bandwidth selection
    # OPTIMIZATION: Estimate DOF for all bandwidths at once (vectorized)
    # This is 5-10x faster than sequential estimation
    from .regression import estimate_dof_nufft_vectorized

    # num_dof_sample: Number of random vectors for Hutchinson's trace estimation
    # NOTE: This is DIFFERENT from 'dstd' which controls the 1-SE rule!
    num_dof_samples = int(opts.get("num_dof_sample", 10))

    # Get random_matrix if provided (for exact verification with MATLAB)
    random_matrix = opts.get("random_matrix", None)

    # Get DOF random seed (default: 42, can be overridden for exact MATLAB matching)
    dof_seed = opts.get("dof_seed", 42)

    # OPTIMIZATION: Compute design matrix S once for DOF estimation
    # S depends only on (x, h), not on y (random samples)
    with timer(f"Computing design matrix S for DOF estimation"):
        from .regression import compute_design_matrix

        S, S_params = compute_design_matrix(
            model.x_scaled,
            bandwidths_filtered,
            grid_shape,
            kernel_type=kernel_type,
            order=order,
            flag_power2=flag_power2,
        )

    # Get DOF interpolation method from options
    # MATLAB uses 'spline' (cubic) by default in fastLPR_gridinterp.m
    # For 1D: Use linear (faster, Numba-accelerated)
    # For 2D+: Use cubic (matches MATLAB's spline, required for DOF accuracy)
    # User can override with dof_interp_method option
    if dims == 1:
        default_dof_interp = "linear"  # Numba-optimized path for 1D
    else:
        default_dof_interp = "cubic"  # Match MATLAB spline for 2D+
    dof_interp_method = opts.get("dof_interp_method", default_dof_interp)

    # Check if y is complex-valued
    # Complex data requires 2x DOF adjustment
    y_is_complex = np.iscomplexobj(model.y_raw)

    # Estimate DOF for all filtered bandwidths using cached S
    with timer(
        f"DOF estimation (vectorized, {dh_filtered} bandwidths, {num_dof_samples} samples)"
    ):
        dof_all, dof_stderr_all, penalty_std_all = estimate_dof_nufft_vectorized(
            model.x_scaled,
            model.x_raw,  # Must use RAW space for interpolation (from MATLAB!)
            bandwidths_filtered,  # Use filtered bandwidths
            grid_shape,
            tuple(model.grid_axes_raw),  # Must use RAW grid axes (from MATLAB!)
            order,
            regularization,
            num_samples=num_dof_samples,
            random_state=dof_seed,
            random_matrix=random_matrix,
            S_precomputed=S,  # Reuse design matrix for all samples
            params_precomputed=S_params,
            interp_method=dof_interp_method,  # 'linear' for speed (default), 'cubic' for exact MATLAB match
            y_is_complex=y_is_complex,  # Complex data needs 2x DOF adjustment
        )

    # Compute regression for ALL bandwidths at once using precomputed KDF
    with timer(f"NUFFT regression for {dh_filtered} bandwidths (vectorized)"):
        from .regression import nufft_regression_with_precomputed_kdf

        # Compute regression for all bandwidths at once
        # m shape: (N1, N2, ..., Nd, dh) for single response
        #          (N1, N2, ..., Nd, dh, dy) for multiple responses
        m_all, params = nufft_regression_with_precomputed_kdf(
            model.x_scaled,
            model.y_raw,
            kd_ft,
            kdf_params,
            grid_shape,
            order=order,
            accuracy=opts.get("accuracy"),
            regularization=regularization,
        )

    # Get GCV interpolation method from options
    # MATLAB uses 'spline' (cubic) by default in fastLPR_gridinterp.m
    # For 1D: Use linear (faster, negligible accuracy difference)
    # For 2D+: Use cubic (matches MATLAB's spline, required for GCV accuracy)
    # User can override with gcv_interp_method option
    if dims == 1:
        default_gcv_interp = "linear"  # Fast path for 1D
    else:
        default_gcv_interp = "cubic"  # Match MATLAB spline for 2D+
    gcv_interp_method = opts.get("gcv_interp_method", default_gcv_interp)

    # Extract grid values and compute GCV for each bandwidth
    gcv_scores = []
    gcv_stderr = []
    grid_values_list = []

    with timer(f"Computing GCV scores for {dh_filtered} bandwidths"):
        # OPTIMIZATION: Batch interpolation for all bandwidths at once
        # This is 10-20x faster than looping over bandwidths!
        # Build grid_values_list first (needed for final interpolant)
        for idx in range(dh_filtered):
            if dh_filtered == 1:
                if m_all.ndim == len(grid_shape):
                    m = m_all
                else:
                    m = m_all
            elif m_all.ndim == len(grid_shape) + 1:
                m = m_all[..., idx]
            else:
                m = m_all[..., idx, :]
            grid_values_list.append(m)

        # Stack all grid values for batch interpolation
        # m_stacked shape: (grid_shape..., dh_filtered) or (grid_shape..., dh_filtered, dy)
        m_stacked = np.stack(grid_values_list, axis=-1 if grid_values_list[0].ndim == len(grid_shape) else -2)

        # Create single interpolator for all bandwidths
        from scipy.interpolate import RegularGridInterpolator
        interp_batch = RegularGridInterpolator(
            tuple(model.grid_axes_raw),
            m_stacked,
            method=gcv_interp_method,
            bounds_error=False,
            fill_value=None,
        )

        # Batch evaluate at all data points
        # yhat_all shape: (n, dh_filtered) or (n, dh_filtered, dy)
        yhat_all = interp_batch(model.x_raw)

        # Compute GCV for each bandwidth
        gcv_selection = str(opts.get("gcv_selection", "first"))
        for idx in range(dh_filtered):
            # Extract yhat for this bandwidth
            if yhat_all.ndim == 2:
                yhat = yhat_all[:, idx]
            else:
                yhat = yhat_all[:, idx, :]

            yhat = np.asarray(yhat)

            # Get DOF from vectorized estimation
            dof = dof_all[idx]
            dof_stderr = dof_stderr_all[idx]
            penalty_std = penalty_std_all[idx]

            # Compute GCV score with aggregation for multi-response
            gcv, info = compute_gcv(
                model.y_raw, yhat, dof, dof_stderr=dof_stderr, penalty_std=penalty_std,
                aggregation=gcv_selection
            )

            gcv_scores.append(gcv)
            gcv_stderr.append(info.get("stderr", 0.0))

    gcv_scores = np.asarray(gcv_scores)
    gcv_stderr = np.asarray(gcv_stderr)

    # Use filtered bandwidths for selection
    bandwidths_for_selection = bandwidths_filtered

    # Select bandwidth using 1-SE rule (prefer smaller bandwidth)
    # MATLAB default: dstd=0 (select minimum GCV, no 1-SE rule)
    dstd = float(opts.get("dstd", 0.0))
    # For mean estimation, use dstd=1 (standard 1-SE rule)
    # For variance estimation, MATLAB uses dstd=5 for more conservative selection
    # When dstd=0, MATLAB selects minimum GCV without 1-SE rule (threshold = min_GCV + 0*SE = min_GCV)
    selected_idx = _one_se_rule(
        gcv_scores, gcv_stderr, bandwidths_for_selection, prefer_smaller=True, dstd=dstd
    )
    selected_bandwidth = bandwidths_for_selection[selected_idx]
    grid_values = grid_values_list[selected_idx]

    # Extract s_0 (zero-th kernel moment) at selected bandwidth for CI/PI
    # S shape: (grid..., dh) for order 0, S[0] shape: (grid..., dh, 1) for order >= 1
    # Strategy: squeeze trailing singletons first, then select bandwidth
    if order == 0:
        s0_squeezed = np.real(np.squeeze(S))
    else:
        s0_squeezed = np.real(np.squeeze(S[0]))  # remove trailing ns=1

    # Now s0_squeezed is (grid..., dh) or (grid...) if dh=1
    if s0_squeezed.ndim > dims:
        s0_selected = s0_squeezed[..., selected_idx]
    else:
        s0_selected = s0_squeezed

    # Create interpolator in raw (original) space
    # The grid_values are computed on the scaled grid, but we create the interpolator
    # with grid_axes_raw so it can be evaluated at raw data points directly
    interpolant = create_interpolator(tuple(model.grid_axes_raw), grid_values)

    # Evaluate at original data points (in raw space)
    fitted_at_samples = interpolant(model.x_raw)

    # Get DOF from the selected bandwidth (from vectorized estimation)
    dof = dof_all[selected_idx]
    dof_stderr = dof_stderr_all[selected_idx]

    # Apply inverse transform for variance estimation
    d_normalization = None
    if y_type_out == "variance":
        # Ty is the number of DATA POINTS, not grid points!
        # MATLAB: d=1./((1/regs.Ty).*sum(yraw.*exp(-regs.yhat)))
        Ty = len(y_raw_original)  # Number of data points
        # Compute normalization factor
        # d = 1 / ((1/Ty) * sum(yraw * exp(-yhat)))
        d_normalization = 1.0 / (
            (1.0 / Ty) * np.sum(y_raw_original * np.exp(-fitted_at_samples))
        )

        # Transform back from log scale
        fitted_at_samples = np.exp(fitted_at_samples) / d_normalization
        grid_values = np.exp(grid_values) / d_normalization

        # Update interpolator with transformed values
        interpolant = create_interpolator(tuple(model.grid_axes_raw), grid_values)

    idmin = int(np.argmin(gcv_scores))
    gcv_result = {
        "bandwidths": bandwidths_for_selection,  # Use filtered bandwidths
        "gcv_m": gcv_scores,
        "gcv_sd": gcv_stderr,
        "idmin": idmin,
        "id1se": selected_idx,
        "h1se": selected_bandwidth,
        "hmin": bandwidths_for_selection[idmin],
        "dof": dof_all,  # DOF for all bandwidths
        "dof_stderr": dof_stderr_all,  # DOF stderr for all bandwidths
    }

    # Compute residuals using original y (not log-transformed)
    # CRITICAL: Use ravel() to ensure 1D arrays - prevents broadcasting bug
    # where (N, 1) - (N,) would broadcast to (N, N) instead of (N,)
    if y_type_out == "variance":
        residuals = y_raw_original - np.asarray(fitted_at_samples).ravel()
    else:
        residuals = model.y_raw.ravel() - np.asarray(fitted_at_samples).ravel()

    metadata = {
        "options": opts,
        "order": order,
        "bandwidths_tested": bandwidths_for_selection,  # Use filtered bandwidths
        "fitted_at_samples": fitted_at_samples,
        "residuals": residuals,
        "mean": model.mean,
        "std": model.std,
        "dof": dof,
        "dof_stderr": dof_stderr,
    }

    return RegressionOutput(
        yhat=fitted_at_samples,
        grid=tuple(model.grid_axes_raw),
        h=selected_bandwidth,
        gcv=gcv_result,
        fpp_yhat=interpolant,
        metadata=metadata,
        dof=dof,
        dof_stderr=dof_stderr,
        d=d_normalization,
        s0=s0_selected,
    )


def cv_fastkde(
    x: np.ndarray,
    h: Optional[Sequence[Sequence[float]]] = None,
    options: Optional[dict] = None,
) -> KDEOutput:
    """
    Kernel density estimation with automatic bandwidth selection via LCV.

    Port from fastLPR/utility/cv_fastKDE.m.

    Uses NUFFT-based fast convolution for O(N + M log M) complexity, making it
    suitable for large datasets (N > 5000). Automatically selects bandwidth via
    Likelihood Cross-Validation (LCV) with 1-SE rule for robust smoothing.

    The density is normalized so that the integral equals 1.

    Parameters
    ----------
    x : np.ndarray
        N x d matrix of sample data. Can be 1D (column vector) or multi-dimensional.
        Complex values are split into [real, imag] columns automatically.
    h : Optional[Sequence[Sequence[float]]]
        Bandwidth parameter(s) (optional, default: automatic selection).
        - Scalar: same bandwidth for all dimensions
        - 1 x d vector: different bandwidth per dimension
        - k x d matrix: grid of k bandwidth combinations for LCV selection
        If not provided, generates logarithmic grid from 0.1*std to 1.0*std.
    options : Optional[dict]
        Options dictionary with fields:
        - kernel_type: Kernel function (default: 'gaussian')
        - dh: Number of bandwidth points for automatic selection (default: 20)
        - N: Grid resolution for evaluation (default: 200 for 1D, 50x50 for 2D)
             Controls accuracy vs speed tradeoff.
        - xrange: Evaluation range (default: data range)
        - dstd: Number of standard errors for 1-SE bandwidth selection rule (default: 0)
                The 1-SE rule selects the largest bandwidth h such that:
                LCV(h) >= LCV_max - dstd * SE(LCV_max)
                where SE is the standard error of the LCV score.
                - dstd=0: Select bandwidth with maximum LCV (no 1-SE rule)
                - dstd=1: Standard 1-SE rule (prefer smoother densities)
                - dstd>1: More aggressive smoothing (larger bandwidths)
        - verbose: Print progress information (default: False)

    Returns
    -------
    KDEOutput
        Structure containing KDE results:
        - fhat: Density estimates at evaluation grid points
        - grid: Evaluation grid points (tuple of arrays)
        - bandwidth: Selected bandwidth (float or array)
        - cv_result: LCV results including all tested bandwidths and scores
        - interpolant: Interpolator for density evaluation at any point
        - metadata: Additional information (options, normalization, etc.)

    Examples
    --------
    Basic 1D density estimation:

    >>> import numpy as np
    >>> from fastlpr import cv_fastkde, get_hlist
    >>> np.random.seed(42)
    >>> x = np.random.randn(100, 1)
    >>> hlist = get_hlist(5, [0.2, 1.0])
    >>> kde = cv_fastkde(x, h=hlist)
    >>> kde.fhat.shape[0] > 0  # Grid has density values
    True
    >>> len(kde.grid) == 1  # 1D data
    True
    >>> kde.h.shape
    (1,)

    Evaluate density at new points using the interpolant:

    >>> x_new = np.linspace(-3, 3, 20).reshape(-1, 1)
    >>> density_new = kde.fpp(x_new)
    >>> density_new.shape
    (20,)
    >>> bool(np.all(density_new >= 0))  # Density is non-negative
    True

    Access LCV cross-validation results:

    >>> 'lcv_m' in kde.lcv  # LCV scores available
    True
    >>> len(kde.lcv['hlist']) == 5  # All bandwidths tested
    True

    Notes
    -----
    - Uses z-score normalization internally for numerical stability
    - Bandwidth selection is via Likelihood Cross-Validation (LCV):
      LCV(h) = (1/N) * sum_i log(f_-i(x_i))
      where f_-i is density estimate leaving out point i
    - 1-SE rule prefers larger (smoother) bandwidths among those with
      similar LCV scores, reducing overfitting
    - Grid size N controls accuracy vs speed: use 200+ for 1D publication plots,
      50x50 for 2D, reduce for faster computation

    See Also
    --------
    cv_fastlpr : Local polynomial regression with GCV bandwidth selection
    get_hlist : Generate logarithmic bandwidth grids
    fastkde_plot : Visualization functions for KDE results
    """
    from .regression import nufft_regression

    x_arr = _as_column_matrix(x)
    n, dims = x_arr.shape

    if n < 2:
        raise ValueError(
            f"Input x must have at least 2 observations. Got {n} observations."
        )

    # Input validation: check for NaN/Inf
    if np.any(np.isnan(x_arr)):
        raise ValueError("Input x contains NaN values. Please remove or impute NaN values before calling cv_fastkde.")
    if np.any(np.isinf(x_arr)):
        raise ValueError("Input x contains Inf values. Please remove or handle Inf values before calling cv_fastkde.")

    # Set KDE-specific options
    # KDE is equivalent to order=0 regression with y=1
    opts = set_defaults(options, order=0, calc_dof=False, y_type_out="mean")
    opts = set_defaults(opts, kernel_type="gaussian", verbose=False, dh=20)

    # Adaptive accuracy like MATLAB: accuracy = max(6 - ceil(log10(N)), 1)
    # Small N needs higher accuracy for better numerical precision
    # Minimum accuracy set to 4 to match MATLAB (was 4)
    if opts.get("accuracy") is None:
        adaptive_accuracy = max(6 - int(np.ceil(np.log10(n))), 4)
        opts["accuracy"] = adaptive_accuracy

    # Grid size N: match MATLAB fastLPR_create default
    # MATLAB: opt.N = ceil(opt.Nratio * (Tx^(1/dx))) with opt.Nratio default 1
    # Here: N = ceil(n^(1/dims)) where n = sample size
    if opts.get("N") is None:
        # Adaptive grid size: N = ceil(n^(1/dims)) for each dimension
        adaptive_N = int(np.ceil(n ** (1.0 / dims)))
        opts["N"] = [adaptive_N] * dims if dims > 1 else adaptive_N

    # Standardize data (z-score normalization)
    # Without this, the NUFFT produces incorrect density shapes
    # MATLAB: [x,~,x_std]=zscore(x);
    x_raw = x_arr.copy()  # Save original data
    x_mean = x_arr.mean(axis=0)
    x_std = x_arr.std(axis=0, ddof=1)  # Use ddof=1 to match MATLAB
    x_std[x_std == 0] = 1.0  # Avoid division by zero
    x_scaled = (x_arr - x_mean) / x_std

    # Center data at origin (for symmetric padding)
    # MATLAB: x=x-(x_max+x_min)/2;
    x_min_scaled = x_scaled.min(axis=0)
    x_max_scaled = x_scaled.max(axis=0)
    x_scaled = x_scaled - (x_max_scaled + x_min_scaled) / 2

    # Prepare bandwidths (in raw space)
    default_range = _default_hrange(x_arr)
    bandwidths = _prepare_bandwidths(h, dims, opts, default_range)
    dh = len(bandwidths)

    # Do NOT scale bandwidths to standardized space!
    # MATLAB uses raw bandwidths with standardized data
    # This is intentional - the bandwidth is specified in the original data scale

    # Compute density estimates using NUFFT
    # MATLAB: regs = fastLPR_create(x, y_dummy, h, opt)
    # MATLAB: regs = fastLPR_s(regs)
    # This computes sum_i K_h(x - x_i) at each grid point
    # For KDE, we want the design matrix S (sum of kernel weights), NOT the regression result m = t/s
    with timer(f"KDE for {dh} bandwidths"):
        # Compute design matrix S using NUFFT
        # For order=0, S = sum_i K_h(x - x_i) at each grid point
        from .regression import compute_design_matrix

        # Use standardized data for NUFFT computation
        # compute_design_matrix will create grid from x_scaled.min() to x_scaled.max()
        grid_shape = (
            opts["N"]
            if isinstance(opts["N"], int)
            else opts["N"][0]
            if dims == 1
            else opts["N"]
        )
        grid_shape = np.atleast_1d(np.asarray(grid_shape, dtype=int))
        if grid_shape.size == 1:
            grid_shape = np.repeat(grid_shape, dims)

        s_grid, params = compute_design_matrix(
            x_scaled,  # Use standardized data, not x_arr
            bandwidths,  # Use raw bandwidths (NOT scaled)
            grid_shape,
            kernel_type=opts.get("kernel_type", "gaussian"),
            order=0,
            accuracy=opts.get("accuracy", None),
            flag_power2=opts.get("flag_power2", True),
        )

        # Extract the grid axes that compute_design_matrix used (in scaled space)
        # compute_design_matrix creates grid from x_scaled.min() to x_scaled.max()
        grid_axes_scaled = []
        for d in range(dims):
            grid_axes_scaled.append(
                np.linspace(
                    x_scaled.min(axis=0)[d], x_scaled.max(axis=0)[d], grid_shape[d]
                )
            )

        # Transform grid axes back to raw space for output
        # Reverse the transformations: x_raw = x_scaled * x_std + x_mean (before centering)
        # But x_scaled was centered, so we need to uncenter first
        grid_axes_raw = []
        for d in range(dims):
            # Uncenter: add back the centering offset
            axis_uncentered = (
                grid_axes_scaled[d] + (x_max_scaled[d] + x_min_scaled[d]) / 2
            )
            # Reverse z-score: multiply by std and add mean
            axis_raw = axis_uncentered * x_std[d] + x_mean[d]
            grid_axes_raw.append(axis_raw)

    # s_grid shape from compute_design_matrix: (grid_size, dh, 1) for 1D, (N1, N2, dh, 1) for 2D
    # Squeeze the last dimension (dy=1 for KDE with y=ones)
    # Result: (grid_size, dh) for 1D, (N1, N2, dh) for 2D
    s_grid = np.squeeze(s_grid, axis=-1)

    # Normalize each bandwidth separately so integral = 1

    # Compute grid spacing (product of spacing in each dimension)
    # Use RAW grid spacing, not scaled!
    # The density integral should be computed in the original data space
    # MATLAB uses xlist (raw grid) for output, so dx is in raw space
    dx_grid = np.prod([ax[1] - ax[0] for ax in grid_axes_raw])

    # Normalize densities
    densities = []
    for ih in range(dh):
        # Extract density for this bandwidth
        s_current = s_grid[..., ih]

        # Take real part to remove numerical noise from FFT
        s_current = np.real(s_current)

        # Normalize so integral equals 1
        sum_s_dx = np.sum(s_current) * dx_grid
        if sum_s_dx > 0:
            fhat = s_current / sum_s_dx
        else:
            fhat = s_current

        # Clip to non-negative to remove numerical noise
        # KDE density is always >= 0 by definition
        fhat = np.maximum(fhat, 0.0)

        densities.append(fhat)

    # Compute LCV scores for bandwidth selection
    # Use efficient leave-one-out formula: f_{-i}(x_i) = (n/(n-1)) * [f(x_i) - K_h(0)/n]
    lcv_scores = []
    lcv_stderr = []

    for ih in range(dh):
        # Get current bandwidth
        h_current = bandwidths[ih]

        # Get density on grid for this bandwidth
        fhat_current = densities[ih]

        # Create interpolator for this bandwidth.
        # MATLAB uses regs.xlist (raw grid) and evaluates at raw x.
        # Using the raw grid here improves cross-language LCV consistency.
        interp_temp = create_interpolator(tuple(grid_axes_raw), fhat_current, method="linear")

        # Evaluate density at data points using interpolation (raw space)
        f_at_data = interp_temp(x_arr)

        # Compute K_h(0) - kernel value at zero distance
        # For Gaussian kernel: K(0) = 1/sqrt(2*pi)
        # Normalized by bandwidth: K_h(0) = K(0) / prod(h)
        K0 = 1.0 / np.sqrt(2.0 * np.pi)  # Gaussian kernel at 0
        K0_normalized = K0 / np.prod(h_current)

        # Compute leave-one-out density using efficient formula
        # f_{-i}(x_i) = (n/(n-1)) * [f(x_i) - K_h(0)/n]
        f_loo = (n / (n - 1)) * (f_at_data - K0_normalized / n)

        # Avoid log(0)
        f_loo = np.maximum(f_loo, 1e-10)

        # LCV score (higher is better)
        log_f_loo = np.log(f_loo)
        score = float(np.sum(log_f_loo))
        stderr = float(np.std(log_f_loo) * np.sqrt(n))  # Standard error

        lcv_scores.append(score)
        lcv_stderr.append(stderr)

    lcv_scores = np.asarray(lcv_scores)
    lcv_stderr = np.asarray(lcv_stderr)
    selected_idx = _one_se_rule(
        lcv_scores, lcv_stderr, bandwidths, prefer_smaller=False
    )
    selected_bandwidth = bandwidths[selected_idx]
    selected_density = densities[selected_idx]
    # Create interpolator using RAW grid (consistent with MATLAB's xlist)
    # MATLAB creates fpp using xlist which is based on xraw (original input)
    # Use linear interpolation by default for speed (especially 3D)
    # User can override with output_interp_method option
    output_interp_method = opts.get("output_interp_method", "linear")
    interpolant = create_interpolator(tuple(grid_axes_raw), selected_density, method=output_interp_method)

    unique_axes = [np.unique(bandwidths[:, d]) for d in range(dims)]
    grid_shape_lcv = tuple(len(axis) for axis in unique_axes)
    matrix = None
    if np.prod(grid_shape_lcv, dtype=int) == len(lcv_scores):
        matrix = lcv_scores.reshape(grid_shape_lcv)

    idmax = int(np.argmax(lcv_scores))
    lcv_result = {
        "hlist": bandwidths,
        "lcv_m": lcv_scores,
        "lcv_sd": lcv_stderr,
        "idmax": idmax,
        "id1se": selected_idx,
        "hmax": bandwidths[idmax],
        "h1se": selected_bandwidth,
        "unique_axes": unique_axes,
        "matrix": matrix,
    }
    # Store all densities (from MATLAB's kde.fhat_all)
    fhat_all = np.stack(densities, axis=-1)  # Shape: (grid_size, dh) for 1D

    metadata = {
        "options": opts,
        "bandwidths_tested": bandwidths,
        "fhat_all": fhat_all,
    }

    # Return grid in RAW space (for user output), not scaled space
    return KDEOutput(
        fhat=selected_density,
        grid=tuple(grid_axes_raw),
        h=selected_bandwidth,
        lcv=lcv_result,
        fpp=interpolant,
        metadata=metadata,
    )
