# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Batch processing utilities for fastLPR to reduce function call overhead.

This module provides optimized batch processing for multiple bandwidths:
1. Precompute shared data once (grid, standardization, KDF)
2. Process all bandwidths in a single vectorized pass
3. Batch interpolation for GCV computation

Usage:
    from fastlpr.batch import precompute_shared, cv_fastlpr_batch

    # Precompute shared data
    shared = precompute_shared(x, y, hlist, opt)

    # Run batch CV (faster than calling cv_fastlpr multiple times)
    results = cv_fastlpr_batch(shared)

NOTE: This module is designed for advanced users who need fine-grained
control over the computation workflow. For most use cases, the standard
cv_fastlpr() function is recommended as it handles all optimization
internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .bandwidth import get_hlist
from .convolution import compute_kernel_fourier_cached
from .core import create_grid, create_interpolator
from .regression import (
    compute_design_matrix,
    compute_gcv,
    estimate_dof_nufft_vectorized,
    nufft_regression_with_precomputed_kdf,
)
from .structures import RegressionOutput
from .utils import set_defaults
from .utils.helpers import _as_column_matrix, _default_hrange


@dataclass
class SharedData:
    """
    Container for precomputed shared data used across multiple bandwidths.

    This data is computed once and reused for all bandwidth evaluations,
    significantly reducing redundant computation.

    Attributes
    ----------
    x_raw : np.ndarray
        Original (unstandardized) predictor data
    y_raw : np.ndarray
        Original response data (or log-transformed for variance estimation)
    x_scaled : np.ndarray
        Standardized predictor data (z-score normalized)
    mean : np.ndarray
        Mean of original x (for reverse transform)
    std : np.ndarray
        Std of original x (for reverse transform)
    grid_axes_raw : list of np.ndarray
        Grid axes in original space
    grid_axes_scaled : list of np.ndarray
        Grid axes in standardized space
    kd_ft : np.ndarray or list
        Kernel in Fourier domain (precomputed for all bandwidths)
    kdf_params : dict
        Parameters from kernel computation (L, qin, qout, etc.)
    bandwidths_filtered : np.ndarray
        Valid bandwidths (after filtering out too-small ones)
    grid_shape : np.ndarray
        Shape of evaluation grid
    options : dict
        Merged options dictionary
    S : np.ndarray or list, optional
        Design matrix (precomputed for DOF estimation)
    S_params : dict, optional
        Parameters from design matrix computation
    y_raw_original : np.ndarray, optional
        Original y for variance estimation (before log transform)
    """

    x_raw: np.ndarray
    y_raw: np.ndarray
    x_scaled: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    grid_axes_raw: List[np.ndarray]
    grid_axes_scaled: List[np.ndarray]
    kd_ft: Union[np.ndarray, list]
    kdf_params: dict
    bandwidths_filtered: np.ndarray
    grid_shape: np.ndarray
    options: dict
    S: Optional[Union[np.ndarray, list]] = None
    S_params: Optional[dict] = None
    y_raw_original: Optional[np.ndarray] = None


# _as_column_matrix and _default_hrange are imported from .utils.helpers


def _prepare_bandwidths(
    h: Optional[np.ndarray],
    dims: int,
    options: Dict[str, object],
    default_range: np.ndarray,
) -> np.ndarray:
    """Prepare bandwidth array from various input formats."""
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
    return bandwidths


def precompute_shared(
    x: np.ndarray,
    y: np.ndarray,
    h: Optional[np.ndarray] = None,
    options: Optional[dict] = None,
    compute_design_matrix_for_dof: bool = True,
) -> SharedData:
    """
    Precompute shared data for batch bandwidth evaluation.

    This function computes all data that can be shared across bandwidth
    evaluations, including:
    - Data standardization (z-score normalization)
    - Grid creation
    - Kernel FFT (KDF) for all bandwidths
    - Design matrix S (optional, for DOF estimation)

    Parameters
    ----------
    x : np.ndarray
        N x d matrix of predictors
    y : np.ndarray
        Response vector (N samples) or matrix (N x dy for multiple responses)
    h : np.ndarray, optional
        Bandwidth candidates. If None, uses automatic grid.
    options : dict, optional
        Options dictionary (order, regularization, kernel_type, etc.)
    compute_design_matrix_for_dof : bool, default=True
        Whether to precompute design matrix S for DOF estimation.
        Set False if you don't need DOF (e.g., fast benchmarking).

    Returns
    -------
    SharedData
        Container with precomputed data for batch processing
    """
    x_arr = _as_column_matrix(x)

    # Validate and convert y
    y_arr = np.asarray(y)
    if y_arr.ndim == 1:
        y_arr = y_arr[:, None]
    elif y_arr.ndim == 2:
        if y_arr.shape[0] == 1 and y_arr.shape[1] > 1:
            y_arr = y_arr.T
    else:
        raise ValueError(f"Response y must be 1D or 2D array. Got {y_arr.ndim}D.")

    if y_arr.shape[0] != x_arr.shape[0]:
        raise ValueError(
            f"Response y must have same number of samples as x. "
            f"x has {x_arr.shape[0]} samples, y has {y_arr.shape[0]} samples."
        )

    # Set default options
    opts = set_defaults(
        options,
        order=0,
        regularization=1e-8,
        num_dof_sample=10,  # Number of DOF samples (matches MATLAB)
        kernel_type="gaussian",
        y_type_out="mean",
        calc_dof=True,
        gcv_selection="first",
    )

    # Adaptive accuracy
    n_samples, dims = x_arr.shape
    if opts.get("accuracy") is None:
        adaptive_accuracy = max(6 - int(np.ceil(np.log10(n_samples))), 4)
        opts["accuracy"] = adaptive_accuracy

    order = int(opts["order"])

    # Prepare bandwidths
    default_range = _default_hrange(x_arr)
    bandwidths = _prepare_bandwidths(h, dims, opts, default_range)

    # Handle variance estimation mode
    y_type_out = str(opts.get("y_type_out", "mean"))
    y_raw_original = None
    if y_type_out == "variance":
        y_raw_original = y_arr.ravel().copy()

    # ===== USE create_regression_model (matching api.py exactly) =====
    # This handles standardization, grid creation, and initial bandwidth filtering
    from .model import create_regression_model
    model = create_regression_model(x_arr, y_arr, bandwidths, opts)

    # Apply log transform for variance estimation AFTER model setup (matching api.py)
    if y_type_out == "variance":
        Ty = len(y_arr)
        y_raw_log = np.log(y_arr + 1.0 / Ty)
        model.y_raw = y_raw_log

    # Get grid shape (matching api.py line 372)
    grid_shape = np.array([len(ax) for ax in model.grid_axes_scaled])

    # ===== KDF COMPUTATION (matching api.py lines 380-391 exactly) =====
    # Note: model.bandwidths is ALREADY filtered by create_regression_model
    # Using cached version ensures same KDF is reused for DOF estimation later
    flag_power2 = bool(opts.get("flag_power2", True))
    kernel_type = str(opts["kernel_type"])

    kd_ft, kdf_params = compute_kernel_fourier_cached(
        model.x_scaled,
        model.bandwidths,  # Use model.bandwidths (already filtered)
        grid_shape,
        order=order,
        kernel_type=kernel_type,
        flag_power2=flag_power2,
    )

    # Get filtered bandwidths (matching api.py line 394)
    bandwidths_filtered = kdf_params["h"]

    # ===== DESIGN MATRIX FOR DOF (optional) =====
    S = None
    S_params = None
    if compute_design_matrix_for_dof and bool(opts.get("calc_dof", True)):
        S, S_params = compute_design_matrix(
            model.x_scaled,
            bandwidths_filtered,
            grid_shape,
            kernel_type=kernel_type,
            order=order,
            flag_power2=flag_power2,
        )

    return SharedData(
        x_raw=model.x_raw,
        y_raw=model.y_raw,
        x_scaled=model.x_scaled,
        mean=model.mean,
        std=model.std,
        grid_axes_raw=model.grid_axes_raw,
        grid_axes_scaled=model.grid_axes_scaled,
        kd_ft=kd_ft,
        kdf_params=kdf_params,
        bandwidths_filtered=bandwidths_filtered,
        grid_shape=grid_shape,
        options=opts,
        S=S,
        S_params=S_params,
        y_raw_original=y_raw_original,
    )


def cv_fastlpr_batch(
    shared: SharedData,
    return_all_grid_values: bool = False,
) -> RegressionOutput:
    """
    Run cross-validation for all bandwidths using precomputed shared data.

    This is the main batch processing function. It uses the precomputed
    shared data to evaluate all bandwidths efficiently in a single pass.

    Parameters
    ----------
    shared : SharedData
        Precomputed shared data from precompute_shared()
    return_all_grid_values : bool, default=False
        If True, include grid values for all bandwidths in metadata

    Returns
    -------
    RegressionOutput
        Regression results with selected bandwidth
    """
    kd_ft = shared.kd_ft
    kdf_params = shared.kdf_params
    bandwidths_filtered = shared.bandwidths_filtered
    grid_shape = shared.grid_shape
    opts = shared.options

    order = int(opts["order"])
    regularization = float(opts["regularization"])
    calc_dof = bool(opts.get("calc_dof", True))
    dims = shared.x_scaled.shape[1]
    dh_filtered = kdf_params["dh"]

    y_type_out = str(opts.get("y_type_out", "mean"))
    y_raw_original = shared.y_raw_original

    # FAST PATH: Skip DOF when calc_dof=False
    if not calc_dof:
        m_all, params = nufft_regression_with_precomputed_kdf(
            shared.x_scaled,
            shared.y_raw,
            kd_ft,
            kdf_params,
            grid_shape,
            order=order,
            regularization=regularization,
        )

        # Use first bandwidth
        if dh_filtered == 1:
            m = m_all
        else:
            m = m_all[..., 0]

        selected_bandwidth = bandwidths_filtered[0]
        grid_values = m

        interp_method = opts.get("interp_method", "linear")
        interpolant = create_interpolator(
            tuple(shared.grid_axes_raw), grid_values, method=interp_method
        )
        fitted_at_samples = interpolant(shared.x_raw)

        d_normalization = None
        if y_type_out == "variance" and y_raw_original is not None:
            Ty = len(y_raw_original)
            d_normalization = 1.0 / (
                (1.0 / Ty) * np.sum(y_raw_original * np.exp(-fitted_at_samples))
            )
            fitted_at_samples = np.exp(fitted_at_samples) / d_normalization
            grid_values = np.exp(grid_values) / d_normalization
            interpolant = create_interpolator(
                tuple(shared.grid_axes_raw), grid_values, method=interp_method
            )

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

        if y_type_out == "variance" and y_raw_original is not None:
            residuals = y_raw_original - np.asarray(fitted_at_samples).ravel()
        else:
            residuals = shared.y_raw.ravel() - np.asarray(fitted_at_samples).ravel()

        metadata = {
            "options": opts,
            "order": order,
            "bandwidths_tested": bandwidths_filtered,
            "fitted_at_samples": fitted_at_samples,
            "residuals": residuals,
            "mean": shared.mean,
            "std": shared.std,
            "dof": np.nan,
            "dof_stderr": np.nan,
        }

        return RegressionOutput(
            yhat=fitted_at_samples,
            grid=tuple(shared.grid_axes_raw),
            h=selected_bandwidth,
            gcv_yhat=gcv_result,
            fpp_yhat=interpolant,
            metadata=metadata,
            dof=np.nan,
            dof_stderr=np.nan,
            d=d_normalization,
        )

    # FULL PATH: DOF and GCV computation
    # num_dof_sample: Number of DOF samples (matches MATLAB option name)
    num_dof_samples = int(opts.get("num_dof_sample", 10))
    random_matrix = opts.get("random_matrix", None)
    dof_seed = opts.get("dof_seed", 42)

    default_dof_interp = "linear" if dims == 1 else "cubic"
    dof_interp_method = opts.get("dof_interp_method", default_dof_interp)

    y_is_complex = np.iscomplexobj(shared.y_raw)

    # Estimate DOF using cached S (or compute S if not cached)
    if shared.S is not None and shared.S_params is not None:
        dof_all, dof_stderr_all, penalty_std_all = estimate_dof_nufft_vectorized(
            shared.x_scaled,
            shared.x_raw,
            bandwidths_filtered,
            grid_shape,
            tuple(shared.grid_axes_raw),
            order,
            regularization,
            num_samples=num_dof_samples,
            random_state=dof_seed,
            random_matrix=random_matrix,
            S_precomputed=shared.S,
            params_precomputed=shared.S_params,
            interp_method=dof_interp_method,
            y_is_complex=y_is_complex,
        )
    else:
        dof_all, dof_stderr_all, penalty_std_all = estimate_dof_nufft_vectorized(
            shared.x_scaled,
            shared.x_raw,
            bandwidths_filtered,
            grid_shape,
            tuple(shared.grid_axes_raw),
            order,
            regularization,
            num_samples=num_dof_samples,
            random_state=dof_seed,
            random_matrix=random_matrix,
            interp_method=dof_interp_method,
            y_is_complex=y_is_complex,
        )

    # Compute regression for ALL bandwidths at once
    m_all, params = nufft_regression_with_precomputed_kdf(
        shared.x_scaled,
        shared.y_raw,
        kd_ft,
        kdf_params,
        grid_shape,
        order=order,
        regularization=regularization,
    )

    # Build grid values list
    grid_values_list = []
    for idx in range(dh_filtered):
        if dh_filtered == 1:
            m = m_all
        elif m_all.ndim == len(grid_shape) + 1:
            m = m_all[..., idx]
        else:
            m = m_all[..., idx, :]
        grid_values_list.append(m)

    # BATCH INTERPOLATION: Stack all grid values and interpolate at once
    default_gcv_interp = "linear" if dims == 1 else "cubic"
    gcv_interp_method = opts.get("gcv_interp_method", default_gcv_interp)

    m_stacked = np.stack(
        grid_values_list,
        axis=-1 if grid_values_list[0].ndim == len(grid_shape) else -2,
    )

    interp_batch = RegularGridInterpolator(
        tuple(shared.grid_axes_raw),
        m_stacked,
        method=gcv_interp_method,
        bounds_error=False,
        fill_value=None,
    )

    yhat_all = interp_batch(shared.x_raw)

    # Compute GCV for each bandwidth
    gcv_selection = str(opts.get("gcv_selection", "first"))
    gcv_scores = []
    gcv_stderr = []

    for idx in range(dh_filtered):
        if yhat_all.ndim == 2:
            yhat = yhat_all[:, idx]
        else:
            yhat = yhat_all[:, idx, :]

        yhat = np.asarray(yhat)
        dof = dof_all[idx]
        dof_stderr = dof_stderr_all[idx]
        penalty_std = penalty_std_all[idx]

        gcv, info = compute_gcv(
            shared.y_raw,
            yhat,
            dof,
            dof_stderr=dof_stderr,
            penalty_std=penalty_std,
            aggregation=gcv_selection,
        )

        gcv_scores.append(gcv)
        gcv_stderr.append(info.get("stderr", 0.0))

    gcv_scores = np.asarray(gcv_scores)
    gcv_stderr = np.asarray(gcv_stderr)

    # Select bandwidth using 1-SE rule
    from .api import _one_se_rule

    dstd_float = float(opts.get("dstd", 1.0))
    selected_idx = _one_se_rule(
        gcv_scores, gcv_stderr, bandwidths_filtered, prefer_smaller=True, dstd=dstd_float
    )
    selected_bandwidth = bandwidths_filtered[selected_idx]
    grid_values = grid_values_list[selected_idx]

    # Create final interpolator
    interpolant = create_interpolator(tuple(shared.grid_axes_raw), grid_values)
    fitted_at_samples = interpolant(shared.x_raw)

    dof = dof_all[selected_idx]
    dof_stderr = dof_stderr_all[selected_idx]

    # Variance estimation transform
    d_normalization = None
    if y_type_out == "variance" and y_raw_original is not None:
        Ty = len(y_raw_original)
        d_normalization = 1.0 / (
            (1.0 / Ty) * np.sum(y_raw_original * np.exp(-fitted_at_samples))
        )
        fitted_at_samples = np.exp(fitted_at_samples) / d_normalization
        grid_values = np.exp(grid_values) / d_normalization
        interpolant = create_interpolator(tuple(shared.grid_axes_raw), grid_values)

    idmin = int(np.argmin(gcv_scores))
    gcv_result = {
        "bandwidths": bandwidths_filtered,
        "gcv_m": gcv_scores,
        "gcv_sd": gcv_stderr,
        "idmin": idmin,
        "id1se": selected_idx,
        "h1se": selected_bandwidth,
        "hmin": bandwidths_filtered[idmin],
        "dof": dof_all,
        "dof_stderr": dof_stderr_all,
    }

    if y_type_out == "variance" and y_raw_original is not None:
        residuals = y_raw_original - np.asarray(fitted_at_samples).ravel()
    else:
        residuals = shared.y_raw.ravel() - np.asarray(fitted_at_samples).ravel()

    metadata = {
        "options": opts,
        "order": order,
        "bandwidths_tested": bandwidths_filtered,
        "fitted_at_samples": fitted_at_samples,
        "residuals": residuals,
        "mean": shared.mean,
        "std": shared.std,
        "dof": dof,
        "dof_stderr": dof_stderr,
    }

    if return_all_grid_values:
        metadata["all_grid_values"] = grid_values_list

    return RegressionOutput(
        yhat=fitted_at_samples,
        grid=tuple(shared.grid_axes_raw),
        h=selected_bandwidth,
        gcv_yhat=gcv_result,
        fpp_yhat=interpolant,
        metadata=metadata,
        dof=dof,
        dof_stderr=dof_stderr,
        d=d_normalization,
    )


def batch_evaluate_at_points(
    shared: SharedData,
    eval_points: np.ndarray,
    bandwidth_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Evaluate regression at custom points for multiple bandwidths.

    This is useful when you need to evaluate the fitted surface at
    points other than the training data.

    Parameters
    ----------
    shared : SharedData
        Precomputed shared data from precompute_shared()
    eval_points : np.ndarray
        Points at which to evaluate (M x dims)
    bandwidth_indices : list of int, optional
        Which bandwidths to evaluate. If None, evaluates all.

    Returns
    -------
    np.ndarray
        Evaluated values, shape (M, len(bandwidth_indices))
    """
    kd_ft = shared.kd_ft
    kdf_params = shared.kdf_params
    grid_shape = shared.grid_shape
    opts = shared.options

    order = int(opts["order"])
    regularization = float(opts["regularization"])

    # Compute regression
    m_all, _ = nufft_regression_with_precomputed_kdf(
        shared.x_scaled,
        shared.y_raw,
        kd_ft,
        kdf_params,
        grid_shape,
        order=order,
        regularization=regularization,
    )

    dh = kdf_params["dh"]
    if bandwidth_indices is None:
        bandwidth_indices = list(range(dh))

    # Build stacked grid values for selected bandwidths
    grid_values_list = []
    for idx in bandwidth_indices:
        if dh == 1:
            m = m_all
        elif m_all.ndim == len(grid_shape) + 1:
            m = m_all[..., idx]
        else:
            m = m_all[..., idx, :]
        grid_values_list.append(m)

    m_stacked = np.stack(grid_values_list, axis=-1)

    # Batch interpolation
    dims = shared.x_raw.shape[1]
    default_interp = "linear" if dims == 1 else "cubic"
    interp_method = opts.get("interp_method", default_interp)

    interp_batch = RegularGridInterpolator(
        tuple(shared.grid_axes_raw),
        m_stacked,
        method=interp_method,
        bounds_error=False,
        fill_value=None,
    )

    return interp_batch(eval_points)


def compute_gcv_scores_batch(
    shared: SharedData,
    m_all: np.ndarray,
    dof_all: np.ndarray,
    dof_stderr_all: np.ndarray,
    penalty_std_all: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute GCV scores for all bandwidths in batch.

    Parameters
    ----------
    shared : SharedData
        Precomputed shared data
    m_all : np.ndarray
        Regression results for all bandwidths
    dof_all : np.ndarray
        DOF for all bandwidths
    dof_stderr_all : np.ndarray
        DOF stderr for all bandwidths
    penalty_std_all : np.ndarray
        Penalty std for all bandwidths

    Returns
    -------
    gcv_scores : np.ndarray
        GCV scores for each bandwidth
    gcv_stderr : np.ndarray
        GCV standard errors for each bandwidth
    """
    grid_shape = shared.grid_shape
    opts = shared.options
    dh = shared.kdf_params["dh"]
    dims = shared.x_raw.shape[1]

    # Build grid values list
    grid_values_list = []
    for idx in range(dh):
        if dh == 1:
            m = m_all
        elif m_all.ndim == len(grid_shape) + 1:
            m = m_all[..., idx]
        else:
            m = m_all[..., idx, :]
        grid_values_list.append(m)

    # Batch interpolation
    default_gcv_interp = "linear" if dims == 1 else "cubic"
    gcv_interp_method = opts.get("gcv_interp_method", default_gcv_interp)

    m_stacked = np.stack(
        grid_values_list,
        axis=-1 if grid_values_list[0].ndim == len(grid_shape) else -2,
    )

    interp_batch = RegularGridInterpolator(
        tuple(shared.grid_axes_raw),
        m_stacked,
        method=gcv_interp_method,
        bounds_error=False,
        fill_value=None,
    )

    yhat_all = interp_batch(shared.x_raw)

    # Compute GCV for each bandwidth
    gcv_selection = str(opts.get("gcv_selection", "first"))
    gcv_scores = []
    gcv_stderr = []

    for idx in range(dh):
        if yhat_all.ndim == 2:
            yhat = yhat_all[:, idx]
        else:
            yhat = yhat_all[:, idx, :]

        yhat = np.asarray(yhat)

        gcv, info = compute_gcv(
            shared.y_raw,
            yhat,
            dof_all[idx],
            dof_stderr=dof_stderr_all[idx],
            penalty_std=penalty_std_all[idx],
            aggregation=gcv_selection,
        )

        gcv_scores.append(gcv)
        gcv_stderr.append(info.get("stderr", 0.0))

    return np.asarray(gcv_scores), np.asarray(gcv_stderr)
