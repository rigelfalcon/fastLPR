# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Interval computation for fastLPR.

Port from fastLPR/utility/fastLPR_interval.m.

This module provides functions to compute and visualize pointwise intervals
for local polynomial regression estimates. Two types are supported:

- **confidence**: Approximate asymptotic pointwise confidence interval for the
  regression mean m(x), based on the standard error of the estimator.
- **prediction**: Pointwise prediction interval for a new observation y|x,
  accounting for both observation noise and estimation uncertainty.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy.stats import norm

from .structures import RegressionOutput

# ---------------------------------------------------------------------------
# Precomputed asymptotic variance constants nu(d, ell) for Gaussian kernel.
#
# nu = e_1^T M_K^{-1} M_{K^2} M_K^{-1} e_1
# where M_K and M_{K^2} are theoretical kernel moment matrices.
#
# Valid at interior points under product Gaussian kernel.
# Computed via multi-index matrix algebra; see dev/option_c_implementation_plan.
# ---------------------------------------------------------------------------
_NU_TABLE = {
    # (d, ell): nu
    (1, 0): 0.282094791774,  # 1/(2*sqrt(pi))
    (1, 1): 0.282094791774,  # 1/(2*sqrt(pi))
    (1, 2): 0.476034961118,  # 27/(32*sqrt(pi))
    (2, 0): 0.079577471546,  # 1/(4*pi)
    (2, 1): 0.079577471546,  # 1/(4*pi)
    (2, 2): 0.198943678865,
    (3, 0): 0.022448390266,  # 1/(8*pi^{3/2})
    (3, 1): 0.022448390266,  # 1/(8*pi^{3/2})
    (3, 2): 0.077166341538,
}


def _get_nu(d: int, ell: int) -> float:
    """Look up the asymptotic variance constant for given (d, ell)."""
    key = (d, ell)
    if key not in _NU_TABLE:
        raise ValueError(
            f"nu constant not available for d={d}, order={ell}. "
            f"Supported: d in {{1,2,3}}, order in {{0,1,2}}."
        )
    return _NU_TABLE[key]


def fastlpr_interval(
    mu: RegressionOutput,
    sigma: Union[RegressionOutput, float, np.ndarray],
    alpha: float = 0.05,
    type: str = "confidence",
) -> np.ndarray:
    """
    Compute pointwise intervals for local polynomial regression estimates.

    Two interval types are supported:

    - ``type='confidence'``: Approximate asymptotic pointwise confidence
      interval for the regression mean *m(x)*:

        CI = m_hat +/- z * se(m_hat)

      where se^2 = sigma^2 * nu / (|H| * s_0), with s_0 the zero-th order
      kernel moment (stored in ``mu.s0`` by ``cv_fastlpr``).

    - ``type='prediction'``: Pointwise prediction interval for a new
      observation *y | x*, accounting for both noise and estimation
      uncertainty:

        PI = m_hat +/- z * sqrt(sigma^2 + se^2)

    Parameters
    ----------
    mu : RegressionOutput
        Mean estimate from cv_fastlpr. Must contain ``s0`` field (populated
        automatically by cv_fastlpr).

    sigma : RegressionOutput, float, or ndarray
        Variance estimate from cv_fastlpr (with y_type_out='variance'),
        or a scalar / array of variance values.

    alpha : float, default=0.05
        Significance level in (0, 1).

    type : str, default='confidence'
        Interval type: ``'confidence'`` or ``'prediction'``.

    Returns
    -------
    ci : ndarray
        Interval bounds with shape (..., 2).
        ``ci[..., 0]`` = upper, ``ci[..., 1]`` = lower.

    Notes
    -----
    - Intervals are pointwise (not simultaneous).
    - The confidence interval is an approximate asymptotic formula valid at
      interior points. Boundary behavior is not corrected.
    - Bias is not subtracted; at MSE-optimal bandwidth, actual coverage
      may be below the nominal level.
    - The design density f(x) is estimated internally from s_0 = n * f_hat(x),
      the zero-th order kernel moment already computed during regression.
      No separate KDE call is needed.

    Examples
    --------
    >>> # Step 1: mean estimation
    >>> regs_mu = cv_fastlpr(x, y, hlist, {'order': 1})
    >>>
    >>> # Step 2: variance estimation
    >>> residuals = y - regs_mu.yhat
    >>> regs_var = cv_fastlpr(x, residuals**2, hlist,
    ...                       {'y_type_out': 'variance', 'dstd': 10})
    >>>
    >>> # Step 3: intervals (no extra KDE needed)
    >>> ci = fastlpr_interval(regs_mu, regs_var, 0.05, type='confidence')
    >>> pi = fastlpr_interval(regs_mu, regs_var, 0.05, type='prediction')
    """
    # --- input validation ------------------------------------------------
    if not isinstance(mu, RegressionOutput):
        raise TypeError("mu must be a RegressionOutput object from cv_fastlpr")

    if mu.yhat is None:
        raise ValueError("mu must have .yhat field (fitted values)")

    if not isinstance(sigma, (RegressionOutput, float, int, np.ndarray)):
        raise TypeError("sigma must be RegressionOutput, float, or ndarray")

    if isinstance(sigma, RegressionOutput) and sigma.yhat is None:
        raise ValueError("sigma must have .yhat field (fitted values)")

    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    type = type.lower()
    if type not in ("confidence", "prediction"):
        raise ValueError(
            f"type must be 'confidence' or 'prediction', got '{type}'"
        )

    if mu.s0 is None:
        raise ValueError(
            "mu.s0 is None. The RegressionOutput must contain s_0 "
            "(zero-th kernel moment). Re-run cv_fastlpr with a recent version."
        )

    # --- z-score ----------------------------------------------------------
    z = norm.ppf(1 - alpha / 2)

    # --- get values on the native grid ------------------------------------
    # Operate on the grid values from fpp_yhat (same grid as s_0)
    if mu.fpp_yhat is not None and hasattr(mu.fpp_yhat, "values"):
        mu_values = mu.fpp_yhat.values
    else:
        mu_values = mu.yhat

    if isinstance(sigma, RegressionOutput):
        if sigma.fpp_yhat is not None and hasattr(sigma.fpp_yhat, "values"):
            sigma_values = sigma.fpp_yhat.values
        else:
            sigma_values = sigma.yhat
    else:
        sigma_values = np.asarray(sigma)

    # --- compute se^2 from s_0 -------------------------------------------
    sigma_pos = np.maximum(sigma_values, 0)
    s0 = np.maximum(mu.s0, 1e-10)  # clamp away from zero

    # Grid compatibility check: mu, sigma, and s_0 must be on the same grid
    if sigma_pos.shape != s0.shape:
        raise ValueError(
            f"Shape mismatch: sigma has shape {sigma_pos.shape} but s0 has "
            f"shape {s0.shape}. Ensure mu and sigma use the same grid."
        )
    if mu_values.shape != s0.shape:
        raise ValueError(
            f"Shape mismatch: mu has shape {mu_values.shape} but s0 has "
            f"shape {s0.shape}. Ensure mu grid values and s0 are aligned."
        )

    d = len(mu.grid)
    ell = int(mu.metadata["order"]) if mu.metadata else 0
    h = np.atleast_1d(mu.h).astype(float)
    prod_h = float(np.prod(h))
    nu = _get_nu(d, ell)

    # se^2(m_hat) = sigma^2 * nu / (|H| * s_0)
    se_sq = sigma_pos * nu / (prod_h * s0)

    # --- compute half-width -----------------------------------------------
    if type == "confidence":
        # CI = m_hat +/- z * sqrt(se^2)
        half_width = z * np.sqrt(se_sq)
    else:  # prediction
        # PI = m_hat +/- z * sqrt(sigma^2 + se^2)
        half_width = z * np.sqrt(sigma_pos + se_sq)

    # --- assemble bounds --------------------------------------------------
    upper = mu_values + half_width
    lower = mu_values - half_width
    ci = np.stack([upper, lower], axis=-1)

    return ci


def fastlpr_plot_interval(
    ci: np.ndarray,
    grid: Optional[tuple[np.ndarray, ...]] = None,
    ax=None,
    **kwargs,
):
    """
    Plot interval bands for regression estimates.

    Visualizes intervals computed by :func:`fastlpr_interval`.
    For 1D data, creates a filled area; for 2D, transparent surfaces.

    Parameters
    ----------
    ci : ndarray
        Interval bounds from fastlpr_interval, shape (..., 2).

    grid : tuple of ndarrays
        Grid axes for evaluation.

    ax : Axes, optional
        Matplotlib axes. If None, uses current axes.

    **kwargs
        color : str, default='g'
        alpha : float, default=0.2
        label : str, default='95% interval'

    Returns
    -------
    ax : Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    upper = ci[..., 0]
    lower = ci[..., 1]

    color = kwargs.get("color", "g")
    alpha_val = kwargs.get("alpha", 0.2)
    label = kwargs.get("label", "95% interval")

    if grid is None:
        raise ValueError("grid must be provided")

    dims = len(grid)

    if dims == 1:
        x_grid = grid[0]
        sort_idx = np.argsort(x_grid)
        x_sorted = x_grid[sort_idx]
        lower_sorted = lower.ravel()[sort_idx]
        upper_sorted = upper.ravel()[sort_idx]

        ax.fill_between(
            x_sorted,
            lower_sorted,
            upper_sorted,
            color=color,
            alpha=alpha_val,
            edgecolor="none",
            label=label,
        )

    elif dims == 2:
        x1_grid, x2_grid = grid
        X1, X2 = np.meshgrid(x1_grid, x2_grid, indexing="ij")

        ax.plot_surface(
            X1, X2, upper,
            color=color, alpha=alpha_val, edgecolor="none",
            label=f"{label} (upper)",
        )
        ax.plot_surface(
            X1, X2, lower,
            color=color, alpha=alpha_val, edgecolor="none",
            label=f"{label} (lower)",
        )
    else:
        raise ValueError(
            f"Unsupported dimensionality: {dims}. Only 1D and 2D supported."
        )

    return ax
