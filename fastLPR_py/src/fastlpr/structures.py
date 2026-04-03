# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Shared dataclasses for fastlpr outputs.

Unified API (v2.0):
    - h: Selected bandwidth
    - fpp_yhat: Interpolant for LPR prediction (RegressionOutput)
    - fpp: Interpolant for KDE density (KDEOutput)
    - gcv/lcv: Cross-validation results dict

Field names are consistent across MATLAB, Python, and R implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class RegressionOutput:
    """
    Output structure for cv_fastlpr().

    Attributes:
        yhat: Fitted values at data points
        grid: Evaluation grid as tuple of arrays
        h: Selected bandwidth (scalar or array)
        gcv: GCV results dict with keys: h1se, idmin, id1se, gcv_mean, gcv_dof, gcv_scores
        fpp_yhat: Interpolant object for prediction at new points
        metadata: Additional info (options, dimensions, etc.)
        dof: Estimated degrees of freedom
        dof_stderr: Standard error of DOF estimate
        d: Normalization factor for variance estimation
    """

    yhat: np.ndarray
    grid: tuple[np.ndarray, ...]
    h: np.ndarray
    gcv: Optional[Dict[str, object]] = None
    fpp_yhat: Optional[object] = None
    metadata: Optional[Dict[str, object]] = None
    dof: Optional[float] = None
    dof_stderr: Optional[float] = None
    d: Optional[float] = None
    s0: Optional[np.ndarray] = None  # Zero-th order kernel moment s_0(x) at grid points (= n * f_hat(x))


@dataclass
class KDEOutput:
    """
    Output structure for cv_fastkde().

    Attributes:
        fhat: Density estimates at evaluation grid points
        grid: Evaluation grid as tuple of arrays
        h: Selected bandwidth (scalar or array)
        lcv: LCV results dict with keys: h1se, idmax, id1se, lcv_mean, lcv_scores
        fpp: Interpolant object for density evaluation at new points
        metadata: Additional info (options, dimensions, etc.)
    """

    fhat: np.ndarray
    grid: tuple[np.ndarray, ...]
    h: np.ndarray
    lcv: Optional[Dict[str, object]] = None
    fpp: Optional[object] = None
    metadata: Optional[Dict[str, object]] = None
