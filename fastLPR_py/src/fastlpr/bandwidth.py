# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Bandwidth utilities for fastlpr.

This module provides bandwidth grid generation for cross-validation
in local polynomial regression (GCV) and kernel density estimation (LCV).

API (Unified v2.0):
    - Parameter names unified across MATLAB, Python, R
    - Unified names: n (count), range, spacing
    - Default spacing: logspace (better for bandwidth search)
"""

from __future__ import annotations

from typing import Callable, Sequence, Union

import numpy as np


def get_hlist(
    n: Union[Sequence[int], int],
    range: np.ndarray,
    spacing: Callable[[float, float, int], np.ndarray] | None = None,
) -> np.ndarray:
    """
    Generate bandwidth candidates for cross-validation.

    Creates a grid of bandwidth values for automatic bandwidth selection
    via GCV (regression) or LCV (KDE). Default uses logarithmic spacing
    which works well for exploring bandwidth scales.

    Parameters
    ----------
    n : int or sequence of ints
        Number of bandwidth candidates per dimension.
        - Scalar: same number of points for all dimensions
        - Sequence: specify number of points per dimension
        Typical values: 20 for 1D, [15, 15] for 2D.
    range : array-like
        Bandwidth range [h_min, h_max] per dimension.
        Shape: (n_dims, 2) or (2,) for 1D.
        Rule of thumb:
        - h_min = 0.1 * std(x): fine details
        - h_max = 1.0 * std(x): smooth trend
    spacing : callable, optional
        Spacing function: (start, stop, num) -> vector.
        Default: logarithmic spacing (logspace).
        Use np.linspace for linear spacing.

    Returns
    -------
    np.ndarray
        Bandwidth grid with shape (prod(n), n_dims).
        Each row is a bandwidth vector [h1, h2, ..., hd].

    Examples
    --------
    1D bandwidth grid with 5 points (logarithmic spacing by default):

    >>> import numpy as np
    >>> from fastlpr.bandwidth import get_hlist  # May print version banner
    >>> hlist = get_hlist(5, [0.1, 1.0])
    >>> hlist.shape
    (5, 1)
    >>> bool(hlist[0, 0] < hlist[-1, 0])  # Monotonically increasing
    True

    2D bandwidth grid (Cartesian product):

    >>> hlist = get_hlist([3, 3], [[0.1, 1.0], [0.2, 2.0]])
    >>> hlist.shape
    (9, 2)

    Linear spacing instead of logarithmic:

    >>> hlist = get_hlist(5, [0.5, 1.0], spacing=np.linspace)
    >>> bool(np.allclose(hlist[1, 0] - hlist[0, 0], hlist[2, 0] - hlist[1, 0]))
    True

    Single bandwidth value:

    >>> hlist = get_hlist(1, [0.5, 0.5])
    >>> hlist
    array([[0.5]])

    Notes
    -----
    - Logarithmic spacing (default) explores bandwidth scales evenly
    - For multi-D, generates Cartesian product of per-dimension grids
    - Ordering matches MATLAB's ndgrid (column-major flattening)

    See Also
    --------
    cv_fastlpr : Regression with automatic bandwidth selection
    cv_fastkde : KDE with automatic bandwidth selection
    """

    # ============================================================
    # Main implementation
    # ============================================================

    range_arr = np.asarray(range, dtype=float)

    if range_arr.ndim == 1:
        range_arr = range_arr.reshape(1, 2)

    dims = range_arr.shape[0]

    if np.isscalar(n):
        n_list = [int(n)] * dims
    else:
        n_list = [int(v) for v in n]

    if len(n_list) != dims:
        raise ValueError("n must match the number of rows in range.")

    if spacing is None:
        spacing = lambda lo, hi, num: np.logspace(np.log10(lo), np.log10(hi), num)

    grids = [
        spacing(lo, hi, num) if num > 1 else np.array([lo], dtype=float)
        for (lo, hi), num in zip(range_arr, n_list)
    ]

    # Use meshgrid with 'ij' indexing to match MATLAB's ndgrid
    # MATLAB's get_hlist uses get_ndgrid which uses reshape(..., [], 1)
    # MATLAB's reshape uses column-major (Fortran) order, so we must use ravel('F')
    # to match MATLAB's ordering exactly.
    #
    # Example for 3x2 grid:
    # MATLAB order: [h1[0],h2[0]], [h1[1],h2[0]], [h1[2],h2[0]], [h1[0],h2[1]], [h1[1],h2[1]], [h1[2],h2[1]]
    # This is achieved by ravel('F') which iterates along the first dimension fastest
    mesh = np.meshgrid(*grids, indexing="ij")
    hlist = np.column_stack([axis.ravel("F") for axis in mesh])
    return hlist
