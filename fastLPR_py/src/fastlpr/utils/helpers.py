# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Shared helper functions for fastLPR.

This module contains utility functions that are used across multiple modules
to avoid code duplication.
"""

from __future__ import annotations

import numpy as np


def is_numba_available() -> bool:
    """
    Check if Numba is available and properly configured.

    Returns
    -------
    bool
        True if Numba is installed and can be imported, False otherwise.
    """
    try:
        import numba
        return True
    except ImportError:
        return False


def _as_column_matrix(x: np.ndarray) -> np.ndarray:
    """
    Convert input to column matrix, handling complex predictors.

    Complex predictors are split into [real, imag] columns (like MATLAB).
    Example: x = [1+2j, 3+4j] becomes x = [[1, 2], [3, 4]]

    Parameters
    ----------
    x : np.ndarray
        Input array (1D or 2D)

    Returns
    -------
    np.ndarray
        2D column matrix (N x D)

    Raises
    ------
    ValueError
        If input is not 1D or 2D array
    """
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("Input must be a 1D or 2D array.")

    # Handle complex predictors (MATLAB line 59-60 in fastLPR_create.m)
    if np.iscomplexobj(arr):
        # Split complex into [real, imag] columns
        real_part = arr.real
        imag_part = arr.imag
        # Only include imaginary columns for complex columns
        complex_cols = np.any(arr.imag != 0, axis=0)
        if np.any(complex_cols):
            arr = np.column_stack([real_part, imag_part[:, complex_cols]])
        else:
            arr = real_part

    return arr.astype(float)


def _default_hrange(points: np.ndarray) -> np.ndarray:
    """
    Compute default bandwidth range based on data scale.

    Parameters
    ----------
    points : np.ndarray
        Data points (N x D matrix)

    Returns
    -------
    np.ndarray
        Bandwidth range matrix (D x 2) with [hmin, hmax] for each dimension
    """
    scale = np.std(points, axis=0, ddof=1)
    scale[scale == 0] = 1.0
    hmin = 0.1 * scale
    hmax = 1.0 * scale
    return np.column_stack((hmin, hmax))
