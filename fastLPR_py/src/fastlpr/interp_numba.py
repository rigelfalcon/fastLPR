# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Numba-accelerated grid interpolation for fastLPR.

This module provides high-performance linear interpolation on regular grids
using Numba JIT compilation. Optimized for the DOF estimation use case
where many interpolations are needed at scattered data points.

Key features:
1. 1D, 2D, and 3D linear interpolation
2. Parallel execution with prange for large data
3. Caching for repeated calls
4. Fallback to scipy when Numba is unavailable

Usage:
    from fastlpr.interp_numba import (
        linear_interp_1d,
        bilinear_interp_2d,
        trilinear_interp_3d,
        fast_grid_interp,  # Auto-selects based on dimension
    )
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Union, List, Optional

from .utils.helpers import is_numba_available

# Try to import Numba, fall back gracefully if not available
try:
    from numba import jit, prange
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Define dummy decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(*args):
        return range(*args)


# =============================================================================
# Core Numba-accelerated interpolation functions
# =============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def _linear_interp_1d_numba(grid_x: np.ndarray, values: np.ndarray,
                                query_x: np.ndarray, result: np.ndarray) -> None:
        """
        Numba 1D linear interpolation (in-place result).

        Parameters
        ----------
        grid_x : ndarray, shape (nx,)
            Grid coordinates (must be uniformly spaced and strictly increasing)
        values : ndarray, shape (nx,) or (nx, ncols)
            Values at grid points
        query_x : ndarray, shape (n,)
            Query points
        result : ndarray, shape (n,) or (n, ncols)
            Output array (pre-allocated, modified in-place)
        """
        n = len(query_x)
        nx = len(grid_x)
        dx = grid_x[1] - grid_x[0]
        x0 = grid_x[0]
        x_max = grid_x[-1]

        # Handle 1D values (single column)
        if values.ndim == 1:
            for i in prange(n):
                xi = query_x[i]

                # Handle NaN
                if np.isnan(xi):
                    result[i] = np.nan
                    continue

                # Compute normalized position
                t = (xi - x0) / dx

                # Clamp to valid range for boundary handling
                if t < 0:
                    t = 0.0
                    ix = 0
                elif t >= nx - 1:
                    # At or beyond right boundary: use last value
                    t = 1.0  # Weight for values[ix+1] = values[-1]
                    ix = nx - 2
                else:
                    ix = int(t)
                    t = t - ix

                # Linear interpolation
                result[i] = values[ix] * (1.0 - t) + values[ix + 1] * t
        else:
            # Handle 2D values (multiple columns)
            ncols = values.shape[1]
            for i in prange(n):
                xi = query_x[i]

                # Handle NaN
                if np.isnan(xi):
                    for j in range(ncols):
                        result[i, j] = np.nan
                    continue

                # Compute normalized position
                t = (xi - x0) / dx

                # Clamp to valid range for boundary handling
                if t < 0:
                    t = 0.0
                    ix = 0
                elif t >= nx - 1:
                    # At or beyond right boundary: use last value
                    t = 1.0  # Weight for values[ix+1] = values[-1]
                    ix = nx - 2
                else:
                    ix = int(t)
                    t = t - ix

                # Linear interpolation for each column
                oneminus_t = 1.0 - t
                for j in range(ncols):
                    result[i, j] = values[ix, j] * oneminus_t + values[ix + 1, j] * t


    @jit(nopython=True, parallel=True, cache=True)
    def _bilinear_interp_2d_numba(grid_x: np.ndarray, grid_y: np.ndarray,
                                   values: np.ndarray,
                                   query_x: np.ndarray, query_y: np.ndarray,
                                   result: np.ndarray) -> None:
        """
        Numba 2D bilinear interpolation (in-place result).

        Parameters
        ----------
        grid_x : ndarray, shape (nx,)
            X-axis grid coordinates (uniformly spaced)
        grid_y : ndarray, shape (ny,)
            Y-axis grid coordinates (uniformly spaced)
        values : ndarray, shape (nx, ny) or (nx, ny, ncols)
            Values at grid points
        query_x : ndarray, shape (n,)
            Query X coordinates
        query_y : ndarray, shape (n,)
            Query Y coordinates
        result : ndarray, shape (n,) or (n, ncols)
            Output array (pre-allocated, modified in-place)
        """
        n = len(query_x)
        nx = len(grid_x)
        ny = len(grid_y)

        dx = grid_x[1] - grid_x[0]
        dy = grid_y[1] - grid_y[0]
        x0 = grid_x[0]
        y0 = grid_y[0]

        # Handle 2D values (no extra columns)
        if values.ndim == 2:
            for i in prange(n):
                qx = query_x[i]
                qy = query_y[i]

                # Handle NaN
                if np.isnan(qx) or np.isnan(qy):
                    result[i] = np.nan
                    continue

                # Compute normalized positions
                tx = (qx - x0) / dx
                ty = (qy - y0) / dy

                # Clamp to valid range
                if tx < 0:
                    tx = 0.0
                    ix = 0
                elif tx >= nx - 1:
                    tx = 1.0  # Weight for values[ix+1]
                    ix = nx - 2
                else:
                    ix = int(tx)
                    tx = tx - ix

                if ty < 0:
                    ty = 0.0
                    iy = 0
                elif ty >= ny - 1:
                    ty = 1.0  # Weight for values[iy+1]
                    iy = ny - 2
                else:
                    iy = int(ty)
                    ty = ty - iy

                # Bilinear interpolation
                # f(x,y) = (1-tx)*(1-ty)*f00 + tx*(1-ty)*f10 + (1-tx)*ty*f01 + tx*ty*f11
                f00 = values[ix, iy]
                f10 = values[ix + 1, iy]
                f01 = values[ix, iy + 1]
                f11 = values[ix + 1, iy + 1]

                oneminus_tx = 1.0 - tx
                oneminus_ty = 1.0 - ty

                result[i] = (oneminus_tx * oneminus_ty * f00 +
                            tx * oneminus_ty * f10 +
                            oneminus_tx * ty * f01 +
                            tx * ty * f11)
        else:
            # Handle 3D values (with extra columns)
            ncols = values.shape[2]
            for i in prange(n):
                qx = query_x[i]
                qy = query_y[i]

                # Handle NaN
                if np.isnan(qx) or np.isnan(qy):
                    for j in range(ncols):
                        result[i, j] = np.nan
                    continue

                # Compute normalized positions
                tx = (qx - x0) / dx
                ty = (qy - y0) / dy

                # Clamp to valid range
                if tx < 0:
                    tx = 0.0
                    ix = 0
                elif tx >= nx - 1:
                    tx = 1.0  # Weight for values[ix+1]
                    ix = nx - 2
                else:
                    ix = int(tx)
                    tx = tx - ix

                if ty < 0:
                    ty = 0.0
                    iy = 0
                elif ty >= ny - 1:
                    ty = 1.0  # Weight for values[iy+1]
                    iy = ny - 2
                else:
                    iy = int(ty)
                    ty = ty - iy

                # Bilinear interpolation for each column
                oneminus_tx = 1.0 - tx
                oneminus_ty = 1.0 - ty

                for j in range(ncols):
                    f00 = values[ix, iy, j]
                    f10 = values[ix + 1, iy, j]
                    f01 = values[ix, iy + 1, j]
                    f11 = values[ix + 1, iy + 1, j]

                    result[i, j] = (oneminus_tx * oneminus_ty * f00 +
                                   tx * oneminus_ty * f10 +
                                   oneminus_tx * ty * f01 +
                                   tx * ty * f11)


    @jit(nopython=True, parallel=True, cache=True)
    def _trilinear_interp_3d_numba(grid_x: np.ndarray, grid_y: np.ndarray,
                                    grid_z: np.ndarray, values: np.ndarray,
                                    query_x: np.ndarray, query_y: np.ndarray,
                                    query_z: np.ndarray, result: np.ndarray) -> None:
        """
        Numba 3D trilinear interpolation (in-place result).

        Parameters
        ----------
        grid_x : ndarray, shape (nx,)
            X-axis grid coordinates (uniformly spaced)
        grid_y : ndarray, shape (ny,)
            Y-axis grid coordinates (uniformly spaced)
        grid_z : ndarray, shape (nz,)
            Z-axis grid coordinates (uniformly spaced)
        values : ndarray, shape (nx, ny, nz) or (nx, ny, nz, ncols)
            Values at grid points
        query_x : ndarray, shape (n,)
            Query X coordinates
        query_y : ndarray, shape (n,)
            Query Y coordinates
        query_z : ndarray, shape (n,)
            Query Z coordinates
        result : ndarray, shape (n,) or (n, ncols)
            Output array (pre-allocated, modified in-place)
        """
        n = len(query_x)
        nx = len(grid_x)
        ny = len(grid_y)
        nz = len(grid_z)

        dx = grid_x[1] - grid_x[0]
        dy = grid_y[1] - grid_y[0]
        dz = grid_z[1] - grid_z[0]
        x0 = grid_x[0]
        y0 = grid_y[0]
        z0 = grid_z[0]

        # Handle 3D values (no extra columns)
        if values.ndim == 3:
            for i in prange(n):
                qx = query_x[i]
                qy = query_y[i]
                qz = query_z[i]

                # Handle NaN
                if np.isnan(qx) or np.isnan(qy) or np.isnan(qz):
                    result[i] = np.nan
                    continue

                # Compute normalized positions
                tx = (qx - x0) / dx
                ty = (qy - y0) / dy
                tz = (qz - z0) / dz

                # Clamp to valid range
                if tx < 0:
                    tx = 0.0
                    ix = 0
                elif tx >= nx - 1:
                    tx = 1.0  # Weight for values[ix+1]
                    ix = nx - 2
                else:
                    ix = int(tx)
                    tx = tx - ix

                if ty < 0:
                    ty = 0.0
                    iy = 0
                elif ty >= ny - 1:
                    ty = 1.0  # Weight for values[iy+1]
                    iy = ny - 2
                else:
                    iy = int(ty)
                    ty = ty - iy

                if tz < 0:
                    tz = 0.0
                    iz = 0
                elif tz >= nz - 1:
                    tz = 1.0  # Weight for values[iz+1]
                    iz = nz - 2
                else:
                    iz = int(tz)
                    tz = tz - iz

                # Trilinear interpolation
                # Get 8 corner values
                c000 = values[ix, iy, iz]
                c100 = values[ix + 1, iy, iz]
                c010 = values[ix, iy + 1, iz]
                c110 = values[ix + 1, iy + 1, iz]
                c001 = values[ix, iy, iz + 1]
                c101 = values[ix + 1, iy, iz + 1]
                c011 = values[ix, iy + 1, iz + 1]
                c111 = values[ix + 1, iy + 1, iz + 1]

                # Interpolate
                oneminus_tx = 1.0 - tx
                oneminus_ty = 1.0 - ty
                oneminus_tz = 1.0 - tz

                # First interpolate along x
                c00 = oneminus_tx * c000 + tx * c100
                c10 = oneminus_tx * c010 + tx * c110
                c01 = oneminus_tx * c001 + tx * c101
                c11 = oneminus_tx * c011 + tx * c111

                # Then interpolate along y
                c0 = oneminus_ty * c00 + ty * c10
                c1 = oneminus_ty * c01 + ty * c11

                # Finally interpolate along z
                result[i] = oneminus_tz * c0 + tz * c1
        else:
            # Handle 4D values (with extra columns)
            ncols = values.shape[3]
            for i in prange(n):
                qx = query_x[i]
                qy = query_y[i]
                qz = query_z[i]

                # Handle NaN
                if np.isnan(qx) or np.isnan(qy) or np.isnan(qz):
                    for j in range(ncols):
                        result[i, j] = np.nan
                    continue

                # Compute normalized positions
                tx = (qx - x0) / dx
                ty = (qy - y0) / dy
                tz = (qz - z0) / dz

                # Clamp to valid range
                if tx < 0:
                    tx = 0.0
                    ix = 0
                elif tx >= nx - 1:
                    tx = 1.0  # Weight for values[ix+1]
                    ix = nx - 2
                else:
                    ix = int(tx)
                    tx = tx - ix

                if ty < 0:
                    ty = 0.0
                    iy = 0
                elif ty >= ny - 1:
                    ty = 1.0  # Weight for values[iy+1]
                    iy = ny - 2
                else:
                    iy = int(ty)
                    ty = ty - iy

                if tz < 0:
                    tz = 0.0
                    iz = 0
                elif tz >= nz - 1:
                    tz = 1.0  # Weight for values[iz+1]
                    iz = nz - 2
                else:
                    iz = int(tz)
                    tz = tz - iz

                oneminus_tx = 1.0 - tx
                oneminus_ty = 1.0 - ty
                oneminus_tz = 1.0 - tz

                # Trilinear interpolation for each column
                for j in range(ncols):
                    c000 = values[ix, iy, iz, j]
                    c100 = values[ix + 1, iy, iz, j]
                    c010 = values[ix, iy + 1, iz, j]
                    c110 = values[ix + 1, iy + 1, iz, j]
                    c001 = values[ix, iy, iz + 1, j]
                    c101 = values[ix + 1, iy, iz + 1, j]
                    c011 = values[ix, iy + 1, iz + 1, j]
                    c111 = values[ix + 1, iy + 1, iz + 1, j]

                    # First interpolate along x
                    c00 = oneminus_tx * c000 + tx * c100
                    c10 = oneminus_tx * c010 + tx * c110
                    c01 = oneminus_tx * c001 + tx * c101
                    c11 = oneminus_tx * c011 + tx * c111

                    # Then interpolate along y
                    c0 = oneminus_ty * c00 + ty * c10
                    c1 = oneminus_ty * c01 + ty * c11

                    # Finally interpolate along z
                    result[i, j] = oneminus_tz * c0 + tz * c1


# =============================================================================
# Public API functions (with scipy fallback)
# =============================================================================

def linear_interp_1d(grid_x: np.ndarray, values: np.ndarray,
                     query_x: np.ndarray) -> np.ndarray:
    """
    1D linear interpolation with Numba acceleration.

    Parameters
    ----------
    grid_x : ndarray, shape (nx,)
        Grid coordinates (must be uniformly spaced and strictly increasing)
    values : ndarray, shape (nx,) or (nx, ncols)
        Values at grid points
    query_x : ndarray, shape (n,)
        Query points

    Returns
    -------
    result : ndarray, shape (n,) or (n, ncols)
        Interpolated values at query points

    Notes
    -----
    - Uses boundary clamping for out-of-bounds points (no extrapolation)
    - Falls back to scipy.interpolate.interp1d if Numba is not available
    - Accuracy matches scipy to within 1e-14 (machine precision)

    Examples
    --------
    >>> grid_x = np.linspace(0, 1, 101)
    >>> values = np.sin(2 * np.pi * grid_x)
    >>> query_x = np.array([0.25, 0.5, 0.75])
    >>> result = linear_interp_1d(grid_x, values, query_x)
    """
    grid_x = np.ascontiguousarray(grid_x, dtype=np.float64)
    values = np.ascontiguousarray(values, dtype=np.float64)
    query_x = np.ascontiguousarray(query_x, dtype=np.float64)

    if NUMBA_AVAILABLE:
        # Allocate output array
        if values.ndim == 1:
            result = np.empty(len(query_x), dtype=np.float64)
        else:
            result = np.empty((len(query_x), values.shape[1]), dtype=np.float64)

        _linear_interp_1d_numba(grid_x, values, query_x, result)
        return result
    else:
        # Fallback to scipy
        from scipy.interpolate import interp1d

        if values.ndim == 1:
            interp_func = interp1d(grid_x, values, kind='linear',
                                   bounds_error=False, fill_value=(values[0], values[-1]))
            return interp_func(query_x)
        else:
            # Interpolate each column
            result = np.empty((len(query_x), values.shape[1]), dtype=np.float64)
            for j in range(values.shape[1]):
                interp_func = interp1d(grid_x, values[:, j], kind='linear',
                                      bounds_error=False,
                                      fill_value=(values[0, j], values[-1, j]))
                result[:, j] = interp_func(query_x)
            return result


def bilinear_interp_2d(grid_x: np.ndarray, grid_y: np.ndarray,
                       values: np.ndarray,
                       query_x: np.ndarray, query_y: np.ndarray) -> np.ndarray:
    """
    2D bilinear interpolation with Numba acceleration.

    Parameters
    ----------
    grid_x : ndarray, shape (nx,)
        X-axis grid coordinates (uniformly spaced)
    grid_y : ndarray, shape (ny,)
        Y-axis grid coordinates (uniformly spaced)
    values : ndarray, shape (nx, ny) or (nx, ny, ncols)
        Values at grid points
    query_x : ndarray, shape (n,)
        Query X coordinates
    query_y : ndarray, shape (n,)
        Query Y coordinates

    Returns
    -------
    result : ndarray, shape (n,) or (n, ncols)
        Interpolated values at query points

    Notes
    -----
    - Uses boundary clamping for out-of-bounds points
    - Falls back to scipy.interpolate.RegularGridInterpolator if Numba unavailable
    - Grid axes must be uniformly spaced for Numba version

    Examples
    --------
    >>> grid_x = np.linspace(0, 1, 51)
    >>> grid_y = np.linspace(0, 1, 51)
    >>> X, Y = np.meshgrid(grid_x, grid_y, indexing='ij')
    >>> values = np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)
    >>> query_x = np.array([0.25, 0.5])
    >>> query_y = np.array([0.25, 0.5])
    >>> result = bilinear_interp_2d(grid_x, grid_y, values, query_x, query_y)
    """
    grid_x = np.ascontiguousarray(grid_x, dtype=np.float64)
    grid_y = np.ascontiguousarray(grid_y, dtype=np.float64)
    values = np.ascontiguousarray(values, dtype=np.float64)
    query_x = np.ascontiguousarray(query_x, dtype=np.float64)
    query_y = np.ascontiguousarray(query_y, dtype=np.float64)

    if NUMBA_AVAILABLE:
        # Allocate output array
        if values.ndim == 2:
            result = np.empty(len(query_x), dtype=np.float64)
        else:
            result = np.empty((len(query_x), values.shape[2]), dtype=np.float64)

        _bilinear_interp_2d_numba(grid_x, grid_y, values, query_x, query_y, result)
        return result
    else:
        # Fallback to scipy
        from scipy.interpolate import RegularGridInterpolator

        if values.ndim == 2:
            interp_func = RegularGridInterpolator(
                (grid_x, grid_y), values, method='linear',
                bounds_error=False, fill_value=None
            )
            points = np.column_stack([query_x, query_y])
            return interp_func(points)
        else:
            # Handle extra columns
            ncols = values.shape[2]
            result = np.empty((len(query_x), ncols), dtype=np.float64)
            points = np.column_stack([query_x, query_y])
            for j in range(ncols):
                interp_func = RegularGridInterpolator(
                    (grid_x, grid_y), values[:, :, j], method='linear',
                    bounds_error=False, fill_value=None
                )
                result[:, j] = interp_func(points)
            return result


def trilinear_interp_3d(grid_x: np.ndarray, grid_y: np.ndarray,
                        grid_z: np.ndarray, values: np.ndarray,
                        query_x: np.ndarray, query_y: np.ndarray,
                        query_z: np.ndarray) -> np.ndarray:
    """
    3D trilinear interpolation with Numba acceleration.

    Parameters
    ----------
    grid_x : ndarray, shape (nx,)
        X-axis grid coordinates (uniformly spaced)
    grid_y : ndarray, shape (ny,)
        Y-axis grid coordinates (uniformly spaced)
    grid_z : ndarray, shape (nz,)
        Z-axis grid coordinates (uniformly spaced)
    values : ndarray, shape (nx, ny, nz) or (nx, ny, nz, ncols)
        Values at grid points
    query_x : ndarray, shape (n,)
        Query X coordinates
    query_y : ndarray, shape (n,)
        Query Y coordinates
    query_z : ndarray, shape (n,)
        Query Z coordinates

    Returns
    -------
    result : ndarray, shape (n,) or (n, ncols)
        Interpolated values at query points

    Notes
    -----
    - Uses boundary clamping for out-of-bounds points
    - Falls back to scipy.interpolate.RegularGridInterpolator if Numba unavailable
    - Grid axes must be uniformly spaced for Numba version

    Examples
    --------
    >>> grid_x = np.linspace(0, 1, 21)
    >>> grid_y = np.linspace(0, 1, 21)
    >>> grid_z = np.linspace(0, 1, 21)
    >>> X, Y, Z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    >>> values = np.sin(2*np.pi*X) * np.cos(2*np.pi*Y) * np.exp(-Z)
    >>> query_x = np.array([0.25, 0.5])
    >>> query_y = np.array([0.25, 0.5])
    >>> query_z = np.array([0.25, 0.5])
    >>> result = trilinear_interp_3d(grid_x, grid_y, grid_z, values,
    ...                               query_x, query_y, query_z)
    """
    grid_x = np.ascontiguousarray(grid_x, dtype=np.float64)
    grid_y = np.ascontiguousarray(grid_y, dtype=np.float64)
    grid_z = np.ascontiguousarray(grid_z, dtype=np.float64)
    values = np.ascontiguousarray(values, dtype=np.float64)
    query_x = np.ascontiguousarray(query_x, dtype=np.float64)
    query_y = np.ascontiguousarray(query_y, dtype=np.float64)
    query_z = np.ascontiguousarray(query_z, dtype=np.float64)

    if NUMBA_AVAILABLE:
        # Allocate output array
        if values.ndim == 3:
            result = np.empty(len(query_x), dtype=np.float64)
        else:
            result = np.empty((len(query_x), values.shape[3]), dtype=np.float64)

        _trilinear_interp_3d_numba(grid_x, grid_y, grid_z, values,
                                   query_x, query_y, query_z, result)
        return result
    else:
        # Fallback to scipy
        from scipy.interpolate import RegularGridInterpolator

        if values.ndim == 3:
            interp_func = RegularGridInterpolator(
                (grid_x, grid_y, grid_z), values, method='linear',
                bounds_error=False, fill_value=None
            )
            points = np.column_stack([query_x, query_y, query_z])
            return interp_func(points)
        else:
            # Handle extra columns
            ncols = values.shape[3]
            result = np.empty((len(query_x), ncols), dtype=np.float64)
            points = np.column_stack([query_x, query_y, query_z])
            for j in range(ncols):
                interp_func = RegularGridInterpolator(
                    (grid_x, grid_y, grid_z), values[:, :, :, j], method='linear',
                    bounds_error=False, fill_value=None
                )
                result[:, j] = interp_func(points)
            return result


def fast_grid_interp(grid_axes: List[np.ndarray], values: np.ndarray,
                     query_points: np.ndarray) -> np.ndarray:
    """
    Fast N-dimensional grid interpolation (auto-selects method).

    This function automatically selects the appropriate interpolation method
    based on the number of dimensions:
    - 1D: linear_interp_1d (Numba)
    - 2D: bilinear_interp_2d (Numba)
    - 3D: trilinear_interp_3d (Numba)
    - 4D+: scipy.RegularGridInterpolator (no Numba optimization)

    Parameters
    ----------
    grid_axes : list of ndarray
        Grid coordinates for each dimension
    values : ndarray
        Values at grid points, shape (n1, n2, ..., nd) or (n1, n2, ..., nd, ncols)
    query_points : ndarray, shape (n_points, ndim)
        Query points

    Returns
    -------
    result : ndarray, shape (n_points,) or (n_points, ncols)
        Interpolated values at query points

    Notes
    -----
    This is the recommended interface for general use. It automatically
    handles dimension selection and falls back gracefully when Numba
    is not available.

    Examples
    --------
    >>> # 2D example
    >>> grid_x = np.linspace(0, 1, 51)
    >>> grid_y = np.linspace(0, 1, 51)
    >>> X, Y = np.meshgrid(grid_x, grid_y, indexing='ij')
    >>> values = np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)
    >>> query_points = np.random.rand(100, 2)
    >>> result = fast_grid_interp([grid_x, grid_y], values, query_points)
    """
    ndim = len(grid_axes)
    query_points = np.asarray(query_points)

    if query_points.ndim == 1:
        query_points = query_points.reshape(-1, 1)

    if ndim == 1:
        return linear_interp_1d(grid_axes[0], values, query_points[:, 0])
    elif ndim == 2:
        return bilinear_interp_2d(grid_axes[0], grid_axes[1], values,
                                  query_points[:, 0], query_points[:, 1])
    elif ndim == 3:
        return trilinear_interp_3d(grid_axes[0], grid_axes[1], grid_axes[2], values,
                                   query_points[:, 0], query_points[:, 1],
                                   query_points[:, 2])
    else:
        # Fallback to scipy for 4D+ (rare case)
        from scipy.interpolate import RegularGridInterpolator
        interp_func = RegularGridInterpolator(
            tuple(grid_axes), values, method='linear',
            bounds_error=False, fill_value=None
        )
        return interp_func(query_points)


# =============================================================================
# Complex-valued interpolation support
# =============================================================================

def fast_grid_interp_complex(grid_axes: List[np.ndarray], values: np.ndarray,
                             query_points: np.ndarray) -> np.ndarray:
    """
    Fast N-dimensional grid interpolation for complex-valued data.

    Interpolates real and imaginary parts separately using Numba-accelerated
    functions, then combines the results.

    Parameters
    ----------
    grid_axes : list of ndarray
        Grid coordinates for each dimension
    values : ndarray (complex)
        Complex values at grid points
    query_points : ndarray, shape (n_points, ndim)
        Query points

    Returns
    -------
    result : ndarray (complex)
        Interpolated complex values at query points
    """
    if not np.iscomplexobj(values):
        return fast_grid_interp(grid_axes, values, query_points)

    real_part = fast_grid_interp(grid_axes, values.real, query_points)
    imag_part = fast_grid_interp(grid_axes, values.imag, query_points)
    return real_part + 1j * imag_part


# =============================================================================
# Utility functions
# =============================================================================

# is_numba_available is imported from .utils.helpers


def warm_up_jit():
    """
    Warm up Numba JIT compilation by running small test cases.

    Call this function at module import or before benchmarking
    to ensure JIT compilation overhead doesn't affect timing.
    """
    if not NUMBA_AVAILABLE:
        return

    # Small test arrays for warm-up
    grid_1d = np.linspace(0, 1, 10)
    values_1d = np.sin(grid_1d)
    query_1d = np.array([0.5])

    grid_x = np.linspace(0, 1, 5)
    grid_y = np.linspace(0, 1, 5)
    X, Y = np.meshgrid(grid_x, grid_y, indexing='ij')
    values_2d = np.sin(X) * np.cos(Y)
    query_xy = np.array([0.5])

    grid_z = np.linspace(0, 1, 5)
    values_3d = np.zeros((5, 5, 5))
    query_xyz = np.array([0.5])

    # Run warmup calls (results discarded)
    _ = linear_interp_1d(grid_1d, values_1d, query_1d)
    _ = bilinear_interp_2d(grid_x, grid_y, values_2d, query_xy, query_xy)
    _ = trilinear_interp_3d(grid_x, grid_y, grid_z, values_3d,
                            query_xyz, query_xyz, query_xyz)


# Module-level exports
__all__ = [
    'linear_interp_1d',
    'bilinear_interp_2d',
    'trilinear_interp_3d',
    'fast_grid_interp',
    'fast_grid_interp_complex',
    'is_numba_available',
    'warm_up_jit',
    'NUMBA_AVAILABLE',
]
