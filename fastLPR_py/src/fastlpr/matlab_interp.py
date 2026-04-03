# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
MATLAB-compatible interpolation wrapper.

This module provides interpolation that matches MATLAB's griddedInterpolant
behavior, including:
- Spline interpolation for interior points
- Linear extrapolation for exterior points using finite difference slopes

Key insight (from debugging):
MATLAB's griddedInterpolant with 'linear' extrapolation uses FINITE DIFFERENCE
slopes at the boundary, NOT the spline's analytical derivative!

For example, at the left boundary:
  slope = (values[1] - values[0]) / (grid[1] - grid[0])  # Forward difference
At the right boundary:
  slope = (values[-1] - values[-2]) / (grid[-1] - grid[-2])  # Backward difference

This is different from scipy.CubicSpline which uses the analytical spline derivative.

Key differences between scipy and MATLAB:
1. scipy.RegularGridInterpolator with fill_value=None uses nearest-neighbor extrapolation
2. MATLAB uses finite difference slopes for linear extrapolation (verified experimentally)
3. scipy.CubicSpline matches MATLAB spline for interior, but extrapolation differs

For interior points: Both match very closely (< 1e-5 error)
For boundary points: Both match exactly
For exterior points: Now matches MATLAB with finite difference extrapolation
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from typing import Tuple, Sequence, Union


class MatlabInterpolator:
    """
    MATLAB-compatible interpolator for 1D, 2D, and 3D grids.

    This class implements interpolation that matches MATLAB's griddedInterpolant
    with 'spline' interpolation and 'linear' extrapolation.

    For 1D: Uses scipy.CubicSpline with 'not-a-knot' boundary conditions
            (matches MATLAB spline very closely)
    For 2D+: Uses scipy.RegularGridInterpolator with 'cubic' method
            (minor differences from MATLAB, especially for extrapolation)

    Parameters
    ----------
    grid_axes : tuple of 1D arrays
        Grid coordinates for each dimension
    values : ndarray
        Values at grid points
    method : str, default='spline'
        'spline' for cubic spline (matches MATLAB), 'linear' for linear interpolation
    extrap_method : str, default='linear'
        'linear' for linear extrapolation (like MATLAB), 'nearest' for nearest-neighbor

    Examples
    --------
    >>> x = np.linspace(0, 1, 11)
    >>> y = np.sin(2*np.pi*x)
    >>> interp = MatlabInterpolator((x,), y)
    >>> interp(np.array([[0.5], [1.5]]))  # Interior and exterior points
    """

    def __init__(
        self,
        grid_axes: Tuple[np.ndarray, ...],
        values: np.ndarray,
        method: str = "spline",
        extrap_method: str = "linear"
    ):
        self.grid_axes = tuple(np.asarray(ax).ravel() for ax in grid_axes)
        self.ndim = len(self.grid_axes)
        self.values = np.asarray(values)
        self.method = method
        self.extrap_method = extrap_method

        # Compute bounds for extrapolation detection
        self.bounds = [(ax.min(), ax.max()) for ax in self.grid_axes]

        # Handle complex values
        self.is_complex = np.iscomplexobj(values)

        if self.ndim == 1:
            # 1D: Use CubicSpline which matches MATLAB spline very closely
            self._init_1d(method)
        else:
            # 2D+: Use RegularGridInterpolator
            self._init_nd(method)

    def _init_1d(self, method):
        """Initialize 1D interpolator using CubicSpline."""
        grid = self.grid_axes[0]
        vals = self.values

        if method == "spline" or method == "cubic":
            bc_type = 'not-a-knot'  # Matches MATLAB's default spline
        else:
            bc_type = None  # Not used for linear

        if self.is_complex:
            if method == "linear":
                self.interp_real = self._create_1d_linear(grid, vals.real)
                self.interp_imag = self._create_1d_linear(grid, vals.imag)
            else:
                self.interp_real = CubicSpline(grid, vals.real, bc_type=bc_type)
                self.interp_imag = CubicSpline(grid, vals.imag, bc_type=bc_type)
        else:
            if method == "linear":
                self.interp_real = self._create_1d_linear(grid, vals)
            else:
                self.interp_real = CubicSpline(grid, vals, bc_type=bc_type)
            self.interp_imag = None

        # Pre-compute FINITE DIFFERENCE slopes for extrapolation (like MATLAB)
        # This is the key insight: MATLAB uses finite differences, not spline derivatives!
        h_left = grid[1] - grid[0]
        h_right = grid[-1] - grid[-2]

        if self.is_complex:
            self.slope_left_real = (vals.real[1] - vals.real[0]) / h_left
            self.slope_left_imag = (vals.imag[1] - vals.imag[0]) / h_left
            self.slope_right_real = (vals.real[-1] - vals.real[-2]) / h_right
            self.slope_right_imag = (vals.imag[-1] - vals.imag[-2]) / h_right
            self.val_left_real = vals.real[0]
            self.val_left_imag = vals.imag[0]
            self.val_right_real = vals.real[-1]
            self.val_right_imag = vals.imag[-1]
        else:
            self.slope_left = (vals[1] - vals[0]) / h_left
            self.slope_right = (vals[-1] - vals[-2]) / h_right
            self.val_left = vals[0]
            self.val_right = vals[-1]

    def _create_1d_linear(self, x, y):
        """Create a simple 1D linear interpolator."""
        from scipy.interpolate import interp1d
        return interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')

    def _init_nd(self, method):
        """Initialize N-D interpolator using RegularGridInterpolator."""
        scipy_method = 'cubic' if method in ('spline', 'cubic') else 'linear'

        if self.is_complex:
            self.interp_real = RegularGridInterpolator(
                self.grid_axes, self.values.real,
                method=scipy_method, bounds_error=False, fill_value=None
            )
            self.interp_imag = RegularGridInterpolator(
                self.grid_axes, self.values.imag,
                method=scipy_method, bounds_error=False, fill_value=None
            )
        else:
            self.interp_real = RegularGridInterpolator(
                self.grid_axes, self.values,
                method=scipy_method, bounds_error=False, fill_value=None
            )
            self.interp_imag = None

        # Pre-compute grid spacing for each dimension (like MATLAB's finite difference)
        self.grid_h = []
        for d, ax in enumerate(self.grid_axes):
            h_left = ax[1] - ax[0] if len(ax) > 1 else 1.0
            h_right = ax[-1] - ax[-2] if len(ax) > 1 else 1.0
            self.grid_h.append((h_left, h_right))

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate interpolator at given points.

        Parameters
        ----------
        points : ndarray, shape (n_points, ndim) or (n_points,) for 1D
            Query points

        Returns
        -------
        values : ndarray, shape (n_points,)
            Interpolated values
        """
        points = np.atleast_2d(points)
        n_points = points.shape[0]

        if self.ndim == 1:
            result = self._eval_1d(points.ravel())
        else:
            result = self._eval_nd(points)

        return result

    def _eval_1d(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate 1D interpolation with MATLAB-style linear extrapolation.

        CRITICAL: MATLAB uses FINITE DIFFERENCE slopes for extrapolation,
        not the spline's analytical derivative!
        """
        bounds_low, bounds_high = self.bounds[0]

        # Identify in-bounds vs out-of-bounds points
        in_bounds = (xi >= bounds_low) & (xi <= bounds_high)

        if self.is_complex:
            result = np.zeros(len(xi), dtype=complex)
        else:
            result = np.zeros(len(xi))

        # Evaluate in-bounds points directly using spline
        if np.any(in_bounds):
            if self.is_complex:
                result[in_bounds] = (
                    self.interp_real(xi[in_bounds]) +
                    1j * self.interp_imag(xi[in_bounds])
                )
            else:
                result[in_bounds] = self.interp_real(xi[in_bounds])

        # Handle out-of-bounds points with linear extrapolation
        # using PRE-COMPUTED FINITE DIFFERENCE slopes (not spline derivatives!)
        out_low = xi < bounds_low
        out_high = xi > bounds_high

        if np.any(out_low):
            # Linear extrapolation from left boundary
            if self.is_complex:
                result[out_low] = (
                    (self.val_left_real + self.slope_left_real * (xi[out_low] - bounds_low)) +
                    1j * (self.val_left_imag + self.slope_left_imag * (xi[out_low] - bounds_low))
                )
            else:
                result[out_low] = self.val_left + self.slope_left * (xi[out_low] - bounds_low)

        if np.any(out_high):
            # Linear extrapolation from right boundary
            if self.is_complex:
                result[out_high] = (
                    (self.val_right_real + self.slope_right_real * (xi[out_high] - bounds_high)) +
                    1j * (self.val_right_imag + self.slope_right_imag * (xi[out_high] - bounds_high))
                )
            else:
                result[out_high] = self.val_right + self.slope_right * (xi[out_high] - bounds_high)

        return result

    def _eval_nd(self, points: np.ndarray) -> np.ndarray:
        """Evaluate N-D interpolation with improved extrapolation."""
        n_points = points.shape[0]

        # Check which points are out of bounds
        bounds_lower = np.array([b[0] for b in self.bounds])
        bounds_upper = np.array([b[1] for b in self.bounds])
        out_of_bounds = np.any((points < bounds_lower) | (points > bounds_upper), axis=1)

        if self.is_complex:
            result = np.zeros(n_points, dtype=complex)
        else:
            result = np.zeros(n_points)

        # Evaluate in-bounds points directly
        in_bounds = ~out_of_bounds
        if np.any(in_bounds):
            if self.is_complex:
                result[in_bounds] = (
                    self.interp_real(points[in_bounds]) +
                    1j * self.interp_imag(points[in_bounds])
                )
            else:
                result[in_bounds] = self.interp_real(points[in_bounds])

        # For out-of-bounds points, use linear extrapolation based on gradient at boundary
        if np.any(out_of_bounds):
            oob_points = points[out_of_bounds]
            oob_result = self._linear_extrapolate_nd(oob_points, bounds_lower, bounds_upper)
            result[out_of_bounds] = oob_result

        return result

    def _linear_extrapolate_nd(
        self,
        points: np.ndarray,
        bounds_lower: np.ndarray,
        bounds_upper: np.ndarray
    ) -> np.ndarray:
        """
        Linear extrapolation for N-D points outside the grid.

        Uses finite difference slopes at the boundary (like MATLAB).
        The slope is computed using the actual grid spacing, not an arbitrary h.
        """
        n_points = points.shape[0]

        # Clip points to grid boundary
        clipped = np.clip(points, bounds_lower, bounds_upper)

        # Evaluate at boundary
        if self.is_complex:
            val_at_boundary = self.interp_real(clipped) + 1j * self.interp_imag(clipped)
        else:
            val_at_boundary = self.interp_real(clipped)

        gradients = np.zeros((n_points, self.ndim), dtype=val_at_boundary.dtype)

        for d in range(self.ndim):
            # Use actual grid spacing for finite differences (like MATLAB)
            h_left, h_right = self.grid_h[d]

            # Compute step direction based on where point is out of bounds
            at_lower = points[:, d] < bounds_lower[d]
            at_upper = points[:, d] > bounds_upper[d]

            # Forward difference for points at or below lower boundary
            if np.any(at_lower):
                pt_plus = clipped[at_lower].copy()
                pt_plus[:, d] = np.minimum(clipped[at_lower, d] + h_left, bounds_upper[d])
                if self.is_complex:
                    val_plus = self.interp_real(pt_plus) + 1j * self.interp_imag(pt_plus)
                else:
                    val_plus = self.interp_real(pt_plus)
                gradients[at_lower, d] = (val_plus - val_at_boundary[at_lower]) / h_left

            # Backward difference for points at or above upper boundary
            if np.any(at_upper):
                pt_minus = clipped[at_upper].copy()
                pt_minus[:, d] = np.maximum(clipped[at_upper, d] - h_right, bounds_lower[d])
                if self.is_complex:
                    val_minus = self.interp_real(pt_minus) + 1j * self.interp_imag(pt_minus)
                else:
                    val_minus = self.interp_real(pt_minus)
                gradients[at_upper, d] = (val_at_boundary[at_upper] - val_minus) / h_right

        # Linear extrapolation: f(x) = f(boundary) + gradient · (x - boundary)
        displacement = points - clipped
        result = val_at_boundary + np.sum(gradients * displacement, axis=1)

        return result


def matlab_interp(
    grid_axes: Tuple[np.ndarray, ...],
    values: np.ndarray,
    query_points: np.ndarray,
    method: str = "spline"
) -> np.ndarray:
    """
    MATLAB-compatible interpolation function.

    This is a convenience function that creates an interpolator and evaluates it.

    Parameters
    ----------
    grid_axes : tuple of 1D arrays
        Grid coordinates for each dimension
    values : ndarray
        Values at grid points
    query_points : ndarray
        Points at which to interpolate
    method : str, default='spline'
        Interpolation method

    Returns
    -------
    result : ndarray
        Interpolated values at query points
    """
    interp = MatlabInterpolator(grid_axes, values, method=method)
    return interp(query_points)
