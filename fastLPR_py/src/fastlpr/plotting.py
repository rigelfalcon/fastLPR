# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Matplotlib-based plotting utilities for fastLPR.

Port from fastLPR/utility/fastLPR_plot.m,. fastLPR_plot_interval.m, fastKDE_plot.m

This module provides plotting functions that match MATLAB visualization output.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.interpolate import RegularGridInterpolator

from .structures import RegressionOutput


def fastlpr_plot(
    interpolant: Union[RegularGridInterpolator, callable],
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    grid: Optional[Tuple[np.ndarray, ...]] = None,
    ax: Optional[Axes] = None,
    **kwargs,
) -> Axes:
    """
    Visualize regression results.

    Port from fastLPR/utility/fastLPR_plot.m.

    Supports:
    - 1D line plots
    - 2D surface/contour plots
    - Complex-valued data (magnitude/phase)

    Parameters
    ----------
    interpolant : RegularGridInterpolator or callable
        Fitted interpolator from regression
    x : ndarray, optional
        Original x data for scatter plot
    y : ndarray, optional
        Original y data for scatter plot
    grid : tuple of ndarrays, optional
        Grid axes for evaluation
    ax : Axes, optional
        Matplotlib axes to plot on
    **kwargs
        Additional plotting arguments (color, linestyle, linewidth, etc.)

    Returns
    -------
    ax : Axes
        Matplotlib axes object

    Examples
    --------
    Plot 1D regression result:

    >>> import numpy as np
    >>> import matplotlib
    >>> matplotlib.use('Agg')  # Non-interactive backend for doctest
    >>> import matplotlib.pyplot as plt
    >>> from fastlpr import cv_fastlpr, get_hlist
    >>> from fastlpr.plotting import fastlpr_plot
    >>> np.random.seed(42)
    >>> x = np.random.rand(50, 1)
    >>> y = np.sin(2*np.pi*x[:,0]) + 0.1*np.random.randn(50)
    >>> result = cv_fastlpr(x, y, h=get_hlist(3, [0.1, 0.3]), options={'order': 1, 'calc_dof': False})
    >>> ax = fastlpr_plot(result.fpp_yhat, x=x, y=y, grid=result.grid)
    >>> ax is not None
    True
    >>> plt.close('all')
    """
    if ax is None:
        ax = plt.gca()

    # Determine dimensionality
    if hasattr(interpolant, "grid"):
        grid_axes = interpolant.grid
        dims = len(grid_axes)
    elif grid is not None:
        grid_axes = grid
        dims = len(grid_axes)
    else:
        raise ValueError(
            "Either interpolant must have 'grid' attribute or grid must be provided"
        )

    if dims == 1:
        # 1D line plot
        x_eval = grid_axes[0]

        # Evaluate interpolant
        if hasattr(interpolant, "__call__"):
            y_eval = interpolant(x_eval[:, None])
        else:
            y_eval = interpolant(x_eval)

        # Plot fitted curve
        line_kwargs = {
            "color": kwargs.get("color", "r"),
            "linestyle": kwargs.get("linestyle", "-"),
            "linewidth": kwargs.get("LineWidth", kwargs.get("linewidth", 2)),
            "label": kwargs.get("DisplayName", kwargs.get("label", None)),
        }
        ax.plot(x_eval, y_eval, **line_kwargs)

        # Plot original data if provided
        if x is not None and y is not None:
            scatter_kwargs = {
                "c": kwargs.get("scatter_color", "k"),
                "s": kwargs.get("scatter_size", 20),
                "alpha": kwargs.get("scatter_alpha", 0.5),
            }
            ax.scatter(x, y, **scatter_kwargs)

        ax.set_xlabel(kwargs.get("xlabel", "X"))
        ax.set_ylabel(kwargs.get("ylabel", "Y"))

    elif dims == 2:
        # 2D surface/contour plot
        x1_eval, x2_eval = grid_axes
        X1, X2 = np.meshgrid(x1_eval, x2_eval, indexing="ij")

        # Evaluate interpolant
        points = np.column_stack([X1.ravel(), X2.ravel()])
        if hasattr(interpolant, "__call__"):
            Z = interpolant(points).reshape(X1.shape)
        else:
            Z = interpolant(points).reshape(X1.shape)

        # Choose plot type
        plot_type = kwargs.get("plot_type", "contourf")

        if plot_type == "surface" or plot_type == "surf":
            # 3D surface plot
            from mpl_toolkits.mplot3d import Axes3D

            if not hasattr(ax, "plot_surface"):
                fig = plt.gcf()
                ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(
                X1,
                X2,
                Z,
                cmap=kwargs.get("cmap", "viridis"),
                alpha=kwargs.get("alpha", 0.8),
            )
        elif plot_type == "contour":
            # Contour lines
            levels = kwargs.get("levels", 10)
            cs = ax.contour(
                X1, X2, Z, levels=levels, cmap=kwargs.get("cmap", "viridis")
            )
            ax.clabel(cs, inline=True, fontsize=8)
        else:
            # Filled contours (default)
            levels = kwargs.get("levels", 15)
            cs = ax.contourf(
                X1, X2, Z, levels=levels, cmap=kwargs.get("cmap", "viridis")
            )
            plt.colorbar(cs, ax=ax)

        ax.set_xlabel(kwargs.get("xlabel", "X1"))
        ax.set_ylabel(kwargs.get("ylabel", "X2"))

    else:
        raise ValueError(f"Plotting for {dims}D data not supported")

    if kwargs.get("title"):
        ax.set_title(kwargs["title"])

    if kwargs.get("grid", True):
        ax.grid(True, alpha=0.3)

    return ax
