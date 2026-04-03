# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
KDE-specific plotting functions for fastlpr.

Port from fastLPR/utility/fastKDE_plot.m. and fastKDE_plot_bandwidth.m

Author: Ying Wang, Min Li (MATLAB), Python port by fastLPR team
Copyright (c): 2020-2025 Ying Wang, Min Li
License: GNU General Public License v3.0
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .structures import KDEOutput


def fastkde_plot(
    kde: KDEOutput,
    x_data: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    **kwargs,
) -> Axes:
    """
    Visualize kernel density estimation results.

    Port from fastLPR/utility/fastKDE_plot.m.

    Parameters
    ----------
    kde : KDEOutput
        KDE structure from cv_fastkde
    x_data : ndarray, optional
        Original data points for visualization
    ax : Axes, optional
        Matplotlib axes to plot on
    **kwargs
        Additional plotting options:
        - plot_type: '1d', '2d_contour', '2d_surface', 'auto' (default)
        - show_data: Show raw data points (default: True)
        - show_histogram: Show histogram for 1D (default: True)
        - n_bins: Number of histogram bins (default: 30)
        - alpha: Transparency for data points (default: 0.4)
        - linewidth: Line width for 1D plots (default: 2)
        - title: Custom title
        - cmap: Colormap for 2D plots (default: 'jet')

    Returns
    -------
    ax : Axes
        Matplotlib axes object
    """
    if ax is None:
        ax = plt.gca()

    # Get options
    plot_type = kwargs.get("plot_type", "auto")
    show_data = kwargs.get("show_data", True)
    show_histogram = kwargs.get("show_histogram", True)
    n_bins = kwargs.get("n_bins", 30)
    alpha = kwargs.get("alpha", 0.4)
    linewidth = kwargs.get("linewidth", 2)
    title = kwargs.get("title", "")
    cmap = kwargs.get("cmap", "jet")

    # Determine dimensionality
    dims = len(kde.grid)

    # Auto-detect plot type
    if plot_type == "auto":
        plot_type = "1d" if dims == 1 else "2d_contour"

    if plot_type == "1d":
        # 1D density plot
        x_grid = kde.grid[0]

        # Plot histogram
        if show_histogram and x_data is not None:
            ax.hist(
                x_data.ravel(),
                bins=n_bins,
                density=True,
                color=[0.7, 0.7, 0.7],
                edgecolor="none",
                alpha=0.5,
                label="Histogram",
            )

        # Plot density
        ax.plot(
            x_grid,
            kde.fhat,
            "b-",
            linewidth=linewidth,
            label=f"KDE (h={kde.h[0]:.3f})",
        )

        # Plot data points
        if show_data and x_data is not None:
            y_pos = kde.fhat.min() - 0.05 * (kde.fhat.max() - kde.fhat.min())
            ax.plot(
                x_data.ravel(),
                np.full(len(x_data), y_pos),
                "k|",
                markersize=10,
                label="Data",
            )

        ax.set_xlabel(kwargs.get("xlabel", "x"), fontsize=12)
        ax.set_ylabel(kwargs.get("ylabel", "Density"), fontsize=12)
        if not title:
            title = "Kernel Density Estimate"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True)

    elif plot_type == "2d_contour":
        # 2D contour plot
        x1_grid, x2_grid = kde.grid
        X1, X2 = np.meshgrid(x1_grid, x2_grid, indexing="ij")

        # Plot contour
        cs = ax.contourf(X1, X2, kde.fhat, 20, cmap=cmap)
        plt.colorbar(cs, ax=ax)

        # Plot data points
        if show_data and x_data is not None:
            ax.scatter(
                x_data[:, 0], x_data[:, 1], s=5, c="k", alpha=alpha, label="Data"
            )

        ax.set_xlabel(kwargs.get("xlabel", "x_1"), fontsize=12)
        ax.set_ylabel(kwargs.get("ylabel", "x_2"), fontsize=12)
        if not title:
            title = f"2D KDE (h=[{kde.h[0]:.2f}, {kde.h[1]:.2f}])"
        ax.set_title(title, fontsize=13, fontweight="bold")

    elif plot_type == "2d_surface":
        # 2D surface plot
        from mpl_toolkits.mplot3d import Axes3D

        if not hasattr(ax, "plot_surface"):
            fig = plt.gcf()
            ax = fig.add_subplot(111, projection="3d")

        x1_grid, x2_grid = kde.grid
        X1, X2 = np.meshgrid(x1_grid, x2_grid, indexing="ij")

        ax.plot_surface(X1, X2, kde.fhat, cmap=cmap, alpha=0.8)

        ax.set_xlabel(kwargs.get("xlabel", "x_1"), fontsize=12)
        ax.set_ylabel(kwargs.get("ylabel", "x_2"), fontsize=12)
        ax.set_zlabel(kwargs.get("zlabel", "Density"), fontsize=12)
        if not title:
            title = f"2D KDE (h=[{kde.h[0]:.2f}, {kde.h[1]:.2f}])"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.view_init(elev=30, azim=45)

    else:
        raise ValueError(f"Invalid plot_type: {plot_type}")

    return ax


def fastkde_plot_bandwidth(kde: KDEOutput, ax: Optional[Axes] = None, **kwargs) -> Axes:
    """
    Visualize bandwidth selection results from cv_fastkde.

    Port from fastLPR/utility/fastKDE_plot_bandwidth.m.

    Parameters
    ----------
    kde : KDEOutput
        KDE structure from cv_fastkde with lcv
    ax : Axes, optional
        Matplotlib axes to plot on
    **kwargs
        Additional plotting options:
        - title: Custom title
        - xlabel: Custom x-label
        - ylabel: Custom y-label
        - cmap: Colormap for 2D plots (default: 'viridis')

    Returns
    -------
    ax : Axes
        Matplotlib axes object
    """
    if ax is None:
        ax = plt.gca()

    if kde.lcv is None or not kde.lcv:
        ax.text(
            0.5,
            0.5,
            "Single bandwidth (no selection needed)",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.axis("off")
        return ax

    # Get options
    title = kwargs.get("title", "")
    xlabel = kwargs.get("xlabel", "")
    ylabel = kwargs.get("ylabel", "")
    cmap = kwargs.get("cmap", "viridis")

    # Get LCV results
    lcv = kde.lcv
    dims = len(kde.h)

    if dims == 1:
        # 1D: Line plot
        hlist_vec = lcv["hlist"][:, 0]
        h_selected = kde.h[0]

        ax.semilogx(
            hlist_vec,
            lcv["lcv_m"],
            "ko-",
            linewidth=1.5,
            markersize=6,
            label="LCV scores",
        )
        ax.semilogx(
            h_selected,
            lcv["lcv_m"][lcv["id1se"]],
            "r*",
            markersize=20,
            linewidth=3,
            label="Selected",
        )

        if not xlabel:
            xlabel = "Bandwidth h"
        if not ylabel:
            ylabel = "Log-Likelihood CV Score"

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc="best", fontsize=10)

    else:
        # 2D: Heatmap
        if lcv["matrix"] is not None:
            lcv_matrix = lcv["matrix"]
        else:
            n_h = int(np.sqrt(len(lcv["lcv_m"])))
            lcv_matrix = lcv["lcv_m"].reshape(n_h, n_h)

        h1_vals = lcv["unique_axes"][0]
        h2_vals = lcv["unique_axes"][1]

        # Create heatmap
        im = ax.imshow(
            lcv_matrix.T,
            origin="lower",
            cmap=cmap,
            extent=[
                np.log10(h1_vals.min()),
                np.log10(h1_vals.max()),
                np.log10(h2_vals.min()),
                np.log10(h2_vals.max()),
            ],
            aspect="auto",
        )
        plt.colorbar(im, ax=ax)

        # Mark selected bandwidth
        ax.plot(
            np.log10(kde.h[0]),
            np.log10(kde.h[1]),
            "r*",
            markersize=25,
            linewidth=3,
        )
        ax.text(
            np.log10(kde.h[0]),
            np.log10(kde.h[1]) - 0.15,
            f"h=[{kde.h[0]:.2f}, {kde.h[1]:.2f}]",
            color="r",
            fontsize=10,
            fontweight="bold",
            ha="center",
        )

        if not xlabel:
            xlabel = "log₁₀(h₁)"
        if not ylabel:
            ylabel = "log₁₀(h₂)"

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

    # Title
    if not title:
        title = "Bandwidth Selection (LCV)"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True)

    return ax
