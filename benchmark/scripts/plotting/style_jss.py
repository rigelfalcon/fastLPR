"""
style_jss.py - JSS (Journal of Statistical Software) publication style module

Defines colorblind-friendly palettes, line styles, markers, and typography
for publication-quality benchmark figures.

Usage:
    from style_jss import apply_jss_style, COLORS, LINE_STYLES, MARKERS

Author: Ying Wang, Min Li
Create Time: 2025-12-19
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List, Optional, Tuple
import numpy as np


# =============================================================================
# Color Palette (Colorblind-friendly)
# =============================================================================

# Method colors - distinct and accessible
COLORS: Dict[str, str] = {
    # Our method (fastLPR family) - Blue tones
    'fastLPR': '#1f77b4',      # Blue
    'fastKDE': '#1f77b4',      # Blue (same as fastLPR)
    'fastLPR (MATLAB)': '#1f77b4',
    'fastLPR (Python)': '#1f77b4',
    'fastLPR (R)': '#1f77b4',
    'fastKDE (MATLAB)': '#1f77b4',
    'fastKDE (Python)': '#1f77b4',
    'fastKDE (R)': '#1f77b4',

    # Competitor methods - Distinct colors
    'ks': '#ff7f0e',           # Orange
    'npregfast': '#2ca02c',    # Green
    'FKSUM': '#9467bd',        # Purple
    'StOpt-NW': '#d62728',     # Red
    'StOpt': '#d62728',        # Red

    # Direct/baseline methods - Gray (dotted)
    'direct': '#7f7f7f',       # Gray
    'direct_kde': '#7f7f7f',
    'direct_nw': '#7f7f7f',
    'DirectKDE': '#7f7f7f',
    'DirectNW': '#7f7f7f',

    # Additional methods if needed
    'scipy': '#8c564b',        # Brown
    'statsmodels': '#e377c2',  # Pink
}


# =============================================================================
# Line Styles by Language
# =============================================================================

LINE_STYLES: Dict[str, str] = {
    'MATLAB': '-',      # Solid
    'Python': '--',     # Dashed
    'R': '-.',          # Dash-dot
    'direct': ':',      # Dotted (for baseline)
}


# =============================================================================
# Markers by Language
# =============================================================================

MARKERS: Dict[str, str] = {
    'MATLAB': 'o',      # Circle
    'Python': 's',      # Square
    'R': 'D',           # Diamond
    'direct': 'x',      # X (for baseline)
}


# =============================================================================
# Complete Style Dictionary (method -> {color, linestyle, marker})
# =============================================================================

def get_style(method: str, lang: Optional[str] = None) -> Dict[str, str]:
    """Get complete style for a method/language combination.

    Args:
        method: Method name (e.g., 'fastLPR', 'ks', 'direct')
        lang: Language (e.g., 'MATLAB', 'Python', 'R')

    Returns:
        Dictionary with 'color', 'linestyle', 'marker', 'label'
    """
    # Determine color
    if method in COLORS:
        color = COLORS[method]
    elif 'direct' in method.lower():
        color = COLORS['direct']
    else:
        color = '#333333'  # Default dark gray

    # Determine line style
    if 'direct' in method.lower():
        linestyle = ':'
    elif lang and lang in LINE_STYLES:
        linestyle = LINE_STYLES[lang]
    else:
        linestyle = '-'

    # Determine marker
    if 'direct' in method.lower():
        marker = 'x'
    elif lang and lang in MARKERS:
        marker = MARKERS[lang]
    else:
        marker = 'o'

    # Construct label
    if lang:
        label = f"{method} ({lang})"
    else:
        label = method

    return {
        'color': color,
        'linestyle': linestyle,
        'marker': marker,
        'label': label,
        'linewidth': 1.5,
        'markersize': 5,
    }


# =============================================================================
# Typography Settings
# =============================================================================

# JSS-style typography (serif fonts)
JSS_FONT_CONFIG = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'font.size': 20,           # doubled from 10
    'axes.titlesize': 24,      # doubled from 12
    'axes.labelsize': 22,      # doubled from 11
    'legend.fontsize': 18,     # doubled from 9
    'xtick.labelsize': 18,     # doubled from 9
    'ytick.labelsize': 18,     # doubled from 9
    'figure.titlesize': 28,    # doubled from 14
}

# Figure dimensions (inches) - typical JSS column width
JSS_FIGURE_CONFIG = {
    'figure.figsize': (6.5, 4.5),  # Single column
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
}

# Axes styling
JSS_AXES_CONFIG = {
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'axes.spines.top': True,
    'axes.spines.right': True,
}


def apply_jss_style(ax: Optional[plt.Axes] = None) -> None:
    """Apply JSS publication styling to axes or globally.

    Args:
        ax: Matplotlib axes (if None, applies globally via rcParams)
    """
    # Apply global style
    plt.rcParams.update(JSS_FONT_CONFIG)
    plt.rcParams.update(JSS_FIGURE_CONFIG)
    plt.rcParams.update(JSS_AXES_CONFIG)

    if ax is not None:
        # Apply to specific axes
        ax.tick_params(direction='in', top=True, right=True)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        # Ensure log-log axes have proper ticks
        if ax.get_xscale() == 'log':
            ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10))
        if ax.get_yscale() == 'log':
            ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10))


def create_figure(nrows: int = 1, ncols: int = 1,
                  figsize: Optional[Tuple[float, float]] = None,
                  **kwargs) -> Tuple[plt.Figure, np.ndarray]:
    """Create a figure with JSS styling.

    Args:
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        figsize: Figure size in inches (default: auto-scale)
        **kwargs: Additional arguments to plt.subplots

    Returns:
        fig, axes: Figure and axes array
    """
    apply_jss_style()

    if figsize is None:
        # Auto-scale based on number of subplots
        width = 6.5 if ncols == 1 else 10
        height = 4.5 * nrows / ncols if nrows > 1 else 4.5
        figsize = (width, height)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)

    # Apply style to all axes
    if isinstance(axes, np.ndarray):
        for ax in axes.flatten():
            apply_jss_style(ax)
    else:
        apply_jss_style(axes)

    return fig, axes


def create_global_legend(fig: plt.Figure, handles: List, labels: List,
                         ncol: int = 4, loc: str = 'lower center',
                         bbox_to_anchor: Tuple[float, float] = (0.5, -0.05),
                         **kwargs) -> None:
    """Create a single global legend at bottom of figure.

    Args:
        fig: Matplotlib figure
        handles: List of line handles
        labels: List of labels
        ncol: Number of columns
        loc: Legend location
        bbox_to_anchor: Anchor point for legend
        **kwargs: Additional legend arguments
    """
    fig.legend(
        handles, labels,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        frameon=True,
        framealpha=0.9,
        edgecolor='gray',
        **kwargs
    )


def add_error_bands(ax: plt.Axes, x: np.ndarray, y_mean: np.ndarray,
                    y_std: np.ndarray, color: str, alpha: float = 0.2,
                    label: Optional[str] = None) -> None:
    """Add shaded error bands to a plot.

    Args:
        ax: Matplotlib axes
        x: X values
        y_mean: Mean Y values
        y_std: Standard deviation of Y values
        color: Fill color
        alpha: Transparency
        label: Optional label for legend
    """
    ax.fill_between(
        x,
        y_mean - y_std,
        y_mean + y_std,
        color=color,
        alpha=alpha,
        label=label
    )


def format_log_axis(ax: plt.Axes, axis: str = 'both',
                    base: int = 10) -> None:
    """Format axis for log scale with nice tick labels.

    Args:
        ax: Matplotlib axes
        axis: Which axis ('x', 'y', or 'both')
        base: Logarithm base
    """
    if axis in ['x', 'both']:
        ax.set_xscale('log', base=base)
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())

    if axis in ['y', 'both']:
        ax.set_yscale('log', base=base)
        ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())


# =============================================================================
# Predefined Style Collections
# =============================================================================

# Complete style definitions for common methods
BENCHMARK_STYLES = {
    'fastKDE (MATLAB)': {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o'},
    'fastKDE (Python)': {'color': '#1f77b4', 'linestyle': '--', 'marker': 's'},
    'fastKDE (R)': {'color': '#1f77b4', 'linestyle': '-.', 'marker': 'D'},
    'fastLPR (MATLAB)': {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o'},
    'fastLPR (Python)': {'color': '#1f77b4', 'linestyle': '--', 'marker': 's'},
    'fastLPR (R)': {'color': '#1f77b4', 'linestyle': '-.', 'marker': 'D'},
    'ks': {'color': '#ff7f0e', 'linestyle': '-', 'marker': '^'},
    'npregfast': {'color': '#2ca02c', 'linestyle': '-', 'marker': 'v'},
    'FKSUM': {'color': '#9467bd', 'linestyle': '-', 'marker': '<'},
    'StOpt-NW': {'color': '#d62728', 'linestyle': '-', 'marker': '>'},
    'direct': {'color': '#7f7f7f', 'linestyle': ':', 'marker': 'x'},
    'DirectKDE': {'color': '#7f7f7f', 'linestyle': ':', 'marker': 'x'},
    'DirectNW': {'color': '#7f7f7f', 'linestyle': ':', 'marker': 'x'},
}


if __name__ == '__main__':
    # Demo the style system
    print("=== JSS Style Module Demo ===")
    print()

    # Show color palette
    print("Color Palette:")
    for name, color in list(COLORS.items())[:8]:
        print(f"  {name}: {color}")

    # Show styles
    print("\nBenchmark Styles:")
    for name, style in list(BENCHMARK_STYLES.items())[:5]:
        print(f"  {name}: {style}")

    # Create demo figure
    print("\nCreating demo figure...")
    fig, ax = create_figure(1, 1, figsize=(8, 5))

    # Demo data
    x = np.array([512, 1024, 2048, 4096, 8192, 16384])

    methods = ['fastKDE (MATLAB)', 'ks', 'FKSUM', 'direct']
    for i, method in enumerate(methods):
        style = BENCHMARK_STYLES.get(method, get_style(method))
        y = x ** (1.5 + 0.2 * i) / 1e6  # Fake timing data
        ax.plot(x, y, label=method,
                color=style['color'],
                linestyle=style['linestyle'],
                marker=style.get('marker', 'o'),
                linewidth=1.5, markersize=5)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Sample Size N')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Demo: Benchmark Comparison')
    ax.legend(loc='upper left')

    # Save demo
    output_path = 'benchmark/figures/style_demo.png'
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved to: {output_path}")

    plt.close()
    print("\nStyle module ready for use.")
