"""
complexity_lines.py - Add O(N), O(N log N), O(N^2) reference lines to plots

These faint dashed lines help visually verify algorithmic complexity claims
in log-log benchmark plots.

Usage:
    from complexity_lines import add_complexity_lines
    add_complexity_lines(ax, n_range=(512, 131072), complexities=['N', 'NlogN', 'N2'])

Author: Ying Wang, Min Li
Create Time: 2025-12-19
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict


# =============================================================================
# Complexity Reference Lines
# =============================================================================

# Complexity function definitions
COMPLEXITY_FUNCS = {
    'N': lambda n: n,
    'NlogN': lambda n: n * np.log2(n),
    'N2': lambda n: n ** 2,
    'N3': lambda n: n ** 3,
    'logN': lambda n: np.log2(n),
    'sqrtN': lambda n: np.sqrt(n),
    'MlogM': lambda n: n * np.log2(n),  # Alias for NlogN (when M~N)
}

# Display labels for each complexity
COMPLEXITY_LABELS = {
    'N': r'$O(N)$',
    'NlogN': r'$O(N \log N)$',
    'N2': r'$O(N^2)$',
    'N3': r'$O(N^3)$',
    'logN': r'$O(\log N)$',
    'sqrtN': r'$O(\sqrt{N})$',
    'MlogM': r'$O(M \log M)$',
}

# Default line styles
COMPLEXITY_STYLES = {
    'N': {'color': '#888888', 'linestyle': '--', 'alpha': 0.5},
    'NlogN': {'color': '#666666', 'linestyle': '-.', 'alpha': 0.5},
    'N2': {'color': '#444444', 'linestyle': ':', 'alpha': 0.5},
    'N3': {'color': '#333333', 'linestyle': ':', 'alpha': 0.4},
    'logN': {'color': '#aaaaaa', 'linestyle': '--', 'alpha': 0.5},
    'sqrtN': {'color': '#999999', 'linestyle': '--', 'alpha': 0.5},
    'MlogM': {'color': '#666666', 'linestyle': '-.', 'alpha': 0.5},
}


def add_complexity_lines(
    ax: plt.Axes,
    n_range: Tuple[float, float],
    complexities: List[str] = ['N', 'NlogN', 'N2'],
    y_range: Optional[Tuple[float, float]] = None,
    scale_to_data: bool = True,
    label_position: str = 'right',
    linewidth: float = 1.0,
    fontsize: int = 14,
    show_labels: bool = True
) -> Dict[str, plt.Line2D]:
    """Add complexity reference lines to a log-log plot.

    Lines are automatically scaled to fit within the visible y-range.

    Args:
        ax: Matplotlib axes (should already be log-log scale)
        n_range: (n_min, n_max) range for the reference lines
        complexities: List of complexity types to draw
        y_range: Optional (y_min, y_max) to constrain lines
        scale_to_data: If True, scale lines to fit existing data
        label_position: 'right', 'left', or 'end' for label placement
        linewidth: Line width
        fontsize: Font size for labels
        show_labels: Whether to show complexity labels

    Returns:
        Dictionary mapping complexity name to Line2D objects
    """
    n_min, n_max = n_range

    # Generate N values (geometric spacing for log scale)
    n_points = 100
    n_values = np.logspace(np.log10(n_min), np.log10(n_max), n_points)

    # Determine y_range from existing data if not provided
    if y_range is None and scale_to_data:
        # Get y limits from axes
        y_min, y_max = ax.get_ylim()
    elif y_range is not None:
        y_min, y_max = y_range
    else:
        y_min, y_max = 1e-3, 1e3  # Default range

    lines = {}

    for complexity in complexities:
        if complexity not in COMPLEXITY_FUNCS:
            print(f"Warning: Unknown complexity '{complexity}', skipping")
            continue

        # Compute complexity values
        func = COMPLEXITY_FUNCS[complexity]
        c_values = func(n_values)

        # Scale to fit in y_range
        # We want the line to pass through the middle of the log-log plot
        log_y_mid = (np.log10(y_min) + np.log10(y_max)) / 2
        log_c_mid = (np.log10(c_values.min()) + np.log10(c_values.max())) / 2
        scale_factor = 10 ** (log_y_mid - log_c_mid)
        c_scaled = c_values * scale_factor

        # Clip to y_range
        mask = (c_scaled >= y_min * 0.5) & (c_scaled <= y_max * 2)
        if not np.any(mask):
            continue

        # Get style
        style = COMPLEXITY_STYLES.get(complexity, {})

        # Plot line
        line, = ax.plot(
            n_values[mask], c_scaled[mask],
            color=style.get('color', '#888888'),
            linestyle=style.get('linestyle', '--'),
            alpha=style.get('alpha', 0.5),
            linewidth=linewidth,
            zorder=1  # Behind data
        )
        lines[complexity] = line

        # Add label
        if show_labels:
            label = COMPLEXITY_LABELS.get(complexity, complexity)

            if label_position == 'right':
                # Label at right end
                idx = np.where(mask)[0][-1]
                x_pos = n_values[idx] * 1.1
                y_pos = c_scaled[idx]
            elif label_position == 'left':
                # Label at left end
                idx = np.where(mask)[0][0]
                x_pos = n_values[idx] * 0.9
                y_pos = c_scaled[idx]
            else:  # 'end' or other
                # Label near maximum visible point
                idx = np.where(mask)[0][-1]
                x_pos = n_values[idx]
                y_pos = c_scaled[idx]

            ax.annotate(
                label,
                xy=(x_pos, y_pos),
                fontsize=fontsize,
                color=style.get('color', '#888888'),
                alpha=0.8,
                va='center',
                ha='left' if label_position == 'right' else 'right'
            )

    return lines


def add_speedup_annotation(
    ax: plt.Axes,
    x: float,
    y_fast: float,
    y_slow: float,
    text_format: str = '{:.0f}x',
    color: str = '#333333',
    fontsize: int = 8,
    offset: Tuple[float, float] = (10, 0)
) -> plt.Annotation:
    """Add a speedup annotation between two data points.

    Args:
        ax: Matplotlib axes
        x: X position
        y_fast: Y value of fast method
        y_slow: Y value of slow method
        text_format: Format string for speedup value
        color: Text color
        fontsize: Font size
        offset: Text offset in points

    Returns:
        Annotation object
    """
    speedup = y_slow / y_fast
    text = text_format.format(speedup)

    # Position at geometric mean of y values
    y_pos = np.sqrt(y_fast * y_slow)

    ann = ax.annotate(
        text,
        xy=(x, y_pos),
        xytext=offset,
        textcoords='offset points',
        fontsize=fontsize,
        color=color,
        ha='left',
        va='center',
        arrowprops=dict(arrowstyle='-', color=color, alpha=0.3)
    )

    return ann


def verify_complexity(
    n_values: np.ndarray,
    time_values: np.ndarray,
    expected: str,
    tolerance: float = 0.2
) -> Tuple[bool, float, str]:
    """Verify that timing data matches expected complexity.

    Uses linear regression on log-log data to estimate exponent.

    Args:
        n_values: Sample sizes
        time_values: Measured times
        expected: Expected complexity ('N', 'NlogN', 'N2')
        tolerance: Allowed deviation from expected exponent

    Returns:
        (is_match, estimated_exponent, complexity_guess)
    """
    # Expected exponents
    expected_exponents = {
        'N': 1.0,
        'NlogN': 1.0,  # Approximate as N for small ranges
        'N2': 2.0,
        'N3': 3.0,
        'logN': 0.0,
        'sqrtN': 0.5,
    }

    # Filter valid data
    mask = (time_values > 0) & np.isfinite(time_values)
    if np.sum(mask) < 3:
        return False, np.nan, 'insufficient_data'

    n_valid = n_values[mask]
    t_valid = time_values[mask]

    # Linear regression on log-log
    log_n = np.log(n_valid)
    log_t = np.log(t_valid)

    # Fit: log(t) = a * log(n) + b
    coeffs = np.polyfit(log_n, log_t, 1)
    estimated_exponent = coeffs[0]

    # Determine complexity guess
    if estimated_exponent < 0.3:
        guess = 'logN'
    elif estimated_exponent < 0.7:
        guess = 'sqrtN'
    elif estimated_exponent < 1.3:
        # Could be N or NlogN - check residuals
        guess = 'NlogN' if estimated_exponent > 1.1 else 'N'
    elif estimated_exponent < 2.3:
        guess = 'N2'
    else:
        guess = 'N3'

    # Check if matches expected
    exp_expected = expected_exponents.get(expected, 1.0)
    is_match = abs(estimated_exponent - exp_expected) <= tolerance

    return is_match, estimated_exponent, guess


if __name__ == '__main__':
    # Demo the complexity lines
    print("=== Complexity Lines Demo ===")
    print()

    import matplotlib
    matplotlib.use('Agg')

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create log-log axes
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    # Generate fake benchmark data
    n_values = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])

    # O(N log N) method
    t_fast = n_values * np.log2(n_values) / 1e7
    ax.plot(n_values, t_fast, 'o-', color='#1f77b4', linewidth=2,
            markersize=6, label='fastLPR (MATLAB)')

    # O(N^2) method
    t_slow = (n_values ** 2) / 1e9
    ax.plot(n_values, t_slow, 'x:', color='#7f7f7f', linewidth=1.5,
            markersize=6, label='Direct (baseline)')

    # Set axis limits
    ax.set_xlim(256, 131072)
    ax.set_ylim(1e-4, 1e2)

    # Add complexity reference lines
    print("Adding complexity reference lines...")
    add_complexity_lines(ax, n_range=(512, 65536),
                         complexities=['N', 'NlogN', 'N2'],
                         show_labels=True)

    # Add speedup annotation at largest N
    add_speedup_annotation(ax, n_values[-1], t_fast[-1], t_slow[-1])

    ax.set_xlabel('Sample Size N')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Benchmark with Complexity Reference Lines')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Save demo
    import os
    output_path = 'benchmark/figures/complexity_lines_demo.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")

    # Test complexity verification
    print("\nComplexity verification:")
    is_match, exp, guess = verify_complexity(n_values, t_fast, 'NlogN')
    print(f"  fastLPR: exponent={exp:.2f}, guess={guess}, matches NlogN: {is_match}")

    is_match, exp, guess = verify_complexity(n_values, t_slow, 'N2')
    print(f"  Direct: exponent={exp:.2f}, guess={guess}, matches N2: {is_match}")

    plt.close()
    print("\nComplexity lines module ready.")
