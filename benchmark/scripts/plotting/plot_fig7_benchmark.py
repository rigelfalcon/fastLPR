"""
plot_fig7_benchmark.py - Figure 7: Publication-quality benchmark figures for JSS

Generates benchmark figures showing:
- Row 1: SPEED (computational time vs N)
- Row 2: MEMORY (memory usage vs N)
- Row 3: ACCURACY (ISE/MSE vs N)

Features (Updated 2025-12-20):
- Uses style_jss.py for consistent JSS publication styling
- Uses complexity_lines.py for O(N), O(N log N), O(N^2) reference lines
- Includes direct methods as O(N^2) baseline
- Error bands for timing variance (when available)
- Phase 3: Okabe-Ito colorblind-friendly palette
- Phase 3: Color = Method, Marker = Language implementation
- PDF primary output with PNG preview

Usage:
    python plot_fig7_benchmark.py

Output:
    benchmark/figures/fig7_kde_benchmark.pdf
    benchmark/figures/fig7_lpr_benchmark.pdf
    benchmark/figures/fig7_kde_benchmark.png (preview)
    benchmark/figures/fig7_lpr_benchmark.png (preview)

Author: Ying Wang, Min Li
Create Time: 2025-12-19
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.gridspec import GridSpec

# Add plotting module to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Import JSS style modules
from style_jss import (
    apply_jss_style,
    create_global_legend,
    add_error_bands,
    format_log_axis,
)
from complexity_lines import add_complexity_lines, add_speedup_annotation

# Paths
BENCHMARK_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..")
)  # benchmark/scripts/plotting -> benchmark/
REPO_ROOT = os.path.abspath(os.path.join(BENCHMARK_DIR, ".."))
DATA_FILE_DEFAULT = os.path.join(
    BENCHMARK_DIR, "data", "benchmark_results_grid.csv"
)  # Default: grid mode
OUTPUT_DIR = os.path.join(BENCHMARK_DIR, "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Phase 3: Okabe-Ito Colorblind-Friendly Palette (per Codex recommendation)
# =============================================================================

# METHOD → COLOR (colorblind-friendly Okabe-Ito palette)
# Reference: https://jfly.uni-koeln.de/color/
OKABE_ITO_COLORS = {
    # Our methods - Blue (distinguishable)
    "fastKDE": "#0072B2",
    "fastLPR": "#0072B2",
    # Direct/baseline methods - Black
    "DirectKDE": "#000000",
    "DirectLPR": "#000000",
    # Competitors - distinct colors
    "ks": "#E69F00",  # Orange
    "FKSUM": "#D55E00",  # Vermillion (red-orange)
    "npregfast": "#009E73",  # Bluish green
    "locfit": "#F0E442",  # Yellow (Okabe-Ito)
    "StOpt-NW": "#CC79A7",  # Reddish purple
    "statsmodels": "#56B4E9",  # Sky blue
}

# LANGUAGE → MARKER (filled markers)
# Note: lang values in CSV are uppercase (PYTHON, MATLAB, R, C++)
LANG_MARKERS = {
    "MATLAB": "o",  # Circle (filled)
    "PYTHON": "s",  # Square (filled) - uppercase to match CSV
    "R": "D",  # Diamond (filled)
    "C++": "X",  # X marker (filled)
}


# CATEGORY → LINE STYLE
# Our methods: Solid, thick
# Direct/baseline: Dotted, thinner
# Competitors: Dashed, thinner
def get_line_style(method: str) -> tuple:
    """Return (linestyle, linewidth) based on method category."""
    # Our methods: fastKDE, fastLPR (but NOT npregfast)
    if method in ("fastKDE", "fastLPR"):
        return "-", 2.5  # Solid, thick
    elif any(x in method.lower() for x in ["direct", "naive"]):
        return ":", 1.5  # Dotted, thinner
    else:
        return "--", 1.5  # Dashed, thinner (competitors)


def get_plot_style(method: str, lang: str = "MATLAB") -> dict:
    """Get complete plot style for a method/language combination.

    Phase 3 styling rules:
    - COLOR distinguishes METHOD
    - MARKER distinguishes LANGUAGE
    - LINE STYLE distinguishes CATEGORY (ours/direct/competitor)
    """
    # Method → Color
    color = OKABE_ITO_COLORS.get(method, "#333333")

    # Category → Line style
    linestyle, linewidth = get_line_style(method)

    # Language → Marker
    marker = LANG_MARKERS.get(lang, "o")

    return {
        "color": color,
        "linestyle": linestyle,
        "linewidth": linewidth,
        "marker": marker,
        "markersize": 7 if method in ("fastKDE", "fastLPR") else 5,
        "markerfacecolor": color,
        "markeredgecolor": "white",
        "markeredgewidth": 0.5,
        "alpha": 0.95,
    }


def plot_methods(ax, data_d, methods, value_col, log_scale=True, scatter_only=False):
    """Plot benchmark data for multiple methods with JSS styling.

    Phase 3 update:
    - Uses Okabe-Ito colorblind palette
    - Color = Method, Marker = Language

    Args:
        log_scale: DEPRECATED - scales are now set by caller via ax.set_xscale/set_yscale.
                   Kept for backward compatibility but ignored.
        scatter_only: If True, plot markers only without connecting lines.
                      Useful for accuracy plots where values span many orders of magnitude.
    """
    legend_handles = []
    legend_labels = []

    for method in methods:
        data_m = data_d[data_d["method"] == method]
        if data_m.empty:
            continue

        for lang in ["R", "PYTHON", "Python", "C++", "MATLAB"]:
            data_ml = data_m[data_m["lang"] == lang].sort_values("N")
            if "status" in data_ml.columns:
                data_ml = data_ml[data_ml["status"] == "success"]
            # Filter out rows without accuracy ONLY for accuracy plots
            # For speed/memory plots, keep all successful runs (including N > 65536 without ground truth)
            if (
                value_col == "accuracy_vs_direct"
                and "accuracy_vs_direct" in data_ml.columns
            ):
                data_ml = data_ml[data_ml["accuracy_vs_direct"].notna()]
            if data_ml.empty:
                continue

            style = get_plot_style(method, lang)
            x = data_ml["N"].values
            y = data_ml[value_col].values

            # Filter unreliable memory measurements
            # Clamp low values to 1MB (internal methods can measure small allocations accurately)
            if value_col == "mem_max":
                y = np.maximum(y, 1.0)

            # Filter valid data
            mask = (y > 0) & np.isfinite(y)
            if not np.any(mask):
                continue

            # Create label: Method (Language) for our methods, just Method for competitors
            if "fast" in method.lower():
                label = f"{method} ({lang})"
            else:
                # For competitors, only add lang if multiple langs exist
                label = method

            # For scatter_only mode, remove linestyle
            if scatter_only:
                style = style.copy()
                style["linestyle"] = "None"
                style["markersize"] = 8  # Larger markers for scatter
                markevery = 1  # Show all markers
            else:
                markevery = 1  # Show ALL markers at every data point

            # Plot using ax.plot() to preserve base=2 x-axis set by caller
            # NOTE: ax.loglog()/ax.semilogx() would reset x-axis base from 2 to 10
            (line,) = ax.plot(
                x[mask], y[mask], label=label, **style, markevery=markevery
            )

            legend_handles.append(line)
            legend_labels.append(label)

    return legend_handles, legend_labels


def create_method_language_legend(
    fig, ax, unique_handles, unique_labels, task_name="KDE"
):
    """Create a two-part legend: Method colors + Language markers.

    Phase 3: Separate legend sections for clarity.

    Args:
        task_name: 'KDE' or 'LPR' - determines which method name to show
    """
    from matplotlib.lines import Line2D

    # Create method legend entries
    method_handles = []
    method_labels = []

    # Our methods - show fastKDE or fastLPR based on task_name
    our_method_name = "fastKDE" if "KDE" in task_name else "fastLPR"
    method_handles.append(
        Line2D(
            [0],
            [0],
            color="#0072B2",
            linestyle="-",
            linewidth=2.5,
            label=our_method_name,
        )
    )
    method_labels.append(f"{our_method_name} (ours) O(N+M log M)")

    # Direct baseline
    method_handles.append(
        Line2D([0], [0], color="#000000", linestyle=":", linewidth=1.5, label="Direct")
    )
    method_labels.append("Direct O(N²)")

    # Competitors (if present in labels)
    competitor_colors = {
        "ks": ("#E69F00", "ks O(N + M log M)"),
        "FKSUM": ("#D55E00", "FKSUM O(N log N)"),
        "locfit": ("#F0E442", "locfit O(M log N)"),
        "npregfast": ("#009E73", "npregfast O(N+M)"),
        "StOpt-NW": ("#CC79A7", "StOpt-NW O(N log N)"),
        "statsmodels": ("#56B4E9", "statsmodels O(N²)"),
    }

    for method, (color, name) in competitor_colors.items():
        if any(method in lbl for lbl in unique_labels):
            method_handles.append(
                Line2D([0], [0], color=color, linestyle="--", linewidth=1.5, label=name)
            )
            method_labels.append(name)

    # Create language marker legend entries
    lang_handles = []
    lang_labels = []

    # Determine which languages are present based on method-language mapping
    r_methods = {"ks", "FKSUM", "npregfast", "locfit"}
    python_methods = {"fastKDE", "fastLPR", "DirectKDE", "DirectLPR", "statsmodels"}
    matlab_methods = {"fastKDE", "fastLPR", "DirectKDE", "DirectLPR"}
    cpp_methods = {"StOpt-NW"}

    present_langs = set()
    for lbl in unique_labels:
        lbl_upper = lbl.upper()
        for method in r_methods:
            if method in lbl and "R" in lbl:
                present_langs.add("R")
        for method in python_methods:
            if method in lbl and ("PYTHON" in lbl_upper or "PY" in lbl_upper):
                present_langs.add("PYTHON")
        for method in matlab_methods:
            if method in lbl and "MATLAB" in lbl:
                present_langs.add("MATLAB")
        for method in cpp_methods:
            if method in lbl:
                present_langs.add("C++")

    # Check lang field directly from labels if available
    for lbl in unique_labels:
        lbl_upper = lbl.upper()
        if "R)" in lbl or lbl.endswith("R"):
            present_langs.add("R")
        if "PYTHON" in lbl_upper or "PY)" in lbl_upper:
            present_langs.add("PYTHON")
        if "MATLAB" in lbl:
            present_langs.add("MATLAB")
        if "C++" in lbl or "CPP" in lbl_upper:
            present_langs.add("C++")

    for lang, marker in LANG_MARKERS.items():
        if lang in present_langs:
            lang_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="gray",
                    marker=marker,
                    markersize=6,
                    linestyle="None",
                    markerfacecolor="gray",
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                )
            )
            lang_labels.append(lang)

    # Combine handles (no spacer — let ncol handle the layout)
    all_handles = method_handles + lang_handles
    all_labels = method_labels + lang_labels

    return all_handles, all_labels


def add_limit_annotation(ax, limit_n=65536):
    """Add vertical line and shading marking O(N^2) memory limit."""
    ylim = ax.get_ylim()

    # Green shaded region where fast methods continue
    ax.axvspan(limit_n, ax.get_xlim()[1], alpha=0.08, color="green", zorder=0)

    # Dashed vertical line at limit
    ax.axvline(x=limit_n, color="#d62728", linestyle="--", alpha=0.6, lw=1.5, zorder=1)

    # Label
    import math

    if ylim[0] > 0 and ylim[1] > 0:
        label_y = math.sqrt(ylim[0] * ylim[1])
    else:
        label_y = (ylim[0] + ylim[1]) / 2
    ax.text(
        limit_n * 1.15,
        label_y,
        "Direct\nlimit",
        color="#d62728",
        fontsize=14,
        ha="left",
        va="center",
        fontweight="bold",
        alpha=0.7,
    )


def plot_3x3_jss(df_task, task_name, methods, outfile_base):
    """Generate 3x3 figure with JSS styling."""
    # Apply JSS style
    apply_jss_style()

    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1], hspace=0.30, wspace=0.28)
    axes = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(3)]
    axes = np.array(axes)

    acc_label = "MSE vs Direct"
    row_labels = ["Time (seconds)", "Memory (MB)", acc_label]

    for col, d in enumerate([1, 2, 3]):
        data_d = df_task[df_task["d"] == d]
        if data_d.empty:
            continue

        N_range = sorted(data_d["N"].unique())

        # Row 0: SPEED
        ax = axes[0, col]
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        plot_methods(ax, data_d, methods, "time_sec", True)

        # Add complexity reference lines
        if len(N_range) > 1:
            add_complexity_lines(
                ax,
                n_range=(min(N_range), max(N_range)),
                complexities=["N", "NlogN", "N2"],
                show_labels=(col == 2),
            )

        # Add limit annotation
        if max(N_range) > 65536:
            add_limit_annotation(ax)

        ax.set_xlabel("N (sample size)")
        ax.set_ylabel(row_labels[0] if col == 0 else "")
        ax.set_title(f"d = {d}", fontweight="bold", fontsize=20)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(direction="in", top=True, right=True)
        if col == 2:
            ax.yaxis.set_ticks_position("right")
            ax.yaxis.set_label_position("right")
            ax.tick_params(axis="y", which="both", labelleft=False, labelright=True)

        # Row 1: MEMORY
        ax = axes[1, col]
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        plot_methods(ax, data_d, methods, "mem_max", True)

        if len(N_range) > 1:
            add_complexity_lines(
                ax,
                n_range=(min(N_range), max(N_range)),
                complexities=["N", "N2"],
                show_labels=(col == 2),
            )

        if max(N_range) > 65536:
            add_limit_annotation(ax)

        ax.set_xlabel("N (sample size)")
        ax.set_ylabel(row_labels[1] if col == 0 else "")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(direction="in", top=True, right=True)
        if col == 2:
            ax.yaxis.set_ticks_position("right")
            ax.yaxis.set_label_position("right")
            ax.tick_params(axis="y", which="both", labelleft=False, labelright=True)

        # Row 2: ACCURACY (scatter only - no lines connecting points)
        ax = axes[2, col]
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")

        # Check for accuracy data
        acc_col = "accuracy_vs_direct"
        if acc_col in data_d.columns:
            data_valid = data_d[
                ~data_d[acc_col].isna()
                & (data_d[acc_col] > 0)
                & (data_d[acc_col] != 0.001)  # Filter placeholder
            ]
            if len(data_valid) > 0:
                plot_methods(ax, data_valid, methods, acc_col, True)
                if len(N_range) > 0:
                    ax.set_xlim(min(N_range), max(N_range))
                if max(N_range) > 65536:
                    add_limit_annotation(ax)

        ax.set_xlabel("N (sample size)")
        ax.set_ylabel(row_labels[2] if col == 0 else "")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(direction="in", top=True, right=True)
        if col == 2:
            ax.yaxis.set_ticks_position("right")
            ax.yaxis.set_label_position("right")
            ax.tick_params(axis="y", which="both", labelleft=False, labelright=True)

    # Collect legend handles from all plots
    all_handles = []
    all_labels = []
    for row in range(3):
        for col in range(3):
            h, l = axes[row, col].get_legend_handles_labels()
            all_handles.extend(h)
            all_labels.extend(l)

    # Remove duplicates while preserving order
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(all_handles, all_labels):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)

    # Create Phase 3 structured legend (Method colors + Language markers)
    legend_handles, legend_labels = create_method_language_legend(
        fig, axes[0, 0], unique_handles, unique_labels, task_name
    )

    # Add global legend at bottom with two columns: Methods | Languages
    # Legend: anchor TOP of legend box at y=0.04 (below x-axis labels which end ~0.06)
    # bbox_inches='tight' in savefig will auto-extend to include the full legend
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=5,
        frameon=True,
        framealpha=0.95,
        edgecolor="gray",
        fontsize=18,
        columnspacing=1.2,
    )

    plt.subplots_adjust(bottom=0.12, top=0.97)

    # Save PDF (primary) and PNG (preview)
    pdf_path = os.path.join(OUTPUT_DIR, f"{outfile_base}.pdf")
    png_path = os.path.join(OUTPUT_DIR, f"{outfile_base}.png")

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(png_path, format="png", bbox_inches="tight", dpi=150)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

    plt.close()


def main():
    """Generate all benchmark figures with JSS styling."""
    parser = argparse.ArgumentParser(description="Generate JSS benchmark figures")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=DATA_FILE_DEFAULT,
        help=f"Input CSV file (default: benchmark_results_grid.csv)",
    )
    args = parser.parse_args()

    data_file = args.input

    print("=" * 60)
    print("Generating JSS Publication-Quality Benchmark Figures")
    print("=" * 60)
    print()

    print(f"Loading data from: {data_file}")
    if not os.path.exists(data_file):
        print(f"ERROR: Data file not found: {data_file}")
        print("Run benchmark first or check path.")
        return

    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} records")
    print(f"N range: {df['N'].min():,} to {df['N'].max():,}")
    print(f"Methods: {df['method'].unique().tolist()}")

    # Data quality validation
    if "mem_method" not in df.columns:
        print(
            "\n⚠️  WARNING: 'mem_method' column missing - data from OLD code (before Job Objects)"
        )
        print("   Memory tracking may be unreliable. Consider re-running benchmark.")

    # Check memory data coverage
    if "mem_max" in df.columns:
        mem_valid = (df["mem_max"] > 0).sum()
        mem_total = len(df)
        mem_pct = 100 * mem_valid / mem_total
        if mem_pct < 70:
            print(
                f"\n⚠️  WARNING: Only {mem_pct:.1f}% ({mem_valid}/{mem_total}) have valid memory data"
            )
            print(
                "   Memory plots will have missing points. Consider re-running benchmark."
            )
    print()

    # KDE figure
    kde = df[df["task"] == "KDE"]
    if len(kde) > 0:
        # Standard KDE methods
        kde_methods = ["fastKDE", "ks", "FKSUM"]
        if "DirectKDE" in set(kde["method"].tolist()):
            kde_methods.append("DirectKDE")

        print("Generating KDE benchmark figure...")
        plot_3x3_jss(kde, "KDE", kde_methods, "fig7_kde_benchmark")
        print()

    # LPR figure
    lpr = df[df["task"] == "LPR"]
    if len(lpr) > 0:
        # Standard LPR methods
        lpr_methods = ["fastLPR", "npregfast", "StOpt-NW", "locfit"]
        if "DirectLPR" in set(lpr["method"].tolist()):
            lpr_methods.append("DirectLPR")

        print("Generating LPR benchmark figure...")
        plot_3x3_jss(lpr, "LPR", lpr_methods, "fig7_lpr_benchmark")
        print()

    # Print summary
    print("=" * 60)
    print("Method Coverage Summary")
    print("=" * 60)
    for task in ["KDE", "LPR"]:
        print(f"\n{task}:")
        task_df = df[df["task"] == task]
        for method in pd.unique(task_df["method"]):
            for d in [1, 2, 3]:
                data = task_df[(task_df["method"] == method) & (task_df["d"] == d)]
                if len(data) > 0:
                    max_n = data["N"].max()
                    print(f"  {method:15s} d={d}: N_max={int(max_n):>10,}")

    print("\nDone!")


if __name__ == "__main__":
    main()
