# Copyright (c) 2024-2025 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
fastlpr - Fast Local Polynomial Regression and Kernel Density Estimation using NUFFT.

Python implementation of the fastLPR algorithm for efficient nonparametric regression
and density estimation on large, scattered, or complex-valued datasets.

Basic Usage:
    from fastlpr import cv_fastlpr, cv_fastkde, get_hlist

    # Regression with automatic bandwidth selection
    hlist = get_hlist(20, [0.01, 1.0])
    regs = cv_fastlpr(x, y, hlist, {'order': 1})

    # Kernel density estimation
    kde = cv_fastkde(x, hlist)

For more examples, see the examples/ directory.
"""

__version__ = "1.0.0"


# =============================================================================
# Startup Banner - Show acceleration backend status
# =============================================================================
def _print_startup_banner():
    """Print startup banner with backend status (called once on import)."""
    import sys
    import os

    # Check Python version compatibility
    py_version = sys.version_info
    if py_version >= (3, 14):
        print(f"[fastlpr] WARNING: Python {py_version.major}.{py_version.minor} detected.")
        print(f"[fastlpr]          Cython acceleration requires Python 3.9-3.13.")
        print(f"[fastlpr]          Falling back to Numba/NumPy backend.")

    # Use centralized backend detection
    from .backend_selection import (
        HAS_NUMBA, HAS_CYTHON, NUMBA_VERSION,
        select_accumulation_backend,
    )

    # Get the active backend
    active_backend = select_accumulation_backend()

    # Report actual backend
    if active_backend == 'cython':
        backend = "Cython+OpenMP"
        parallel = False  # Sequential due to race conditions
    elif active_backend == 'numba':
        backend = f"Numba {NUMBA_VERSION}"
        parallel = False  # Sequential due to race conditions
    else:
        backend = "NumPy (pure Python)"
        parallel = False

    # Print status
    status = "parallel" if parallel else "single-threaded"
    print(f"[fastlpr] v{__version__} | Backend: {backend} ({status})")


# Print banner on first import (can be disabled by setting FASTLPR_QUIET=1)
import os as _os
if not _os.environ.get("FASTLPR_QUIET"):
    _print_startup_banner()

# =============================================================================
# Public API - User-facing functions
# =============================================================================
from .api import cv_fastkde, cv_fastlpr
from .bandwidth import get_hlist
from .caching import clear_fastlpr_caches, get_cache_stats
from .confidence import fastlpr_interval
from .predict import fastlpr_predict
from .structures import KDEOutput, RegressionOutput


# Lazy imports for plotting functions (avoid forcing matplotlib at import time)
def __getattr__(name):
    _lazy = {
        "fastlpr_plot": (".plotting", "fastlpr_plot"),
        "fastlpr_plot_interval": (".confidence", "fastlpr_plot_interval"),
        "fastkde_plot": (".kde_plotting", "fastkde_plot"),
        "fastkde_plot_bandwidth": (".kde_plotting", "fastkde_plot_bandwidth"),
    }
    if name in _lazy:
        module_path, attr = _lazy[name]
        import importlib
        mod = importlib.import_module(module_path, __package__)
        val = getattr(mod, attr)
        globals()[name] = val  # cache for subsequent access
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# =============================================================================
# Internal imports - Available for advanced users but not in __all__
# =============================================================================
# These are not part of the public API but can be imported explicitly:
#   from fastlpr.batch import cv_fastlpr_batch
#   from fastlpr.nufft_numba import nufftn_type1_numba
# =============================================================================

__all__ = [
    # Version
    "__version__",
    # Main entry points
    "cv_fastlpr",
    "cv_fastkde",
    "get_hlist",
    # Prediction and intervals (CI and PI)
    "fastlpr_predict",
    "fastlpr_interval",
    # Visualization
    "fastlpr_plot",
    "fastlpr_plot_interval",
    "fastkde_plot",
    "fastkde_plot_bandwidth",
    # Output structures
    "RegressionOutput",
    "KDEOutput",
    # Caching utilities
    "clear_fastlpr_caches",
    "get_cache_stats",
]
