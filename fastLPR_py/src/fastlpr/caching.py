# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Caching utilities for fastLPR to avoid repeated computations.

This module provides caching mechanisms for expensive computations that are
repeated across bandwidth iterations during cross-validation. Key cacheable items:

1. Kernel FFTs - Same kernel shape and bandwidth combination
2. NUFFT of input data - Computed once per cv_fastlpr call
3. Grid interpolator objects - Reused across evaluations
4. Design matrix templates - Structure depends only on order and dims

Caching can provide 2-5x speedup for cross-validation with many bandwidth candidates.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Union
import weakref

import numpy as np


class ArrayCache:
    """
    LRU cache for numpy arrays and complex objects.

    Unlike functools.lru_cache, this can handle unhashable numpy arrays
    by using a custom key generator based on array properties.

    Parameters
    ----------
    maxsize : int, default=32
        Maximum number of items to cache.

    Examples
    --------
    >>> cache = ArrayCache(maxsize=16)
    >>>
    >>> def compute_kernel(shape, bandwidth):
    ...     # Expensive computation
    ...     return np.random.randn(*shape)
    >>>
    >>> # First call computes and caches
    >>> result1 = cache.get_or_compute(
    ...     key=("kernel", (100,), 0.1),
    ...     compute_fn=lambda: compute_kernel((100,), 0.1)
    ... )
    >>>
    >>> # Second call returns cached result
    >>> result2 = cache.get_or_compute(
    ...     key=("kernel", (100,), 0.1),
    ...     compute_fn=lambda: compute_kernel((100,), 0.1)
    ... )
    """

    def __init__(self, maxsize: int = 32):
        self._cache: OrderedDict[Any, Any] = OrderedDict()
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0

    def get_or_compute(
        self,
        key: Any,
        compute_fn: Callable[[], Any],
    ) -> Any:
        """
        Get cached value or compute and cache it.

        Parameters
        ----------
        key : hashable
            Cache key. Must be hashable (use tuples for array shapes, etc.)
        compute_fn : callable
            Function to compute the value if not cached

        Returns
        -------
        result : Any
            Cached or computed result
        """
        if key in self._cache:
            self._hits += 1
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]

        self._misses += 1

        # Compute new value
        result = compute_fn()

        # Add to cache
        if len(self._cache) >= self._maxsize:
            # Remove oldest entry (LRU)
            self._cache.popitem(last=False)

        self._cache[key] = result
        return result

    def get(self, key: Any, default: Any = None) -> Any:
        """Get value from cache without computing."""
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        return default

    def put(self, key: Any, value: Any) -> None:
        """Put value into cache directly."""
        if len(self._cache) >= self._maxsize and key not in self._cache:
            self._cache.popitem(last=False)
        self._cache[key] = value
        self._cache.move_to_end(key)

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def __contains__(self, key: Any) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    @property
    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": hit_rate,
            "size": len(self._cache),
            "maxsize": self._maxsize,
        }


def array_key(arr: np.ndarray) -> Tuple:
    """
    Generate a hashable key for a numpy array.

    Uses shape, dtype, and a hash of the data for uniqueness.

    Parameters
    ----------
    arr : np.ndarray
        Array to generate key for

    Returns
    -------
    key : tuple
        Hashable tuple (shape, dtype, data_hash)
    """
    if arr is None:
        return (None,)

    arr = np.asarray(arr)

    # Use shape and dtype
    shape = arr.shape
    dtype = str(arr.dtype)

    # Hash the raw data bytes for uniqueness
    # Use tobytes() which is fast for contiguous arrays
    try:
        data_bytes = arr.tobytes()
        data_hash = hashlib.md5(data_bytes).hexdigest()[:16]
    except (ValueError, TypeError):
        # Fallback for non-contiguous arrays
        data_hash = hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]

    return (shape, dtype, data_hash)


def make_cache_key(*args, **kwargs) -> Tuple:
    """
    Create a hashable cache key from mixed arguments.

    Handles numpy arrays by converting to array_key.

    Parameters
    ----------
    *args : Any
        Positional arguments
    **kwargs : Any
        Keyword arguments (sorted by key for consistency)

    Returns
    -------
    key : tuple
        Hashable cache key
    """
    processed_args = []

    for arg in args:
        if isinstance(arg, np.ndarray):
            processed_args.append(array_key(arg))
        elif isinstance(arg, (list, tuple)):
            processed_args.append(tuple(arg))
        else:
            processed_args.append(arg)

    processed_kwargs = []
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        if isinstance(v, np.ndarray):
            processed_kwargs.append((k, array_key(v)))
        elif isinstance(v, (list, tuple)):
            processed_kwargs.append((k, tuple(v)))
        else:
            processed_kwargs.append((k, v))

    return tuple(processed_args) + tuple(processed_kwargs)


# ============================================================================
# Global caches for fastLPR computations
# ============================================================================

# Cache for kernel FFTs: key = (kernel_shape, bandwidth_tuple, kernel_type, order)
_kernel_fft_cache = ArrayCache(maxsize=128)

# Cache for NUFFT transforms: key = (x_hash, y_hash, grid_shape, accuracy)
_nufft_cache = ArrayCache(maxsize=64)

# Cache for design matrices: key = (x_hash, h_hash, grid_shape, order)
_design_matrix_cache = ArrayCache(maxsize=32)

# Cache for interpolators: weak references to avoid memory leaks
_interpolator_cache: Dict[Tuple, weakref.ref] = {}


def get_kernel_fft_cached(
    x: np.ndarray,
    h: np.ndarray,
    grid_shape: np.ndarray,
    kernel_type: str,
    order: int,
    flag_power2: bool,
    accuracy: int,
    compute_fn: Callable[[], Tuple[np.ndarray, dict]],
) -> Tuple[np.ndarray, dict]:
    """
    Get kernel FFT from cache or compute and cache it.

    Parameters
    ----------
    x : np.ndarray
        Data points (used for cache key)
    h : np.ndarray
        Bandwidths
    grid_shape : np.ndarray
        Grid shape
    kernel_type : str
        Kernel type
    order : int
        Polynomial order
    flag_power2 : bool
        Whether to use power-of-2 padding
    accuracy : int
        NUFFT accuracy
    compute_fn : callable
        Function to compute kernel FFT if not cached

    Returns
    -------
    kd_ft : np.ndarray or list
        Kernel in Fourier domain
    params : dict
        Kernel parameters
    """
    # Create cache key
    key = (
        array_key(x),
        array_key(h),
        tuple(grid_shape),
        kernel_type,
        order,
        flag_power2,
        accuracy,
    )

    return _kernel_fft_cache.get_or_compute(key, compute_fn)


def get_nufft_cached(
    x: np.ndarray,
    y: np.ndarray,
    grid_shape: np.ndarray,
    qin: np.ndarray,
    accuracy: int,
    compute_fn: Callable[[], np.ndarray],
) -> np.ndarray:
    """
    Get NUFFT transform from cache or compute and cache it.

    This caches the NUFFT of input data, which is expensive and only needs
    to be computed once per cv_fastlpr call (same x, y for all bandwidths).

    Parameters
    ----------
    x : np.ndarray
        Data points
    y : np.ndarray
        Response values
    grid_shape : np.ndarray
        Grid shape
    qin : np.ndarray
        Padding indices
    accuracy : int
        NUFFT accuracy
    compute_fn : callable
        Function to compute NUFFT if not cached

    Returns
    -------
    y_ft : np.ndarray
        Fourier-transformed y
    """
    key = (
        array_key(x),
        array_key(y),
        tuple(grid_shape),
        array_key(qin),
        accuracy,
    )

    return _nufft_cache.get_or_compute(key, compute_fn)


def get_design_matrix_cached(
    x: np.ndarray,
    h: np.ndarray,
    grid_shape: np.ndarray,
    order: int,
    compute_fn: Callable[[], Tuple[Any, dict]],
) -> Tuple[Any, dict]:
    """
    Get design matrix S from cache or compute and cache it.

    The design matrix S depends only on x and h, not on y.
    This is already partially implemented via S_precomputed parameter,
    but this provides a more general caching mechanism.

    Parameters
    ----------
    x : np.ndarray
        Data points
    h : np.ndarray
        Bandwidths
    grid_shape : np.ndarray
        Grid shape
    order : int
        Polynomial order
    compute_fn : callable
        Function to compute design matrix if not cached

    Returns
    -------
    s : ndarray or list
        Design matrix elements
    params : dict
        Parameters
    """
    key = (
        array_key(x),
        array_key(h),
        tuple(grid_shape),
        order,
    )

    return _design_matrix_cache.get_or_compute(key, compute_fn)


def clear_fastlpr_caches() -> Dict[str, Dict[str, int]]:
    """
    Clear all fastLPR caches and return statistics.

    Call this to free memory after cross-validation is complete,
    or before starting a new analysis with different data.

    Returns
    -------
    stats : dict
        Cache statistics before clearing
    """
    stats = {
        "kernel_fft": _kernel_fft_cache.stats,
        "nufft": _nufft_cache.stats,
        "design_matrix": _design_matrix_cache.stats,
        "interpolator": {"size": len(_interpolator_cache)},
    }

    _kernel_fft_cache.clear()
    _nufft_cache.clear()
    _design_matrix_cache.clear()
    _interpolator_cache.clear()

    return stats


def get_cache_stats() -> Dict[str, Dict[str, int]]:
    """
    Get current cache statistics.

    Returns
    -------
    stats : dict
        Statistics for each cache
    """
    return {
        "kernel_fft": _kernel_fft_cache.stats,
        "nufft": _nufft_cache.stats,
        "design_matrix": _design_matrix_cache.stats,
        "interpolator": {"size": len(_interpolator_cache)},
    }


# ============================================================================
# Context manager for scoped caching
# ============================================================================

@dataclass
class CacheContext:
    """
    Context manager for scoped caching within a single cv_fastlpr call.

    This provides a clean way to cache intermediate results that are
    specific to a single cross-validation run and should be cleared
    after the run completes.

    Attributes
    ----------
    x_ft : np.ndarray, optional
        Cached NUFFT of input x (ones vector)
    y_ft : np.ndarray, optional
        Cached NUFFT of response y
    kernel_ft : dict
        Cached kernel FFTs by bandwidth index
    design_matrix : tuple, optional
        Cached design matrix (S, params)
    interpolators : dict
        Cached interpolator objects by bandwidth index

    Examples
    --------
    >>> with CacheContext() as ctx:
    ...     # First bandwidth - compute and cache y_ft
    ...     if ctx.y_ft is None:
    ...         ctx.y_ft = nufft_transform(x, y, L)
    ...
    ...     # All subsequent bandwidths reuse cached y_ft
    ...     result = convolve(ctx.y_ft, kernel_ft)
    """

    x_ft: Optional[np.ndarray] = None
    y_ft: Optional[np.ndarray] = None
    kernel_ft: Dict[int, np.ndarray] = field(default_factory=dict)
    design_matrix: Optional[Tuple[Any, dict]] = None
    interpolators: Dict[int, Any] = field(default_factory=dict)
    grid_axes: Optional[Tuple[np.ndarray, ...]] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def __enter__(self) -> "CacheContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.clear()

    def clear(self) -> None:
        """Clear all cached data."""
        self.x_ft = None
        self.y_ft = None
        self.kernel_ft.clear()
        self.design_matrix = None
        self.interpolators.clear()
        self.grid_axes = None
        self.params.clear()

    def cache_kernel_ft(self, idx: int, kernel_ft: np.ndarray) -> None:
        """Cache kernel FFT for a specific bandwidth index."""
        self.kernel_ft[idx] = kernel_ft

    def get_kernel_ft(self, idx: int) -> Optional[np.ndarray]:
        """Get cached kernel FFT for a specific bandwidth index."""
        return self.kernel_ft.get(idx)

    def cache_interpolator(self, idx: int, interp: Any) -> None:
        """Cache interpolator for a specific bandwidth index."""
        self.interpolators[idx] = interp

    def get_interpolator(self, idx: int) -> Optional[Any]:
        """Get cached interpolator for a specific bandwidth index."""
        return self.interpolators.get(idx)


# ============================================================================
# Memoization decorator for array-returning functions
# ============================================================================

def memoize_array(maxsize: int = 32):
    """
    Decorator for memoizing functions that return numpy arrays.

    Parameters
    ----------
    maxsize : int, default=32
        Maximum number of results to cache

    Returns
    -------
    decorator : callable
        Decorator function

    Examples
    --------
    >>> @memoize_array(maxsize=16)
    ... def expensive_kernel(shape, bandwidth):
    ...     return np.exp(-np.arange(shape[0])**2 / (2 * bandwidth**2))
    >>>
    >>> # First call computes
    >>> k1 = expensive_kernel((100,), 0.1)
    >>>
    >>> # Second call returns cached
    >>> k2 = expensive_kernel((100,), 0.1)
    """
    def decorator(func: Callable) -> Callable:
        cache = ArrayCache(maxsize=maxsize)

        def wrapper(*args, **kwargs):
            key = make_cache_key(*args, **kwargs)
            return cache.get_or_compute(key, lambda: func(*args, **kwargs))

        # Attach cache methods to wrapper
        wrapper.cache = cache
        wrapper.clear_cache = cache.clear
        wrapper.cache_stats = lambda: cache.stats

        return wrapper

    return decorator
