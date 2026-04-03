# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
FFT Backend for fastLPR - Supports pyfftw, scipy.fft, and numpy.fft.

This module provides a unified interface for FFT operations with automatic
backend selection and fallback.

Backend selection & fallback order:

1. `pyfftw` (optional) — ONLY used when installed *and* explicitly enabled via
   `fastlpr.fft_backend.enable_pyfftw()`.
2. `scipy.fft` — default backend.
3. `numpy.fft` — fallback when SciPy is unavailable.

Rationale for NOT auto-enabling `pyfftw`:

- For the benchmark configurations used in this repo (including `accuracy=0` binning
  mode), FFTs are often not the dominant cost; switching to `pyfftw` may be neutral or
  slower due to planning/dispatch overhead.
- When users want to experiment with `pyfftw` anyway, prefer making it an explicit
  opt-in so results remain reproducible.

Usage:
    from fastlpr.fft_backend import fft, ifft, fftshift, ifftshift, get_backend_info

    # Use like numpy.fft
    result = fft(data, axis=0)
    result = ifft(result, axis=0)

    # Enable/disable pyfftw
    from fastlpr import fft_backend
    fft_backend.enable_pyfftw()   # Enable pyfftw if available
    fft_backend.disable_pyfftw()  # Force scipy/numpy

    # Get backend info
    info = get_backend_info()
    print(info['backend'])  # 'pyfftw', 'scipy', or 'numpy'

Performance Notes:
    - pyfftw can be 1.5-3x faster than numpy.fft for large arrays (>1024)
    - For small arrays (<256), planning overhead may reduce benefit
    - scipy.fft is typically 2x faster than numpy.fft
    - Enable pyfftw caching (wisdom) for repeated transforms of same size

Author: Ying Wang, Min Li
"""

import os
import logging
import numpy as np
from typing import Optional, Union, Tuple

# Configure module logger
logger = logging.getLogger(__name__)

# =============================================================================
# Backend State
# =============================================================================

# Global flag to enable/disable pyfftw (disabled by default for best performance)
# pyfftw is beneficial for large multi-dimensional FFTs (>128^2) but adds overhead
# for small 1D FFTs typical in fastLPR cross-validation loops.
# Enable manually with fft_backend.enable_pyfftw() for 2D/3D regression tasks.
_PYFFTW_ENABLED = False

# Try to import pyfftw
try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as pyfftw_numpy_fft

    _PYFFTW_AVAILABLE = True

    # Enable caching (wisdom) by default for repeated transforms
    pyfftw.interfaces.cache.enable()

    # Configure threading - use all available cores
    # Can be overridden with environment variable PYFFTW_NUM_THREADS
    _num_threads = os.environ.get("PYFFTW_NUM_THREADS", None)
    if _num_threads is not None:
        pyfftw.config.NUM_THREADS = int(_num_threads)
    else:
        # Use all cores (-1 means all in some versions, or use cpu_count)
        try:
            import multiprocessing

            pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
        except Exception:
            pyfftw.config.NUM_THREADS = 4  # Safe default

except ImportError:
    _PYFFTW_AVAILABLE = False
    pyfftw = None
    pyfftw_numpy_fft = None

# Try to import scipy.fft
try:
    from scipy import fft as scipy_fft

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    scipy_fft = None


# =============================================================================
# Backend Control Functions
# =============================================================================


def enable_pyfftw():
    """Enable pyfftw backend (if installed)."""
    global _PYFFTW_ENABLED
    _PYFFTW_ENABLED = True
    if not _PYFFTW_AVAILABLE:
        logger.warning("pyfftw is not installed. Install with: pip install pyfftw")


def disable_pyfftw():
    """Disable pyfftw backend (will use scipy or numpy instead)."""
    global _PYFFTW_ENABLED
    _PYFFTW_ENABLED = False


def is_pyfftw_enabled():
    """Check if pyfftw is currently enabled and available."""
    return _PYFFTW_ENABLED and _PYFFTW_AVAILABLE


def get_backend_info():
    """
    Get information about the current FFT backend.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'backend': str - Name of active backend ('pyfftw', 'scipy', 'numpy')
        - 'pyfftw_available': bool - Whether pyfftw is installed
        - 'pyfftw_enabled': bool - Whether pyfftw is enabled
        - 'scipy_available': bool - Whether scipy.fft is available
        - 'num_threads': int - Number of threads (pyfftw only)
        - 'cache_enabled': bool - Whether wisdom caching is enabled (pyfftw only)
    """
    info = {
        "pyfftw_available": _PYFFTW_AVAILABLE,
        "pyfftw_enabled": _PYFFTW_ENABLED,
        "scipy_available": _SCIPY_AVAILABLE,
    }

    if _PYFFTW_ENABLED and _PYFFTW_AVAILABLE:
        info["backend"] = "pyfftw"
        info["num_threads"] = pyfftw.config.NUM_THREADS
        info["cache_enabled"] = pyfftw.interfaces.cache.is_enabled()
    elif _SCIPY_AVAILABLE:
        info["backend"] = "scipy"
        info["num_threads"] = 1
        info["cache_enabled"] = False
    else:
        info["backend"] = "numpy"
        info["num_threads"] = 1
        info["cache_enabled"] = False

    return info


def set_num_threads(n: int):
    """
    Set the number of threads for pyfftw.

    Parameters
    ----------
    n : int
        Number of threads to use. Use -1 for all available cores.
    """
    if _PYFFTW_AVAILABLE:
        if n == -1:
            import multiprocessing

            n = multiprocessing.cpu_count()
        pyfftw.config.NUM_THREADS = n
    else:
        logger.warning("pyfftw not available. Thread setting has no effect.")


def clear_cache():
    """Clear the pyfftw wisdom cache."""
    if _PYFFTW_AVAILABLE:
        pyfftw.interfaces.cache.disable()
        pyfftw.interfaces.cache.enable()


# =============================================================================
# FFT Functions
# =============================================================================


def fft(
    a: np.ndarray,
    n: Optional[int] = None,
    axis: int = -1,
    overwrite_x: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Compute the one-dimensional discrete Fourier Transform.

    Parameters
    ----------
    a : array_like
        Input array
    n : int, optional
        Length of the transformed axis of the output
    axis : int, optional
        Axis over which to compute the FFT (default: -1)
    overwrite_x : bool, optional
        If True, the contents of x can be destroyed (default: False)
        Only used by scipy backend.
    **kwargs
        Additional keyword arguments passed to the backend

    Returns
    -------
    ndarray
        FFT result
    """
    if _PYFFTW_ENABLED and _PYFFTW_AVAILABLE:
        # pyfftw has threads parameter
        threads = kwargs.pop("threads", pyfftw.config.NUM_THREADS)
        return pyfftw_numpy_fft.fft(a, n=n, axis=axis, threads=threads)
    elif _SCIPY_AVAILABLE:
        return scipy_fft.fft(a, n=n, axis=axis, overwrite_x=overwrite_x)
    else:
        return np.fft.fft(a, n=n, axis=axis)


def ifft(
    a: np.ndarray,
    n: Optional[int] = None,
    axis: int = -1,
    overwrite_x: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Compute the one-dimensional inverse discrete Fourier Transform.

    Parameters
    ----------
    a : array_like
        Input array
    n : int, optional
        Length of the transformed axis of the output
    axis : int, optional
        Axis over which to compute the IFFT (default: -1)
    overwrite_x : bool, optional
        If True, the contents of x can be destroyed (default: False)
        Only used by scipy backend.
    **kwargs
        Additional keyword arguments passed to the backend

    Returns
    -------
    ndarray
        IFFT result
    """
    if _PYFFTW_ENABLED and _PYFFTW_AVAILABLE:
        threads = kwargs.pop("threads", pyfftw.config.NUM_THREADS)
        return pyfftw_numpy_fft.ifft(a, n=n, axis=axis, threads=threads)
    elif _SCIPY_AVAILABLE:
        return scipy_fft.ifft(a, n=n, axis=axis, overwrite_x=overwrite_x)
    else:
        return np.fft.ifft(a, n=n, axis=axis)


def fft2(
    a: np.ndarray,
    s: Optional[Tuple[int, int]] = None,
    axes: Tuple[int, int] = (-2, -1),
    overwrite_x: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Compute the 2-dimensional discrete Fourier Transform.

    Parameters
    ----------
    a : array_like
        Input array
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
    axes : sequence of ints, optional
        Axes over which to compute the FFT (default: (-2, -1))
    overwrite_x : bool, optional
        If True, the contents of x can be destroyed (default: False)
    **kwargs
        Additional keyword arguments passed to the backend

    Returns
    -------
    ndarray
        FFT2 result
    """
    if _PYFFTW_ENABLED and _PYFFTW_AVAILABLE:
        threads = kwargs.pop("threads", pyfftw.config.NUM_THREADS)
        return pyfftw_numpy_fft.fft2(a, s=s, axes=axes, threads=threads)
    elif _SCIPY_AVAILABLE:
        return scipy_fft.fft2(a, s=s, axes=axes, overwrite_x=overwrite_x)
    else:
        return np.fft.fft2(a, s=s, axes=axes)


def ifft2(
    a: np.ndarray,
    s: Optional[Tuple[int, int]] = None,
    axes: Tuple[int, int] = (-2, -1),
    overwrite_x: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Compute the 2-dimensional inverse discrete Fourier Transform.

    Parameters
    ----------
    a : array_like
        Input array
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
    axes : sequence of ints, optional
        Axes over which to compute the IFFT (default: (-2, -1))
    overwrite_x : bool, optional
        If True, the contents of x can be destroyed (default: False)
    **kwargs
        Additional keyword arguments passed to the backend

    Returns
    -------
    ndarray
        IFFT2 result
    """
    if _PYFFTW_ENABLED and _PYFFTW_AVAILABLE:
        threads = kwargs.pop("threads", pyfftw.config.NUM_THREADS)
        return pyfftw_numpy_fft.ifft2(a, s=s, axes=axes, threads=threads)
    elif _SCIPY_AVAILABLE:
        return scipy_fft.ifft2(a, s=s, axes=axes, overwrite_x=overwrite_x)
    else:
        return np.fft.ifft2(a, s=s, axes=axes)


def fftn(
    a: np.ndarray,
    s: Optional[Tuple[int, ...]] = None,
    axes: Optional[Tuple[int, ...]] = None,
    overwrite_x: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Compute the N-dimensional discrete Fourier Transform.

    Parameters
    ----------
    a : array_like
        Input array
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
    axes : sequence of ints, optional
        Axes over which to compute the FFT (default: all axes)
    overwrite_x : bool, optional
        If True, the contents of x can be destroyed (default: False)
    **kwargs
        Additional keyword arguments passed to the backend

    Returns
    -------
    ndarray
        FFTN result
    """
    if _PYFFTW_ENABLED and _PYFFTW_AVAILABLE:
        threads = kwargs.pop("threads", pyfftw.config.NUM_THREADS)
        return pyfftw_numpy_fft.fftn(a, s=s, axes=axes, threads=threads)
    elif _SCIPY_AVAILABLE:
        return scipy_fft.fftn(a, s=s, axes=axes, overwrite_x=overwrite_x)
    else:
        return np.fft.fftn(a, s=s, axes=axes)


def ifftn(
    a: np.ndarray,
    s: Optional[Tuple[int, ...]] = None,
    axes: Optional[Tuple[int, ...]] = None,
    overwrite_x: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Compute the N-dimensional inverse discrete Fourier Transform.

    Parameters
    ----------
    a : array_like
        Input array
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
    axes : sequence of ints, optional
        Axes over which to compute the IFFT (default: all axes)
    overwrite_x : bool, optional
        If True, the contents of x can be destroyed (default: False)
    **kwargs
        Additional keyword arguments passed to the backend

    Returns
    -------
    ndarray
        IFFTN result
    """
    if _PYFFTW_ENABLED and _PYFFTW_AVAILABLE:
        threads = kwargs.pop("threads", pyfftw.config.NUM_THREADS)
        return pyfftw_numpy_fft.ifftn(a, s=s, axes=axes, threads=threads)
    elif _SCIPY_AVAILABLE:
        return scipy_fft.ifftn(a, s=s, axes=axes, overwrite_x=overwrite_x)
    else:
        return np.fft.ifftn(a, s=s, axes=axes)


# =============================================================================
# Real FFT Functions (rfft/irfft for real-valued data)
# =============================================================================


def rfft(
    a: np.ndarray,
    n: Optional[int] = None,
    axis: int = -1,
    overwrite_x: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Compute the one-dimensional FFT for real input.

    This is optimized for real-valued data and returns only the positive
    frequency terms (N//2 + 1 complex values for N real inputs).

    Parameters
    ----------
    a : array_like
        Real input array
    n : int, optional
        Length of the transformed axis of the output
    axis : int, optional
        Axis over which to compute the FFT (default: -1)
    overwrite_x : bool, optional
        If True, the contents of x can be destroyed (default: False)
    **kwargs
        Additional keyword arguments passed to the backend

    Returns
    -------
    ndarray
        Complex FFT result with shape[axis] = n//2 + 1
    """
    if _PYFFTW_ENABLED and _PYFFTW_AVAILABLE:
        threads = kwargs.pop("threads", pyfftw.config.NUM_THREADS)
        return pyfftw_numpy_fft.rfft(a, n=n, axis=axis, threads=threads)
    elif _SCIPY_AVAILABLE:
        return scipy_fft.rfft(a, n=n, axis=axis, overwrite_x=overwrite_x)
    else:
        return np.fft.rfft(a, n=n, axis=axis)


def irfft(
    a: np.ndarray,
    n: Optional[int] = None,
    axis: int = -1,
    overwrite_x: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Compute the inverse FFT for real output.

    Parameters
    ----------
    a : array_like
        Complex input array (from rfft)
    n : int, optional
        Length of the transformed axis of the output
    axis : int, optional
        Axis over which to compute the IFFT (default: -1)
    overwrite_x : bool, optional
        If True, the contents of x can be destroyed (default: False)
    **kwargs
        Additional keyword arguments passed to the backend

    Returns
    -------
    ndarray
        Real IFFT result
    """
    if _PYFFTW_ENABLED and _PYFFTW_AVAILABLE:
        threads = kwargs.pop("threads", pyfftw.config.NUM_THREADS)
        return pyfftw_numpy_fft.irfft(a, n=n, axis=axis, threads=threads)
    elif _SCIPY_AVAILABLE:
        return scipy_fft.irfft(a, n=n, axis=axis, overwrite_x=overwrite_x)
    else:
        return np.fft.irfft(a, n=n, axis=axis)


def rfft2(
    a: np.ndarray,
    s: Optional[Tuple[int, int]] = None,
    axes: Tuple[int, int] = (-2, -1),
    overwrite_x: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Compute the 2-dimensional FFT for real input.

    Parameters
    ----------
    a : array_like
        Real input array
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
    axes : sequence of ints, optional
        Axes over which to compute the FFT (default: (-2, -1))
    overwrite_x : bool, optional
        If True, the contents of x can be destroyed (default: False)
    **kwargs
        Additional keyword arguments passed to the backend

    Returns
    -------
    ndarray
        Complex FFT2 result
    """
    if _PYFFTW_ENABLED and _PYFFTW_AVAILABLE:
        threads = kwargs.pop("threads", pyfftw.config.NUM_THREADS)
        return pyfftw_numpy_fft.rfft2(a, s=s, axes=axes, threads=threads)
    elif _SCIPY_AVAILABLE:
        return scipy_fft.rfft2(a, s=s, axes=axes, overwrite_x=overwrite_x)
    else:
        return np.fft.rfft2(a, s=s, axes=axes)


def irfft2(
    a: np.ndarray,
    s: Optional[Tuple[int, int]] = None,
    axes: Tuple[int, int] = (-2, -1),
    overwrite_x: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Compute the 2-dimensional inverse FFT for real output.

    Parameters
    ----------
    a : array_like
        Complex input array (from rfft2)
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
    axes : sequence of ints, optional
        Axes over which to compute the IFFT (default: (-2, -1))
    overwrite_x : bool, optional
        If True, the contents of x can be destroyed (default: False)
    **kwargs
        Additional keyword arguments passed to the backend

    Returns
    -------
    ndarray
        Real IFFT2 result
    """
    if _PYFFTW_ENABLED and _PYFFTW_AVAILABLE:
        threads = kwargs.pop("threads", pyfftw.config.NUM_THREADS)
        return pyfftw_numpy_fft.irfft2(a, s=s, axes=axes, threads=threads)
    elif _SCIPY_AVAILABLE:
        return scipy_fft.irfft2(a, s=s, axes=axes, overwrite_x=overwrite_x)
    else:
        return np.fft.irfft2(a, s=s, axes=axes)


def rfftn(
    a: np.ndarray,
    s: Optional[Tuple[int, ...]] = None,
    axes: Optional[Tuple[int, ...]] = None,
    overwrite_x: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Compute the N-dimensional FFT for real input.

    Parameters
    ----------
    a : array_like
        Real input array
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
    axes : sequence of ints, optional
        Axes over which to compute the FFT (default: all axes)
    overwrite_x : bool, optional
        If True, the contents of x can be destroyed (default: False)
    **kwargs
        Additional keyword arguments passed to the backend

    Returns
    -------
    ndarray
        Complex FFTN result
    """
    if _PYFFTW_ENABLED and _PYFFTW_AVAILABLE:
        threads = kwargs.pop("threads", pyfftw.config.NUM_THREADS)
        return pyfftw_numpy_fft.rfftn(a, s=s, axes=axes, threads=threads)
    elif _SCIPY_AVAILABLE:
        return scipy_fft.rfftn(a, s=s, axes=axes, overwrite_x=overwrite_x)
    else:
        return np.fft.rfftn(a, s=s, axes=axes)


def irfftn(
    a: np.ndarray,
    s: Optional[Tuple[int, ...]] = None,
    axes: Optional[Tuple[int, ...]] = None,
    overwrite_x: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Compute the N-dimensional inverse FFT for real output.

    Parameters
    ----------
    a : array_like
        Complex input array (from rfftn)
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
    axes : sequence of ints, optional
        Axes over which to compute the IFFT (default: all axes)
    overwrite_x : bool, optional
        If True, the contents of x can be destroyed (default: False)
    **kwargs
        Additional keyword arguments passed to the backend

    Returns
    -------
    ndarray
        Real IFFTN result
    """
    if _PYFFTW_ENABLED and _PYFFTW_AVAILABLE:
        threads = kwargs.pop("threads", pyfftw.config.NUM_THREADS)
        return pyfftw_numpy_fft.irfftn(a, s=s, axes=axes, threads=threads)
    elif _SCIPY_AVAILABLE:
        return scipy_fft.irfftn(a, s=s, axes=axes, overwrite_x=overwrite_x)
    else:
        return np.fft.irfftn(a, s=s, axes=axes)


# =============================================================================
# fftshift/ifftshift Functions
# =============================================================================


def fftshift(
    x: np.ndarray, axes: Optional[Union[int, Tuple[int, ...]]] = None
) -> np.ndarray:
    """
    Shift the zero-frequency component to the center of the spectrum.

    Parameters
    ----------
    x : array_like
        Input array
    axes : int or sequence of ints, optional
        Axes over which to shift (default: all axes)

    Returns
    -------
    ndarray
        Shifted array

    Note
    ----
    fftshift is a simple array reordering operation that is the same
    across all backends. We use scipy.fft.fftshift or np.fft.fftshift.
    """
    if _SCIPY_AVAILABLE:
        return scipy_fft.fftshift(x, axes=axes)
    else:
        return np.fft.fftshift(x, axes=axes)


def ifftshift(
    x: np.ndarray, axes: Optional[Union[int, Tuple[int, ...]]] = None
) -> np.ndarray:
    """
    The inverse of fftshift.

    Parameters
    ----------
    x : array_like
        Input array
    axes : int or sequence of ints, optional
        Axes over which to shift (default: all axes)

    Returns
    -------
    ndarray
        Shifted array
    """
    if _SCIPY_AVAILABLE:
        return scipy_fft.ifftshift(x, axes=axes)
    else:
        return np.fft.ifftshift(x, axes=axes)


# =============================================================================
# Advanced Features - FFT Objects with Plan Caching
# =============================================================================


class FFTPlan:
    """
    FFT plan object for repeated transforms of the same shape.

    This class provides explicit control over FFT planning and can be more
    efficient than the function interface for repeated transforms.

    Usage:
        plan = FFTPlan(shape=(1024,), axes=(0,), direction='forward')
        for data in data_list:
            result = plan(data)

    Note: Currently only provides benefit with pyfftw backend.
    With other backends, it falls back to standard functions.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        axes: Optional[Tuple[int, ...]] = None,
        direction: str = "forward",
        dtype: np.dtype = np.complex128,
        threads: Optional[int] = None,
        planning_timelimit: Optional[float] = None,
    ):
        """
        Create an FFT plan.

        Parameters
        ----------
        shape : tuple of ints
            Shape of the input array
        axes : tuple of ints, optional
            Axes over which to compute FFT (default: all)
        direction : str
            'forward' for FFT, 'backward' for IFFT
        dtype : numpy dtype
            Data type of input (default: complex128)
        threads : int, optional
            Number of threads (default: use global setting)
        planning_timelimit : float, optional
            Time limit for planning in seconds (pyfftw only)
        """
        self.shape = shape
        self.axes = axes if axes is not None else tuple(range(len(shape)))
        self.direction = direction
        self.dtype = dtype
        self.threads = threads
        self._plan = None
        self._input_array = None
        self._output_array = None

        if _PYFFTW_ENABLED and _PYFFTW_AVAILABLE:
            self._create_pyfftw_plan(planning_timelimit)

    def _create_pyfftw_plan(self, planning_timelimit: Optional[float]):
        """Create pyfftw FFTW object."""
        # Create aligned input and output arrays
        self._input_array = pyfftw.empty_aligned(
            self.shape, dtype=self.dtype, n=pyfftw.simd_alignment
        )
        self._output_array = pyfftw.empty_aligned(
            self.shape, dtype=self.dtype, n=pyfftw.simd_alignment
        )

        # Determine direction
        if self.direction == "forward":
            direction = "FFTW_FORWARD"
        else:
            direction = "FFTW_BACKWARD"

        # Set threads
        threads = (
            self.threads if self.threads is not None else pyfftw.config.NUM_THREADS
        )

        # Create plan with FFTW_MEASURE for optimal performance
        # Use FFTW_ESTIMATE for faster planning if timelimit is very short
        flags = ["FFTW_MEASURE"]
        if planning_timelimit is not None and planning_timelimit < 0.1:
            flags = ["FFTW_ESTIMATE"]

        self._plan = pyfftw.FFTW(
            self._input_array,
            self._output_array,
            axes=self.axes,
            direction=direction,
            threads=threads,
            flags=flags,
            planning_timelimit=planning_timelimit,
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Execute the FFT plan on input data.

        Parameters
        ----------
        x : ndarray
            Input array (must match plan shape)

        Returns
        -------
        ndarray
            FFT result
        """
        if self._plan is not None:
            # Use pyfftw plan
            self._input_array[:] = x
            self._plan()
            return self._output_array.copy()
        else:
            # Fallback to function interface
            if self.direction == "forward":
                return fftn(x, axes=self.axes)
            else:
                return ifftn(x, axes=self.axes)


# =============================================================================
# Convenience Functions
# =============================================================================


def benchmark_backends(
    sizes: Union[int, list] = None,
    repeat: int = 100,
    warmup: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Comprehensive benchmark of available FFT backends.

    Parameters
    ----------
    sizes : int or list of int, optional
        Size(s) of test arrays (1D). Default: [256, 1024, 4096, 16384, 65536]
    repeat : int, default=100
        Number of repetitions for timing
    warmup : int, default=5
        Number of warmup iterations (excluded from timing)
    verbose : bool, default=True
        Whether to print progress and results

    Returns
    -------
    dict
        Nested dictionary with structure:
        {
            'sizes': [256, 1024, ...],
            'backends': ['numpy', 'scipy', 'pyfftw'],
            'fft': {
                256: {'numpy': 0.01, 'scipy': 0.008, 'pyfftw': 0.005},
                ...
            },
            'ifft': { ... },
            'fftn_2d': { ... },
            'speedup': {  # Relative to numpy
                'scipy': { 256: 1.25, 1024: 1.5, ... },
                'pyfftw': { 256: 2.0, 1024: 3.0, ... }
            }
        }
    """
    import time

    if sizes is None:
        sizes = [256, 1024, 4096, 16384, 65536]
    elif isinstance(sizes, int):
        sizes = [sizes]

    results = {
        "sizes": sizes,
        "backends": [],
        "fft": {},
        "ifft": {},
        "fftn_2d": {},
        "speedup": {},
    }

    # Determine available backends
    backends = ["numpy"]
    if _SCIPY_AVAILABLE:
        backends.append("scipy")
    if _PYFFTW_AVAILABLE:
        backends.append("pyfftw")
    results["backends"] = backends

    np.random.seed(42)

    def time_func(func, data, warmup_n, repeat_n):
        """Time a function with warmup."""
        for _ in range(warmup_n):
            _ = func(data)
        times = []
        for _ in range(repeat_n):
            start = time.perf_counter()
            _ = func(data)
            times.append(time.perf_counter() - start)
        return np.median(times) * 1000  # ms

    # Benchmark 1D FFT
    for size in sizes:
        if verbose:
            print(f"Benchmarking 1D FFT size {size}...", end=" ")

        data = np.random.randn(size) + 1j * np.random.randn(size)
        results["fft"][size] = {}
        results["ifft"][size] = {}

        # numpy
        results["fft"][size]["numpy"] = time_func(np.fft.fft, data, warmup, repeat)
        results["ifft"][size]["numpy"] = time_func(np.fft.ifft, data, warmup, repeat)

        # scipy
        if _SCIPY_AVAILABLE:
            results["fft"][size]["scipy"] = time_func(
                scipy_fft.fft, data, warmup, repeat
            )
            results["ifft"][size]["scipy"] = time_func(
                scipy_fft.ifft, data, warmup, repeat
            )

        # pyfftw
        if _PYFFTW_AVAILABLE:
            results["fft"][size]["pyfftw"] = time_func(
                pyfftw_numpy_fft.fft, data, warmup, repeat
            )
            results["ifft"][size]["pyfftw"] = time_func(
                pyfftw_numpy_fft.ifft, data, warmup, repeat
            )

        if verbose:
            numpy_time = results["fft"][size]["numpy"]
            print(f"numpy: {numpy_time:.3f}ms", end="")
            if "scipy" in results["fft"][size]:
                scipy_time = results["fft"][size]["scipy"]
                print(
                    f", scipy: {scipy_time:.3f}ms ({numpy_time / scipy_time:.2f}x)",
                    end="",
                )
            if "pyfftw" in results["fft"][size]:
                pyfftw_time = results["fft"][size]["pyfftw"]
                print(
                    f", pyfftw: {pyfftw_time:.3f}ms ({numpy_time / pyfftw_time:.2f}x)",
                    end="",
                )
            print()

    # Benchmark 2D FFT
    for size in sizes:
        if size > 4096:
            # Skip very large 2D for memory
            continue

        if verbose:
            print(f"Benchmarking 2D FFT size {size}x{size}...", end=" ")

        side = int(np.sqrt(size))
        data_2d = np.random.randn(side, side) + 1j * np.random.randn(side, side)
        results["fftn_2d"][size] = {}

        # numpy
        results["fftn_2d"][size]["numpy"] = time_func(
            np.fft.fft2, data_2d, warmup, repeat
        )

        # scipy
        if _SCIPY_AVAILABLE:
            results["fftn_2d"][size]["scipy"] = time_func(
                scipy_fft.fft2, data_2d, warmup, repeat
            )

        # pyfftw
        if _PYFFTW_AVAILABLE:
            results["fftn_2d"][size]["pyfftw"] = time_func(
                pyfftw_numpy_fft.fft2, data_2d, warmup, repeat
            )

        if verbose:
            numpy_time = results["fftn_2d"][size]["numpy"]
            print(f"numpy: {numpy_time:.3f}ms", end="")
            if "scipy" in results["fftn_2d"][size]:
                scipy_time = results["fftn_2d"][size]["scipy"]
                print(
                    f", scipy: {scipy_time:.3f}ms ({numpy_time / scipy_time:.2f}x)",
                    end="",
                )
            if "pyfftw" in results["fftn_2d"][size]:
                pyfftw_time = results["fftn_2d"][size]["pyfftw"]
                print(
                    f", pyfftw: {pyfftw_time:.3f}ms ({numpy_time / pyfftw_time:.2f}x)",
                    end="",
                )
            print()

    # Compute speedup factors
    for backend in ["scipy", "pyfftw"]:
        if backend not in backends:
            continue
        results["speedup"][backend] = {}
        for size in sizes:
            numpy_time = results["fft"][size]["numpy"]
            backend_time = results["fft"][size].get(backend)
            if backend_time:
                results["speedup"][backend][size] = numpy_time / backend_time

    return results


def benchmark_comprehensive(verbose: bool = True) -> dict:
    """
    Run comprehensive FFT benchmarks covering various data types and dimensions.

    This benchmark tests:
    - 1D complex FFT (typical NUFFT use case)
    - 1D real FFT (rfft, optimized for real data)
    - 2D FFT (image processing, 2D regression)
    - 3D FFT (volume data, 3D KDE)

    Returns
    -------
    dict
        Comprehensive benchmark results with timing and speedup information.
    """
    import time

    results = {"info": get_backend_info(), "tests": {}}

    np.random.seed(42)

    test_cases = [
        (
            "1D_complex_small",
            lambda: np.random.randn(1024) + 1j * np.random.randn(1024),
        ),
        (
            "1D_complex_medium",
            lambda: np.random.randn(16384) + 1j * np.random.randn(16384),
        ),
        (
            "1D_complex_large",
            lambda: np.random.randn(131072) + 1j * np.random.randn(131072),
        ),
        ("1D_real_small", lambda: np.random.randn(1024)),
        ("1D_real_medium", lambda: np.random.randn(16384)),
        ("1D_real_large", lambda: np.random.randn(131072)),
        ("2D_64x64", lambda: np.random.randn(64, 64) + 1j * np.random.randn(64, 64)),
        (
            "2D_128x128",
            lambda: np.random.randn(128, 128) + 1j * np.random.randn(128, 128),
        ),
        (
            "2D_256x256",
            lambda: np.random.randn(256, 256) + 1j * np.random.randn(256, 256),
        ),
        (
            "3D_32x32x32",
            lambda: np.random.randn(32, 32, 32) + 1j * np.random.randn(32, 32, 32),
        ),
        (
            "3D_64x64x64",
            lambda: np.random.randn(64, 64, 64) + 1j * np.random.randn(64, 64, 64),
        ),
    ]

    for name, data_fn in test_cases:
        if verbose:
            print(f"Testing {name}...")

        data = data_fn()
        results["tests"][name] = {
            "shape": data.shape,
            "dtype": str(data.dtype),
            "timings": {},
        }

        # Determine which FFT function to use
        ndim = data.ndim
        is_real = np.isrealobj(data)

        if ndim == 1:
            if is_real:
                fft_funcs = {
                    "numpy": np.fft.rfft,
                    "scipy": scipy_fft.rfft if _SCIPY_AVAILABLE else None,
                    "pyfftw": pyfftw_numpy_fft.rfft if _PYFFTW_AVAILABLE else None,
                }
            else:
                fft_funcs = {
                    "numpy": np.fft.fft,
                    "scipy": scipy_fft.fft if _SCIPY_AVAILABLE else None,
                    "pyfftw": pyfftw_numpy_fft.fft if _PYFFTW_AVAILABLE else None,
                }
        elif ndim == 2:
            fft_funcs = {
                "numpy": np.fft.fft2,
                "scipy": scipy_fft.fft2 if _SCIPY_AVAILABLE else None,
                "pyfftw": pyfftw_numpy_fft.fft2 if _PYFFTW_AVAILABLE else None,
            }
        else:
            fft_funcs = {
                "numpy": np.fft.fftn,
                "scipy": scipy_fft.fftn if _SCIPY_AVAILABLE else None,
                "pyfftw": pyfftw_numpy_fft.fftn if _PYFFTW_AVAILABLE else None,
            }

        # Time each backend
        repeat = 50
        warmup = 5

        for backend, func in fft_funcs.items():
            if func is None:
                continue

            # Warmup
            for _ in range(warmup):
                _ = func(data)

            # Timing
            times = []
            for _ in range(repeat):
                start = time.perf_counter()
                _ = func(data)
                times.append(time.perf_counter() - start)

            median_ms = np.median(times) * 1000
            results["tests"][name]["timings"][backend] = median_ms

        # Calculate speedup
        numpy_time = results["tests"][name]["timings"].get("numpy", 1)
        for backend in ["scipy", "pyfftw"]:
            backend_time = results["tests"][name]["timings"].get(backend)
            if backend_time:
                speedup = numpy_time / backend_time
                results["tests"][name][f"{backend}_speedup"] = speedup

    if verbose:
        print("\n" + "=" * 60)
        print("FFT Backend Benchmark Results")
        print("=" * 60)
        print(f"Active backend: {results['info']['backend']}")
        if results["info"]["backend"] == "pyfftw":
            print(f"Threads: {results['info']['num_threads']}")
            print(f"Cache enabled: {results['info']['cache_enabled']}")
        print("-" * 60)
        print(
            f"{'Test':<25} {'numpy':>10} {'scipy':>10} {'pyfftw':>10} {'speedup':>10}"
        )
        print("-" * 60)
        for name, data in results["tests"].items():
            numpy_time = data["timings"].get("numpy", float("nan"))
            scipy_time = data["timings"].get("scipy", float("nan"))
            pyfftw_time = data["timings"].get("pyfftw", float("nan"))
            best_speedup = max(
                data.get("scipy_speedup", 1), data.get("pyfftw_speedup", 1)
            )
            print(
                f"{name:<25} {numpy_time:>9.3f}ms {scipy_time:>9.3f}ms {pyfftw_time:>9.3f}ms {best_speedup:>9.2f}x"
            )
        print("=" * 60)

    return results


def print_backend_info():
    """Print information about the FFT backend configuration."""
    info = get_backend_info()
    print("=" * 50)
    print("fastLPR FFT Backend Configuration")
    print("=" * 50)
    print(f"Active backend:    {info['backend']}")
    print(f"pyfftw available:  {info['pyfftw_available']}")
    print(f"pyfftw enabled:    {info['pyfftw_enabled']}")
    print(f"scipy available:   {info['scipy_available']}")
    print(f"Number of threads: {info['num_threads']}")
    print(f"Cache enabled:     {info['cache_enabled']}")
    print("=" * 50)


# Module initialization: print backend info if verbose mode
if os.environ.get("FASTLPR_FFT_VERBOSE", "").lower() in ("1", "true", "yes"):
    print_backend_info()
