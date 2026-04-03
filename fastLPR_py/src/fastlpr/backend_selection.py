# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Centralized backend detection and selection for fastLPR.

This module is the SINGLE SOURCE OF TRUTH for:
1. Backend availability (Numba, Cython, Fused Spreading)
2. Backend preference (from FASTLPR_BACKEND environment variable)
3. Backend selection logic for accumulation and spreading operations

All other modules should import from here instead of doing their own detection.

Usage:
    from fastlpr.backend_selection import (
        HAS_NUMBA, HAS_CYTHON, HAS_FUSED_SPREADING,
        select_accumulation_backend, select_spreading_backend,
        get_backend_info, get_accumulation_function,
    )

Environment Variables:
    FASTLPR_BACKEND: 'auto' (default), 'numba', 'cython'
        - auto: Prefer Numba if available, otherwise Cython, then Python
        - numba: Force Numba, fall back to Cython if unavailable, then Python
        - cython: Force Cython, fall back to Numba if unavailable, then Python

Author: Ying Wang, Min Li
"""

import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

__all__ = [
    # Availability flags
    'HAS_NUMBA',
    'HAS_CYTHON',
    'HAS_FUSED_SPREADING',
    # Backend info
    'BackendInfo',
    'get_backend_info',
    # Selection functions
    'select_accumulation_backend',
    'select_spreading_backend',
    'get_accumulation_function',
    # Preference
    'get_backend_preference',
]

# =============================================================================
# Backend Availability Detection (SINGLE SOURCE OF TRUTH)
# =============================================================================

# Numba detection
try:
    import numba
    HAS_NUMBA = True
    NUMBA_VERSION = numba.__version__
except ImportError:
    HAS_NUMBA = False
    NUMBA_VERSION = None

# Cython acceleration detection
try:
    from ._nufft_accel import nufft_accumulate_cython
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False
    nufft_accumulate_cython = None

# Fused spreading kernels detection (requires Numba)
try:
    from .spreading_fused import (
        fused_spreading_dispatch as _fused_dispatch,
        HAS_FUSED_SPREADING as _HAS_FUSED,
    )
    HAS_FUSED_SPREADING = _HAS_FUSED
except ImportError:
    HAS_FUSED_SPREADING = False
    _fused_dispatch = None


# =============================================================================
# Backend Preference
# =============================================================================

def get_backend_preference() -> str:
    """
    Get the backend preference from environment variable.

    Returns:
        str: One of 'auto', 'numba', 'cython'
    """
    pref = os.environ.get('FASTLPR_BACKEND', 'auto').lower()
    if pref not in ('auto', 'numba', 'cython'):
        pref = 'auto'
    return pref


# =============================================================================
# Backend Selection Logic
# =============================================================================

def select_accumulation_backend(
    preference: Optional[str] = None,
    dy: int = 1,
) -> str:
    """
    Select the best accumulation backend based on preference and availability.

    The selection logic:
    1. If preference='cython' and Cython available -> 'cython'
    2. If preference='numba' and Numba available -> 'numba'
    3. If preference='numba' and Numba unavailable but Cython available -> 'cython' (fallback)
    4. If preference='auto':
       - If Numba available -> 'numba' (default preference)
       - Else if Cython available -> 'cython'
       - Else -> 'python'
    5. Otherwise -> 'python'

    Note: Both Numba and Cython accumulation are SEQUENTIAL (not parallel)
    due to race conditions in scatter-add operations.

    Args:
        preference: 'auto', 'numba', 'cython', or None (uses env var)
        dy: Number of output columns (Cython has issues with dy > 1)

    Returns:
        str: 'numba', 'cython', or 'python'
    """
    if preference is None:
        preference = get_backend_preference()

    if preference == 'cython':
        if HAS_CYTHON:
            return 'cython'
        elif HAS_NUMBA:
            return 'numba'  # Fallback
        else:
            return 'python'

    elif preference == 'numba':
        if HAS_NUMBA:
            return 'numba'
        elif HAS_CYTHON:
            return 'cython'  # Fallback
        else:
            return 'python'

    else:  # auto
        # Default: prefer Numba (better memory efficiency, similar speed)
        if HAS_NUMBA:
            return 'numba'
        elif HAS_CYTHON:
            return 'cython'
        else:
            return 'python'


def select_spreading_backend(dx: int) -> str:
    """
    Select the best spreading backend based on dimension and availability.

    The fused spreading kernels (Numba) are 20-40x faster than vectorized
    NumPy for dimensions 1-3.

    Args:
        dx: Number of input dimensions

    Returns:
        str: 'fused_numba' or 'vectorized'
    """
    if HAS_FUSED_SPREADING and dx <= 3:
        return 'fused_numba'
    return 'vectorized'


# =============================================================================
# Backend Info
# =============================================================================

@dataclass
class BackendInfo:
    """Information about the active backends."""
    accumulation_backend: str  # 'numba', 'cython', 'python'
    spreading_backend: str     # 'fused_numba', 'vectorized'
    accumulation_parallel: bool  # Always False (race conditions)
    spreading_parallel: bool     # False (single-threaded kernels)

    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility."""
        return {
            'accumulation': {
                'backend': self.accumulation_backend,
                'parallel': self.accumulation_parallel,
                'status': 'active' if self.accumulation_backend != 'python' else 'fallback',
            },
            'spreading': {
                'backend': self.spreading_backend,
                'parallel': self.spreading_parallel,
                'status': 'active' if self.spreading_backend == 'fused_numba' else 'fallback',
                'speedup': '20-40x' if self.spreading_backend == 'fused_numba' else 'N/A',
            },
            # Legacy fields (for backward compatibility)
            'backend': self.spreading_backend if self.spreading_backend == 'fused_numba' else self.accumulation_backend,
            'parallel': self.accumulation_parallel,
            'status': 'active' if self.accumulation_backend != 'python' else 'fallback',
        }


def get_backend_info(
    dx: int = 2,
    preference: Optional[str] = None,
) -> dict:
    """
    Get information about the active backends.

    This function ALWAYS reflects the actual execution path, not just
    what backends are available.

    Args:
        dx: Number of dimensions (affects spreading backend selection)
        preference: Backend preference (default: from env var)

    Returns:
        dict: Backend information including 'accumulation', 'spreading', and legacy fields
    """
    accumulation = select_accumulation_backend(preference=preference)
    spreading = select_spreading_backend(dx)

    info = BackendInfo(
        accumulation_backend=accumulation,
        spreading_backend=spreading,
        accumulation_parallel=False,  # Both Numba and Cython are sequential
        spreading_parallel=False,      # Fused kernels are single-threaded
    )

    return info.to_dict()


# =============================================================================
# Accumulation Function Dispatch
# =============================================================================

def get_accumulation_function(
    preference: Optional[str] = None,
    dy: int = 1,
) -> Tuple[Callable, str]:
    """
    Get the accumulation function and its backend name.

    This is the single point of dispatch for accumulation operations.
    All modules should use this instead of their own if/elif chains.

    Args:
        preference: Backend preference
        dy: Number of output columns

    Returns:
        Tuple of (accumulation_function, backend_name)
        The function signature is: func(Ftau_flat, xindx, ysp, dy, strides)
    """
    backend = select_accumulation_backend(preference=preference, dy=dy)

    if backend == 'numba':
        # Import locally to avoid circular imports
        from .nufftn_type1 import _accumulate_numba
        return _accumulate_numba, 'numba'

    elif backend == 'cython':
        # Wrapper to match the Numba signature
        def _cython_wrapper(Ftau_flat, xindx, ysp, dy, strides):
            nufft_accumulate_cython(Ftau_flat, xindx, ysp, strides)

        return _cython_wrapper, 'cython'

    else:  # python
        def _python_fallback(Ftau_flat, xindx, ysp, dy, strides):
            n_points, dx = xindx.shape
            # Use Python fallback implementation
            for iy in range(dy):
                for idx in range(n_points):
                    linear_idx = 0
                    for d in range(dx):
                        linear_idx += int(xindx[idx, d] * strides[d])
                    Ftau_flat[linear_idx, iy] += ysp[idx, iy]

        return _python_fallback, 'python'


def get_spreading_function(dx: int):
    """
    Get the spreading function for the given dimension.

    Args:
        dx: Number of dimensions

    Returns:
        Tuple of (spreading_function, backend_name) or (None, 'vectorized')
    """
    if HAS_FUSED_SPREADING and dx <= 3:
        return _fused_dispatch, 'fused_numba'
    return None, 'vectorized'
