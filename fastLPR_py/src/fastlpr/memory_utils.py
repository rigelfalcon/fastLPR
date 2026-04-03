# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Memory utilities for adaptive chunked NUFFT processing.

This module provides functions to estimate memory requirements and compute
optimal chunk sizes for memory-efficient NUFFT operations. Similar to MATLAB's
self-aware memory handling for multiple y columns.

Author: Ying Wang, Min Li
"""

import numpy as np
from typing import Tuple, Optional
import gc


def get_available_memory() -> int:
    """
    Get available system memory (excluding swap), in bytes.

    Returns:
        int: Available physical memory in bytes. Returns a large default
             (16GB) if psutil is not available.
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.available  # Physical memory available (excludes swap)
    except ImportError:
        # Fallback if psutil not installed: assume 16GB available
        return 16 * 1024**3


def estimate_nufft_memory(M_chunk: int, dx: int, Msp2_prod: int, dy: int = 1) -> int:
    """
    Estimate peak memory required to process M_chunk points in NUFFT.

    Memory breakdown for NUFFT spreading phase:
    - mpmm: (M_chunk, Msp2_prod, dx) float64 -> M_chunk * Msp2_prod * dx * 8
    - xmod_temp: (M_chunk, Msp2_prod, dx) float64 -> M_chunk * Msp2_prod * dx * 8
    - weight: (M_chunk, Msp2_prod) float64 -> M_chunk * Msp2_prod * 8
    - ysp: (M_chunk * Msp2_prod, dy) complex128 -> M_chunk * Msp2_prod * dy * 16
    - xindx: (M_chunk * Msp2_prod, dx) int64 -> M_chunk * Msp2_prod * dx * 8

    Total per point: Msp2_prod * (dx*8 + dx*8 + 8 + dy*16 + dx*8)
                   = Msp2_prod * (24*dx + 8 + 16*dy)

    Args:
        M_chunk: Number of data points in chunk
        dx: Number of input dimensions
        Msp2_prod: Product of (2*Msp+1) across all dimensions
        dy: Number of output dimensions (y columns)

    Returns:
        int: Estimated peak memory in bytes
    """
    # Memory per point: mpmm(dx*8) + xmod_temp(dx*8) + weight(8) + ysp(dy*16) + xindx(dx*8)
    bytes_per_point = Msp2_prod * (24 * dx + 8 + 16 * dy)
    return M_chunk * bytes_per_point


def compute_optimal_chunk_size(
    M: int,
    dx: int,
    Msp2_prod: int,
    dy: int = 1,
    safety_factor: float = 0.7,
    min_chunk: int = 10000,
) -> int:
    """
    Compute optimal chunk size based on available system memory.

    Uses a safety factor (default 70%) to leave headroom for:
    - FFT workspace allocations
    - Temporary arrays during computation
    - Other system processes

    For small datasets that fit in memory, returns M (no chunking).
    For large datasets, returns the largest chunk that fits safely.

    Args:
        M: Total number of data points
        dx: Number of input dimensions
        Msp2_prod: Product of (2*Msp+1) across all dimensions
        dy: Number of output dimensions
        safety_factor: Fraction of available memory to use (0.0-1.0)
        min_chunk: Minimum chunk size (for efficiency)

    Returns:
        int: Optimal chunk size (M if no chunking needed)

    Example:
        >>> # For N=33M, d=3, accuracy=6 with 16GB available
        >>> Msp2_prod = 13**3  # 2197
        >>> chunk = compute_optimal_chunk_size(33554432, 3, Msp2_prod)
        >>> # Returns ~500K (fits in 70% of 16GB)
    """
    available = get_available_memory() * safety_factor
    bytes_per_point = estimate_nufft_memory(1, dx, Msp2_prod, dy)

    # Maximum chunk size that fits in available memory
    max_chunk = int(available / bytes_per_point) if bytes_per_point > 0 else M

    # Clamp to [min_chunk, M]
    chunk_size = max(min_chunk, min(max_chunk, M))

    # If whole dataset fits, return M (no chunking)
    if chunk_size >= M:
        return M

    return chunk_size


def should_use_chunking(
    M: int, dx: int, Msp2_prod: int, dy: int = 1, threshold: float = 0.5
) -> bool:
    """
    Determine if chunked processing should be used.

    Returns True if the dataset would use more than `threshold` of available
    memory, suggesting chunked processing is beneficial.

    Args:
        M: Total number of data points
        dx: Number of input dimensions
        Msp2_prod: Product of (2*Msp+1) across all dimensions
        dy: Number of output dimensions
        threshold: Memory fraction threshold (default 0.5)

    Returns:
        bool: True if chunking is recommended
    """
    required = estimate_nufft_memory(M, dx, Msp2_prod, dy)
    available = get_available_memory()
    return required > (available * threshold)


def get_memory_info() -> dict:
    """
    Get current system memory information.

    Returns:
        dict: Memory info with keys:
            - total_gb: Total physical memory in GB
            - available_gb: Available memory in GB
            - percent_used: Percentage of memory in use
            - psutil_available: Whether psutil is installed
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            'total_gb': mem.total / 1024**3,
            'available_gb': mem.available / 1024**3,
            'percent_used': mem.percent,
            'psutil_available': True
        }
    except ImportError:
        return {
            'total_gb': None,
            'available_gb': 16.0,  # Default assumption
            'percent_used': None,
            'psutil_available': False
        }


def force_gc():
    """Force garbage collection to free memory."""
    gc.collect()
