# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Type utilities for memory-efficient NUFFT processing.

This module provides functions to select optimal data types for arrays,
matching MATLAB's bit_convert and min_numerical_type functions.

Author: Ying Wang, Min Li
"""

import numpy as np
from typing import Tuple, Union, List


def bit_convert(accuracy: int, *arrays) -> Tuple[np.ndarray, ...]:
    """
    Convert arrays to appropriate precision based on accuracy requirement.

    Mirrors MATLAB's bit_convert function exactly.

    Args:
        accuracy: Required accuracy in decimal digits
                  < 5: convert to float32 (single precision)
                  >= 5: convert to float64 (double precision)
        *arrays: Input arrays to convert

    Returns:
        Tuple of converted arrays

    Example:
        >>> x, y = bit_convert(3, x, y)  # Convert to float32 (3 < 5)
        >>> x, y = bit_convert(10, x, y)  # Convert to float64 (10 >= 5)

    Notes:
        - float32: 4 bytes, ~7 decimal digits precision
        - float64: 8 bytes, ~15 decimal digits precision
        - Threshold of 5 digits balances accuracy and performance
    """
    if accuracy < 5:
        # Low accuracy: use single precision (faster, less memory)
        dtype = np.float32
    else:
        # High accuracy: use double precision (more accurate)
        dtype = np.float64

    return tuple(arr.astype(dtype) if arr.dtype != dtype else arr for arr in arrays)


def min_numerical_type(minval: int, maxval: int, redundancy: float = 1.0,
                       isunsign: bool = None, isfloating: bool = None) -> Tuple[np.dtype, int]:
    """
    Find smallest numeric type that can represent a range [minval, maxval].

    Mirrors MATLAB's min_numerical_type function exactly.

    Args:
        minval: Minimum value to represent (default: 0)
        maxval: Maximum value to represent (required)
        redundancy: Safety factor for max value (default: 1)
        isunsign: Force unsigned type (default: auto from minval >= 0)
        isfloating: Force floating-point type (default: auto from integer check)

    Returns:
        Tuple of (dtype, bytes_per_element)

    Example:
        >>> dtype, nbytes = min_numerical_type(0, 200)
        >>> # Returns: (np.uint8, 1) - can hold 0-255
        >>> dtype, nbytes = min_numerical_type(0, 1000)
        >>> # Returns: (np.uint16, 2) - can hold 0-65535

    Type Selection Priority:
        1. Unsigned integers: uint8, uint16, uint32, uint64
        2. Signed integers: int8, int16, int32, int64
        3. Floating-point: float32, float64
    """
    # Auto-detect unsigned if minval >= 0
    if isunsign is None:
        isunsign = minval >= 0

    # Auto-detect floating if values are not integers
    if isfloating is None:
        isfloating = not (float(minval).is_integer() and float(maxval).is_integer())

    # Define type candidates with their max values
    uint_types = [
        (np.uint8, np.iinfo(np.uint8).max),
        (np.uint16, np.iinfo(np.uint16).max),
        (np.uint32, np.iinfo(np.uint32).max),
        (np.uint64, np.iinfo(np.uint64).max),
    ]
    int_types = [
        (np.int8, np.iinfo(np.int8).max),
        (np.int16, np.iinfo(np.int16).max),
        (np.int32, np.iinfo(np.int32).max),
        (np.int64, np.iinfo(np.int64).max),
    ]
    float_types = [
        (np.float32, np.finfo(np.float32).max),
        (np.float64, np.finfo(np.float64).max),
    ]

    # Select type candidates based on requirements
    if isunsign and not isfloating:
        # Unsigned integers preferred, then signed, then float
        candidates = uint_types + int_types + float_types
    elif not isunsign and not isfloating:
        # Signed integers preferred, then float
        candidates = int_types + float_types
    else:
        # Only floating-point types
        candidates = float_types

    # Find smallest type that can hold maxval with redundancy
    for dtype, type_max in candidates:
        if type_max * redundancy > maxval:
            return dtype, np.dtype(dtype).itemsize

    raise ValueError(f"No numeric type can represent maxval={maxval} with redundancy={redundancy}")


def get_optimal_index_dtype(max_grid_size: int) -> np.dtype:
    """
    Get optimal integer dtype for grid indices.

    This is a convenience wrapper for min_numerical_type for the common case
    of grid indexing in NUFFT.

    Args:
        max_grid_size: Maximum grid dimension (max of Mr array)

    Returns:
        Optimal numpy dtype for indices

    Example:
        >>> dtype = get_optimal_index_dtype(200)  # Returns np.uint8
        >>> dtype = get_optimal_index_dtype(1000)  # Returns np.uint16
        >>> dtype = get_optimal_index_dtype(100000)  # Returns np.uint32
    """
    dtype, _ = min_numerical_type(0, max_grid_size)
    return dtype


def get_float_dtype(accuracy: int) -> np.dtype:
    """
    Get optimal float dtype based on accuracy requirement.

    Args:
        accuracy: NUFFT accuracy parameter (number of decimal digits)

    Returns:
        np.float32 if accuracy < 5, else np.float64
    """
    return np.float32 if accuracy < 5 else np.float64


def get_complex_dtype(accuracy: int) -> np.dtype:
    """
    Get optimal complex dtype based on accuracy requirement.

    Args:
        accuracy: NUFFT accuracy parameter (number of decimal digits)

    Returns:
        np.complex64 if accuracy < 5, else np.complex128
    """
    return np.complex64 if accuracy < 5 else np.complex128
