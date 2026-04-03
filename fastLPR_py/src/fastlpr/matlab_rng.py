# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
MATLAB-compatible random number generation utilities.

This module provides RNG functions that attempt to match MATLAB's Mersenne Twister
random number generator behavior. Both MATLAB and NumPy use MT19937 internally,
but their initialization differs, so sequences will NOT be identical.

The purpose of this module is:
1. Provide a consistent interface for seeded RNG in Python
2. Document the differences between MATLAB and NumPy RNG
3. Enable reproducibility within Python while acknowledging cross-language differences

IMPORTANT NOTES:
- MATLAB's rng(0) uses the "default" state (equivalent to seed 5489), not seed 0
- MATLAB and NumPy initialize MT19937 differently from the same seed
- For exact reproducibility, use saved random vectors from MATLAB reference files

Author: Ying Wang, Min Li
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from numpy.random import Generator, MT19937


def matlab_rng(seed: int) -> Generator:
    """
    Create a NumPy random Generator using Mersenne Twister, similar to MATLAB's rng().

    This creates a Generator backed by MT19937, which is the same algorithm MATLAB uses.
    However, the state initialization differs between MATLAB and NumPy, so sequences
    will not match exactly.

    Parameters
    ----------
    seed : int
        Seed for the random number generator.

    Returns
    -------
    Generator
        NumPy random Generator backed by MT19937.

    Examples
    --------
    >>> rng = matlab_rng(42)
    >>> rng.random(5)  # Uniform random numbers
    >>> rng.standard_normal(5)  # Normal random numbers

    Notes
    -----
    MATLAB's initialization differs from NumPy's:
    - MATLAB: rng(seed, 'twister') uses a proprietary initialization
    - NumPy: MT19937(seed) uses the standard reference implementation

    For reproducibility across languages, save and load the actual random vectors
    rather than relying on seed matching.
    """
    return Generator(MT19937(seed))


def matlab_rand(
    n: Union[int, Tuple[int, ...]], seed: Optional[int] = None, rng: Optional[Generator] = None
) -> np.ndarray:
    """
    Generate uniform random numbers like MATLAB's rand().

    Produces values uniformly distributed in [0, 1).

    Parameters
    ----------
    n : int or tuple of ints
        Shape of the output array. If int, produces (n,) array.
        If tuple, produces array with that shape.
    seed : int, optional
        Seed for reproducibility. Creates new Generator if provided.
        Ignored if rng is provided.
    rng : Generator, optional
        Existing NumPy Generator to use. If not provided and seed is given,
        creates a new Generator with that seed.

    Returns
    -------
    np.ndarray
        Array of uniform random values in [0, 1).

    Examples
    --------
    >>> # Single call with seed
    >>> x = matlab_rand(100, seed=42)

    >>> # Multiple calls with same Generator
    >>> rng = matlab_rng(42)
    >>> x1 = matlab_rand(10, rng=rng)
    >>> x2 = matlab_rand(10, rng=rng)  # Continues sequence

    >>> # 2D array like MATLAB's rand(100, 5)
    >>> x = matlab_rand((100, 5), seed=42)

    Notes
    -----
    MATLAB's rand() and NumPy's random() both produce U[0,1) values using MT19937,
    but due to initialization differences, sequences will not match exactly.
    """
    if rng is None:
        if seed is not None:
            rng = matlab_rng(seed)
        else:
            # Use global NumPy random state (not recommended for reproducibility)
            return np.random.random(n if isinstance(n, tuple) else (n,))

    shape = n if isinstance(n, tuple) else (n,)
    return rng.random(shape)


def matlab_randn(
    n: Union[int, Tuple[int, ...]], seed: Optional[int] = None, rng: Optional[Generator] = None
) -> np.ndarray:
    """
    Generate standard normal random numbers like MATLAB's randn().

    Produces values from the standard normal distribution N(0, 1).

    Parameters
    ----------
    n : int or tuple of ints
        Shape of the output array. If int, produces (n,) array.
        If tuple, produces array with that shape.
    seed : int, optional
        Seed for reproducibility. Creates new Generator if provided.
        Ignored if rng is provided.
    rng : Generator, optional
        Existing NumPy Generator to use. If not provided and seed is given,
        creates a new Generator with that seed.

    Returns
    -------
    np.ndarray
        Array of standard normal random values.

    Examples
    --------
    >>> # Single call with seed
    >>> z = matlab_randn(100, seed=42)

    >>> # Multiple calls with same Generator
    >>> rng = matlab_rng(42)
    >>> z1 = matlab_randn(10, rng=rng)
    >>> z2 = matlab_randn(10, rng=rng)  # Continues sequence

    >>> # 2D array like MATLAB's randn(100, 5)
    >>> z = matlab_randn((100, 5), seed=42)

    Notes
    -----
    Both MATLAB and NumPy use the Ziggurat algorithm for normal random number
    generation on top of MT19937 uniform values. However, the transformation
    and initialization differences mean sequences will not match exactly.

    For DOF estimation in fastLPR, use the saved dof_random_vectors from
    MATLAB reference files for exact reproducibility.
    """
    if rng is None:
        if seed is not None:
            rng = matlab_rng(seed)
        else:
            # Use global NumPy random state (not recommended for reproducibility)
            return np.random.standard_normal(n if isinstance(n, tuple) else (n,))

    shape = n if isinstance(n, tuple) else (n,)
    return rng.standard_normal(shape)


def compare_rng_sequences(seed: int, n: int = 100) -> dict:
    """
    Generate RNG comparison data for verification against MATLAB.

    This is a diagnostic function to help understand the differences between
    MATLAB and NumPy random number generation.

    Parameters
    ----------
    seed : int
        Seed to test.
    n : int, default=100
        Number of random values to generate.

    Returns
    -------
    dict
        Dictionary containing:
        - 'seed': The seed used
        - 'randn': Standard normal values from NumPy
        - 'rand': Uniform values from NumPy
        - 'first_5': First 5 randn values for quick comparison

    Examples
    --------
    >>> data = compare_rng_sequences(42)
    >>> print(data['first_5'])
    """
    rng = matlab_rng(seed)
    randn_values = rng.standard_normal(n)

    rng2 = matlab_rng(seed)
    rand_values = rng2.random(n)

    return {
        "seed": seed,
        "randn": randn_values,
        "rand": rand_values,
        "first_5": randn_values[:5],
    }


__all__ = [
    "matlab_rng",
    "matlab_rand",
    "matlab_randn",
    "compare_rng_sequences",
]
