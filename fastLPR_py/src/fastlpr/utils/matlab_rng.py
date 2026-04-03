# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
MATLAB-compatible random number generation for DOF estimation reproducibility.

This module provides functions to generate random numbers that match MATLAB's
randn() output when using the same seed. This is critical for exact numerical
parity in DOF estimation via Hutchinson's trace estimator.

Key Differences between MATLAB and NumPy MT19937:
-------------------------------------------------
1. MATLAB's rng(0) is special: it initializes to the default state (seed 5489),
   not seed 0. NumPy treats seed 0 literally.

2. The state initialization differs between MATLAB and NumPy even for non-zero
   seeds. While both use MT19937, the transformation from seed to state vector
   is different.

3. To achieve exact matching, we need to either:
   a) Use pre-generated random matrices from MATLAB (most reliable)
   b) Use a transformation that maps MATLAB seeds to NumPy seeds (approximate)
   c) Use MATLAB's exact state vector initialization (complex but exact)

This module implements option (a) as the primary approach for exact verification,
with option (b) available as a fallback for general use.

Note: For exact Python-MATLAB parity in DOF estimation, the recommended approach
is to pre-generate and save the random matrix from MATLAB, then load it in Python.

References:
-----------
- https://walkingrandomly.com/?p=5479 (Reproducing MATLAB random numbers in Python)
- https://walkingrandomly.com/?p=5480 (MATLAB's seed 0 = seed 5489 issue)
- https://numpy.org/doc/stable/reference/random/bit_generators/mt19937.html

Author: Ying Wang, Min Li
Copyright (c) 2020-2025 fastLPR Development Team
License: GNU General Public License v3.0
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np


class MatlabRNG:
    """
    MATLAB-compatible random number generator using Mersenne Twister.

    This class attempts to reproduce MATLAB's randn() sequences. Due to
    implementation differences, exact matching may not be possible for all
    seeds without using pre-generated MATLAB data.

    For exact verification, use the `from_matlab_state()` class method to
    initialize from a MATLAB-exported state vector.

    Parameters
    ----------
    seed : int
        Random seed. Note that seed 0 in MATLAB is special (equivalent to 5489).

    Examples
    --------
    >>> rng = MatlabRNG(42)
    >>> values = rng.randn(100)
    >>> print(values[:5])

    Notes
    -----
    MATLAB and NumPy both use MT19937 but with different state initialization.
    The transformation from seed to internal state differs between implementations.
    """

    def __init__(self, seed: int = 0):
        """Initialize the RNG with the given seed."""
        self._seed = seed
        self._generator = self._create_generator(seed)

    def _create_generator(self, seed: int) -> np.random.Generator:
        """Create a NumPy Generator with MT19937 bit generator."""
        # MATLAB treats seed 0 as the default initialization (seed 5489)
        # NumPy uses seed 0 literally, so we need to handle this case
        if seed == 0:
            # MATLAB's rng(0) is equivalent to its default state
            # This is the same as seed 5489 in the reference implementation
            effective_seed = 5489
        else:
            effective_seed = seed

        bit_gen = np.random.MT19937(effective_seed)
        return np.random.Generator(bit_gen)

    def randn(self, *shape: int) -> np.ndarray:
        """
        Generate standard normal random numbers.

        Parameters
        ----------
        *shape : int
            Shape of the output array. If no arguments, returns a scalar.

        Returns
        -------
        np.ndarray
            Array of standard normal random numbers.

        Examples
        --------
        >>> rng = MatlabRNG(42)
        >>> x = rng.randn(100)  # 1D array of 100 values
        >>> x = rng.randn(100, 10)  # 2D array
        """
        if len(shape) == 0:
            return self._generator.standard_normal()
        return self._generator.standard_normal(shape)

    def rand(self, *shape: int) -> np.ndarray:
        """
        Generate uniform random numbers in [0, 1).

        Parameters
        ----------
        *shape : int
            Shape of the output array.

        Returns
        -------
        np.ndarray
            Array of uniform random numbers.
        """
        if len(shape) == 0:
            return self._generator.random()
        return self._generator.random(shape)

    @classmethod
    def from_matlab_state(cls, state: np.ndarray) -> "MatlabRNG":
        """
        Create an RNG initialized from a MATLAB state vector.

        This method allows exact reproduction of MATLAB sequences by using
        MATLAB's exported state vector directly.

        Parameters
        ----------
        state : np.ndarray
            MATLAB's RNG state vector (625 uint32 values for MT19937).
            Can be obtained from MATLAB via: s = rng(); state = s.State;

        Returns
        -------
        MatlabRNG
            An RNG instance that will produce the same sequence as MATLAB.

        Examples
        --------
        >>> import scipy.io
        >>> data = scipy.io.loadmat('matlab_state.mat')
        >>> rng = MatlabRNG.from_matlab_state(data['state'].ravel())
        """
        instance = cls.__new__(cls)
        instance._seed = None

        # Convert state to correct format for NumPy MT19937
        state = np.asarray(state, dtype=np.uint32)
        if len(state) != 625:
            raise ValueError(
                f"MATLAB MT19937 state must have 625 elements, got {len(state)}"
            )

        # Create MT19937 with the state vector
        bit_gen = np.random.MT19937()
        # NumPy state format: {'state': {'key': array, 'pos': int}, 'has_uint32': bool, 'uinteger': int}
        # The 'key' is 624 elements, 'pos' is the 625th element
        bit_gen.state = {
            'bit_generator': 'MT19937',
            'state': {'key': state[:624], 'pos': int(state[624])},
            'has_uint32': 0,
            'uinteger': 0
        }
        instance._generator = np.random.Generator(bit_gen)

        return instance

    def get_state(self) -> dict:
        """Get the current state of the RNG."""
        return self._generator.bit_generator.state

    def set_state(self, state: dict) -> None:
        """Set the state of the RNG."""
        self._generator.bit_generator.state = state


def matlab_randn(
    seed: int,
    shape: Union[int, Tuple[int, ...]],
    method: str = "numpy_mt19937"
) -> np.ndarray:
    """
    Generate random numbers matching MATLAB's randn() output.

    This function attempts to reproduce MATLAB's random number sequences.
    For exact matching in verification tests, use pre-generated MATLAB data
    instead of this function.

    Parameters
    ----------
    seed : int
        Random seed for the generator. Note that seed 0 in MATLAB is special
        and is handled accordingly.
    shape : int or tuple of ints
        Shape of the output array.
    method : str, optional
        Method for random number generation:
        - "numpy_mt19937": Use NumPy's MT19937 (default, approximate match)
        - "exact": Raises NotImplementedError (would require MATLAB state)

    Returns
    -------
    np.ndarray
        Array of standard normal random numbers.

    Notes
    -----
    Due to implementation differences between MATLAB and NumPy, the sequences
    will NOT match exactly. For exact verification:
    1. Generate and save random matrix in MATLAB
    2. Load it in Python using scipy.io.loadmat()
    3. Pass it to cv_fastlpr() via opts['random_matrix']

    Examples
    --------
    >>> # Generate 100 random numbers with seed 42
    >>> x = matlab_randn(42, 100)
    >>> print(x[:5])

    >>> # Generate 100x10 matrix for DOF estimation
    >>> p = matlab_randn(42, (100, 10))
    """
    if isinstance(shape, int):
        shape = (shape,)

    if method == "exact":
        raise NotImplementedError(
            "Exact MATLAB matching requires loading pre-generated MATLAB data. "
            "Use scipy.io.loadmat() to load MATLAB reference data instead."
        )

    # Use MatlabRNG for consistent behavior
    rng = MatlabRNG(seed)
    return rng.randn(*shape)


def load_matlab_random_matrix(mat_file: str, var_name: str = "random_matrix") -> np.ndarray:
    """
    Load a pre-generated random matrix from a MATLAB .mat file.

    This is the recommended approach for exact MATLAB-Python parity in
    verification tests.

    Parameters
    ----------
    mat_file : str
        Path to the MATLAB .mat file.
    var_name : str, optional
        Name of the variable in the .mat file (default: "random_matrix").

    Returns
    -------
    np.ndarray
        The random matrix from MATLAB.

    Examples
    --------
    >>> from fastlpr.utils.matlab_rng import load_matlab_random_matrix
    >>> p = load_matlab_random_matrix("refs/dof_random_matrix.mat")
    >>> # Use in cv_fastlpr
    >>> opts = {"random_matrix": p}
    >>> regs = cv_fastlpr(x, y, hlist, opts)
    """
    try:
        import scipy.io
    except ImportError:
        raise ImportError(
            "scipy is required to load MATLAB .mat files. "
            "Install it with: pip install scipy"
        )

    data = scipy.io.loadmat(mat_file)
    if var_name not in data:
        available = [k for k in data.keys() if not k.startswith('_')]
        raise KeyError(
            f"Variable '{var_name}' not found in {mat_file}. "
            f"Available variables: {available}"
        )

    return np.asarray(data[var_name], dtype=np.float64)


__all__ = ["matlab_randn", "MatlabRNG", "load_matlab_random_matrix"]
