# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Utility functions for fastLPR Python package.

This subpackage provides utilities for MATLAB compatibility,
including random number generation that matches MATLAB's output.
"""

from .matlab_rng import matlab_randn, MatlabRNG, load_matlab_random_matrix
from .helpers import is_numba_available, _as_column_matrix, _default_hrange

# Import set_defaults from the parent module (legacy compatibility)
# This allows `from .utils import set_defaults` to work
from typing import Any, Dict


def set_defaults(
    options: Dict[str, Any] | None = None, *args: Any, **kwargs: Any
) -> Dict[str, Any]:
    """
    Populate default values without overriding user-specified entries.

    Parameters
    ----------
    options : dict, optional
        Existing option dictionary. A shallow copy is returned.
    *args :
        Optional alternating key/value pairs, mirroring the MATLAB usage
        `set_defaults(opt, 'field', value, ...)`.
    **kwargs :
        Additional defaults expressed as keyword arguments.

    Returns
    -------
    dict
        Updated dictionary containing both the original entries and any
        defaults that were previously missing.
    """

    result: Dict[str, Any] = dict(options) if options is not None else {}

    if args:
        if len(args) % 2 != 0:
            raise ValueError(
                "set_defaults expects an even number of positional arguments (key/value pairs)."
            )
        for key, value in zip(args[0::2], args[1::2]):
            if not isinstance(key, str):
                raise TypeError(
                    f"Default option names must be strings, got {type(key)!r}."
                )
            result.setdefault(key, value)

    for key, value in kwargs.items():
        result.setdefault(key, value)

    return result


__all__ = [
    "matlab_randn",
    "MatlabRNG",
    "load_matlab_random_matrix",
    "set_defaults",
    "is_numba_available",
    "_as_column_matrix",
    "_default_hrange",
]
