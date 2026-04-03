# SPDX-License-Identifier: GPL-3.0-or-later
"""
Grid utilities mirroring MATLAB helpers such as `get_ndgrid` and `nufftfreqs`.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .core import multispace


def _as_vector_list(
    vectors: Iterable[Iterable[float]] | np.ndarray,
) -> List[np.ndarray]:
    if isinstance(vectors, np.ndarray) and vectors.ndim == 2:
        return [vectors[:, dim] for dim in range(vectors.shape[1])]
    result = []
    for vec in vectors:
        arr = np.asarray(vec, dtype=float)
        result.append(arr)
    return result


def get_ndgrid(
    vectors: Iterable[Iterable[float]] | np.ndarray,
    output: str = "cell",
) -> List[np.ndarray] | np.ndarray:
    """
    Generate multi-dimensional grid representations.
    """

    vec_list = _as_vector_list(vectors)
    grids = np.meshgrid(*vec_list, indexing="ij")

    if output == "cell":
        return grids
    if output == "array":
        return np.stack(grids, axis=-1)
    if output == "list":
        return np.stack(grids, axis=-1).reshape(-1, len(vec_list))
    if output == "cellList":
        return [grid.reshape(-1, 1) for grid in grids]

    raise ValueError(f"Unknown output format: {output}")


def nufftfreqs(
    N: Sequence[int], df: Sequence[float] | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate centered frequency grid used by NUFFT routines.
    """

    N = np.asarray(N, dtype=int)
    dx = len(N)
    if df is None:
        df = np.ones(dx, dtype=float)
    else:
        df = np.asarray(df, dtype=float)
        if df.size != dx:
            raise ValueError("df must match the dimensionality of N.")

    start = np.floor(-N / 2).astype(int)
    stop = N - np.floor(N / 2).astype(int) - 1
    vectors = multispace(start, stop, N, np.linspace, "cell", True)

    for dim in range(dx):
        vectors[dim] = df[dim] * vectors[dim]

    grid = np.stack(np.meshgrid(*vectors, indexing="ij"), axis=-1).reshape(-1, dx)
    return grid, df
