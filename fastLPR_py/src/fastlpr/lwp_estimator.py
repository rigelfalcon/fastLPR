# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Local Weighted Polynomial (LWP) estimator for fastLPR.

Port from fastLPR/utility/core/get_lwp_estimator.m.

This module provides functions for computing design matrices and solving
local polynomial regression systems for orders 0, 1, and 2.

OPTIMIZED VERSION (2025-12):
- Pre-computed polynomial terms using broadcasting
- Cached column-major indices for mat2tril/vec2tril
- scipy.linalg.solve for efficient matrix solving
- Eliminated full diagonal matrix creation (W @ X)
- Vectorized tensor contractions
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, List, Tuple

import numpy as np
from scipy import linalg


# =============================================================================
# Optimized polynomial term computation using broadcasting
# =============================================================================

def get_polynomial_terms(x: np.ndarray, order: int) -> np.ndarray:
    """
    Generate polynomial design matrix for local polynomial regression.

    OPTIMIZED: Uses pre-allocated arrays and vectorized operations.

    Parameters
    ----------
    x : ndarray, shape (N, d)
        Normalized predictor variables (x - x0) / h
    order : int
        Polynomial order (0, 1, or 2)

    Returns
    -------
    X : ndarray, shape (N, n_terms)
        Design matrix with polynomial terms

    Notes
    -----
    For order 0: X = [1]
    For order 1: X = [1, x1, x2, ..., xd]
    For order 2: X = [1, x1, x2, ..., xd, x1^2, x1*x2, ..., xd^2]
    """
    x = np.atleast_2d(x)
    N, dims = x.shape

    # Check for unsupported high-dimensional, high-order cases
    if dims > 2 and order > 2:
        raise ValueError(
            f"Polynomial order {order} for {dims}D data is not supported. "
            f"High-dimensional (dims > 2) high-order (order > 2) regression requires "
            f"large computational power for symbolic formula generation and "
            f"design matrix inversion. Please use order <= 2 for dims > 2."
        )

    if order == 0:
        # Order 0: Nadaraya-Watson (local constant)
        return np.ones((N, 1), dtype=x.dtype)

    elif order == 1:
        # Order 1: Local linear - [1, x1, x2, ..., xd]
        # Pre-allocate and fill - faster than np.column_stack
        n_terms = 1 + dims
        X = np.empty((N, n_terms), dtype=x.dtype)
        X[:, 0] = 1.0
        X[:, 1:] = x
        return X

    elif order == 2:
        # Order 2: Local quadratic
        # Include all quadratic terms (lower triangular of x*x')
        # Pre-allocate with exact size
        n_quad = dims * (dims + 1) // 2
        n_terms = 1 + dims + n_quad
        X = np.empty((N, n_terms), dtype=x.dtype)

        # Fill in order: constant, linear, quadratic
        X[:, 0] = 1.0
        X[:, 1:dims+1] = x

        # PERFORMANCE NOTE: For small dimensions (dims <= 5), explicit loops are faster
        # than np.triu_indices due to index creation overhead. Since fastLPR rarely
        # uses dims > 3, we use explicit loops which are ~1.5x faster in practice.
        # For dims > 5, vectorized approach would be beneficial but is rare.
        idx = dims + 1
        for i in range(dims):
            for j in range(i, dims):
                X[:, idx] = x[:, i] * x[:, j]
                idx += 1

        return X

    else:
        raise ValueError(f"Order {order} not supported. Use 0, 1, or 2.")


def get_polynomial_terms_1d_order1(x: np.ndarray) -> np.ndarray:
    """Optimized polynomial terms for 1D order 1: [1, x]"""
    N = len(x)
    X = np.empty((N, 2), dtype=x.dtype)
    X[:, 0] = 1.0
    X[:, 1] = x.ravel()
    return X


def get_polynomial_terms_1d_order2(x: np.ndarray) -> np.ndarray:
    """Optimized polynomial terms for 1D order 2: [1, x, x^2]"""
    N = len(x)
    x_flat = x.ravel()
    X = np.empty((N, 3), dtype=x.dtype)
    X[:, 0] = 1.0
    X[:, 1] = x_flat
    X[:, 2] = x_flat * x_flat
    return X


def get_polynomial_terms_2d_order1(x: np.ndarray) -> np.ndarray:
    """Optimized polynomial terms for 2D order 1: [1, x1, x2]"""
    N = x.shape[0]
    X = np.empty((N, 3), dtype=x.dtype)
    X[:, 0] = 1.0
    X[:, 1] = x[:, 0]
    X[:, 2] = x[:, 1]
    return X


def get_polynomial_terms_2d_order2(x: np.ndarray) -> np.ndarray:
    """Optimized polynomial terms for 2D order 2: [1, x1, x2, x1^2, x1*x2, x2^2]"""
    N = x.shape[0]
    x1, x2 = x[:, 0], x[:, 1]
    X = np.empty((N, 6), dtype=x.dtype)
    X[:, 0] = 1.0
    X[:, 1] = x1
    X[:, 2] = x2
    X[:, 3] = x1 * x1
    X[:, 4] = x1 * x2
    X[:, 5] = x2 * x2
    return X


def get_polynomial_terms_3d_order1(x: np.ndarray) -> np.ndarray:
    """Optimized polynomial terms for 3D order 1: [1, x1, x2, x3]"""
    N = x.shape[0]
    X = np.empty((N, 4), dtype=x.dtype)
    X[:, 0] = 1.0
    X[:, 1] = x[:, 0]
    X[:, 2] = x[:, 1]
    X[:, 3] = x[:, 2]
    return X


# =============================================================================
# Cached column-major index computation for mat2tril/vec2tril
# =============================================================================

@lru_cache(maxsize=16)
def _get_tril_indices_column_major(n: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Get lower triangular indices in COLUMN-MAJOR order (cached).

    This matches MATLAB's find(tril(ones(n))) ordering.
    """
    row_indices = []
    col_indices = []
    for j in range(n):  # Column-major: iterate columns first
        for i in range(j, n):  # Only lower triangle (i >= j)
            row_indices.append(i)
            col_indices.append(j)
    return tuple(row_indices), tuple(col_indices)


def mat2tril(A: np.ndarray) -> np.ndarray:
    """
    Extract lower triangular elements of a matrix in COLUMN-MAJOR order.

    OPTIMIZED: Uses cached index computation.

    This function matches MATLAB's mat2tril behavior, which extracts elements
    in column-major (Fortran) order. This is critical for correct indexing
    of the design matrix elements in LWP formulas.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Symmetric matrix

    Returns
    -------
    v : ndarray, shape (n*(n+1)/2,)
        Lower triangular elements in column-major order
    """
    n = A.shape[0]
    row_idx, col_idx = _get_tril_indices_column_major(n)
    return A[row_idx, col_idx]


def vec2tril(v: np.ndarray) -> np.ndarray:
    """
    Reconstruct symmetric matrix from lower triangular elements in COLUMN-MAJOR order.

    OPTIMIZED: Uses cached index computation and efficient symmetric fill.

    Parameters
    ----------
    v : ndarray, shape (n*(n+1)/2,)
        Lower triangular elements in column-major order

    Returns
    -------
    A : ndarray, shape (n, n)
        Symmetric matrix
    """
    # Determine matrix size from vector length
    n = int((-1 + np.sqrt(1 + 8 * len(v))) / 2)

    # MEMORY OPTIMIZATION: Use np.empty instead of np.zeros since we'll fill all values
    A = np.empty((n, n), dtype=v.dtype)
    row_idx, col_idx = _get_tril_indices_column_major(n)

    # Fill lower triangle
    A[row_idx, col_idx] = v

    # Fill upper triangle by transposing indices (symmetric matrix)
    A[col_idx, row_idx] = v

    return A


# =============================================================================
# Optimized design matrix computation (avoid full diagonal matrix)
# =============================================================================

def compute_design_matrix_elements(
    x: np.ndarray,
    weights: np.ndarray,
    order: int,
) -> List[np.ndarray]:
    """
    Compute design matrix S = X' * W * X for local polynomial regression.

    OPTIMIZED: Avoids creating full N x N diagonal matrix W.
    Uses weighted X directly: sqrt(W) * X, then X'WX = (sqrt(W)X)' @ (sqrt(W)X)

    Parameters
    ----------
    x : ndarray, shape (N, d)
        Normalized predictor variables
    weights : ndarray, shape (N,)
        Kernel weights
    order : int
        Polynomial order

    Returns
    -------
    s_elements : list of ndarrays
        Lower triangular elements of design matrix
    """
    X = get_polynomial_terms(x, order)
    N, n_terms = X.shape

    # OPTIMIZATION: Avoid np.diag(weights) @ X which creates N x N matrix
    # Instead, use broadcasting: (weights[:, None] * X) applies weights row-wise
    # S = X' * W * X = X' * (weights * X) = (X.T) @ (weights[:, None] * X)
    WX = weights[:, np.newaxis] * X  # Shape: (N, n_terms)
    S = X.T @ WX  # Shape: (n_terms, n_terms)

    if order == 0:
        # Order 0: S is scalar
        return [S[0, 0]]
    else:
        # Order >= 1: Extract lower triangular elements
        s_vec = mat2tril(S)

        # Split into list for compatibility with MATLAB cell array format
        return list(s_vec)


def compute_weighted_response(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    order: int,
) -> List[np.ndarray]:
    """
    Compute weighted response T = X' * W * y for local polynomial regression.

    OPTIMIZED: Avoids creating full N x N diagonal matrix W.

    Parameters
    ----------
    x : ndarray, shape (N, d)
        Normalized predictor variables
    y : ndarray, shape (N,)
        Response values
    weights : ndarray, shape (N,)
        Kernel weights
    order : int
        Polynomial order

    Returns
    -------
    t_elements : list of ndarrays
        Elements of weighted response vector
    """
    X = get_polynomial_terms(x, order)
    N, n_terms = X.shape

    # OPTIMIZATION: T = X' * W * y = X' * (weights * y) = X.T @ (weights * y)
    Wy = weights * y  # Element-wise multiplication
    T = X.T @ Wy  # Shape: (n_terms,)

    # Split into list for compatibility with MATLAB cell array format
    return list(T)


# =============================================================================
# Optimized system solver
# =============================================================================

def solve_lwp_system(
    s_elements: List[np.ndarray],
    t_elements: List[np.ndarray],
    order: int,
    regularization: float = 1e-6,
) -> np.ndarray:
    """
    Solve local polynomial regression system: S * beta = T.

    OPTIMIZED: Uses scipy.linalg.solve with assume_a='pos' for SPD matrices.

    Parameters
    ----------
    s_elements : list of ndarrays
        Lower triangular elements of design matrix
    t_elements : list of ndarrays
        Elements of weighted response
    order : int
        Polynomial order
    regularization : float, default=1e-6
        Regularization parameter

    Returns
    -------
    m : ndarray
        Regression estimate (beta[0], the constant term)
    """
    if order == 0:
        # Order 0: Nadaraya-Watson
        s = s_elements[0]
        t = t_elements[0]

        # Adaptive threshold
        s_threshold = np.maximum(s, np.abs(s) * regularization)
        return t / s_threshold

    else:
        # Order >= 1: Local polynomial
        # Reconstruct symmetric matrix S from lower triangular elements
        # Detect if input is complex and preserve dtype
        is_complex = any(np.iscomplexobj(s) for s in s_elements) or any(np.iscomplexobj(t) for t in t_elements)
        dtype = np.complex128 if is_complex else np.float64
        s_vec = np.asarray(s_elements, dtype=dtype)
        S = vec2tril(s_vec)

        # Weighted response vector
        T = np.asarray(t_elements, dtype=dtype)

        # Apply adaptive regularization
        max_diag = np.max(np.abs(np.diag(S)))
        lambda_reg = regularization * max_diag
        S_reg = S + lambda_reg * np.eye(len(S), dtype=S.dtype)

        # OPTIMIZATION: Use explicit Cholesky + triangular solves
        # This is marginally faster than scipy.linalg.solve with assume_a='pos'
        # because it avoids internal checks and uses optimized triangular solvers
        from .solvers import solve_spd
        try:
            beta = solve_spd(S_reg, T)
        except linalg.LinAlgError:
            # Fall back to pseudo-inverse if still singular
            beta = np.linalg.lstsq(S_reg, T, rcond=None)[0]

        # Return constant term (beta[0])
        return beta[0]


# =============================================================================
# Factory function for LWP functions
# =============================================================================

def create_lwp_functions(
    dims: int,
    order: int,
) -> Tuple[Callable, Callable, Callable, int, int]:
    """
    Create functions for local polynomial regression.

    Port from fastLPR/utility/core/get_lwp_estimator.m.

    Parameters
    ----------
    dims : int
        Number of predictor dimensions
    order : int
        Polynomial order (0, 1, or 2)

    Returns
    -------
    Sxfun : callable
        Function to compute design matrix elements
    Txfun : callable
        Function to compute weighted response elements
    mfun : callable
        Function to solve regression system (uses precomputed symbolic formula if available)
    ns : int
        Number of unique elements in lower triangular S
    nt : int
        Number of polynomial terms

    Notes
    -----
    This function tries to use precomputed symbolic formulas (like MATLAB's
    estimated_lwp_estimator.m). If no precomputed formula is available,
    it falls back to numerical computation.
    """
    from .lwp_formulas import get_lwp_formula

    # Determine number of terms
    if order == 0:
        nt = 1
        ns = 1
    elif order == 1:
        nt = 1 + dims
        ns = nt * (nt + 1) // 2
    elif order == 2:
        nt = 1 + dims + dims * (dims + 1) // 2
        ns = nt * (nt + 1) // 2
    else:
        raise ValueError(f"Order {order} not supported")

    # Create lambda functions for design matrix and weighted response
    Sxfun = lambda weights, x: compute_design_matrix_elements(x, weights, order)
    Txfun = lambda weights, x, y: compute_weighted_response(x, y, weights, order)

    # Try to get precomputed symbolic formula
    symbolic_mfun = get_lwp_formula(dims, order, is_complex=False)

    if symbolic_mfun is not None:
        # Use precomputed symbolic formula (VECTORIZED, like MATLAB)
        mfun = symbolic_mfun
    else:
        # Fall back to numerical solver
        mfun = lambda s, t, reg=1e-6: solve_lwp_system(s, t, order, reg)

    return Sxfun, Txfun, mfun, ns, nt
