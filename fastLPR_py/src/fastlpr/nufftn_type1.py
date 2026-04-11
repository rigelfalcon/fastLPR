# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
nufftn_type1 - Non-Uniform FFT Type 1

Line-by-line translation of MATLAB's nufftn_type1.m to ensure exact correspondence.

Memory-efficient implementation with adaptive chunked processing:
- Small datasets: Direct processing (no overhead)
- Large datasets: Automatic chunking based on available system memory

Acceleration backends (in order of preference):
1. Numba JIT compilation (default, better memory efficiency)
2. Cython with OpenMP (optional, better for N > 500K)
3. Pure Python/NumPy fallback

Environment variables:
- FASTLPR_BACKEND=cython|numba|auto  (default: auto, prefers Numba)

Author: Ying Wang, Min Li
"""

import numpy as np
from typing import Tuple, Optional
import gc
from . import fft_backend
from .type_utils import get_optimal_index_dtype, get_complex_dtype, get_float_dtype

# =============================================================================
# Backend Detection and Selection (from centralized module)
# =============================================================================
from .backend_selection import (
    HAS_NUMBA,
    HAS_CYTHON,
    HAS_FUSED_SPREADING,
    get_backend_preference,
    select_accumulation_backend,
    get_backend_info,
)

# Import Cython accumulation if available
if HAS_CYTHON:
    from ._nufft_accel import nufft_accumulate_cython
else:
    nufft_accumulate_cython = None

# Import Numba module for JIT compilation
if HAS_NUMBA:
    import numba

# Import fused spreading kernels if available
if HAS_FUSED_SPREADING:
    from .spreading_fused import fused_spreading_dispatch
else:
    fused_spreading_dispatch = None

# Import Numba-optimized scale_knots if available
if HAS_NUMBA:
    from .nufft_numba import _scale_knots_numba
else:
    _scale_knots_numba = None


def _use_cython():
    """Determine if Cython backend should be used based on preference."""
    return select_accumulation_backend() == "cython"


# Numba-optimized accumulation function for ANY dimension
if HAS_NUMBA:

    @numba.jit(nopython=True, cache=True, fastmath=False)
    def _accumulate_numba(Ftau_flat, xindx, ysp, dy, strides):
        """
        Accumulate spreading values into grid using Numba.

        This works for ANY dimension (1D, 2D, 3D, 4D, ...) by using flattened array
        and linear indexing with unrolled loops for common cases (1D/2D/3D).

        IMPORTANT: This function CANNOT use parallel=True because multiple points
        can spread to the same grid cell, causing race conditions. The sequential
        loop ensures correct accumulation.

        Args:
            Ftau_flat: Flattened output array (prod(Mr), dy)
            xindx: Multi-dimensional indices (n_points, dx)
            ysp: Spreading values (n_points, dy)
            dy: Number of output dimensions
            strides: Strides for converting multi-dim to linear index (dx,)

        This replaces dimension-specific functions and matches MATLAB's accumarray approach.
        Performance: Comparable or faster than dimension-specific implementations.
        """
        n_points = xindx.shape[0]
        dx = xindx.shape[1]

        # Unroll for common dimensions (1D, 2D, 3D) for best performance
        # NOTE: Cannot parallelize over idx because multiple points may write to same grid cell
        if dx == 1:
            for idx in range(n_points):
                linear_idx = xindx[idx, 0]
                for iy in range(dy):
                    Ftau_flat[linear_idx, iy] += ysp[idx, iy]
        elif dx == 2:
            for idx in range(n_points):
                linear_idx = xindx[idx, 0] * strides[0] + xindx[idx, 1]
                for iy in range(dy):
                    Ftau_flat[linear_idx, iy] += ysp[idx, iy]
        elif dx == 3:
            for idx in range(n_points):
                linear_idx = (
                    xindx[idx, 0] * strides[0]
                    + xindx[idx, 1] * strides[1]
                    + xindx[idx, 2]
                )
                for iy in range(dy):
                    Ftau_flat[linear_idx, iy] += ysp[idx, iy]
        else:
            # General case for dx > 3 (rare but supported)
            for idx in range(n_points):
                linear_idx = 0
                for d in range(dx):
                    linear_idx += xindx[idx, d] * strides[d]
                for iy in range(dy):
                    Ftau_flat[linear_idx, iy] += ysp[idx, iy]

else:
    # Fallback to pure Python if Numba is not available
    def _accumulate_numba(Ftau_flat, xindx, ysp, dy, strides):
        n_points = xindx.shape[0]
        dx = xindx.shape[1]
        for iy in range(dy):
            for idx in range(n_points):
                linear_idx = 0
                for d in range(dx):
                    linear_idx += xindx[idx, d] * strides[d]
                Ftau_flat[linear_idx, iy] += ysp[idx, iy]


def scale_knots(knots: np.ndarray, N: np.ndarray, Fs: np.ndarray) -> np.ndarray:
    """
    Scale and shift knot locations for NUFFT.

    Mirrors MATLAB's scale_knots.m exactly.

    Args:
        knots: Original knot locations (M x dx)
        N: Grid size (dx,)
        Fs: Sampling frequency (dx,)

    Returns:
        Scaled knot locations in [0, 2π]

    MEMORY OPTIMIZATION: Uses in-place operations to minimize temporary allocations.
    Reduces memory usage by ~40% compared to original implementation.
    """
    # Make a working copy to allow in-place operations
    knots = knots.copy()

    # Scale by frequency and grid size
    # MEMORY OPTIMIZATION: In-place multiply instead of knots = knots * Fs / N
    scale_factor = Fs / N  # Small temporary (dx,)
    np.multiply(knots, scale_factor, out=knots)

    # Center around 0
    # MEMORY OPTIMIZATION: Compute shift in single expression
    kmin = (knots.min(axis=0) + knots.max(axis=0)) * 0.5 - 0.5
    shift = -0.5 - kmin
    np.add(knots, shift, out=knots)  # Now in [-1/2, 1/2]

    # Convert to [0, 2π] convention (Greengard)
    # MEMORY OPTIMIZATION: In-place operations
    np.multiply(knots, 2 * np.pi, out=knots)
    np.mod(knots, 2 * np.pi, out=knots)

    return knots


# Numba-optimized heat kernel for better performance
if HAS_NUMBA:

    @numba.jit(nopython=True, parallel=True, cache=True, fastmath=False)
    def _heat_kernel_numba(x: np.ndarray, tau_scalar: float) -> np.ndarray:
        """
        Numba-optimized heat kernel computation.

        This is 2-3x faster than the NumPy version for typical NUFFT sizes.

        Args:
            x: Distance from grid points, shape (M, N, dx)
            tau_scalar: Spreading parameter (scalar)

        Returns:
            Heat kernel values, shape (M, N)
        """
        M, N, dx = x.shape
        result = np.zeros((M, N), dtype=np.float64)

        # Parallel loop over M (number of input points)
        for i in numba.prange(M):
            for j in range(N):
                # Compute sum of squared distances
                dist_sq = 0.0
                for d in range(dx):
                    dist_sq += x[i, j, d] ** 2

                # Gaussian kernel: exp(-dist_sq / (4*tau))
                result[i, j] = np.exp(-dist_sq / (4.0 * tau_scalar))

        return result


def heat_kernel(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Compute heat kernel (Gaussian spreading function).

    Mirrors MATLAB's heat_kernel function.
    Uses fully vectorized NumPy operations for best performance.

    Args:
        x: Distance from grid points
        h: Spreading parameter tau

    Returns:
        Heat kernel values
    """
    # OPTIMIZATION: Always use vectorized NumPy version
    # This mirrors MATLAB's vectorized implementation exactly:
    # x=exp( -sum((x.^2)./(4*h),3))
    #
    # Benchmarking shows vectorized NumPy is faster than Numba with nested loops
    # for typical NUFFT sizes (M~10k, N~100, dx~2-3)

    # Line 174: h=permute(h(:),[3:-1:1])
    # This reverses the dimensions for broadcasting
    h = h.reshape(1, 1, -1)

    # Line 175: x=exp( -sum((x.^2)./(4*h),3))
    # MEMORY OPTIMIZATION: Compute in-place to reduce allocations
    # Instead of: result = np.exp(-np.sum((x**2) / (4 * h), axis=2))
    # Use: compute x**2 / (4*h), sum, negate, exp - all with minimal temporaries
    x_sq = np.square(x)  # In-place would need copy first, so use square
    h_safe = np.maximum(4 * h, np.finfo(x.dtype).tiny)  # Guard against h=0
    np.divide(x_sq, h_safe, out=x_sq)  # In-place divide
    result = np.sum(x_sq, axis=2)  # Sum along last axis
    np.negative(result, out=result)  # In-place negate
    np.exp(result, out=result)  # In-place exp

    return result


def _compute_chunk_size(M: int, dx: int, Msp2_prod: int, dy: int = 1) -> int:
    """
    Compute optimal chunk size based on available system memory.

    Uses memory_utils module for memory estimation. Falls back to
    conservative defaults if psutil is not available.

    Args:
        M: Total number of data points
        dx: Number of input dimensions
        Msp2_prod: Product of (2*Msp+1) across all dimensions
        dy: Number of output dimensions

    Returns:
        int: Optimal chunk size (M if no chunking needed)
    """
    try:
        from .memory_utils import compute_optimal_chunk_size

        return compute_optimal_chunk_size(M, dx, Msp2_prod, dy)
    except ImportError:
        # Fallback: use 100K points per chunk as safe default
        return min(100000, M)


def _process_spreading_chunk(
    x_chunk: np.ndarray,
    y_chunk: np.ndarray,
    xmod_chunk: np.ndarray,
    Ftau_flat: np.ndarray,
    Msp: np.ndarray,
    Msp2: np.ndarray,
    Mr_orig: np.ndarray,
    hx: np.ndarray,
    tau: np.ndarray,
    strides: np.ndarray,
    dx: int,
    dy: int,
    index_dtype: np.dtype = np.int64,
) -> None:
    """
    Process a single chunk of data for NUFFT spreading.

    This function handles the memory-intensive spreading operation for a
    subset of data points, accumulating results into the shared Ftau_flat array.

    Args:
        x_chunk: Chunk of input locations (M_chunk x dx)
        y_chunk: Chunk of input values (M_chunk x dy)
        xmod_chunk: Scaled knot locations for chunk (M_chunk x dx)
        Ftau_flat: Flattened output grid to accumulate into
        Msp, Msp2, Mr_orig, hx, tau: Grid parameters
        strides: Strides for linear indexing
        dx, dy: Dimensions
        index_dtype: Optimal index dtype (uint8/uint16/uint32/int64)
    """
    M_chunk = x_chunk.shape[0]
    Msp2_prod = int(np.prod(Msp2))

    # Line 61: m = round(xmod ./ hx)
    m = np.round(xmod_chunk / hx.flatten()).astype(int)

    # Permute for broadcasting
    m = m[:, None, :]  # (M_chunk, 1, dx)
    hx_bc = hx.reshape(1, 1, -1)  # (1, 1, dx)
    xmod_bc = xmod_chunk[:, None, :]  # (M_chunk, 1, dx)

    # Build spreading neighborhood
    mpmm = m.copy()
    for i in range(dx):
        m_temp = mpmm.copy()
        mm = np.zeros((1, Msp2[i], dx))
        mm[0, :, i] = np.round(np.linspace(-Msp[i], Msp[i], Msp2[i]))
        mpmm = (m_temp + mm).reshape(-1, 1, dx)

    mpmm = mpmm.reshape(M_chunk, Msp2_prod, dx)

    # Compute heat kernel weights
    weight = heat_kernel(xmod_bc - hx_bc * mpmm, tau)

    # Cleanup intermediate arrays
    del xmod_bc, hx_bc, m
    gc.collect()

    # Compute weighted values
    y_bc = y_chunk[:, None, :]  # (M_chunk, 1, dy)
    ysp = y_bc * weight[:, :, None]  # (M_chunk, Msp2_prod, dy)
    ysp = ysp.reshape(-1, dy)

    del weight, y_bc
    gc.collect()

    # Compute grid indices with optimal dtype
    xindx = np.mod(mpmm, Mr_orig).astype(index_dtype)
    xindx = xindx.reshape(-1, dx)

    del mpmm
    gc.collect()

    # Accumulate into Ftau_flat using centralized backend selection
    sz = tuple(Mr_orig.flatten().astype(int))
    backend = select_accumulation_backend()

    if backend == "cython":
        # Cython requires exact complex128/int64 dtypes
        xindx_c = xindx.astype(np.int64, copy=False)
        ysp_c = ysp.astype(np.complex128, copy=False)
        strides_c = strides.astype(np.int64, copy=False)
        if Ftau_flat.dtype == np.complex128:
            nufft_accumulate_cython(Ftau_flat, xindx_c, ysp_c, strides_c)
        else:
            # Low-precision mode: accumulate in complex128 temp, then write back
            Ftau_tmp = Ftau_flat.astype(np.complex128)
            nufft_accumulate_cython(Ftau_tmp, xindx_c, ysp_c, strides_c)
            Ftau_flat[:] = Ftau_tmp.astype(Ftau_flat.dtype)
    elif backend == "numba":
        _accumulate_numba(Ftau_flat, xindx, ysp, dy, strides)
    else:  # python fallback
        for iy in range(dy):
            linear_idx = np.ravel_multi_index(tuple(xindx[:, i] for i in range(dx)), sz)
            np.add.at(Ftau_flat[:, iy], linear_idx, ysp[:, iy])

    del xindx, ysp
    gc.collect()


def nufftn_type1(
    x: np.ndarray,
    y: np.ndarray,
    Fs: Optional[np.ndarray] = None,
    df: Optional[np.ndarray] = None,
    iflag: int = -1,
    acc: int = 6,
    isdeconv: bool = True,
) -> np.ndarray:
    """
    Non-Uniform FFT Type 1 - mirrors MATLAB's nufftn_type1.m

    This function is a line-by-line translation of the MATLAB code.
    Comments reference the original MATLAB line numbers.

    Memory-efficient: Uses adaptive chunked processing for large datasets.
    Small datasets are processed directly without overhead.

    Args:
        x: Non-equispaced locations (knots) where y lies on (M x dx)
        y: Non-equispaced function values (M x dy)
        Fs: Sampling frequency (dx,) - corresponds to grid size in frequency domain
        df: Frequency spacing (dx,)
        iflag: Sign for exponential (-1 for forward, +1 for inverse)
        acc: Accuracy parameter (default 6)
        isdeconv: Whether to apply deconvolution (default True)

    Returns:
        Yq: Transformed values on uniform grid (N x dy)
    """
    # Lines 15-16: Get dimensions
    M, dx = x.shape
    if y.ndim == 1:
        y = y[:, None]
    _, dy = y.shape

    # Lines 19-27: Handle df, N, and Fs
    # Priority: Fs > df > auto-compute from M
    if Fs is not None:
        # Fs is provided - use it to determine N
        Fs = np.atleast_1d(Fs).ravel().astype(int)
        N = Fs.copy()
        if df is None:
            df = 1.0 / N
    elif df is not None:
        # df is provided - compute N from df
        df = np.atleast_1d(df).ravel()
        N = np.round(1.0 / df).astype(int)
        Fs = N.copy()
    else:
        # Neither Fs nor df provided - auto-compute from M
        N = np.ceil(M ** (1 / dx)) * np.ones(dx, dtype=int)
        df = 1.0 / N
        Fs = N.copy()

    # Line 46: Compute grid parameters
    from .kernel import compute_grid_params

    # Ensure N is int array
    N = N.astype(int)

    Msp, Mr, tau, R = compute_grid_params(N, acc)

    # Binning mode: accuracy=0 means no spreading, direct accumarray
    if acc == 0:
        # Ensure Mr is array for consistent handling
        Mr_arr = np.atleast_1d(Mr)
        hx = 2 * np.pi / Mr_arr
        # Use Numba-optimized scale_knots if available (6x faster for large N)
        if HAS_NUMBA and _scale_knots_numba is not None:
            xmod = _scale_knots_numba(
                np.ascontiguousarray(x, dtype=np.float64),
                N.astype(np.float64),
                Fs.astype(np.float64),
            )
        else:
            xmod = scale_knots(x, N, Fs)
        m_orig = np.round(xmod / hx).astype(int)

        # Compute grid indices (mod for periodic boundary)
        Mr_vec = Mr_arr.flatten().astype(int)
        xindx = np.mod(m_orig, Mr_vec)

        # Direct binning - use np.bincount (23x faster than np.add.at)
        sz = tuple(Mr_vec)
        Mr_total = int(np.prod(Mr_vec))
        
        # Compute linear indices
        if dx == 1:
            linear_idx = xindx[:, 0]
        else:
            strides = np.zeros(dx, dtype=np.int64)
            strides[-1] = 1
            for d in range(dx - 2, -1, -1):
                strides[d] = strides[d + 1] * Mr_vec[d + 1]
            linear_idx = np.sum(xindx * strides, axis=1)
        
        # Use np.bincount for binning (handles real/complex separately)
        Ftau = np.zeros((Mr_total, dy), dtype=np.complex128)
        y_is_complex = np.iscomplexobj(y)
        
        for iy in range(dy):
            y_col = y[:, iy]
            if y_is_complex:
                # Complex: bin real and imaginary parts separately
                real_part = np.bincount(linear_idx, weights=y_col.real, minlength=Mr_total)
                imag_part = np.bincount(linear_idx, weights=y_col.imag, minlength=Mr_total)
                Ftau[:, iy] = real_part + 1j * imag_part
            else:
                # Real: direct bincount
                Ftau[:, iy] = np.bincount(linear_idx, weights=y_col.real, minlength=Mr_total)
        
        Ftau = Ftau.reshape(sz + (dy,))

        # FFT (same as spreading mode)
        fft_func = fft_backend.fft
        fftshift_func = fft_backend.fftshift
        ifft_func = fft_backend.ifft
        ifftshift_func = fft_backend.ifftshift

        if iflag < 0:
            for ix in range(dx - 1, -1, -1):
                Ftau = fft_func(Ftau, axis=ix, overwrite_x=True)
            Ftau = fftshift_func(Ftau, axes=tuple(range(dx)))
            Ftau = Ftau / (M * (R**dx))
        else:
            Ftau = ifftshift_func(Ftau, axes=tuple(range(dx)))
            for ix in range(dx - 1, -1, -1):
                Ftau = ifft_func(Ftau, axis=ix, overwrite_x=True)

        # Extract output region (no deconvolution needed for binning)
        N_vec = N if isinstance(N, np.ndarray) else np.array([N])
        q = (Mr_vec - N_vec) / 2
        q_start = np.ceil(q).astype(int)
        q_end = (q_start + N_vec).astype(int)
        slices = tuple(slice(q_start[i], q_end[i]) for i in range(dx)) + (slice(None),)
        Yq = Ftau[slices]
        Yq = Yq.reshape(tuple(N_vec.astype(int)) + (dy,))
        return Yq

    # Ensure arrays
    if not isinstance(Msp, np.ndarray):
        Msp = np.array([Msp])
    if not isinstance(Mr, np.ndarray):
        Mr = np.array([Mr])
    if not isinstance(tau, np.ndarray):
        tau = np.array([tau])

    # Save original Mr for later use
    Mr_orig = Mr.copy()

    # Line 58: hx = 2*pi./Mr
    hx = 2 * np.pi / Mr

    # Line 59: xmod = scale_knots(x,N,Fs)
    xmod = scale_knots(x, N, Fs)

    # Line 62: Msp2=Msp*2+1
    Msp2 = Msp * 2 + 1
    Msp2_prod = int(np.prod(Msp2))

    # Compute strides for linear indexing (C-style, row-major)
    strides = np.zeros(dx, dtype=np.int64)
    strides[-1] = 1
    for d in range(dx - 2, -1, -1):
        strides[d] = strides[d + 1] * Mr_orig[d + 1]

    sz = tuple(Mr_orig.flatten().astype(int))

    # Determine optimal dtypes based on accuracy (matching MATLAB's bit_convert)
    # acc < 5: use single precision (float32/complex64) for 4x memory savings
    # acc >= 5: use double precision (float64/complex128) for full accuracy
    complex_dtype = get_complex_dtype(acc)
    float_dtype = get_float_dtype(acc)
    # Optimal index dtype based on max grid size (matching MATLAB's min_numerical_type)
    # For Mr=200: uint8 (1 byte), Mr=1000: uint16 (2 bytes), Mr=100000: uint32 (4 bytes)
    index_dtype = get_optimal_index_dtype(int(np.max(Mr_orig)))

    # Initialize output grid with optimal dtype
    Ftau_flat = np.zeros((np.prod(Mr_orig), dy), dtype=complex_dtype)

    # Determine chunk size based on available memory
    chunk_size = _compute_chunk_size(M, dx, Msp2_prod, dy)

    if chunk_size >= M:
        # Small dataset: process all at once
        # Use fused spreading if available (20-40x faster for d<=3)
        if HAS_FUSED_SPREADING and dx <= 3:
            # Fused spreading: compute neighborhood + weights + accumulate in one pass
            # Avoids massive intermediate arrays (e.g., 270MB for d=2, N=100k)
            fused_spreading_dispatch(
                dx,
                xmod,
                y,
                Ftau_flat,
                Msp.astype(np.int64),
                Mr_orig.astype(np.int64),
                hx.astype(np.float64),
                tau.astype(np.float64),
                strides.astype(np.int64),
            )
        else:
            # Fallback: vectorized approach (original algorithm)
            # Line 61: m = round(xmod ./ hx)
            m = np.round(xmod / hx).astype(int)

            # Lines 64-67: Permute for broadcasting
            m = m[:, None, :]
            Mr_bc = Mr.reshape(1, 1, -1)
            hx_bc = hx.reshape(1, 1, -1)
            xmod_bc = xmod[:, None, :]

            # Lines 68-75: Build spreading neighborhood
            mpmm = m.copy()
            for i in range(dx):
                m_temp = mpmm.copy()
                mm = np.zeros((1, Msp2[i], dx))
                mm[0, :, i] = np.round(np.linspace(-Msp[i], Msp[i], Msp2[i]))
                mpmm = (m_temp + mm).reshape(-1, 1, dx)

            mpmm = mpmm.reshape(M, Msp2_prod, dx)

            # Line 80: weight = heat_kernel(xmod-hx.* mpmm,tau)
            weight = heat_kernel(xmod_bc - hx_bc * mpmm, tau)

            # Lines 81-82: y=permute(y,[1,3,2]); ysp = y .* weight
            y_bc = y[:, None, :]  # (M, 1, dy)
            ysp = y_bc * weight[:, :, None]  # (M, prod(Msp2), dy)

            # Line 84: ysp=reshape(ysp,[],dy)
            ysp = ysp.reshape(-1, dy)

            # Lines 90-92: xindx = mod(mpmm, Mr)+1
            # Use optimal index dtype (uint8/uint16/uint32) based on max(Mr)
            # Memory savings: 4-8x compared to int64 for typical grid sizes
            xindx = np.mod(mpmm, Mr_orig).astype(index_dtype)
            xindx = xindx.reshape(-1, dx)

            # Accumulate using centralized backend selection
            backend = select_accumulation_backend()
            if backend == "cython":
                # Cython requires exact complex128/int64 dtypes
                xindx_c = xindx.astype(np.int64, copy=False)
                ysp_c = ysp.astype(np.complex128, copy=False)
                strides_c = strides.astype(np.int64, copy=False)
                if Ftau_flat.dtype == np.complex128:
                    nufft_accumulate_cython(Ftau_flat, xindx_c, ysp_c, strides_c)
                else:
                    Ftau_tmp = np.zeros_like(Ftau_flat, dtype=np.complex128)
                    nufft_accumulate_cython(Ftau_tmp, xindx_c, ysp_c, strides_c)
                    Ftau_flat += Ftau_tmp.astype(Ftau_flat.dtype)
            elif backend == "numba":
                _accumulate_numba(Ftau_flat, xindx, ysp, dy, strides)
            else:  # python fallback
                for iy in range(dy):
                    linear_idx = np.ravel_multi_index(
                        tuple(xindx[:, i] for i in range(dx)), sz
                    )
                    np.add.at(Ftau_flat[:, iy], linear_idx, ysp[:, iy])

    else:
        # Large dataset: process in chunks to limit memory usage
        n_chunks = (M + chunk_size - 1) // chunk_size

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, M)

            # Extract chunk data
            x_chunk = x[start:end]
            y_chunk = y[start:end]
            xmod_chunk = xmod[start:end]

            # Use fused spreading if available (20-40x faster for d<=3)
            if HAS_FUSED_SPREADING and dx <= 3:
                fused_spreading_dispatch(
                    dx,
                    xmod_chunk,
                    y_chunk,
                    Ftau_flat,
                    Msp.astype(np.int64),
                    Mr_orig.astype(np.int64),
                    hx.astype(np.float64),
                    tau.astype(np.float64),
                    strides.astype(np.int64),
                )
            else:
                # Fallback: vectorized chunked processing
                _process_spreading_chunk(
                    x_chunk,
                    y_chunk,
                    xmod_chunk,
                    Ftau_flat,
                    Msp,
                    Msp2,
                    Mr_orig,
                    hx,
                    tau,
                    strides,
                    dx,
                    dy,
                    index_dtype,
                )

    # Reshape to grid
    Ftau = Ftau_flat.reshape(sz + (dy,))

    # Lines 136-148: FFT
    # Use centralized fft_backend for consistent backend selection
    # (pyfftw > scipy > numpy priority)
    fft_func = fft_backend.fft
    ifft_func = fft_backend.ifft
    fftshift_func = fft_backend.fftshift
    ifftshift_func = fft_backend.ifftshift

    if iflag < 0:
        # Lines 137-140: Forward FFT
        # fftshift is REQUIRED for deconvolution step (see below)
        # After fftshift, DC component is at position M/2+1 (centered)

        # MEMORY OPTIMIZATION: Perform all FFTs first, then single consolidated fftshift
        # This reduces dx fftshift calls to 1, saving (dx-1) large array allocations
        # Note: fftshift(fft(x), axis=i) for all i is equivalent to fftshift(fft_all(x), axes=all)
        for ix in range(dx - 1, -1, -1):
            Ftau = fft_func(Ftau, axis=ix, overwrite_x=True)

        # Single consolidated fftshift for all dimensions (instead of per-dimension)
        # MEMORY OPTIMIZATION: Reduces dx allocations to 1
        Ftau = fftshift_func(Ftau, axes=tuple(range(dx)))

        # Line 140: Ftau=Ftau/(M*(R^dx))
        # MEMORY OPTIMIZATION: Use in-place division
        norm_factor = M * (R**dx)
        np.divide(Ftau, norm_factor, out=Ftau)
    else:
        # Lines 143-147: Inverse FFT
        # MEMORY OPTIMIZATION: Single consolidated ifftshift, then all IFFTs
        Ftau = ifftshift_func(Ftau, axes=tuple(range(dx)))
        for ix in range(dx - 1, -1, -1):
            Ftau = ifft_func(Ftau, axis=ix, overwrite_x=True)

    # Lines 151-157: Extract output indices
    N_vec = N if isinstance(N, np.ndarray) else np.array([N])
    Mr_vec = Mr.flatten()
    q = (Mr_vec - N_vec) / 2
    q_start = np.ceil(q).astype(int)
    q_end = (q_start + N_vec).astype(int)

    # Extract the region - dynamic slicing for arbitrary dimensions
    slices = tuple(slice(q_start[i], q_end[i]) for i in range(dx)) + (slice(None),)
    Ftau = Ftau[slices]

    # Reshape to (N x dy)
    Ftau = Ftau.reshape(tuple(N_vec.astype(int)) + (dy,))

    # Lines 160-168: Deconvolve
    # This step requires Ftau to be centered (DC at M/2+1) because
    # nufftfreqs() generates centered frequencies (fix(-N/2) to N-fix(N/2)-1)
    # This is why fftshift is applied above
    if isdeconv:
        # Line 162: Kn=sqrt(prod(tau(:))/(pi^dx))* exp(-sum(permute(tau(:),[2:dx+1,1]) .* nufftfreqs(N).^2,dx+1))
        from .kernel import nufftfreqs

        freqs = nufftfreqs(N_vec)  # Returns grid of frequencies

        # Compute Kn
        tau_prod = np.prod(tau)
        Kn = np.sqrt(tau_prod / (np.pi**dx))

        # Compute exponential term
        # permute(tau(:),[2:dx+1,1]) creates shape (1, 1, ..., 1, dx)
        tau_reshaped = tau.reshape((1,) * dx + (dx,))
        # MEMORY OPTIMIZATION: Compute in-place
        freq_sq = np.square(freqs)  # freqs^2
        np.multiply(freq_sq, tau_reshaped, out=freq_sq)  # tau * freqs^2
        freq_sq_sum = np.sum(freq_sq, axis=-1)  # Sum along last axis
        np.negative(freq_sq_sum, out=freq_sq_sum)  # Negate
        np.exp(freq_sq_sum, out=freq_sq_sum)  # exp(-...)
        Kn = Kn * freq_sq_sum  # Final Kn

        # Line 163: Kninv=1./Kn
        # MEMORY OPTIMIZATION: Compute in-place (guard against zero/underflow)
        np.maximum(Kn, np.finfo(Kn.dtype).tiny, out=Kn)
        np.reciprocal(Kn, out=Kn)  # Now Kn contains Kninv

        # Line 164: Yq = Kninv .* Ftau
        # MEMORY OPTIMIZATION: Use in-place broadcast multiplication
        # Kn shape: (*grid_shape), Ftau shape: (*grid_shape, dy)
        # Instead of creating new array, multiply Ftau in-place
        np.multiply(Ftau, Kn[..., None], out=Ftau)
        Yq = Ftau  # Just a reference, no copy
    else:
        Yq = Ftau

    return Yq
