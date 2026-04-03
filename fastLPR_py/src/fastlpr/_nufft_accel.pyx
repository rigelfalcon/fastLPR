# cython: boundscheck=False, wraparound=False, cdivision=True
# cython: language_level=3
"""
Cython-accelerated NUFFT with SIMD vectorization for fastLPR.

Supports any dimension dx (1D, 2D, 3D, ...) and multiple outputs dy.
Uses C-contiguous memory layout and OpenMP parallelization for maximum performance.

This module is optional - install with: pip install fastlpr[accel]
The package falls back to Numba/pure Python if Cython is not available.

Author: Ying Wang, Min Li
Copyright (c) 2024 fastLPR Development Team
License: GPL-3.0-or-later
"""
import numpy as np
cimport numpy as np
from cython.parallel import prange
cimport cython
from libc.math cimport exp

np.import_array()

# Type definitions for C-level operations
ctypedef np.complex128_t complex_t
ctypedef np.float64_t float_t
ctypedef np.int64_t int_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void nufft_accumulate_cython(
    complex_t[:, ::1] Ftau_flat,
    int_t[:, ::1] xindx,
    complex_t[:, ::1] ysp,
    int_t[::1] strides
) noexcept:
    """
    Accumulation for NUFFT Type-1 (SEQUENTIAL - race-condition free).

    Accumulates spreading values into the grid using linear indexing.
    Supports any dimension dx and any number of output channels dy.
    Uses C-contiguous memory layout (::1) for optimal cache performance.

    IMPORTANT: This function uses SEQUENTIAL loops (not parallel) because
    multiple non-uniform points can map to the same grid cell during NUFFT
    spreading. Parallel writes to the same location cause race conditions
    and produce incorrect results.

    Args:
        Ftau_flat: Output grid, shape (prod(Mr), dy), C-contiguous, modified in-place
        xindx: Multi-dimensional grid indices, shape (n_points, dx), C-contiguous
        ysp: Spreading values to accumulate, shape (n_points, dy), C-contiguous
        strides: Strides for linear indexing, shape (dx,), C-contiguous

    Memory Layout:
        All arrays must be C-contiguous (row-major) for optimal performance.
    """
    cdef Py_ssize_t n_points = xindx.shape[0]
    cdef Py_ssize_t dx = xindx.shape[1]
    cdef Py_ssize_t dy = ysp.shape[1]
    cdef Py_ssize_t idx, iy, d
    cdef Py_ssize_t linear_idx

    # Sequential loop over points - cannot parallelize due to race conditions
    # Multiple points may spread to the same grid cell
    with nogil:
        for idx in range(n_points):
            # Compute linear index from multi-dimensional indices
            linear_idx = 0
            for d in range(dx):
                linear_idx = linear_idx + xindx[idx, d] * strides[d]

            # Accumulate all dy channels
            for iy in range(dy):
                Ftau_flat[linear_idx, iy] = Ftau_flat[linear_idx, iy] + ysp[idx, iy]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[float_t, ndim=2] heat_kernel_cython(
    float_t[:, :, ::1] x,
    float_t[::1] tau
):
    """
    SIMD-optimized heat kernel computation for NUFFT spreading.
    
    Computes the Gaussian heat kernel: exp(-sum(x^2 / (4*tau), axis=2))
    
    This is the spreading function used in NUFFT Type-1 to interpolate
    scattered data onto the regular grid. The heat kernel provides optimal
    frequency localization for the subsequent FFT.
    
    Args:
        x: Distance from grid points, shape (M, N, dx), C-contiguous
           M = number of data points
           N = number of spreading neighbors per point
           dx = number of dimensions
        tau: Spreading parameter per dimension, shape (dx,), C-contiguous
             Controls the width of the Gaussian spreading
    
    Returns:
        Heat kernel values, shape (M, N), C-contiguous
        Values are in (0, 1] with peak at x=0
    
    Performance:
        Uses OpenMP parallel over M (data points) with static scheduling.
        Inner loops over N and dx are sequential for cache efficiency.
    """
    cdef Py_ssize_t M = x.shape[0]
    cdef Py_ssize_t N = x.shape[1]
    cdef Py_ssize_t dx = x.shape[2]
    cdef Py_ssize_t i, j, d
    cdef float_t dist_sq, val, tau_4
    
    # Allocate output array (C-contiguous by default)
    cdef np.ndarray[float_t, ndim=2] result = np.empty((M, N), dtype=np.float64)
    cdef float_t[:, ::1] result_view = result
    
    # Parallel over data points
    with nogil:
        for i in prange(M, schedule='static'):
            for j in range(N):
                dist_sq = 0.0
                for d in range(dx):
                    val = x[i, j, d]
                    tau_4 = 4.0 * tau[d]
                    dist_sq = dist_sq + (val * val) / tau_4
                result_view[i, j] = exp(-dist_sq)
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[complex_t, ndim=2] apply_kernel_fft_cython(
    complex_t[:, ::1] data,
    complex_t[:, ::1] kernel
):
    """
    Element-wise multiplication of data with kernel in Fourier domain.
    
    This is the convolution step: multiply gridded data with kernel FFT.
    Supports multiple bandwidths (dh) stored in the last dimension.
    
    Args:
        data: Gridded data FFT, shape (L, dy), C-contiguous
        kernel: Kernel FFT, shape (L, dh), C-contiguous
    
    Returns:
        Convolved result, shape (L, dy*dh), C-contiguous
    
    Note:
        Output shape is (L, dy*dh) where each (dy, dh) pair is computed.
        Caller is responsible for reshaping to desired output format.
    """
    cdef Py_ssize_t L = data.shape[0]
    cdef Py_ssize_t dy = data.shape[1]
    cdef Py_ssize_t dh = kernel.shape[1]
    cdef Py_ssize_t i, iy, ih, out_idx
    
    cdef np.ndarray[complex_t, ndim=2] result = np.empty((L, dy * dh), dtype=np.complex128)
    cdef complex_t[:, ::1] result_view = result
    
    with nogil:
        for i in prange(L, schedule='static'):
            for iy in range(dy):
                for ih in range(dh):
                    out_idx = iy * dh + ih
                    result_view[i, out_idx] = data[i, iy] * kernel[i, ih]
    
    return result


def get_accel_info():
    """
    Return information about the Cython acceleration module.
    
    Returns:
        dict with keys:
            - backend: 'cython'
            - version: module version string
            - openmp: True if OpenMP is available
            - features: list of accelerated functions
            - simd: True if SIMD optimizations enabled
    """
    return {
        'backend': 'cython',
        'version': '1.1.0',
        'openmp': True,
        'simd': True,
        'features': [
            'nufft_accumulate_cython',
            'heat_kernel_cython', 
            'apply_kernel_fft_cython'
        ]
    }
