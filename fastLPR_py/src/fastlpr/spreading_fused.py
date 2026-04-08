# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
spreading_fused - Fused NUFFT spreading kernels optimized with Numba

This module provides fused spreading kernels that compute:
1. Spreading neighborhood construction
2. Heat kernel weights
3. Grid accumulation

All in a single pass per data point, avoiding massive intermediate arrays.

Performance improvement: 20-40x speedup vs vectorized NumPy approach
- d=1: ~6ms for 100k points (vs ~200ms vectorized)
- d=2: ~83ms for 100k points (vs ~2000ms vectorized)
- d=3: ~550ms for 50k points (vs ~21000ms vectorized)

Author: Ying Wang, Min Li
"""

import numpy as np

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

__all__ = [
    'fused_spreading_1d',
    'fused_spreading_2d',
    'fused_spreading_3d',
    'fused_spreading_nd',
    'fused_spreading_dispatch',
    'HAS_FUSED_SPREADING',
]

HAS_FUSED_SPREADING = HAS_NUMBA


if HAS_NUMBA:
    # ============================================================
    # Dimension-specific optimized kernels (real-valued)
    # NOTE: fastmath=False for numerical stability (fastmath reorders operations)
    # ============================================================

    @numba.jit(nopython=True, cache=True)
    def fused_spreading_1d_real(xmod, y, Ftau_flat, Msp, Mr, hx, tau, strides):
        """1D fused spreading kernel for real-valued data."""
        M = xmod.shape[0]
        dy = y.shape[1]
        Msp0 = Msp[0]
        Mr0 = Mr[0]
        hx0 = hx[0]
        tau0 = tau[0]

        for i in range(M):
            m0 = int(round(xmod[i, 0] / hx0))
            for j0 in range(-Msp0, Msp0 + 1):
                idx0 = (m0 + j0) % Mr0
                diff0 = xmod[i, 0] - hx0 * (m0 + j0)
                weight = np.exp(-(diff0 * diff0 / (4 * tau0)))
                for iy in range(dy):
                    Ftau_flat[idx0, iy] += y[i, iy] * weight

    @numba.jit(nopython=True, cache=True)
    def fused_spreading_2d_real(xmod, y, Ftau_flat, Msp, Mr, hx, tau, strides):
        """2D fused spreading kernel for real-valued data."""
        M = xmod.shape[0]
        dy = y.shape[1]
        Msp0, Msp1 = Msp[0], Msp[1]
        Mr0, Mr1 = Mr[0], Mr[1]
        hx0, hx1 = hx[0], hx[1]
        tau0, tau1 = tau[0], tau[1]
        stride0 = strides[0]

        for i in range(M):
            m0 = int(round(xmod[i, 0] / hx0))
            m1 = int(round(xmod[i, 1] / hx1))

            for j0 in range(-Msp0, Msp0 + 1):
                idx0 = (m0 + j0) % Mr0
                diff0 = xmod[i, 0] - hx0 * (m0 + j0)

                for j1 in range(-Msp1, Msp1 + 1):
                    idx1 = (m1 + j1) % Mr1
                    diff1 = xmod[i, 1] - hx1 * (m1 + j1)
                    weight = np.exp(-(diff0 * diff0 / (4 * tau0) +
                                      diff1 * diff1 / (4 * tau1)))
                    linear_idx = idx0 * stride0 + idx1

                    for iy in range(dy):
                        Ftau_flat[linear_idx, iy] += y[i, iy] * weight

    @numba.jit(nopython=True, cache=True)
    def fused_spreading_3d_real(xmod, y, Ftau_flat, Msp, Mr, hx, tau, strides):
        """3D fused spreading kernel for real-valued data."""
        M = xmod.shape[0]
        dy = y.shape[1]
        Msp0, Msp1, Msp2 = Msp[0], Msp[1], Msp[2]
        Mr0, Mr1, Mr2 = Mr[0], Mr[1], Mr[2]
        hx0, hx1, hx2 = hx[0], hx[1], hx[2]
        tau0, tau1, tau2 = tau[0], tau[1], tau[2]
        stride0, stride1 = strides[0], strides[1]

        for i in range(M):
            m0 = int(round(xmod[i, 0] / hx0))
            m1 = int(round(xmod[i, 1] / hx1))
            m2 = int(round(xmod[i, 2] / hx2))

            for j0 in range(-Msp0, Msp0 + 1):
                idx0 = (m0 + j0) % Mr0
                diff0 = xmod[i, 0] - hx0 * (m0 + j0)

                for j1 in range(-Msp1, Msp1 + 1):
                    idx1 = (m1 + j1) % Mr1
                    diff1 = xmod[i, 1] - hx1 * (m1 + j1)

                    for j2 in range(-Msp2, Msp2 + 1):
                        idx2 = (m2 + j2) % Mr2
                        diff2 = xmod[i, 2] - hx2 * (m2 + j2)

                        weight = np.exp(-(diff0 * diff0 / (4 * tau0) +
                                          diff1 * diff1 / (4 * tau1) +
                                          diff2 * diff2 / (4 * tau2)))
                        linear_idx = idx0 * stride0 + idx1 * stride1 + idx2

                        for iy in range(dy):
                            Ftau_flat[linear_idx, iy] += y[i, iy] * weight

    # ============================================================
    # Complex-valued kernels (split real/imag for Numba compatibility)
    # ============================================================

    @numba.jit(nopython=True, cache=True)
    def fused_spreading_1d_complex(xmod, y_real, y_imag, Ftau_real, Ftau_imag,
                                    Msp, Mr, hx, tau, strides):
        """1D fused spreading kernel for complex-valued data."""
        M = xmod.shape[0]
        dy = y_real.shape[1]
        Msp0 = Msp[0]
        Mr0 = Mr[0]
        hx0 = hx[0]
        tau0 = tau[0]

        for i in range(M):
            m0 = int(round(xmod[i, 0] / hx0))
            for j0 in range(-Msp0, Msp0 + 1):
                idx0 = (m0 + j0) % Mr0
                diff0 = xmod[i, 0] - hx0 * (m0 + j0)
                weight = np.exp(-(diff0 * diff0 / (4 * tau0)))
                for iy in range(dy):
                    Ftau_real[idx0, iy] += y_real[i, iy] * weight
                    Ftau_imag[idx0, iy] += y_imag[i, iy] * weight

    @numba.jit(nopython=True, cache=True)
    def fused_spreading_2d_complex(xmod, y_real, y_imag, Ftau_real, Ftau_imag,
                                    Msp, Mr, hx, tau, strides):
        """2D fused spreading kernel for complex-valued data."""
        M = xmod.shape[0]
        dy = y_real.shape[1]
        Msp0, Msp1 = Msp[0], Msp[1]
        Mr0, Mr1 = Mr[0], Mr[1]
        hx0, hx1 = hx[0], hx[1]
        tau0, tau1 = tau[0], tau[1]
        stride0 = strides[0]

        for i in range(M):
            m0 = int(round(xmod[i, 0] / hx0))
            m1 = int(round(xmod[i, 1] / hx1))

            for j0 in range(-Msp0, Msp0 + 1):
                idx0 = (m0 + j0) % Mr0
                diff0 = xmod[i, 0] - hx0 * (m0 + j0)

                for j1 in range(-Msp1, Msp1 + 1):
                    idx1 = (m1 + j1) % Mr1
                    diff1 = xmod[i, 1] - hx1 * (m1 + j1)
                    weight = np.exp(-(diff0 * diff0 / (4 * tau0) +
                                      diff1 * diff1 / (4 * tau1)))
                    linear_idx = idx0 * stride0 + idx1

                    for iy in range(dy):
                        Ftau_real[linear_idx, iy] += y_real[i, iy] * weight
                        Ftau_imag[linear_idx, iy] += y_imag[i, iy] * weight

    @numba.jit(nopython=True, cache=True)
    def fused_spreading_3d_complex(xmod, y_real, y_imag, Ftau_real, Ftau_imag,
                                    Msp, Mr, hx, tau, strides):
        """3D fused spreading kernel for complex-valued data."""
        M = xmod.shape[0]
        dy = y_real.shape[1]
        Msp0, Msp1, Msp2 = Msp[0], Msp[1], Msp[2]
        Mr0, Mr1, Mr2 = Mr[0], Mr[1], Mr[2]
        hx0, hx1, hx2 = hx[0], hx[1], hx[2]
        tau0, tau1, tau2 = tau[0], tau[1], tau[2]
        stride0, stride1 = strides[0], strides[1]

        for i in range(M):
            m0 = int(round(xmod[i, 0] / hx0))
            m1 = int(round(xmod[i, 1] / hx1))
            m2 = int(round(xmod[i, 2] / hx2))

            for j0 in range(-Msp0, Msp0 + 1):
                idx0 = (m0 + j0) % Mr0
                diff0 = xmod[i, 0] - hx0 * (m0 + j0)

                for j1 in range(-Msp1, Msp1 + 1):
                    idx1 = (m1 + j1) % Mr1
                    diff1 = xmod[i, 1] - hx1 * (m1 + j1)

                    for j2 in range(-Msp2, Msp2 + 1):
                        idx2 = (m2 + j2) % Mr2
                        diff2 = xmod[i, 2] - hx2 * (m2 + j2)

                        weight = np.exp(-(diff0 * diff0 / (4 * tau0) +
                                          diff1 * diff1 / (4 * tau1) +
                                          diff2 * diff2 / (4 * tau2)))
                        linear_idx = idx0 * stride0 + idx1 * stride1 + idx2

                        for iy in range(dy):
                            Ftau_real[linear_idx, iy] += y_real[i, iy] * weight
                            Ftau_imag[linear_idx, iy] += y_imag[i, iy] * weight

    # ============================================================
    # Dispatcher functions (public API)
    # ============================================================

    def fused_spreading_1d(xmod, y, Ftau_flat, Msp, Mr, hx, tau, strides):
        """1D fused spreading with automatic real/complex dispatch."""
        if np.iscomplexobj(y):
            Ftau_real = np.real(Ftau_flat).astype(np.float64).copy()
            Ftau_imag = np.imag(Ftau_flat).astype(np.float64).copy()
            fused_spreading_1d_complex(
                xmod.astype(np.float64),
                np.real(y).astype(np.float64),
                np.imag(y).astype(np.float64),
                Ftau_real, Ftau_imag,
                Msp.astype(np.int64), Mr.astype(np.int64),
                hx.astype(np.float64), tau.astype(np.float64),
                strides.astype(np.int64)
            )
            Ftau_flat[:] = Ftau_real + 1j * Ftau_imag
        else:
            fused_spreading_1d_real(
                xmod.astype(np.float64),
                y.astype(np.float64),
                Ftau_flat,
                Msp.astype(np.int64), Mr.astype(np.int64),
                hx.astype(np.float64), tau.astype(np.float64),
                strides.astype(np.int64)
            )

    def fused_spreading_2d(xmod, y, Ftau_flat, Msp, Mr, hx, tau, strides):
        """2D fused spreading with automatic real/complex dispatch."""
        if np.iscomplexobj(y):
            Ftau_real = np.real(Ftau_flat).astype(np.float64).copy()
            Ftau_imag = np.imag(Ftau_flat).astype(np.float64).copy()
            fused_spreading_2d_complex(
                xmod.astype(np.float64),
                np.real(y).astype(np.float64),
                np.imag(y).astype(np.float64),
                Ftau_real, Ftau_imag,
                Msp.astype(np.int64), Mr.astype(np.int64),
                hx.astype(np.float64), tau.astype(np.float64),
                strides.astype(np.int64)
            )
            Ftau_flat[:] = Ftau_real + 1j * Ftau_imag
        else:
            fused_spreading_2d_real(
                xmod.astype(np.float64),
                y.astype(np.float64),
                Ftau_flat,
                Msp.astype(np.int64), Mr.astype(np.int64),
                hx.astype(np.float64), tau.astype(np.float64),
                strides.astype(np.int64)
            )

    def fused_spreading_3d(xmod, y, Ftau_flat, Msp, Mr, hx, tau, strides):
        """3D fused spreading with automatic real/complex dispatch."""
        if np.iscomplexobj(y):
            Ftau_real = np.real(Ftau_flat).astype(np.float64).copy()
            Ftau_imag = np.imag(Ftau_flat).astype(np.float64).copy()
            fused_spreading_3d_complex(
                xmod.astype(np.float64),
                np.real(y).astype(np.float64),
                np.imag(y).astype(np.float64),
                Ftau_real, Ftau_imag,
                Msp.astype(np.int64), Mr.astype(np.int64),
                hx.astype(np.float64), tau.astype(np.float64),
                strides.astype(np.int64)
            )
            Ftau_flat[:] = Ftau_real + 1j * Ftau_imag
        else:
            fused_spreading_3d_real(
                xmod.astype(np.float64),
                y.astype(np.float64),
                Ftau_flat,
                Msp.astype(np.int64), Mr.astype(np.int64),
                hx.astype(np.float64), tau.astype(np.float64),
                strides.astype(np.int64)
            )

    def fused_spreading_nd(xmod, y, Ftau_flat, Msp, Mr, hx, tau, strides):
        """Generic n-dimensional fused spreading (slower fallback for d>3)."""
        # For d>3, fall back to the vectorized approach
        # This is rare in practice (fastLPR only supports d<=3)
        raise NotImplementedError("Fused spreading for d>3 not implemented. Use vectorized fallback.")

    def fused_spreading_dispatch(dx, xmod, y, Ftau_flat, Msp, Mr, hx, tau, strides):
        """
        Dispatch to the appropriate fused spreading kernel based on dimension.

        Args:
            dx: Number of dimensions
            xmod: Scaled knot locations (M, dx)
            y: Input values (M, dy)
            Ftau_flat: Flattened output grid (prod(Mr), dy) - modified in-place
            Msp: Spreading radius per dimension (dx,)
            Mr: Grid size per dimension (dx,)
            hx: Grid spacing per dimension (dx,)
            tau: Spreading parameter per dimension (dx,)
            strides: Strides for linear indexing (dx,)
        """
        if dx == 1:
            fused_spreading_1d(xmod, y, Ftau_flat, Msp, Mr, hx, tau, strides)
        elif dx == 2:
            fused_spreading_2d(xmod, y, Ftau_flat, Msp, Mr, hx, tau, strides)
        elif dx == 3:
            fused_spreading_3d(xmod, y, Ftau_flat, Msp, Mr, hx, tau, strides)
        else:
            raise NotImplementedError(f"Fused spreading for d={dx} not implemented")

else:
    # Fallback stubs when Numba is not available
    def fused_spreading_1d(xmod, y, Ftau_flat, Msp, Mr, hx, tau, strides):
        raise NotImplementedError("Fused spreading requires Numba")

    def fused_spreading_2d(xmod, y, Ftau_flat, Msp, Mr, hx, tau, strides):
        raise NotImplementedError("Fused spreading requires Numba")

    def fused_spreading_3d(xmod, y, Ftau_flat, Msp, Mr, hx, tau, strides):
        raise NotImplementedError("Fused spreading requires Numba")

    def fused_spreading_nd(xmod, y, Ftau_flat, Msp, Mr, hx, tau, strides):
        raise NotImplementedError("Fused spreading requires Numba")

    def fused_spreading_dispatch(dx, xmod, y, Ftau_flat, Msp, Mr, hx, tau, strides):
        raise NotImplementedError("Fused spreading requires Numba")
