# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Pure-Python (NumPy/Numba) implementation of the Type-1 NUFFT.

The goal is to mirror the behavior of `nufftn_type1.m` without relying on
compiled extensions.  The implementation follows the reference algorithm
described in Dutt & Rokhlin (1993) and the optimization notes outlined in
Jake VanderPlas' blog post on NUFFT optimisation with NumPy/Numba.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, Tuple

import numpy as np

from . import fft_backend

# Use centralized backend detection
from .backend_selection import HAS_NUMBA

if HAS_NUMBA:
    import numba
else:
    numba = None


class NUFFTBackend(Protocol):
    """Protocol for NUFFT backends."""

    def type1(
        self,
        x: np.ndarray,
        y: np.ndarray,
        grid_shape: Tuple[int, ...],
        iflag: int = -1,
    ) -> np.ndarray:
        ...


@dataclass
class BackendConfig:
    name: str
    version: str
    extra: dict[str, str] | None = None


def _normalise_coordinates(x: np.ndarray) -> np.ndarray:
    """Normalise coordinates to the unit interval."""

    coords = np.asarray(x, dtype=float)
    if coords.ndim == 1:
        coords = coords[:, None]
    coords = np.mod(coords, 1.0)
    return coords


def _frequencies_for_shape(shape: Sequence[int]) -> np.ndarray:
    """Return centred integer frequency offsets for each grid point."""

    shape = np.asarray(shape, dtype=np.int64)
    dims = shape.size
    total = int(np.prod(shape))
    freq = np.zeros((total, dims), dtype=np.float64)

    for idx in range(total):
        remainder = idx
        for d in range(dims - 1, -1, -1):
            m = shape[d]
            freq[idx, d] = remainder % m - m // 2
            remainder //= m
    return freq


def _nudft_numpy(
    x: np.ndarray,
    y: np.ndarray,
    grid_shape: Tuple[int, ...],
    iflag: int,
) -> np.ndarray:
    """Direct non-uniform discrete Fourier transform (NumPy fallback)."""

    dims = x.shape[1]
    freq_axes = [np.arange(m) - m // 2 for m in grid_shape]
    out = np.zeros(grid_shape, dtype=np.complex128)
    factor = -1 if iflag < 0 else 1

    for idx in np.ndindex(grid_shape):
        freq = np.array([freq_axes[d][idx[d]] for d in range(dims)], dtype=float)
        angles = x @ freq
        out[idx] = np.dot(y, np.exp(factor * 2j * np.pi * angles))
    return out


if HAS_NUMBA:  # pragma: no branch

    @numba.njit(cache=True)
    def _nudft_numba(
        x: np.ndarray,
        y: np.ndarray,
        grid_shape: np.ndarray,
        iflag: int,
    ) -> np.ndarray:
        n_samples, dims = x.shape
        total = 1
        for m in grid_shape:
            total *= m

        out = np.empty(total, dtype=np.complex128)
        freq = np.empty(dims, dtype=np.float64)
        factor = -1.0 if iflag < 0 else 1.0

        for idx in range(total):
            remainder = idx
            for d in range(dims - 1, -1, -1):
                m = grid_shape[d]
                freq[d] = remainder % m - m // 2
                remainder //= m

            accum_real = 0.0
            accum_imag = 0.0
            for n in range(n_samples):
                angle = 0.0
                for d in range(dims):
                    angle += freq[d] * x[n, d]
                arg = factor * 2.0 * np.pi * angle
                val = y[n]
                accum_real += val.real * np.cos(arg) - val.imag * np.sin(arg)
                accum_imag += val.real * np.sin(arg) + val.imag * np.cos(arg)
            out[idx] = accum_real + 1j * accum_imag

        return out  # Return flat array, reshape outside Numba


class DirectNUFFTBackend:
    """Pure Python NUFFT backend with optional Numba acceleration."""

    def __init__(self) -> None:
        self._has_numba = HAS_NUMBA

    def type1(
        self,
        x: np.ndarray,
        y: np.ndarray,
        grid_shape: Tuple[int, ...],
        iflag: int = -1,
    ) -> np.ndarray:
        coords = _normalise_coordinates(x)
        values = np.asarray(y, dtype=np.complex128)
        shape = tuple(int(n) for n in grid_shape)

        if self._has_numba:
            # Numba version - reshape manually after
            shape_arr = np.array(shape, dtype=np.int64)
            out_flat = _nudft_numba(coords, values, shape_arr, int(iflag))
            return out_flat.reshape(shape)
        return _nudft_numpy(coords, values, shape, int(iflag))


if HAS_NUMBA:

    @numba.jit(nopython=True, cache=True)
    def _build_grid_numba(
        x: np.ndarray,
        c: np.ndarray,
        tau: float,
        Msp: int,
        ftau: np.ndarray,
        E3: np.ndarray,
    ) -> np.ndarray:
        """
        Build convolved grid using Gaussian spreading (Numba-optimized).

        This is the performance-critical loop for NUFFT Type-1.
        Following Jake VanderPlas's optimization approach.
        """
        Mr = ftau.shape[0]
        hx = 2.0 * np.pi / Mr

        # Precompute exponents
        for j in range(Msp + 1):
            E3[j] = np.exp(-((np.pi * j / Mr) ** 2) / tau)

        # Spread values onto ftau
        for i in range(x.shape[0]):
            xi = x[i] % (2.0 * np.pi)
            m = 1 + int(xi // hx)
            xi = xi - hx * m
            E1 = np.exp(-0.25 * xi**2 / tau)
            E2 = np.exp((xi * np.pi) / (Mr * tau))
            E2mm = 1.0

            for mm in range(Msp):
                ftau[(m + mm) % Mr] += c[i] * E1 * E2mm * E3[mm]
                E2mm *= E2
                ftau[(m - mm - 1) % Mr] += c[i] * E1 / E2mm * E3[mm + 1]

        return ftau

    @numba.jit(nopython=True, cache=True, parallel=True)
    def _build_grid_2d_numba(
        x: np.ndarray,
        y: np.ndarray,
        c: np.ndarray,
        tau: np.ndarray,
        Msp: np.ndarray,
        Mr: np.ndarray,
        ftau: np.ndarray,
    ) -> np.ndarray:
        """
        Build convolved grid for 2D using Gaussian spreading (Numba-optimized).

        Parameters
        ----------
        x : np.ndarray
            Scaled x-coordinates, shape (N,)
        y : np.ndarray
            Scaled y-coordinates, shape (N,)
        c : np.ndarray
            Complex values to spread, shape (N,)
        tau : np.ndarray
            Gaussian spreading parameters, shape (2,)
        Msp : np.ndarray
            Spreading widths, shape (2,)
        Mr : np.ndarray
            Oversampled grid sizes, shape (2,)
        ftau : np.ndarray
            Output grid, shape (Mr[0], Mr[1])
        """
        N = x.shape[0]
        hx = 2.0 * np.pi / Mr[0]
        hy = 2.0 * np.pi / Mr[1]

        Msp_x = int(Msp[0])
        Msp_y = int(Msp[1])
        Mr_x = int(Mr[0])
        Mr_y = int(Mr[1])

        # Spread each point (parallelized over data points)
        for i in numba.prange(N):
            xi = x[i] % (2.0 * np.pi)
            yi = y[i] % (2.0 * np.pi)

            mx = int(np.round(xi / hx))
            my = int(np.round(yi / hy))

            # Spread in 2D neighborhood
            for mmx in range(-Msp_x, Msp_x + 1):
                idx_x = (mx + mmx) % Mr_x
                diff_x = xi - hx * (mx + mmx)
                weight_x = np.exp(-(diff_x**2) / (4.0 * tau[0]))

                for mmy in range(-Msp_y, Msp_y + 1):
                    idx_y = (my + mmy) % Mr_y
                    diff_y = yi - hy * (my + mmy)
                    weight_y = np.exp(-(diff_y**2) / (4.0 * tau[1]))

                    ftau[idx_x, idx_y] += c[i] * weight_x * weight_y

        return ftau


class FastNUFFTBackend:
    """
    Fast NUFFT backend using Gaussian gridding and FFT.

    Implements the algorithm from Dutt & Rokhlin (1993) with optimizations
    from Jake VanderPlas (2015).

    Uses MATLAB-compatible grid parameter computation.
    """

    def __init__(self, accuracy: int = 6) -> None:
        self._accuracy = accuracy
        self._has_numba = HAS_NUMBA

    def _compute_grid_params(self, M: int) -> Tuple[int, int, float, int]:
        """
        Compute NUFFT grid parameters using MATLAB formula.

        Port from fastLPR/utility/core/compute_grid_params.m.

        Returns
        -------
        Msp : int
            Spreading width
        Mr : int
            Oversampled grid size
        tau : float
            Gaussian spreading parameter
        ratio : int
            Oversampling ratio
        """
        from .kernel import compute_grid_params

        Msp, Mr, tau, ratio = compute_grid_params(M, self._accuracy)
        return Msp, Mr, tau, ratio

    def type1(
        self,
        x: np.ndarray,
        y: np.ndarray,
        grid_shape: Tuple[int, ...],
        iflag: int = -1,
    ) -> np.ndarray:
        """
        Fast Type-1 NUFFT using Gaussian gridding.

        Supports arbitrary dimensions using tensor product of 1D spreading.
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        y = np.asarray(y, dtype=complex)

        if x.ndim == 1:
            x = x[:, None]

        N = x.shape[0]  # Number of data points
        dims = x.shape[1]

        # Handle 1D case with optimized path
        if dims == 1:
            return self._type1_1d(x, y, grid_shape, iflag)

        # Multi-dimensional case
        return self._type1_nd(x, y, grid_shape, iflag)

    def _type1_1d(
        self,
        x: np.ndarray,
        y: np.ndarray,
        grid_shape: Tuple[int, ...],
        iflag: int = -1,
    ) -> np.ndarray:
        """Fast Type-1 NUFFT for 1D case (optimized)."""
        N = x.shape[0]
        dims = 1
        M = grid_shape[0]  # Grid size
        Msp, Mr, tau, ratio = self._compute_grid_params(M)

        # Construct the convolved grid
        ftau = np.zeros(Mr, dtype=complex)

        if self._has_numba:
            # Use optimized Numba version
            E3 = np.zeros(Msp + 1, dtype=float)
            x_scaled = x[:, 0] * 2.0 * np.pi  # Scale to [0, 2π]
            ftau = _build_grid_numba(x_scaled, y, tau, Msp, ftau, E3)
        else:
            # Pure NumPy fallback
            hx = 2.0 * np.pi / Mr
            xmod = (x[:, 0] * 2.0 * np.pi) % (2.0 * np.pi)
            m = 1 + (xmod // hx).astype(int)
            mm = np.arange(-Msp, Msp)
            mpmm = m[:, None] + mm[None, :]
            spread = y[:, None] * np.exp(-0.25 * (xmod[:, None] - hx * mpmm) ** 2 / tau)
            np.add.at(ftau, mpmm % Mr, spread)

        # Compute FFT on the convolved grid
        # MATLAB applies fftshift(fft(...)) then normalizes by M*(R^dx)
        # where M = number of data points, R = oversampling ratio
        if iflag < 0:
            # Forward FFT - apply fftshift after FFT
            Ftau_before = fft_backend.fft(ftau)
            Ftau = fft_backend.fftshift(Ftau_before)
            # Normalize by M * R^dx
            norm_factor = N * (ratio**dims)
            Ftau = Ftau / norm_factor
        else:
            Ftau = fft_backend.ifft(ftau)

        # Extract evaluation grid
        # q = (Mr - N) / 2
        # q = [ceil(q)+1, ceil(q)+N]
        # Ftau = Ftau(q(1):q(2))
        q_start = int(np.ceil((Mr - M) / 2))
        q_end = q_start + M
        Ftau = Ftau[q_start:q_end]

        # Deconvolve using convolution theorem
        # MATLAB always applies deconvolution (isdeconv=true by default)
        # Kn = sqrt(tau/pi) * exp(-tau * k^2)
        # Kninv = 1 / Kn
        # Yq = Kninv .* Ftau
        k = np.arange(-(M // 2), M - (M // 2))
        Kn = np.sqrt(tau / np.pi) * np.exp(-tau * k**2)
        Kninv = 1.0 / Kn

        return Kninv * Ftau

    def _type1_nd(
        self,
        x: np.ndarray,
        y: np.ndarray,
        grid_shape: Tuple[int, ...],
        iflag: int = -1,
    ) -> np.ndarray:
        """
        Fast Type-1 NUFFT for multi-dimensional case.

        Uses tensor product of 1D Gaussian spreading following MATLAB implementation.
        Reference: fastLPR/utility/core/nufftn_type1.m
        """
        N = x.shape[0]  # Number of data points
        dims = x.shape[1]  # Number of dimensions

        # Compute grid parameters for each dimension
        M = np.array(grid_shape, dtype=int)
        Msp = np.zeros(dims, dtype=int)
        Mr = np.zeros(dims, dtype=int)
        tau = np.zeros(dims, dtype=float)
        ratio = np.zeros(dims, dtype=int)

        for d in range(dims):
            Msp[d], Mr[d], tau[d], ratio[d] = self._compute_grid_params(M[d])

        # Scale coordinates to [0, 2π]
        hx = 2.0 * np.pi / Mr
        xmod = (x * 2.0 * np.pi) % (2.0 * np.pi)

        # Spread values onto grid
        ftau_shape = tuple(Mr)
        ftau = np.zeros(ftau_shape, dtype=complex)

        # Use Numba-optimized 2D spreading if available and dims == 2
        if dims == 2 and self._has_numba:
            # Use optimized 2D Numba kernel (computes weights internally)
            x_scaled = xmod[:, 0]
            y_scaled = xmod[:, 1]
            ftau = _build_grid_2d_numba(x_scaled, y_scaled, y, tau, Msp, Mr, ftau)
        else:
            # General N-dimensional spreading
            # Compute spreading grid indices
            m = np.round(xmod / hx).astype(int)  # Shape: (N, dims)

            # Build tensor product of spreading neighborhoods
            Msp2 = 2 * Msp + 1  # Spreading width in each dimension
            total_spread = np.prod(
                Msp2
            )  # Total number of spreading points per data point

            # Generate all spreading indices (tensor product)
            # mpmm will have shape (N, total_spread, dims)
            mpmm = np.zeros((N, total_spread, dims), dtype=int)

            # Build tensor product of spreading offsets
            spread_offsets = []
            for d in range(dims):
                spread_offsets.append(np.arange(-Msp[d], Msp[d] + 1, dtype=int))

            # Create meshgrid of all combinations
            offset_grids = np.meshgrid(*spread_offsets, indexing="ij")
            for d in range(dims):
                offset_flat = offset_grids[d].ravel()  # Shape: (total_spread,)
                mpmm[:, :, d] = m[:, d : d + 1] + offset_flat[None, :]  # Broadcasting

            # Compute Gaussian weights
            # weight = heat_kernel(xmod - hx * mpmm, tau)
            # heat_kernel(x, tau) = exp(-x^2 / (4*tau))
            weight = np.ones((N, total_spread), dtype=float)
            for d in range(dims):
                diff = (
                    xmod[:, d : d + 1] - hx[d] * mpmm[:, :, d]
                )  # Shape: (N, total_spread)
                weight *= np.exp(-(diff**2) / (4.0 * tau[d]))

            # ysp = y * weight
            ysp = y[:, None] * weight  # Shape: (N, total_spread)

            # Vectorized spreading using np.add.at with flattened indices
            # Convert multi-dimensional indices to flat indices
            indices_list = []
            for d in range(dims):
                indices_list.append(mpmm[:, :, d] % Mr[d])

            # Use np.ravel_multi_index to convert to flat indices
            flat_indices = np.ravel_multi_index(
                [indices_list[d].ravel() for d in range(dims)], ftau_shape
            )

            # Flatten ysp and use np.add.at for accumulation
            ftau_flat = ftau.ravel()
            np.add.at(ftau_flat, flat_indices, ysp.ravel())
            ftau = ftau_flat.reshape(ftau_shape)

        # Apply FFT along each dimension
        if iflag < 0:
            # Forward transform: Apply FFT then fftshift
            for d in range(dims):
                ftau = fft_backend.fft(ftau, axis=d)
                ftau = fft_backend.fftshift(ftau, axes=d)
            # Normalize by N * R^dims
            norm_factor = N * np.prod(ratio)
            ftau = ftau / norm_factor
        else:
            # Inverse transform
            for d in range(dims):
                ftau = fft_backend.ifft(ftau, axis=d)

        # Extract evaluation grid
        slices = []
        for d in range(dims):
            q_start = int(np.ceil((Mr[d] - M[d]) / 2))
            q_end = q_start + M[d]
            slices.append(slice(q_start, q_end))
        ftau = ftau[tuple(slices)]

        # Deconvolve - MATLAB always applies deconvolution (isdeconv=true by default)
        # Build deconvolution kernel for each dimension
        Kninv = np.ones(M, dtype=float)
        for d in range(dims):
            k = np.arange(-(M[d] // 2), M[d] - (M[d] // 2))
            Kn_d = np.sqrt(tau[d] / np.pi) * np.exp(-tau[d] * k**2)
            Kninv_d = 1.0 / Kn_d
            # Broadcast to full grid
            shape = [1] * dims
            shape[d] = M[d]
            Kninv_d_reshaped = Kninv_d.reshape(shape)
            Kninv = Kninv * Kninv_d_reshaped
        return Kninv * ftau


def get_backend(preferred: Optional[str] = None) -> Tuple[NUFFTBackend, BackendConfig]:
    """
    Return the NUFFT backend.

    Parameters
    ----------
    preferred : str, optional
        Preferred backend: 'fast' (default) or 'direct'

    Returns
    -------
    backend : NUFFTBackend
        NUFFT backend instance
    config : BackendConfig
        Backend configuration
    """
    if preferred == "direct":
        backend = DirectNUFFTBackend()
        config = BackendConfig(
            name="direct-nudft", version="0.1", extra={"numba": str(HAS_NUMBA)}
        )
    else:
        backend = FastNUFFTBackend()
        config = BackendConfig(
            name="fast-nufft",
            version="0.1",
            extra={"numba": str(HAS_NUMBA), "method": "gaussian-gridding"},
        )

    return backend, config


def type1(
    x: np.ndarray,
    y: np.ndarray,
    grid_shape: Tuple[int, ...],
    iflag: int = -1,
    backend: Optional[NUFFTBackend] = None,
    accuracy: int = 6,
) -> np.ndarray:
    """
    Compute the Type-1 NUFFT (non-uniform samples to uniform grid).

    Parameters
    ----------
    x : array-like, shape (N, dim)
        Non-uniform sample locations in [0, 1).
    y : array-like, shape (N,)
        Sample values (real or complex).
    grid_shape : tuple of ints
        Output grid shape per dimension.
    iflag : int, optional
        Sign convention (+1 or -1). Defaults to -1 (physics convention).
    backend : NUFFTBackend, optional
        Custom backend implementing the NUFFT protocol.
    accuracy : int, optional
        NUFFT accuracy in decimal digits (default: 6).
        Matches MATLAB's accuracy parameter.

    Returns
    -------
    F : ndarray, shape grid_shape
        Fourier transform on uniform grid.

    Notes
    -----
    This implements the Type-1 NUFFT: transforming from non-uniform points
    to a uniform frequency grid. The algorithm uses Gaussian gridding
    followed by FFT for O(N + M log M) complexity.

    References
    ----------
    .. [1] Dutt, A., & Rokhlin, V. (1993). Fast Fourier transforms for
           nonequispaced data. SIAM Journal on Scientific computing, 14(6), 1368-1393.
    .. [2] VanderPlas, J. (2015). Optimizing Python in the Real World: NumPy,
           Numba, and the NUFFT. Blog post.
    """
    if backend is None:
        backend = FastNUFFTBackend(accuracy=accuracy)

    return backend.type1(x, y, grid_shape, iflag=iflag)
