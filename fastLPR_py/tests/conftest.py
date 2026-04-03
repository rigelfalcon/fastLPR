# ==============================================================================
# pytest Configuration and Fixtures for fastLPR Unit Tests
# ==============================================================================
#
# This module provides shared fixtures and configuration for testing the fastlpr
# Python module. Modeled after R's test-unit-nufft.R structure.
#
# Author: fastLPR Development Team
# Copyright (c) 2024-2025 fastLPR Development Team
# License: GPL-3.0-or-later
# ==============================================================================

import pytest
import numpy as np
from pathlib import Path
import scipy.io as sio


# =============================================================================
# Constants
# =============================================================================

# Fixed random seed for reproducibility (matches MATLAB/R)
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# Global Tolerance Constants (unified with R's helper-utils.R)
# -----------------------------------------------------------------------------
# These tolerances are used across all tests to ensure consistent error
# thresholds. See CLAUDE.md for pass criteria.

# Strict tolerance for exact numerical matching (machine precision)
# Use for: Pure Python internal consistency tests
TOL_STRICT = 1e-12

# Numerical tolerance for algorithm comparisons
# Use for: Python-to-Python comparisons, self-consistency tests
TOL_NUMERICAL = 1e-6

# Cross-language tolerance for Python vs MATLAB comparisons
# Use for: test_xl_*.py files, cross-validation tests
# Per CLAUDE.md: BW MaxErr < 0.02, Mean MaxErr < 0.05
TOL_CROSSLANG = {
    'bw_maxerr': 0.02,        # Bandwidth selection threshold
    'mean_maxerr': 0.05,      # Mean estimate max error
    'var_maxerr': 0.05,       # Variance estimate max error
    'lcv_relerr': 0.01,       # LCV relative error (1%)
    'gcv_relerr': 0.05,       # GCV relative error (5%)
    'speed_ratio': 8.0,       # Max allowed speed ratio vs MATLAB
}

# NUFFT accuracy level tolerances (expected max error for each accuracy level)
ACCURACY_TOLERANCES = {
    4: 1e-3,
    6: 1e-5,
    8: 1e-7,
    9: 1e-9,
    12: 1e-11,
}

# Default test parameters
DEFAULT_ACCURACY = 6


# =============================================================================
# Path Helpers
# =============================================================================

def get_project_root() -> Path:
    """Get the project root directory (jss-code/)."""
    # conftest.py is at: jss-code/fastLPR_py/tests/conftest.py
    # So we need 3 .parent calls to reach jss-code/
    return Path(__file__).parent.parent.parent


def get_matlab_refs_dir() -> Path:
    """Get the MATLAB reference data directory."""
    return get_project_root() / "fastLPR" / "tests" / "refs"


def get_unit_refs_dir() -> Path:
    """Get the MATLAB unit test reference directory."""
    return get_matlab_refs_dir() / "crosslang_unit"


def get_e2e_refs_dir() -> Path:
    """Get the MATLAB E2E test reference directory."""
    return get_matlab_refs_dir() / "crosslang_e2e"


# =============================================================================
# Skip Helpers (unified with R's helper-utils.R)
# =============================================================================

def skip_if_no_matlab_refs(ref_subdir: str = "crosslang_e2e") -> None:
    """
    Skip test if MATLAB reference data directory is not available.

    This function checks if the MATLAB reference data files exist and
    skips the test if they are not available (e.g., in CI/CD without
    reference .mat files).

    Parameters
    ----------
    ref_subdir : str
        Subdirectory under fastLPR/tests/refs/ (default: "crosslang_e2e")

    Raises
    ------
    pytest.skip
        If reference directory is not available

    Examples
    --------
    >>> def test_xl_1d_order1():
    ...     skip_if_no_matlab_refs()
    ...     # ... test code that requires MATLAB reference data
    """
    ref_dir = get_matlab_refs_dir() / ref_subdir
    if not ref_dir.exists():
        pytest.skip(f"MATLAB reference directory not found: {ref_dir}")

    # Check if directory has any .mat files
    mat_files = list(ref_dir.glob("*.mat"))
    if len(mat_files) == 0:
        pytest.skip(f"No .mat files found in: {ref_dir}")


# =============================================================================
# XL Output Assertions
# =============================================================================

def assert_gcv_fields(gcv: dict) -> None:
    """
    Assert that the expected GCV fields exist and are internally consistent.

    This is used by cross-language tests (test_xl_*.py) to ensure that the Python
    API exposes MATLAB-aligned metadata needed for test unification.
    """
    assert isinstance(gcv, dict), f"gcv must be a dict, got {type(gcv)}"

    for key in ("gcv_m", "gcv_sd", "idmin", "hmin", "bandwidths"):
        assert key in gcv, f"Missing required gcv field: {key}"

    gcv_m = np.asarray(gcv["gcv_m"]).ravel()
    gcv_sd = np.asarray(gcv["gcv_sd"]).ravel()
    assert gcv_m.size > 0, "gcv_m must be non-empty"
    assert gcv_sd.shape == gcv_m.shape, (
        f"gcv_sd shape {gcv_sd.shape} must match gcv_m shape {gcv_m.shape}"
    )
    assert np.all(np.isfinite(gcv_sd)), "gcv_sd must be finite"
    assert np.all(gcv_sd >= 0), "gcv_sd must be >= 0"

    idmin_arr = np.asarray(gcv["idmin"]).ravel()
    assert idmin_arr.size > 0, "idmin must be non-empty"
    idmin = int(idmin_arr[0])
    assert 0 <= idmin < gcv_m.size, f"idmin={idmin} out of range [0, {gcv_m.size})"

    bandwidths = np.asarray(gcv["bandwidths"])
    assert bandwidths.size > 0, "bandwidths must be non-empty"
    expected_hmin = np.asarray(bandwidths[idmin]).ravel()
    hmin = np.asarray(gcv["hmin"]).ravel()

    assert np.all(np.isfinite(hmin)), "hmin must be finite"
    assert np.all(hmin > 0), "hmin must be positive"
    np.testing.assert_allclose(hmin, expected_hmin, rtol=0.0, atol=0.0)


# =============================================================================
# Comparison Helpers
# =============================================================================

def compare_results(py_result: np.ndarray, ref_result: np.ndarray) -> dict:
    """
    Compute comparison metrics between Python and reference results.

    Parameters
    ----------
    py_result : ndarray
        Python implementation output
    ref_result : ndarray
        Reference output (e.g., from MATLAB)

    Returns
    -------
    dict with keys:
        - max_abs_err: Maximum absolute error
        - mean_abs_err: Mean absolute error
        - max_rel_err: Maximum relative error (ignoring near-zero values)
    """
    py_vec = np.ravel(py_result)
    ref_vec = np.ravel(ref_result)

    abs_err = np.abs(py_vec - ref_vec)
    max_abs_err = np.max(abs_err)
    mean_abs_err = np.mean(abs_err)

    # Relative error (avoid division by zero)
    denom = np.maximum(np.abs(ref_vec), 1e-10)
    rel_err = abs_err / denom
    # Only consider relative error where reference is significant
    significant_mask = np.abs(ref_vec) > 1e-10
    if np.any(significant_mask):
        max_rel_err = np.max(rel_err[significant_mask])
    else:
        max_rel_err = 0.0

    return {
        'max_abs_err': max_abs_err,
        'mean_abs_err': mean_abs_err,
        'max_rel_err': max_rel_err,
    }


def compute_naive_dft_1d(x: np.ndarray, y: np.ndarray, N: int) -> np.ndarray:
    """
    Compute naive DFT for 1D data as reference.

    Parameters
    ----------
    x : ndarray, shape (M,) or (M, 1)
        Sample positions in [0, 1) or [-0.5, 0.5]
    y : ndarray, shape (M,) or (M, 1)
        Sample values
    N : int
        Grid size

    Returns
    -------
    naive : ndarray, shape (N,)
        Naive DFT result
    """
    x = np.ravel(x)
    y = np.ravel(y)
    M = len(x)

    k = np.arange(-N//2, N - N//2)
    naive = np.zeros(N, dtype=complex)

    for ik, kval in enumerate(k):
        phase = -2 * np.pi * kval * x
        naive[ik] = np.sum(y * np.exp(1j * phase)) / M

    return naive


def compute_naive_dft_2d(x: np.ndarray, y: np.ndarray, N: tuple) -> np.ndarray:
    """
    Compute naive DFT for 2D data as reference.

    Parameters
    ----------
    x : ndarray, shape (M, 2)
        Sample positions
    y : ndarray, shape (M,) or (M, 1)
        Sample values
    N : tuple of (N1, N2)
        Grid size per dimension

    Returns
    -------
    naive : ndarray, shape (N1, N2)
        Naive DFT result
    """
    y = np.ravel(y)
    M = len(y)
    N1, N2 = N

    k1 = np.arange(-N1//2, N1 - N1//2)
    k2 = np.arange(-N2//2, N2 - N2//2)

    naive = np.zeros((N1, N2), dtype=complex)

    for i1, kv1 in enumerate(k1):
        for i2, kv2 in enumerate(k2):
            phase = -2 * np.pi * (kv1 * x[:, 0] + kv2 * x[:, 1])
            naive[i1, i2] = np.sum(y * np.exp(1j * phase)) / M

    return naive


def compute_naive_dft_3d(x: np.ndarray, y: np.ndarray, N: tuple) -> np.ndarray:
    """
    Compute naive DFT for 3D data as reference.

    Parameters
    ----------
    x : ndarray, shape (M, 3)
        Sample positions
    y : ndarray, shape (M,) or (M, 1)
        Sample values
    N : tuple of (N1, N2, N3)
        Grid size per dimension

    Returns
    -------
    naive : ndarray, shape (N1, N2, N3)
        Naive DFT result
    """
    y = np.ravel(y)
    M = len(y)
    N1, N2, N3 = N

    k1 = np.arange(-N1//2, N1 - N1//2)
    k2 = np.arange(-N2//2, N2 - N2//2)
    k3 = np.arange(-N3//2, N3 - N3//2)

    naive = np.zeros((N1, N2, N3), dtype=complex)

    for i1, kv1 in enumerate(k1):
        for i2, kv2 in enumerate(k2):
            for i3, kv3 in enumerate(k3):
                phase = -2 * np.pi * (kv1 * x[:, 0] + kv2 * x[:, 1] + kv3 * x[:, 2])
                naive[i1, i2, i3] = np.sum(y * np.exp(1j * phase)) / M

    return naive


# =============================================================================
# pytest Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def random_state():
    """Return a seeded random state for reproducibility."""
    return np.random.RandomState(RANDOM_SEED)


@pytest.fixture
def nufft_1d_data(random_state):
    """Generate 1D NUFFT test data."""
    M = 50
    N = 16

    x = random_state.rand(M, 1)
    y = np.sin(2 * np.pi * x) + 0.1 * random_state.randn(M, 1)

    return {
        'x': x,
        'y': y,
        'M': M,
        'N': N,
        'df': 1.0 / N,
    }


@pytest.fixture
def nufft_2d_data(random_state):
    """Generate 2D NUFFT test data."""
    M = 100
    N = (8, 8)

    x = random_state.rand(M, 2)
    y = np.sin(2 * np.pi * x[:, 0:1]) * np.cos(2 * np.pi * x[:, 1:2]) + 0.1 * random_state.randn(M, 1)

    return {
        'x': x,
        'y': y,
        'M': M,
        'N': N,
        'df': np.array([1.0 / N[0], 1.0 / N[1]]),
    }


@pytest.fixture
def nufft_3d_data(random_state):
    """Generate 3D NUFFT test data."""
    M = 200
    N = (4, 4, 4)

    x = random_state.rand(M, 3)
    y = (np.sin(2 * np.pi * x[:, 0:1]) *
         np.cos(2 * np.pi * x[:, 1:2]) *
         np.sin(2 * np.pi * x[:, 2:3]) +
         0.1 * random_state.randn(M, 1))

    return {
        'x': x,
        'y': y,
        'M': M,
        'N': N,
        'df': np.array([1.0 / N[0], 1.0 / N[1], 1.0 / N[2]]),
    }


@pytest.fixture(params=[4, 6, 8, 9, 12])
def accuracy_level(request):
    """Parametrize tests over accuracy levels."""
    return request.param


@pytest.fixture
def edge_case_single_point():
    """Single point (M=1) edge case."""
    return {
        'x': np.array([[0.5]]),
        'y': np.array([[1.0]]),
        'M': 1,
        'N': 8,
    }


@pytest.fixture
def edge_case_two_points():
    """Two points (M=2) edge case."""
    return {
        'x': np.array([[0.25], [0.75]]),
        'y': np.array([[1.0], [2.0]]),
        'M': 2,
        'N': 8,
    }


@pytest.fixture
def edge_case_constant_y():
    """Constant y values edge case."""
    M = 50
    return {
        'x': np.linspace(0.05, 0.95, M).reshape(-1, 1),
        'y': np.ones((M, 1)) * 3.14159,
        'M': M,
        'N': 16,
    }


@pytest.fixture
def edge_case_boundary_x():
    """Points at boundary positions (0 and near 1)."""
    return {
        'x': np.array([[0.0], [0.001], [0.999], [0.5]]),
        'y': np.array([[1.0], [2.0], [3.0], [4.0]]),
        'M': 4,
        'N': 8,
    }


@pytest.fixture
def complex_data(random_state):
    """Complex-valued y data."""
    M = 50
    x = random_state.rand(M, 1)
    y_real = np.sin(2 * np.pi * x)
    y_imag = np.cos(2 * np.pi * x)
    y = y_real + 1j * y_imag

    return {
        'x': x,
        'y': y,
        'M': M,
        'N': 16,
    }


# =============================================================================
# MATLAB Reference Data Loader
# =============================================================================

@pytest.fixture
def load_matlab_ref():
    """Factory fixture to load MATLAB reference data."""
    def _loader(filename: str):
        ref_path = get_unit_refs_dir() / filename
        if not ref_path.exists():
            pytest.skip(f"Reference data not found: {filename}")
        return sio.loadmat(str(ref_path), squeeze_me=True)
    return _loader
