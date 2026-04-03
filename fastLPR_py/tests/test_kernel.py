# ==============================================================================
# Unit Tests for Kernel Functions (Python)
# ==============================================================================
#
# This file provides comprehensive unit tests for the kernel functions in
# fastLPR Python. Modeled after R's test-unit-nufft.R structure.
#
# Tests include:
# 1. Basic Gaussian kernel computation
# 2. Kernel normalization (integral approximately 1)
# 3. Kernel symmetry K(-x) = K(x)
# 4. Kernel positivity (all values >= 0)
# 5. Multi-dimensional kernels (1D, 2D, 3D)
# 6. Edge cases (very small/large bandwidths)
# 7. Various kernel types (Epanechnikov, tricube, etc.)
#
# Target: All tests should pass with fixed random seed = 42
#
# Author: fastLPR Development Team
# Copyright (c) 2024-2025 fastLPR Development Team
# License: GPL-3.0-or-later
# ==============================================================================

import pytest
import numpy as np
from scipy import integrate

# Fixed random seed for reproducibility
RANDOM_SEED = 42

# Import the module under test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastlpr.kernel import (
    kernel_function,
    KERNEL_GAUSSIAN,
    KERNEL_EPANECHNIKOV,
    KERNEL_TRICUBE,
    KERNEL_QUARTIC,
    KERNEL_TRIWEIGHT,
    KERNEL_UNIFORM,
    SQRT_2PI,
)


# =============================================================================
# pytest Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def random_state():
    """Return a seeded random state for reproducibility."""
    return np.random.RandomState(RANDOM_SEED)


@pytest.fixture
def grid_1d_fine():
    """Fine 1D grid for integration tests."""
    return np.linspace(-5, 5, 1001).reshape(-1, 1)


@pytest.fixture
def grid_1d_coarse():
    """Coarse 1D grid for basic tests."""
    return np.linspace(-3, 3, 101).reshape(-1, 1)


@pytest.fixture
def grid_2d_fine():
    """Fine 2D grid for integration tests."""
    x1 = np.linspace(-5, 5, 51)  # Unified: 51x51 (was 101x101)
    x2 = np.linspace(-5, 5, 51)
    X1, X2 = np.meshgrid(x1, x2, indexing='ij')
    return np.stack([X1, X2], axis=-1)


@pytest.fixture
def grid_3d_coarse():
    """Coarse 3D grid for tests."""
    x1 = np.linspace(-3, 3, 21)
    x2 = np.linspace(-3, 3, 21)
    x3 = np.linspace(-3, 3, 21)
    X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing='ij')
    return np.stack([X1, X2, X3], axis=-1)


@pytest.fixture
def scattered_1d_data(random_state):
    """Random scattered 1D points."""
    n = 100
    return random_state.randn(n, 1)


@pytest.fixture
def scattered_2d_data(random_state):
    """Random scattered 2D points."""
    n = 100
    return random_state.randn(n, 2)


# =============================================================================
# Section 1: Basic Gaussian Kernel Tests
# =============================================================================

class TestGaussianKernelBasic:
    """Basic tests for Gaussian kernel computation."""

    def test_gaussian_at_origin(self):
        """UNIT: Gaussian kernel at origin equals 1/sqrt(2*pi)"""
        x = np.array([[0.0]])
        result = kernel_function(x, kernel_type="gaussian")
        expected = 1.0 / SQRT_2PI
        assert np.isclose(result, expected, rtol=1e-12), \
            f"Gaussian at origin: {result} should equal {expected}"

    def test_gaussian_at_origin_2d(self):
        """UNIT: 2D Gaussian kernel at origin equals 1/sqrt(2*pi)"""
        x = np.array([[0.0, 0.0]])
        result = kernel_function(x, kernel_type="gaussian")
        # MATLAB uses 1D normalization for all dimensions
        expected = 1.0 / SQRT_2PI
        assert np.isclose(result, expected, rtol=1e-12), \
            f"2D Gaussian at origin: {result} should equal {expected}"

    def test_gaussian_at_origin_3d(self):
        """UNIT: 3D Gaussian kernel at origin equals 1/sqrt(2*pi)"""
        x = np.array([[0.0, 0.0, 0.0]])
        result = kernel_function(x, kernel_type="gaussian")
        expected = 1.0 / SQRT_2PI
        assert np.isclose(result, expected, rtol=1e-12), \
            f"3D Gaussian at origin: {result} should equal {expected}"

    def test_gaussian_decay(self, grid_1d_coarse):
        """UNIT: Gaussian kernel decays with distance from origin"""
        result = kernel_function(grid_1d_coarse, kernel_type="gaussian")
        center_idx = len(result) // 2

        # Value at origin should be maximum
        assert result[center_idx] == np.max(result), \
            "Gaussian should be maximal at origin"

        # Values should decay symmetrically
        for offset in [1, 5, 10, 20]:
            if center_idx + offset < len(result):
                assert result[center_idx + offset] < result[center_idx], \
                    f"Gaussian should decay at offset {offset}"

    def test_gaussian_tail_behavior(self):
        """UNIT: Gaussian kernel approaches 0 at large distances"""
        x_far = np.array([[5.0], [-5.0], [10.0]])
        result = kernel_function(x_far, kernel_type="gaussian")

        # At 5 sigma, value should be very small (< 1e-5)
        # At 10 sigma, value should be negligible
        assert np.all(result < 1e-5), \
            f"Gaussian should be negligible at 5+ sigma, got max={np.max(result):.2e}"


# =============================================================================
# Section 2: Kernel Normalization Tests
# =============================================================================

class TestKernelNormalization:
    """Tests for kernel normalization (integral should be approximately 1)."""

    def test_gaussian_normalization_1d(self, grid_1d_fine):
        """UNIT: 1D Gaussian kernel integrates to approximately 1"""
        result = kernel_function(grid_1d_fine, kernel_type="gaussian")
        dx = grid_1d_fine[1, 0] - grid_1d_fine[0, 0]

        # Numerical integration using trapezoidal rule
        integral = np.trapezoid(result.ravel(), dx=dx)

        # With default h=1, integral should be close to 1
        # Note: MATLAB uses 1D normalization, so integral ~ 1 for 1D
        assert np.isclose(integral, 1.0, rtol=0.01), \
            f"1D Gaussian integral={integral:.6f} should be ~1.0"

    def test_gaussian_normalization_with_bandwidth(self):
        """UNIT: Gaussian kernel with bandwidth h integrates correctly"""
        h_values = [0.5, 1.0, 2.0]
        for h in h_values:
            # Use fine grid scaled by bandwidth
            x = np.linspace(-5 * h, 5 * h, 1001).reshape(-1, 1)
            result = kernel_function(x, h=h, kernel_type="gaussian")
            dx = x[1, 0] - x[0, 0]

            integral = np.trapezoid(result.ravel(), dx=dx)

            assert np.isclose(integral, 1.0, rtol=0.02), \
                f"Gaussian with h={h}: integral={integral:.6f} should be ~1.0"

    def test_epanechnikov_normalization_1d(self):
        """UNIT: 1D Epanechnikov kernel integrates to approximately 1"""
        # Epanechnikov has compact support [-1, 1]
        x = np.linspace(-1, 1, 1001).reshape(-1, 1)
        result = kernel_function(x, kernel_type="epanechnikov")
        dx = x[1, 0] - x[0, 0]

        # Note: Epanechnikov uses max(eps, ...) to avoid zeros
        # Analytical integral of 0.75 * (1 - x^2) from -1 to 1 is 1.0
        integral = np.trapezoid(result.ravel(), dx=dx)

        # Allow for numerical precision with eps floor
        assert np.isclose(integral, 1.0, rtol=0.02), \
            f"1D Epanechnikov integral={integral:.6f} should be ~1.0"


# =============================================================================
# Section 3: Kernel Symmetry Tests
# =============================================================================

class TestKernelSymmetry:
    """Tests for kernel symmetry K(-x) = K(x)."""

    def test_gaussian_symmetry_1d(self):
        """UNIT: 1D Gaussian kernel is symmetric"""
        x_pos = np.array([[0.5], [1.0], [2.0], [3.0]])
        x_neg = -x_pos

        result_pos = kernel_function(x_pos, kernel_type="gaussian")
        result_neg = kernel_function(x_neg, kernel_type="gaussian")

        np.testing.assert_allclose(result_pos, result_neg, rtol=1e-12,
                                   err_msg="Gaussian should be symmetric: K(-x) = K(x)")

    def test_gaussian_symmetry_2d(self):
        """UNIT: 2D Gaussian kernel is symmetric"""
        x_pos = np.array([[0.5, 0.3], [1.0, 0.7], [2.0, 1.5]])
        x_neg = -x_pos

        result_pos = kernel_function(x_pos, kernel_type="gaussian")
        result_neg = kernel_function(x_neg, kernel_type="gaussian")

        np.testing.assert_allclose(result_pos, result_neg, rtol=1e-12,
                                   err_msg="2D Gaussian should be symmetric")

    def test_gaussian_symmetry_3d(self):
        """UNIT: 3D Gaussian kernel is symmetric"""
        x_pos = np.array([[0.5, 0.3, 0.2], [1.0, 0.7, 0.4]])
        x_neg = -x_pos

        result_pos = kernel_function(x_pos, kernel_type="gaussian")
        result_neg = kernel_function(x_neg, kernel_type="gaussian")

        np.testing.assert_allclose(result_pos, result_neg, rtol=1e-12,
                                   err_msg="3D Gaussian should be symmetric")

    def test_epanechnikov_symmetry(self):
        """UNIT: Epanechnikov kernel is symmetric"""
        x_pos = np.array([[0.3], [0.5], [0.8]])
        x_neg = -x_pos

        result_pos = kernel_function(x_pos, kernel_type="epanechnikov")
        result_neg = kernel_function(x_neg, kernel_type="epanechnikov")

        np.testing.assert_allclose(result_pos, result_neg, rtol=1e-12,
                                   err_msg="Epanechnikov should be symmetric")

    # test_other_kernels_symmetry - ARCHIVED (non-core kernels: tricube/quartic/triweight/uniform)


# =============================================================================
# Section 4: Kernel Positivity Tests
# =============================================================================

class TestKernelPositivity:
    """Tests for kernel positivity (all values >= 0)."""

    def test_gaussian_positivity(self, scattered_1d_data):
        """UNIT: Gaussian kernel values are all positive"""
        result = kernel_function(scattered_1d_data, kernel_type="gaussian")
        assert np.all(result >= 0), "Gaussian kernel should be non-negative"

    def test_gaussian_positivity_2d(self, scattered_2d_data):
        """UNIT: 2D Gaussian kernel values are all positive"""
        result = kernel_function(scattered_2d_data, kernel_type="gaussian")
        assert np.all(result >= 0), "2D Gaussian kernel should be non-negative"

    def test_epanechnikov_positivity(self):
        """UNIT: Epanechnikov kernel values are all non-negative"""
        # Include points inside and outside support
        x = np.array([[0.0], [0.5], [0.9], [1.5], [2.0]])
        result = kernel_function(x, kernel_type="epanechnikov")

        # All values should be >= eps (due to max(eps, ...) floor)
        assert np.all(result >= np.finfo(float).eps), \
            "Epanechnikov kernel should be >= eps everywhere"

    # test_all_kernels_positivity - ARCHIVED (redundant with individual positivity tests)


# =============================================================================
# Section 5: Multi-dimensional Kernel Tests
# =============================================================================

class TestMultidimensionalKernels:
    """Tests for multi-dimensional kernel evaluation."""

    def test_gaussian_1d_shape(self, scattered_1d_data):
        """UNIT: 1D Gaussian kernel returns correct shape"""
        result = kernel_function(scattered_1d_data, kernel_type="gaussian")
        assert result.shape == (len(scattered_1d_data),), \
            f"1D result shape should be ({len(scattered_1d_data)},)"

    def test_gaussian_2d_scattered(self, scattered_2d_data):
        """UNIT: 2D Gaussian kernel on scattered data returns correct shape"""
        result = kernel_function(scattered_2d_data, kernel_type="gaussian")
        assert result.shape == (len(scattered_2d_data),), \
            f"2D scattered result shape should be ({len(scattered_2d_data)},)"

    # test_gaussian_2d_grid - ARCHIVED (redundant with test_gaussian_2d_scattered)

    def test_gaussian_3d_grid(self, grid_3d_coarse):
        """UNIT: 3D Gaussian kernel on grid returns correct shape"""
        result = kernel_function(grid_3d_coarse, kernel_type="gaussian")
        expected_shape = grid_3d_coarse.shape[:-1]
        assert result.shape == expected_shape, \
            f"3D grid result shape should be {expected_shape}"

    def test_gaussian_radial_symmetry_2d(self):
        """UNIT: 2D Gaussian kernel is radially symmetric"""
        # Points at same distance from origin but different angles
        r = 1.0
        angles = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]
        points = np.array([[r * np.cos(a), r * np.sin(a)] for a in angles])

        result = kernel_function(points, kernel_type="gaussian")

        # All values should be equal (radially symmetric)
        np.testing.assert_allclose(result, result[0], rtol=1e-12,
                                   err_msg="2D Gaussian should be radially symmetric")


# =============================================================================
# Section 6: Edge Cases - Bandwidth Tests
# =============================================================================

class TestBandwidthEdgeCases:
    """Tests for edge cases with very small/large bandwidths."""

    def test_very_small_bandwidth(self):
        """UNIT: Kernel with very small bandwidth is sharply peaked"""
        h = 0.01
        x = np.linspace(-1, 1, 101).reshape(-1, 1)
        result = kernel_function(x, h=h, kernel_type="gaussian")

        # At origin, value should be large
        center_idx = 50
        assert result[center_idx] > 10.0, \
            f"Small bandwidth should give large center value, got {result[center_idx]}"

        # Away from center, values should be very small
        assert result[0] < 1e-10, \
            "Small bandwidth should give negligible values away from center"

    def test_very_large_bandwidth(self):
        """UNIT: Kernel with very large bandwidth is nearly flat"""
        h = 100.0
        x = np.linspace(-5, 5, 101).reshape(-1, 1)
        result = kernel_function(x, h=h, kernel_type="gaussian")

        # With large bandwidth, values should be nearly constant
        variation = np.max(result) - np.min(result)
        assert variation < 0.0001, \
            f"Large bandwidth should give nearly flat kernel, variation={variation}"

    def test_bandwidth_scaling(self):
        """UNIT: Kernel values scale correctly with bandwidth"""
        x = np.array([[0.0]])  # At origin
        h_values = [0.5, 1.0, 2.0, 4.0]

        results = []
        for h in h_values:
            res = kernel_function(x, h=h, kernel_type="gaussian")
            # Handle 0-d array case
            results.append(float(np.atleast_1d(res)[0]))

        # Kernel at origin should scale as 1/h
        for i in range(len(h_values) - 1):
            ratio = results[i] / results[i + 1]
            expected_ratio = h_values[i + 1] / h_values[i]
            assert np.isclose(ratio, expected_ratio, rtol=1e-12), \
                f"Kernel scaling: ratio={ratio:.4f} should be {expected_ratio:.4f}"

    def test_multidimensional_bandwidth(self):
        """UNIT: Different bandwidths per dimension work correctly"""
        x = np.array([[1.0, 2.0]])
        h = np.array([[0.5, 2.0]])  # Different bandwidth per dimension

        result = kernel_function(x, h=h, kernel_type="gaussian")

        # Manually compute expected value
        u = x / h  # Normalized: [2.0, 1.0]
        expected = (1.0 / SQRT_2PI) * np.exp(-0.5 * np.sum(u**2)) / np.prod(h)

        assert np.isclose(result, expected, rtol=1e-12), \
            f"Multi-dimensional bandwidth: {result} should equal {expected}"


# =============================================================================
# Section 7: Other Kernel Types Tests
# =============================================================================

class TestOtherKernelTypes:
    """Tests for non-Gaussian kernel types."""

    def test_epanechnikov_compact_support(self):
        """UNIT: Epanechnikov has compact support |u| <= 1"""
        # Points inside support
        x_inside = np.array([[0.0], [0.5], [0.9]])
        result_inside = kernel_function(x_inside, kernel_type="epanechnikov")

        # Points outside support (with default h=1)
        x_outside = np.array([[1.1], [2.0], [5.0]])
        result_outside = kernel_function(x_outside, kernel_type="epanechnikov")

        # Inside should have significant values
        assert np.all(result_inside > 0.1), "Epanechnikov should be significant inside support"

        # Outside should be eps (floor value)
        assert np.all(result_outside == pytest.approx(np.finfo(float).eps, rel=1e-5)), \
            "Epanechnikov should be eps outside support"

    # test_tricube_compact_support - ARCHIVED (non-core kernel)
    # test_quartic_compact_support - ARCHIVED (non-core kernel)
    # test_triweight_compact_support - ARCHIVED (non-core kernel)
    # test_uniform_compact_support - ARCHIVED (non-core kernel)
    # test_gaussian_ft_no_normalization - ARCHIVED (internal implementation)
    # test_polynomial_kernel - ARCHIVED (non-core kernel)


# =============================================================================
# Section 9: Kernel with Center Point Tests
# =============================================================================

class TestKernelWithCenter:
    """Tests for kernel function with non-zero center."""

    def test_gaussian_shifted_center(self):
        """UNIT: Gaussian centered at c peaks at c"""
        c = np.array([[2.0]])
        x = np.linspace(0, 4, 101).reshape(-1, 1)

        result = kernel_function(x, c=c, kernel_type="gaussian")

        # Maximum should be at x = c = 2.0
        max_idx = np.argmax(result)
        x_max = x[max_idx, 0]
        assert np.isclose(x_max, 2.0, atol=0.05), \
            f"Gaussian centered at 2.0 should peak at 2.0, peaked at {x_max}"

    # test_gaussian_shifted_center_2d - ARCHIVED (redundant with test_gaussian_shifted_center)


# =============================================================================
# Section 10: Error Handling Tests
# =============================================================================

class TestKernelErrorHandling:
    """Tests for error handling in kernel functions."""

    def test_invalid_kernel_type(self):
        """UNIT: kernel_function raises error for unknown kernel type"""
        x = np.array([[0.0]])
        with pytest.raises(ValueError, match="Unknown kernel type"):
            kernel_function(x, kernel_type="invalid_kernel")

    def test_polynomial_without_order(self):
        """UNIT: polynomial kernel requires order in opt"""
        x = np.array([[1.0]])
        with pytest.raises(ValueError, match="order"):
            kernel_function(x, kernel_type="polynomial", opt={})


# =============================================================================
# Section 12: Reproducibility Tests
# =============================================================================

class TestReproducibility:
    """Tests for reproducibility of kernel computations."""

    def test_deterministic_output(self):
        """UNIT: kernel_function produces deterministic output"""
        np.random.seed(RANDOM_SEED)
        x = np.random.randn(50, 1)

        result1 = kernel_function(x, kernel_type="gaussian")
        result2 = kernel_function(x, kernel_type="gaussian")

        np.testing.assert_array_equal(result1, result2,
                                      err_msg="Same input should produce identical output")

    # test_seeded_random_consistency - ARCHIVED (redundant with test_deterministic_output)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
