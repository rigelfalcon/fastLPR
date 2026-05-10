# ==============================================================================
# Unit Tests for Bandwidth Selection in fastLPR (Python)
# ==============================================================================
#
# This file provides comprehensive unit tests for bandwidth selection utilities
# including:
# 1. get_hlist - Bandwidth grid generation
# 2. GCV score computation
# 3. Bandwidth selection logic (1-SE rule)
# 4. Multi-dimensional bandwidth grids
#
# Author: fastLPR Development Team
# Copyright (c) 2024-2025 fastLPR Development Team
# License: GPL-3.0-or-later
# ==============================================================================

import pytest
import numpy as np
import sys
from pathlib import Path

# Fixed random seed for reproducibility
RANDOM_SEED = 42

# Add source path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastlpr.bandwidth import get_hlist
from fastlpr.regression import compute_gcv


# =============================================================================
# Section 1: get_hlist - Basic Functionality Tests
# =============================================================================

class TestGetHListBasic:
    """Tests for basic get_hlist functionality."""

    def test_1d_default_logspace(self):
        """UNIT: get_hlist 1D with default logarithmic spacing"""
        n = 10
        h_range = [0.1, 1.0]

        hlist = get_hlist(n, h_range)

        # Check shape: (n, 1) for 1D
        assert hlist.shape == (n, 1), f"Expected shape ({n}, 1), got {hlist.shape}"

        # Check bounds
        assert np.isclose(hlist[0, 0], 0.1), f"First bandwidth should be 0.1, got {hlist[0, 0]}"
        assert np.isclose(hlist[-1, 0], 1.0), f"Last bandwidth should be 1.0, got {hlist[-1, 0]}"

        # Check logarithmic spacing (ratio between consecutive points should be constant)
        ratios = hlist[1:, 0] / hlist[:-1, 0]
        # Unified: rtol=1e-6 (was 1e-10, matches MATLAB/R)
        assert np.allclose(ratios, ratios[0], rtol=1e-6), \
            "Bandwidth ratios should be constant for logspace"

    def test_1d_linear_spacing(self):
        """UNIT: get_hlist 1D with linear spacing"""
        n = 5
        h_range = [0.2, 1.0]

        hlist = get_hlist(n, h_range, spacing=np.linspace)

        # Check shape
        assert hlist.shape == (n, 1)

        # Check linear spacing (differences should be constant)
        diffs = np.diff(hlist[:, 0])
        expected_diff = (1.0 - 0.2) / (n - 1)
        # Unified: rtol=1e-6 (was 1e-10, matches MATLAB/R)
        assert np.allclose(diffs, expected_diff, rtol=1e-6), \
            "Bandwidth differences should be constant for linspace"

    def test_1d_single_bandwidth(self):
        """UNIT: get_hlist 1D with n=1 returns single bandwidth"""
        n = 1
        h_range = [0.5, 1.0]  # When n=1, should use h_min

        hlist = get_hlist(n, h_range)

        assert hlist.shape == (1, 1)
        assert np.isclose(hlist[0, 0], 0.5), \
            "Single bandwidth should equal h_min when n=1"


# =============================================================================
# Section 2: get_hlist - Multi-dimensional Tests
# =============================================================================

class TestGetHListMultiDim:
    """Tests for multi-dimensional bandwidth grids."""

    def test_2d_bandwidth_grid(self):
        """UNIT: get_hlist 2D creates Cartesian product grid"""
        n = [3, 4]  # 3 in dim 1, 4 in dim 2
        h_range = [[0.1, 0.3], [0.2, 0.5]]  # Range for each dimension

        hlist = get_hlist(n, h_range)

        # Total bandwidths = 3 * 4 = 12
        expected_rows = n[0] * n[1]
        assert hlist.shape == (expected_rows, 2), \
            f"Expected shape ({expected_rows}, 2), got {hlist.shape}"

        # Check that all combinations are present
        unique_dim0 = np.unique(hlist[:, 0])
        unique_dim1 = np.unique(hlist[:, 1])
        assert len(unique_dim0) == n[0], f"Expected {n[0]} unique values in dim 0"
        assert len(unique_dim1) == n[1], f"Expected {n[1]} unique values in dim 1"

    def test_2d_scalar_n(self):
        """UNIT: get_hlist 2D with scalar n broadcasts to all dimensions"""
        n = 5  # Same for both dimensions
        h_range = [[0.1, 1.0], [0.2, 2.0]]

        hlist = get_hlist(n, h_range)

        expected_rows = n * n  # 5 * 5 = 25
        assert hlist.shape == (expected_rows, 2)

    def test_3d_bandwidth_grid(self):
        """UNIT: get_hlist 3D creates proper Cartesian product"""
        n = [2, 3, 2]
        h_range = [[0.1, 0.2], [0.1, 0.3], [0.1, 0.2]]

        hlist = get_hlist(n, h_range)

        expected_rows = 2 * 3 * 2  # 12
        assert hlist.shape == (expected_rows, 3)

# =============================================================================
# Section 3: get_hlist - Input Validation Tests
# =============================================================================

class TestGetHListInputValidation:
    """Tests for input validation and edge cases."""

    def test_n_mismatch_raises_error(self):
        """UNIT: get_hlist raises error when n length mismatches range dimensions"""
        n = [3, 4, 5]  # 3 dimensions
        h_range = [[0.1, 1.0], [0.1, 1.0]]  # Only 2 dimensions

        with pytest.raises(ValueError, match="n must match"):
            get_hlist(n, h_range)

    def test_empty_range_handled(self):
        """UNIT: get_hlist handles edge case of h_min == h_max"""
        n = 5
        h_range = [0.5, 0.5]  # Same min and max

        # Should still work, just return same value repeated
        # (logspace will handle this case)
        hlist = get_hlist(n, h_range, spacing=np.linspace)

        assert hlist.shape == (n, 1)
        assert np.allclose(hlist, 0.5), "All bandwidths should equal 0.5"

    def test_positive_bandwidths(self):
        """UNIT: get_hlist produces positive bandwidths"""
        n = 20
        h_range = [0.01, 10.0]

        hlist = get_hlist(n, h_range)

        assert np.all(hlist > 0), "All bandwidths must be positive"

    def test_monotonic_1d(self):
        """UNIT: get_hlist 1D produces monotonically increasing bandwidths"""
        n = 15
        h_range = [0.1, 5.0]

        hlist = get_hlist(n, h_range)

        diffs = np.diff(hlist[:, 0])
        assert np.all(diffs > 0), "Bandwidths should be monotonically increasing"


# =============================================================================
# Section 4: get_hlist - Special Range Tests
# =============================================================================

class TestGetHListRanges:
    """Tests for different range specifications."""

    def test_wide_range_logspace(self):
        """UNIT: get_hlist handles wide range (3 orders of magnitude)"""
        n = 30
        h_range = [0.001, 1.0]  # 3 orders of magnitude

        hlist = get_hlist(n, h_range)

        assert hlist.shape == (n, 1)
        assert hlist[0, 0] >= 0.001
        assert hlist[-1, 0] <= 1.0

        # Check reasonable spread across the range
        log_range_covered = np.log10(hlist[-1, 0]) - np.log10(hlist[0, 0])
        expected_log_range = np.log10(1.0) - np.log10(0.001)
        assert np.isclose(log_range_covered, expected_log_range, rtol=0.01)

    def test_narrow_range(self):
        """UNIT: get_hlist handles narrow range"""
        n = 10
        h_range = [0.48, 0.52]  # Very narrow

        hlist = get_hlist(n, h_range)

        assert hlist.shape == (n, 1)
        assert np.all(hlist >= 0.48)
        assert np.all(hlist <= 0.52)

    def test_large_n(self):
        """UNIT: get_hlist handles large n without memory issues"""
        n = 100
        h_range = [0.1, 10.0]

        hlist = get_hlist(n, h_range)

        assert hlist.shape == (n, 1)
        assert not np.any(np.isnan(hlist))
        assert not np.any(np.isinf(hlist))


# =============================================================================
# Section 5: GCV Score Computation Tests
# =============================================================================

class TestComputeGCV:
    """Tests for GCV score computation."""

    def test_gcv_basic(self):
        """UNIT: compute_gcv returns valid GCV score"""
        np.random.seed(RANDOM_SEED)
        n = 100

        y = np.random.randn(n, 1)
        yhat = y + 0.1 * np.random.randn(n, 1)  # Small noise
        dof = 5.0

        gcv, info = compute_gcv(y, yhat, dof)

        assert np.isfinite(gcv), "GCV score must be finite"
        assert gcv >= 0, "GCV score must be non-negative"
        assert "rss" in info, "Info should contain RSS"

    def test_gcv_perfect_fit(self):
        """UNIT: compute_gcv for perfect fit (yhat == y)"""
        np.random.seed(RANDOM_SEED)
        n = 50

        y = np.random.randn(n, 1)
        yhat = y.copy()  # Perfect fit
        dof = 10.0

        gcv, info = compute_gcv(y, yhat, dof)

        # GCV should be very small (zero RSS)
        assert gcv < 1e-10, "GCV should be ~0 for perfect fit"

    def test_gcv_increases_with_error(self):
        """UNIT: compute_gcv increases with larger prediction errors"""
        np.random.seed(RANDOM_SEED)
        n = 100

        y = np.random.randn(n, 1)
        dof = 5.0

        # Small error
        yhat_small = y + 0.1 * np.random.randn(n, 1)
        gcv_small, _ = compute_gcv(y, yhat_small, dof)

        # Large error
        yhat_large = y + 1.0 * np.random.randn(n, 1)
        gcv_large, _ = compute_gcv(y, yhat_large, dof)

        assert gcv_large > gcv_small, \
            "GCV should increase with larger prediction errors"

    def test_gcv_dof_effect(self):
        """UNIT: compute_gcv penalty increases with DOF (for same RSS)"""
        np.random.seed(RANDOM_SEED)
        n = 100

        y = np.random.randn(n, 1)
        yhat = y + 0.5 * np.random.randn(n, 1)

        # Low DOF (simple model)
        gcv_low, _ = compute_gcv(y, yhat, dof=5.0)

        # High DOF (complex model) - more penalty
        gcv_high, _ = compute_gcv(y, yhat, dof=20.0)

        # For same RSS, higher DOF means smaller effective sample size (n-dof)
        # This leads to larger GCV due to the penalty term (1/(1-dof/n))^2
        assert gcv_high > gcv_low, \
            "GCV should increase with DOF due to complexity penalty"

    def test_gcv_multi_response(self):
        """UNIT: compute_gcv handles multi-response (dy > 1)"""
        np.random.seed(RANDOM_SEED)
        n = 50
        dy = 3

        y = np.random.randn(n, dy)
        yhat = y + 0.2 * np.random.randn(n, dy)
        dof = 5.0

        # Test 'first' aggregation
        gcv_first, info_first = compute_gcv(y, yhat, dof, aggregation='first')
        assert np.isfinite(gcv_first)

        # Test 'average' aggregation
        gcv_avg, info_avg = compute_gcv(y, yhat, dof, aggregation='average')
        assert np.isfinite(gcv_avg)

        # Test 'sum' aggregation
        gcv_sum, info_sum = compute_gcv(y, yhat, dof, aggregation='sum')
        assert np.isfinite(gcv_sum)

        # Sum should be approximately dy * average
        assert np.isclose(gcv_sum, dy * gcv_avg, rtol=0.1)


# =============================================================================
# Section 6: GCV Edge Cases Tests
# =============================================================================

class TestComputeGCVEdgeCases:
    """Tests for GCV computation edge cases."""

    def test_gcv_small_n(self):
        """UNIT: compute_gcv handles small sample size"""
        np.random.seed(RANDOM_SEED)
        n = 10

        y = np.random.randn(n, 1)
        yhat = y + 0.1 * np.random.randn(n, 1)
        dof = 3.0  # DOF < n is required

        gcv, info = compute_gcv(y, yhat, dof)

        assert np.isfinite(gcv), "GCV should be finite for small n"

    def test_gcv_dof_near_n(self):
        """UNIT: compute_gcv handles DOF close to n (high complexity)"""
        np.random.seed(RANDOM_SEED)
        n = 50

        y = np.random.randn(n, 1)
        yhat = y + 0.1 * np.random.randn(n, 1)
        dof = 45.0  # DOF very close to n

        gcv, info = compute_gcv(y, yhat, dof)

        # Should return a large (penalty) but finite value
        assert np.isfinite(gcv), "GCV should be finite even with high DOF"

    def test_gcv_zero_dof(self):
        """UNIT: compute_gcv handles dof=0 (no model complexity)"""
        np.random.seed(RANDOM_SEED)
        n = 50

        y = np.random.randn(n, 1)
        yhat = y + 0.1 * np.random.randn(n, 1)
        dof = 0.0

        gcv, info = compute_gcv(y, yhat, dof)

        # With dof=0, GCV = RSS/n (no penalty)
        expected_rss = np.sum((y - yhat) ** 2)
        expected_gcv = expected_rss / n
        assert np.isclose(gcv, expected_gcv, rtol=0.01), \
            "GCV with dof=0 should equal RSS/n"


# =============================================================================
# Section 7: GCV Numerical Properties Tests
# =============================================================================

class TestComputeGCVNumerical:
    """Tests for numerical properties of GCV."""

    def test_gcv_no_nan(self):
        """UNIT: compute_gcv never returns NaN for valid inputs"""
        np.random.seed(RANDOM_SEED)

        for _ in range(10):
            n = np.random.randint(20, 200)
            dof = np.random.uniform(1, n * 0.5)

            y = np.random.randn(n, 1)
            yhat = y + np.random.uniform(0.01, 1.0) * np.random.randn(n, 1)

            gcv, _ = compute_gcv(y, yhat, dof)

            assert not np.isnan(gcv), f"GCV should not be NaN (n={n}, dof={dof})"

    def test_gcv_scale_invariance(self):
        """UNIT: compute_gcv scales with y^2 (MSE property)"""
        np.random.seed(RANDOM_SEED)
        n = 100

        y = np.random.randn(n, 1)
        yhat = y + 0.3 * np.random.randn(n, 1)
        dof = 5.0

        gcv1, _ = compute_gcv(y, yhat, dof)

        # Scale by factor k
        k = 10.0
        gcv_scaled, _ = compute_gcv(k * y, k * yhat, dof)

        # GCV should scale by k^2 (since it's based on squared residuals)
        assert np.isclose(gcv_scaled, k**2 * gcv1, rtol=0.01), \
            "GCV should scale quadratically with y"

    def test_gcv_std_error(self):
        """UNIT: compute_gcv returns valid standard error when provided"""
        np.random.seed(RANDOM_SEED)
        n = 100

        y = np.random.randn(n, 1)
        yhat = y + 0.2 * np.random.randn(n, 1)
        dof = 5.0
        dof_stderr = 1.0
        penalty_std = 0.1

        gcv, info = compute_gcv(y, yhat, dof, dof_stderr=dof_stderr, penalty_std=penalty_std)

        assert np.isfinite(gcv)
        assert "stderr" in info
        # With non-zero penalty_std, stderr should be non-zero
        assert info["stderr"] >= 0


# =============================================================================
# Section 8: Integration Test with cv_fastlpr
# =============================================================================

class TestBandwidthSelectionIntegration:
    """Integration tests for bandwidth selection in cv_fastlpr."""

    def test_cv_fastlpr_selects_reasonable_bandwidth(self):
        """UNIT: cv_fastlpr selects reasonable bandwidth for smooth function"""
        np.random.seed(RANDOM_SEED)

        # Generate smooth function with noise
        n = 200
        x = np.random.rand(n, 1)
        y_true = np.sin(2 * np.pi * x)
        y = y_true + 0.1 * np.random.randn(n, 1)

        # Create bandwidth grid
        hlist = get_hlist(10, [0.05, 0.5])

        from fastlpr.api import cv_fastlpr

        result = cv_fastlpr(x, y, h=hlist, options={'order': 1})

        # Check that a bandwidth was selected
        assert result.h is not None
        assert np.all(result.h > 0)

        # Check that GCV scores were computed
        assert 'gcv_m' in result.gcv_yhat
        assert len(result.gcv_yhat['gcv_m']) > 0

    def test_cv_fastlpr_bandwidth_affects_smoothness(self):
        """UNIT: Larger bandwidth produces smoother fits"""
        np.random.seed(RANDOM_SEED)

        n = 100
        x = np.linspace(0, 1, n).reshape(-1, 1)
        y = np.sin(4 * np.pi * x) + 0.2 * np.random.randn(n, 1)

        from fastlpr.api import cv_fastlpr

        # Small bandwidth
        result_small = cv_fastlpr(x, y, h=[[0.02]], options={'order': 1, 'calc_dof': False})

        # Large bandwidth
        result_large = cv_fastlpr(x, y, h=[[0.15]], options={'order': 1, 'calc_dof': False})

        # Compute roughness (sum of squared second differences)
        def roughness(yhat):
            d2 = np.diff(np.diff(yhat.ravel()))
            return np.sum(d2 ** 2)

        rough_small = roughness(result_small.yhat)
        rough_large = roughness(result_large.yhat)

        assert rough_large < rough_small, \
            "Larger bandwidth should produce smoother (less rough) fit"


# =============================================================================
# Section 9: Custom Spacing Functions Tests
# =============================================================================

class TestGetHListCustomSpacing:
    """Tests for custom spacing functions."""

    def test_custom_quadratic_spacing(self):
        """UNIT: get_hlist accepts custom spacing function"""
        n = 10
        h_range = [0.1, 1.0]

        # Quadratic spacing (denser at small values)
        def quadspace(lo, hi, num):
            t = np.linspace(0, 1, num)
            return lo + (hi - lo) * t**2

        hlist = get_hlist(n, h_range, spacing=quadspace)

        assert hlist.shape == (n, 1)
        assert np.isclose(hlist[0, 0], 0.1)
        assert np.isclose(hlist[-1, 0], 1.0)

        # Check quadratic property: more values near lo
        midpoint_count = np.sum(hlist[:, 0] < 0.55)
        assert midpoint_count > n // 2, \
            "Quadratic spacing should have more values in lower half"

    def test_geomspace_vs_logspace(self):
        """UNIT: get_hlist logspace equivalent to geomspace"""
        n = 15
        h_range = [0.01, 10.0]

        hlist_default = get_hlist(n, h_range)  # Uses logspace
        hlist_geom = get_hlist(n, h_range, spacing=np.geomspace)

        # Unified: rtol=1e-6 (was 1e-10, matches MATLAB/R)
        assert np.allclose(hlist_default, hlist_geom, rtol=1e-6), \
            "Default logspace should match geomspace"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
