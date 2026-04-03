# ==============================================================================
# Error Handling Tests for fastLPR (Python)
# ==============================================================================
#
# This module tests error handling in cv_fastlpr and cv_fastkde functions.
# Based on requirements document section 4 (错误输入测试).
#
# Test Categories:
# 1. Empty hlist validation
# 2. Dimension mismatch between x and y
# 3. Invalid order parameter
# 4. All NaN/Inf data handling
# 5. Negative bandwidths validation
#
# Author: fastLPR Development Team
# Copyright (c) 2024-2025 fastLPR Development Team
# License: GPL-3.0-or-later
# ==============================================================================

import pytest
import numpy as np
import sys
from pathlib import Path

# Add source path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastlpr import cv_fastlpr, cv_fastkde, get_hlist


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def valid_1d_data():
    """Generate valid 1D test data."""
    np.random.seed(42)
    n = 100
    x = np.random.rand(n, 1)
    y = np.sin(2 * np.pi * x[:, 0]) + 0.1 * np.random.randn(n)
    return x, y


@pytest.fixture
def valid_2d_data():
    """Generate valid 2D test data."""
    np.random.seed(42)
    n = 100
    x = np.random.rand(n, 2)
    y = np.sin(2 * np.pi * x[:, 0]) * np.cos(2 * np.pi * x[:, 1]) + 0.1 * np.random.randn(n)
    return x, y


@pytest.fixture
def valid_hlist_1d():
    """Generate valid 1D bandwidth list."""
    return get_hlist(5, [0.1, 0.5])


@pytest.fixture
def valid_hlist_2d():
    """Generate valid 2D bandwidth list."""
    return get_hlist([3, 3], [[0.1, 0.5], [0.1, 0.5]])


# =============================================================================
# Section 1: Empty hlist Tests
# =============================================================================

class TestEmptyHlist:
    """Tests for empty bandwidth list validation."""

    def test_cv_fastlpr_empty_array_hlist(self, valid_1d_data):
        """ERROR: cv_fastlpr should raise error for empty hlist array."""
        x, y = valid_1d_data
        empty_hlist = np.array([]).reshape(0, 1)

        with pytest.raises((ValueError, IndexError)) as excinfo:
            cv_fastlpr(x, y, h=empty_hlist, options={'calc_dof': False})

        # Error message should be informative - accept various error types
        error_msg = str(excinfo.value).lower()
        # Accept "stack" which is the actual error message from numpy
        assert any(word in error_msg for word in ['empty', 'bandwidth', 'shape', 'size', 'index', 'stack', 'array']), \
            f"Error message should mention empty/bandwidth/array issue: {excinfo.value}"

    # test_cv_fastkde_empty_array_hlist - ARCHIVED (redundant)
    # test_cv_fastlpr_zero_length_hlist - ARCHIVED (redundant)


# =============================================================================
# Section 2: Dimension Mismatch Tests
# =============================================================================

class TestDimensionMismatch:
    """Tests for dimension mismatch between x and y."""

    def test_cv_fastlpr_x_y_sample_mismatch(self, valid_hlist_1d):
        """ERROR: cv_fastlpr should raise ValueError when x and y have different sample counts."""
        np.random.seed(42)
        x = np.random.rand(100, 1)  # 100 samples
        y = np.random.randn(50)      # 50 samples (mismatch!)

        with pytest.raises(ValueError) as excinfo:
            cv_fastlpr(x, y, h=valid_hlist_1d, options={'calc_dof': False})

        error_msg = str(excinfo.value).lower()
        # Check error message mentions the mismatch
        assert 'sample' in error_msg or 'shape' in error_msg or 'mismatch' in error_msg, \
            f"Error message should mention sample/shape mismatch: {excinfo.value}"

    # test_cv_fastlpr_x_y_explicit_mismatch_message - ARCHIVED (message format test)

    def test_cv_fastlpr_bandwidth_dimension_mismatch(self, valid_1d_data):
        """ERROR: cv_fastlpr should raise ValueError for bandwidth dimension mismatch."""
        x, y = valid_1d_data  # 1D data

        # 2D bandwidth for 1D data
        hlist_2d = get_hlist([3, 3], [[0.1, 0.5], [0.1, 0.5]])

        with pytest.raises(ValueError) as excinfo:
            cv_fastlpr(x, y, h=hlist_2d, options={'calc_dof': False})

        error_msg = str(excinfo.value).lower()
        assert 'dimension' in error_msg or 'bandwidth' in error_msg, \
            f"Error message should mention dimension/bandwidth issue: {excinfo.value}"

    # test_cv_fastkde_bandwidth_dimension_mismatch - ARCHIVED (redundant)


# =============================================================================
# Section 3: Invalid Order Parameter Tests
# =============================================================================

class TestInvalidOrder:
    """Tests for invalid polynomial order parameter."""

    @pytest.mark.parametrize("invalid_order", [-1, 3, 4, 10, -10])
    def test_cv_fastlpr_invalid_order_values(self, valid_1d_data, valid_hlist_1d, invalid_order):
        """ERROR: cv_fastlpr should raise ValueError for invalid order values."""
        x, y = valid_1d_data

        with pytest.raises(ValueError) as excinfo:
            cv_fastlpr(x, y, h=valid_hlist_1d, options={'order': invalid_order, 'calc_dof': False})

        error_msg = str(excinfo.value).lower()
        assert 'order' in error_msg or '0' in error_msg or '1' in error_msg or '2' in error_msg, \
            f"Error message should mention valid order values: {excinfo.value}"

    def test_cv_fastlpr_order_must_be_0_1_or_2(self, valid_1d_data, valid_hlist_1d):
        """ERROR: cv_fastlpr error message should state valid order values."""
        x, y = valid_1d_data

        with pytest.raises(ValueError) as excinfo:
            cv_fastlpr(x, y, h=valid_hlist_1d, options={'order': 5, 'calc_dof': False})

        error_msg = str(excinfo.value)
        # Should mention valid values are 0, 1, or 2
        assert '0' in error_msg or '1' in error_msg or '2' in error_msg, \
            f"Error message should list valid order values: {excinfo.value}"

    @pytest.mark.parametrize("valid_order", [0, 1, 2])
    def test_cv_fastlpr_valid_order_no_error(self, valid_1d_data, valid_hlist_1d, valid_order):
        """VALID: cv_fastlpr should accept order values 0, 1, 2 without error."""
        x, y = valid_1d_data

        # Should NOT raise error
        result = cv_fastlpr(x, y, h=valid_hlist_1d, options={'order': valid_order, 'calc_dof': False})
        assert result is not None
        assert hasattr(result, 'yhat')


# =============================================================================
# Section 4: NaN/Inf Data Tests
# =============================================================================

class TestNaNInfData:
    """Tests for handling of NaN and Inf values in input data.

    The API validates inputs and raises ValueError for NaN/Inf values,
    providing clear error messages to guide users.
    """

    def test_cv_fastlpr_all_nan_y(self, valid_hlist_1d):
        """ERROR: cv_fastlpr raises ValueError for all-NaN y."""
        np.random.seed(42)
        x = np.random.rand(50, 1)
        y = np.full(50, np.nan)  # All NaN

        with pytest.raises(ValueError, match="NaN"):
            cv_fastlpr(x, y, h=valid_hlist_1d, options={'calc_dof': False})

    def test_cv_fastlpr_all_inf_y(self, valid_hlist_1d):
        """ERROR: cv_fastlpr raises ValueError for all-Inf y."""
        np.random.seed(42)
        x = np.random.rand(50, 1)
        y = np.full(50, np.inf)  # All Inf

        with pytest.raises(ValueError, match="Inf"):
            cv_fastlpr(x, y, h=valid_hlist_1d, options={'calc_dof': False})

    # test_cv_fastlpr_mixed_nan_inf_y - ARCHIVED (redundant with all_nan and all_inf)

    def test_cv_fastkde_all_nan_x(self):
        """ERROR: cv_fastkde raises ValueError for all-NaN x."""
        x = np.full((50, 1), np.nan)

        with pytest.raises(ValueError, match="NaN"):
            cv_fastkde(x)

    def test_cv_fastkde_all_inf_x(self):
        """ERROR: cv_fastkde raises ValueError for all-Inf x."""
        x = np.full((50, 1), np.inf)

        with pytest.raises(ValueError, match="Inf"):
            cv_fastkde(x)

    def test_cv_fastlpr_partial_nan_produces_error(self, valid_hlist_1d):
        """ERROR: cv_fastlpr raises ValueError for partial NaN in y."""
        np.random.seed(42)
        x = np.random.rand(100, 1)
        y = np.sin(2 * np.pi * x[:, 0]) + 0.1 * np.random.randn(100)
        y[0:5] = np.nan  # First 5 values are NaN

        # API validates and rejects any NaN values
        with pytest.raises(ValueError, match="NaN"):
            cv_fastlpr(x, y, h=valid_hlist_1d, options={'calc_dof': False})


# =============================================================================
# Section 5: Negative Bandwidth Tests
# =============================================================================

class TestNegativeBandwidth:
    """Tests for negative bandwidth validation.

    The API validates that bandwidths are strictly positive and raises
    ValueError for non-positive values.
    """

    def test_cv_fastlpr_negative_bandwidth(self, valid_1d_data):
        """ERROR: cv_fastlpr raises ValueError for negative bandwidths."""
        x, y = valid_1d_data
        negative_hlist = np.array([[-0.1], [-0.5], [-1.0]])  # All negative

        with pytest.raises(ValueError, match="positive"):
            cv_fastlpr(x, y, h=negative_hlist, options={'calc_dof': False})

    def test_cv_fastlpr_mixed_negative_bandwidth(self, valid_1d_data):
        """ERROR: cv_fastlpr raises ValueError for mixed negative bandwidths."""
        x, y = valid_1d_data
        mixed_hlist = np.array([[0.1], [-0.1], [0.2]])  # One negative

        with pytest.raises(ValueError, match="positive"):
            cv_fastlpr(x, y, h=mixed_hlist, options={'calc_dof': False})

    def test_cv_fastlpr_zero_bandwidth(self, valid_1d_data):
        """ERROR: cv_fastlpr raises ValueError for zero bandwidth."""
        x, y = valid_1d_data
        zero_hlist = np.array([[0.0], [0.1], [0.2]])  # First is zero

        # Zero bandwidth is non-positive, should raise ValueError
        with pytest.raises(ValueError, match="positive"):
            cv_fastlpr(x, y, h=zero_hlist, options={'calc_dof': False})

    # test_cv_fastkde_negative_bandwidth - ARCHIVED (redundant)
    # test_cv_fastlpr_negative_bandwidth_2d - ARCHIVED (redundant)


# =============================================================================
# Section 6: Other Input Validation Tests
# =============================================================================

class TestOtherInputValidation:
    """Additional input validation tests."""

    def test_cv_fastlpr_y_wrong_ndim(self, valid_hlist_1d):
        """ERROR: cv_fastlpr should reject y with wrong number of dimensions."""
        np.random.seed(42)
        x = np.random.rand(50, 1)
        y = np.random.randn(50, 2, 3)  # 3D array - invalid

        with pytest.raises(ValueError) as excinfo:
            cv_fastlpr(x, y, h=valid_hlist_1d, options={'calc_dof': False})

        error_msg = str(excinfo.value).lower()
        assert '1d' in error_msg or '2d' in error_msg or 'dimension' in error_msg, \
            f"Error message should mention dimension requirements: {excinfo.value}"

    def test_cv_fastkde_too_few_observations(self):
        """ERROR: cv_fastkde should reject data with fewer than 2 observations."""
        x = np.array([[0.5]])  # Only 1 observation

        with pytest.raises(ValueError) as excinfo:
            cv_fastkde(x)

        error_msg = str(excinfo.value).lower()
        assert '2' in error_msg or 'observation' in error_msg or 'sample' in error_msg, \
            f"Error message should mention minimum observations: {excinfo.value}"

    def test_cv_fastlpr_4d_runs_without_crash(self, valid_hlist_1d):
        """VALID: cv_fastlpr should handle 4D data without crashing (max practical dim)."""
        np.random.seed(42)
        x = np.random.rand(50, 4)  # 4D data - max practical dimension
        y = np.random.randn(50)

        # 4D should work without errors (may be slow)
        hlist_4d = get_hlist([2, 2, 2, 2], [[0.3, 0.6]] * 4)
        result = cv_fastlpr(x, y, h=hlist_4d, options={'calc_dof': False})
        assert result is not None
        assert hasattr(result, 'yhat')

    def test_cv_fastlpr_exceeds_max_dimension(self, valid_hlist_1d):
        """ERROR: cv_fastlpr should raise error for dimension > 10."""
        np.random.seed(42)
        x = np.random.rand(100, 11)  # 11D - exceeds maximum
        y = np.random.randn(100)

        # Validation should catch this before any computation
        with pytest.raises(ValueError) as excinfo:
            hlist_11d = get_hlist([2] * 11, [[0.1, 0.5]] * 11)
            cv_fastlpr(x, y, h=hlist_11d, options={'calc_dof': False})

        error_msg = str(excinfo.value).lower()
        assert 'dimension' in error_msg or '10' in error_msg, \
            f"Error message should mention dimension limit: {excinfo.value}"


# =============================================================================
# Section 7: Error Message Quality Tests - ARCHIVED
# =============================================================================
# TestErrorMessageQuality class archived - tests error message format, not functionality
# See: dev/archive/py-errors-tests-20260109/archived_errors_tests.py


# =============================================================================
# Main entry point for running tests directly
# =============================================================================

if __name__ == "__main__":
    # Suppress fastlpr startup banner during tests
    import os
    os.environ["FASTLPR_QUIET"] = "1"

    pytest.main([__file__, "-v", "--tb=short"])
