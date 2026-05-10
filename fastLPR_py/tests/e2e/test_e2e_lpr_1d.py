# ==============================================================================
# E2E Test: 1D Local Polynomial Regression
# ==============================================================================
# Test file: fastLPR_py/tests/e2e/test_e2e_lpr_1d.py
# Created: 2026-01-08
#
# These tests verify standalone functionality without MATLAB reference data.
# They run independently and are suitable for CI/CD pipelines.
# ==============================================================================

import numpy as np
import pytest


class TestE2ELpr1D:
    """End-to-end tests for 1D Local Polynomial Regression."""

    def test_order0_1d(self):
        """1D Nadaraya-Watson (order 0) regression works."""
        from fastlpr import cv_fastlpr, get_hlist

        np.random.seed(42)
        n = 500
        x = np.random.rand(n, 1)
        y_true = np.sin(2 * np.pi * x).ravel()
        y = y_true + 0.2 * np.random.randn(n)

        hlist = get_hlist(10, [0.01, 0.5])
        opt = {'order': 0}
        regs = cv_fastlpr(x, y, hlist, opt)

        assert regs.yhat is not None
        assert len(regs.yhat) == n
        # Correlation check
        corr = np.corrcoef(regs.yhat.ravel(), y_true)[0, 1]
        assert corr > 0.7

    def test_order1_1d(self):
        """1D Local Linear (order 1) regression works."""
        from fastlpr import cv_fastlpr, get_hlist

        np.random.seed(42)
        n = 500
        x = np.random.rand(n, 1)
        y_true = np.sin(2 * np.pi * x).ravel()
        y = y_true + 0.2 * np.random.randn(n)

        hlist = get_hlist(10, [0.01, 0.5])
        opt = {'order': 1, 'calc_dof': True}
        regs = cv_fastlpr(x, y, hlist, opt)

        assert regs.yhat is not None
        assert len(regs.yhat) == n
        corr = np.corrcoef(regs.yhat.ravel(), y_true)[0, 1]
        assert corr > 0.8

    def test_order2_1d(self):
        """1D Local Quadratic (order 2) regression works."""
        from fastlpr import cv_fastlpr, get_hlist

        np.random.seed(42)
        n = 500
        x = np.random.rand(n, 1)
        y_true = np.sin(2 * np.pi * x).ravel()
        y = y_true + 0.2 * np.random.randn(n)

        hlist = get_hlist(10, [0.01, 0.5])
        opt = {'order': 2, 'calc_dof': True}
        regs = cv_fastlpr(x, y, hlist, opt)

        assert regs.yhat is not None
        mse = np.mean((regs.yhat.ravel() - y_true) ** 2)
        assert mse < 0.05

    def test_single_bandwidth_1d(self):
        """1D regression with single bandwidth works."""
        from fastlpr import cv_fastlpr

        np.random.seed(42)
        n = 100
        x = np.random.rand(n, 1)
        y = np.sin(2 * np.pi * x).ravel() + 0.1 * np.random.randn(n)

        h = np.array([[0.2]])
        opt = {'order': 1}
        regs = cv_fastlpr(x, y, h, opt)

        assert regs.yhat is not None
        assert len(regs.yhat) == n

    def test_linear_function_1d(self):
        """1D regression handles linear function correctly."""
        from fastlpr import cv_fastlpr, get_hlist

        np.random.seed(42)
        n = 150
        x = np.random.rand(n, 1)
        y_true = (2 * x + 1).ravel()
        y = y_true + 0.05 * np.random.randn(n)

        hlist = get_hlist(5, [0.2, 0.8])
        opt = {'order': 1}
        regs = cv_fastlpr(x, y, hlist, opt)

        mse = np.mean((regs.yhat.ravel() - y_true) ** 2)
        assert mse < 0.01

    def test_reproducibility_1d(self):
        """1D regression is reproducible."""
        from fastlpr import cv_fastlpr, get_hlist

        np.random.seed(42)
        n = 200
        x = np.random.rand(n, 1)
        y = np.sin(2 * np.pi * x).ravel() + 0.2 * np.random.randn(n)

        hlist = get_hlist(5, [0.05, 0.3])
        opt = {'order': 1}

        regs1 = cv_fastlpr(x, y, hlist, opt)
        regs2 = cv_fastlpr(x, y, hlist, opt)

        np.testing.assert_array_equal(regs1.yhat, regs2.yhat)

    def test_gcv_bandwidth_selection(self):
        """1D GCV bandwidth selection works (from MATLAB)."""
        from fastlpr import cv_fastlpr, get_hlist

        np.random.seed(42)
        n = 500
        x = np.random.rand(n, 1)
        y_true = np.sin(2 * np.pi * x).ravel()
        y = y_true + 0.2 * np.random.randn(n)

        hlist = get_hlist(20, [0.01, 0.5])
        opt = {'order': 1, 'calc_dof': True}
        regs = cv_fastlpr(x, y, hlist, opt)

        # Selected bandwidth should be in range
        assert min(hlist) <= regs.h <= max(hlist)
        # GCV scores should exist
        assert regs.gcv_yhat is not None

    def test_prediction_accuracy(self):
        """1D prediction accuracy MSE check (from MATLAB)."""
        from fastlpr import cv_fastlpr, get_hlist

        np.random.seed(42)
        n = 500
        x = np.random.rand(n, 1)
        y_true = np.sin(2 * np.pi * x).ravel()
        y = y_true + 0.1 * np.random.randn(n)

        hlist = get_hlist(15, [0.01, 0.3])
        opt = {'order': 1}
        regs = cv_fastlpr(x, y, hlist, opt)

        # MSE should be reasonably small
        mse = np.mean((regs.yhat.ravel() - y_true) ** 2)
        assert mse < 0.02  # Tight tolerance for good fit

    def test_order1_reduces_bias(self):
        """Order 1 reduces bias compared to order 0 (from R)."""
        from fastlpr import cv_fastlpr, get_hlist

        np.random.seed(42)
        n = 200
        x = np.random.rand(n, 1)
        y_true = np.sin(2 * np.pi * x).ravel()
        y = y_true + 0.05 * np.random.randn(n)

        hlist = get_hlist(5, [0.3, 0.8])

        # Order 0 (Nadaraya-Watson)
        opt0 = {'order': 0}
        regs0 = cv_fastlpr(x, y, hlist, opt0)
        mse0 = np.mean((regs0.yhat.ravel() - y_true) ** 2)

        # Order 1 (Local Linear)
        opt1 = {'order': 1}
        regs1 = cv_fastlpr(x, y, hlist, opt1)
        mse1 = np.mean((regs1.yhat.ravel() - y_true) ** 2)

        # Order 1 should have lower or similar MSE (less bias)
        assert mse1 <= mse0 * 1.2  # Allow 20% tolerance

    def test_boundary_behavior(self):
        """1D regression handles boundary correctly (from R)."""
        from fastlpr import cv_fastlpr, get_hlist

        np.random.seed(42)
        n = 200
        x = np.random.rand(n, 1)
        y_true = np.sin(2 * np.pi * x).ravel()
        y = y_true + 0.1 * np.random.randn(n)

        hlist = get_hlist(5, [0.1, 0.5])
        opt = {'order': 1}
        regs = cv_fastlpr(x, y, hlist, opt)

        # Check that fitted values don't have extreme outliers
        yhat_range = [regs.yhat.min(), regs.yhat.max()]
        y_range = [y.min(), y.max()]

        # Fitted values should be within reasonable range of observed data
        assert yhat_range[0] >= y_range[0] - 1
        assert yhat_range[1] <= y_range[1] + 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
