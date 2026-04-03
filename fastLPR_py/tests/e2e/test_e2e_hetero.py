# ==============================================================================
# E2E Test: Heteroscedastic Variance Estimation
# ==============================================================================

import numpy as np
import pytest

# Note: Numba is required for performance. Tests run with Numba backend.


class TestE2EHetero:
    """End-to-end tests for heteroscedastic variance estimation."""

    def test_hetero_1d_mean_and_variance(self):
        """1D can estimate both mean and variance using two-step workflow.

        Correct workflow for heteroscedastic variance estimation:
        1. Estimate mean with y_type_out='mean'
        2. Compute squared residuals: residuals_sq = (y - yhat_mean)^2
        3. Estimate variance with y_type_out='variance' using residuals_sq as input
        """
        # Unified with MATLAB: n=500, h=[0.01,0.5], nh=10
        from fastlpr import cv_fastlpr, get_hlist

        np.random.seed(42)
        n = 500
        x = np.sort(np.random.rand(n, 1), axis=0)
        y_true = np.sin(2 * np.pi * x).ravel()
        var_true = 0.1 + 0.3 * x.ravel() ** 2
        y = y_true + np.sqrt(var_true) * np.random.randn(n)

        hlist = get_hlist(10, [0.01, 0.5])

        # Step 1: Mean estimation
        opt_mean = {'order': 1, 'y_type_out': 'mean', 'calc_dof': True}
        regs_mean = cv_fastlpr(x, y, hlist, opt_mean)

        # Step 2: Compute squared residuals (always positive)
        residuals_sq = (y.reshape(-1, 1) - regs_mean.yhat.reshape(-1, 1)) ** 2

        # Step 3: Variance estimation using squared residuals
        opt_var = {'order': 1, 'y_type_out': 'variance'}
        regs_var = cv_fastlpr(x, residuals_sq, hlist, opt_var)

        # Both should work
        assert regs_mean.yhat is not None
        assert regs_var.yhat is not None
        assert len(regs_mean.yhat) == n
        # Variance should be positive
        assert np.all(regs_var.yhat >= 0)

    def test_hetero_2d_variance(self):
        """2D heteroscedastic variance estimation using two-step workflow."""
        # Unified with MATLAB: n=400, h=[0.05,0.5], nh=5x5
        from fastlpr import cv_fastlpr, get_hlist

        np.random.seed(42)
        n = 400
        x = np.random.rand(n, 2)
        y_true = np.sin(2 * np.pi * x[:, 0]) * np.cos(2 * np.pi * x[:, 1])
        var_true = 0.1 + 0.2 * x[:, 0] ** 2 + 0.2 * x[:, 1] ** 2
        y = y_true + np.sqrt(var_true) * np.random.randn(n)

        hlist = get_hlist([5, 5], [[0.05, 0.5], [0.05, 0.5]])

        # Step 1: Mean estimation
        opt_mean = {'order': 1, 'y_type_out': 'mean'}
        regs_mean = cv_fastlpr(x, y, hlist, opt_mean)

        # Step 2: Compute squared residuals
        residuals_sq = (y.reshape(-1, 1) - regs_mean.yhat.reshape(-1, 1)) ** 2

        # Step 3: Variance estimation
        opt_var = {'order': 1, 'y_type_out': 'variance'}
        regs_var = cv_fastlpr(x, residuals_sq, hlist, opt_var)

        assert regs_var.yhat is not None
        assert len(regs_var.yhat) == n
        assert np.all(regs_var.yhat >= 0)

    def test_hetero_reproducibility(self):
        """Heteroscedastic estimation is reproducible using two-step workflow."""
        # Unified with MATLAB: n=200, h=[0.1,0.5], nh=5
        from fastlpr import cv_fastlpr, get_hlist

        np.random.seed(42)
        n = 200
        x = np.sort(np.random.rand(n, 1), axis=0)
        var_true = 0.1 + 0.3 * x.ravel() ** 2
        y = np.sin(2 * np.pi * x).ravel() + np.sqrt(var_true) * np.random.randn(n)

        hlist = get_hlist(5, [0.1, 0.5])

        # Step 1: Mean estimation (run twice)
        opt_mean = {'order': 1, 'y_type_out': 'mean'}
        regs_mean1 = cv_fastlpr(x, y, hlist, opt_mean)
        regs_mean2 = cv_fastlpr(x, y, hlist, opt_mean)

        # Verify mean estimation is reproducible
        np.testing.assert_array_equal(regs_mean1.yhat, regs_mean2.yhat)

        # Step 2: Compute squared residuals
        residuals_sq = (y.reshape(-1, 1) - regs_mean1.yhat.reshape(-1, 1)) ** 2

        # Step 3: Variance estimation (run twice)
        opt_var = {'order': 1, 'y_type_out': 'variance'}
        regs_var1 = cv_fastlpr(x, residuals_sq, hlist, opt_var)
        regs_var2 = cv_fastlpr(x, residuals_sq, hlist, opt_var)

        # Verify variance estimation is reproducible
        np.testing.assert_array_equal(regs_var1.yhat, regs_var2.yhat)

    def test_hetero_variance_positive(self):
        """Heteroscedastic variance estimates are non-negative (from MATLAB)."""
        # Unified with MATLAB: n=500, h=[0.02,0.4], nh=10
        from fastlpr import cv_fastlpr, get_hlist

        np.random.seed(42)
        n = 500
        x = np.sort(np.random.rand(n, 1), axis=0)
        y_true = x.ravel() ** 2
        var_true = 0.1 + 0.5 * x.ravel()
        y = y_true + np.sqrt(var_true) * np.random.randn(n)

        hlist = get_hlist(10, [0.02, 0.4])

        # Step 1: Mean estimation
        opt_mean = {'order': 1, 'y_type_out': 'mean'}
        regs_mean = cv_fastlpr(x, y, hlist, opt_mean)

        # Step 2: Compute squared residuals
        residuals_sq = (y.reshape(-1, 1) - regs_mean.yhat.reshape(-1, 1)) ** 2

        # Step 3: Variance estimation
        opt_var = {'order': 1, 'y_type_out': 'variance'}
        regs_var = cv_fastlpr(x, residuals_sq, hlist, opt_var)

        # Variance should be non-negative
        assert np.all(regs_var.yhat >= 0)

    def test_hetero_mean_accuracy(self):
        """Heteroscedastic mean estimation MSE check (from MATLAB)."""
        # Unified with MATLAB: n=500, h=[0.01,0.3], nh=15
        from fastlpr import cv_fastlpr, get_hlist

        np.random.seed(42)
        n = 500
        x = np.sort(np.random.rand(n, 1), axis=0)
        y_true = np.sin(2 * np.pi * x).ravel()
        var_true = 0.1 + 0.3 * x.ravel() ** 2
        y = y_true + np.sqrt(var_true) * np.random.randn(n)

        hlist = get_hlist(15, [0.01, 0.3])

        # Mean estimation
        opt_mean = {'order': 1, 'y_type_out': 'mean'}
        regs_mean = cv_fastlpr(x, y, hlist, opt_mean)

        # MSE should be reasonably small despite heteroscedasticity
        mse = np.mean((regs_mean.yhat.ravel() - y_true) ** 2)
        assert mse < 0.05  # Reasonable tolerance for heteroscedastic data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
