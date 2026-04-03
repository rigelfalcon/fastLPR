# ==============================================================================
# E2E Test: 2D Local Polynomial Regression
# ==============================================================================

import numpy as np
import pytest


class TestE2ELpr2D:
    """End-to-end tests for 2D Local Polynomial Regression."""

    def test_order0_2d(self):
        """2D Nadaraya-Watson (order 0) regression works."""
        # Unified with MATLAB: n=500, h=[0.05,0.5], nh=10x10, noise=0.2
        from fastlpr import cv_fastlpr, get_hlist

        np.random.seed(42)
        n = 500
        x = np.random.rand(n, 2)
        y_true = np.sin(2 * np.pi * x[:, 0]) * np.cos(2 * np.pi * x[:, 1])
        y = y_true + 0.2 * np.random.randn(n)

        hlist = get_hlist([10, 10], [[0.05, 0.5], [0.05, 0.5]])
        opt = {'order': 0, 'calc_dof': True}
        regs = cv_fastlpr(x, y, hlist, opt)

        assert regs.yhat is not None
        assert len(regs.yhat) == n
        assert len(regs.h) == 2

    def test_order1_2d(self):
        """2D Local Linear (order 1) regression works."""
        # Unified with MATLAB: n=500, h=[0.05,0.5], nh=10x10, noise=0.2
        from fastlpr import cv_fastlpr, get_hlist

        np.random.seed(42)
        n = 500
        x = np.random.rand(n, 2)
        y_true = np.sin(2 * np.pi * x[:, 0]) * np.cos(2 * np.pi * x[:, 1])
        y = y_true + 0.2 * np.random.randn(n)

        hlist = get_hlist([10, 10], [[0.05, 0.5], [0.05, 0.5]])
        opt = {'order': 1, 'calc_dof': True}
        regs = cv_fastlpr(x, y, hlist, opt)

        assert regs.yhat is not None
        assert len(regs.h) == 2

    def test_order2_2d(self):
        """2D Local Quadratic (order 2) regression works."""
        # Unified with MATLAB: n=500, h=[0.05,0.5], nh=10x10, noise=0.2
        from fastlpr import cv_fastlpr, get_hlist

        np.random.seed(42)
        n = 500
        x = np.random.rand(n, 2)
        y_true = np.sin(2 * np.pi * x[:, 0]) * np.cos(2 * np.pi * x[:, 1])
        y = y_true + 0.2 * np.random.randn(n)

        hlist = get_hlist([10, 10], [[0.05, 0.5], [0.05, 0.5]])
        opt = {'order': 2, 'calc_dof': True}
        regs = cv_fastlpr(x, y, hlist, opt)

        assert regs.yhat is not None

    def test_anisotropic_2d(self):
        """2D regression with anisotropic data works."""
        # Unified with MATLAB: n=500, h=[[0.02,0.3],[0.1,0.8]], nh=10x10, noise=0.1
        from fastlpr import cv_fastlpr, get_hlist

        np.random.seed(42)
        n = 500
        x = np.column_stack([np.random.rand(n) * 0.5, np.random.rand(n) * 2])
        y = x[:, 0] ** 2 + x[:, 1] + 0.1 * np.random.randn(n)

        hlist = get_hlist([10, 10], [[0.02, 0.3], [0.1, 0.8]])
        opt = {'order': 1}
        regs = cv_fastlpr(x, y, hlist, opt)

        assert regs.yhat is not None

    def test_gcv_bandwidth_selection(self):
        """2D GCV bandwidth selection works (from MATLAB)."""
        # Unified with MATLAB: n=400, h=[0.1,0.6], nh=5x5, noise=0.1, N=[50,50]
        from fastlpr import cv_fastlpr, get_hlist

        np.random.seed(42)
        n = 400
        x = 2 * np.random.rand(n, 2) - 1  # [-1, 1]^2
        y_true = np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
        y = y_true + 0.1 * np.random.randn(n)

        hlist = get_hlist([5, 5], [[0.1, 0.6], [0.1, 0.6]])
        opt = {'order': 1, 'N': [50, 50]}
        regs = cv_fastlpr(x, y, hlist, opt)

        # Selected bandwidths should be in range
        assert 0.1 <= regs.h[0] <= 0.6
        assert 0.1 <= regs.h[1] <= 0.6
        # GCV scores should exist
        assert regs.gcv is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
