# ==============================================================================
# E2E Test: 1D Kernel Density Estimation
# ==============================================================================

import numpy as np
import pytest


class TestE2EKde1D:
    """End-to-end tests for 1D Kernel Density Estimation."""

    def test_kde_1d_normal(self):
        """1D KDE on standard normal data works."""
        # Unified with MATLAB: n=500, h=[0.05,1.0], nh=10
        from fastlpr import cv_fastkde, get_hlist

        np.random.seed(42)
        n = 500
        x = np.random.randn(n, 1)

        hlist = get_hlist(10, [0.05, 1.0])
        opt = {'order': 0}
        kde = cv_fastkde(x, hlist, opt)

        assert kde.fhat is not None
        assert len(kde.fhat) == n
        # Density should be positive
        assert np.all(kde.fhat >= 0)

    def test_kde_1d_bimodal(self):
        """1D KDE on bimodal data works."""
        # Unified with MATLAB: n=400, h=[0.1,1.5], nh=10
        from fastlpr import cv_fastkde, get_hlist

        np.random.seed(42)
        n = 400
        x1 = np.random.randn(n // 2, 1) - 2
        x2 = np.random.randn(n // 2, 1) + 2
        x = np.vstack([x1, x2])

        hlist = get_hlist(10, [0.1, 1.5])
        opt = {'order': 0}
        kde = cv_fastkde(x, hlist, opt)

        assert kde.fhat is not None
        assert np.all(kde.fhat >= 0)

    def test_kde_1d_single_bandwidth(self):
        """1D KDE with single bandwidth works."""
        # Unified with MATLAB: n=200, h=0.5
        from fastlpr import cv_fastkde

        np.random.seed(42)
        n = 200
        x = np.random.randn(n, 1)

        h = np.array([[0.5]])
        opt = {'order': 0}
        kde = cv_fastkde(x, h, opt)

        assert kde.fhat is not None
        assert len(kde.fhat) == n

    def test_kde_1d_lcv_selection(self):
        """1D KDE LCV bandwidth selection works."""
        # Unified with MATLAB: n=500, h=[0.05,1.0], nh=20
        from fastlpr import cv_fastkde, get_hlist

        np.random.seed(42)
        n = 500
        x = np.random.randn(n, 1)

        hlist = get_hlist(20, [0.05, 1.0])
        opt = {'order': 0}
        kde = cv_fastkde(x, hlist, opt)

        # Selected bandwidth should be in range
        assert min(hlist) <= kde.h <= max(hlist)

    def test_kde_1d_reproducibility(self):
        """1D KDE is deterministic (reproducibility check from MATLAB)."""
        # Unified with MATLAB: n=200, h=[0.1,0.5], nh=5
        from fastlpr import cv_fastkde, get_hlist

        np.random.seed(42)
        n = 200
        x = np.random.randn(n, 1)

        hlist = get_hlist(5, [0.1, 0.5])
        opt = {'order': 0}

        kde1 = cv_fastkde(x, hlist, opt)
        kde2 = cv_fastkde(x, hlist, opt)

        np.testing.assert_array_equal(kde1.fhat, kde2.fhat)
        np.testing.assert_array_equal(kde1.h, kde2.h)

    def test_kde_1d_density_integrates_to_1(self):
        """1D KDE density integrates to approximately 1 (from MATLAB)."""
        # Unified with MATLAB: n=500, h=[0.2,0.5], nh=5, N=200
        from fastlpr import cv_fastkde, get_hlist

        np.random.seed(42)
        n = 500
        x = np.random.randn(n, 1)

        hlist = get_hlist(5, [0.2, 0.5])
        opt = {'order': 0, 'N': 200}
        kde = cv_fastkde(x, hlist, opt)

        # Numerical integration using trapezoidal rule
        grid = kde.grid[0]
        dx = grid[1] - grid[0]
        integral = np.trapezoid(kde.fhat, dx=dx)

        # Integral should be close to 1 (within 10% tolerance)
        assert 0.9 <= integral <= 1.1

    def test_kde_1d_uniform_distribution(self):
        """1D KDE on uniform data works (from MATLAB)."""
        # Unified with MATLAB: n=200, h=[0.05,0.2], nh=5
        from fastlpr import cv_fastkde, get_hlist

        np.random.seed(42)
        n = 200
        x = np.random.rand(n, 1)  # Uniform [0, 1]

        hlist = get_hlist(5, [0.05, 0.2])
        opt = {'order': 0}
        kde = cv_fastkde(x, hlist, opt)

        assert kde.fhat is not None
        assert np.all(kde.fhat >= 0)
        # Density should be roughly uniform (close to 1)
        fhat_mean = np.mean(kde.fhat)
        assert abs(fhat_mean - 1.0) < 0.3

    def test_kde_1d_density_positive(self):
        """1D KDE density values should be positive (from MATLAB)."""
        # Unified with MATLAB: n=500, h=0.2
        from fastlpr import cv_fastkde

        np.random.seed(42)
        n = 500
        x = np.random.randn(n, 1)

        h = np.array([[0.2]])
        kde = cv_fastkde(x, h)

        # Density should be strictly positive
        assert np.all(kde.fhat > 0), "Density should be positive"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
