# ==============================================================================
# E2E Test: 2D Kernel Density Estimation
# ==============================================================================

import numpy as np
import pytest


class TestE2EKde2D:
    """End-to-end tests for 2D Kernel Density Estimation."""

    def test_kde_2d_normal(self):
        """2D KDE on bivariate normal data works."""
        # Unified with MATLAB: n=500, h=[0.1,1.0], nh=10x10
        from fastlpr import cv_fastkde, get_hlist

        np.random.seed(42)
        n = 500
        x = np.random.randn(n, 2)

        hlist = get_hlist([10, 10], [[0.1, 1.0], [0.1, 1.0]])
        opt = {'order': 0}
        kde = cv_fastkde(x, hlist, opt)

        assert kde.fhat is not None
        assert kde.grid is not None
        assert len(kde.h) == 2

    def test_kde_2d_correlated(self):
        """2D KDE on correlated data works."""
        # Unified with MATLAB: n=300, h=[0.1,1.0], nh=5x5
        from fastlpr import cv_fastkde, get_hlist

        np.random.seed(42)
        n = 300
        # Correlated 2D data
        x1 = np.random.randn(n)
        x2 = 0.8 * x1 + 0.6 * np.random.randn(n)
        x = np.column_stack([x1, x2])

        hlist = get_hlist([5, 5], [[0.1, 1.0], [0.1, 1.0]])
        opt = {'order': 0}
        kde = cv_fastkde(x, hlist, opt)

        assert kde.fhat is not None
        assert np.all(kde.fhat >= -1e-9)

    def test_kde_2d_single_bandwidth(self):
        """2D KDE with single bandwidth works."""
        # Unified with MATLAB: n=200, h=[0.5,0.5]
        from fastlpr import cv_fastkde

        np.random.seed(42)
        n = 200
        x = np.random.randn(n, 2)

        h = np.array([[0.5, 0.5]])
        opt = {'order': 0}
        kde = cv_fastkde(x, h, opt)

        assert kde.fhat is not None
        assert kde.grid is not None

    def test_kde_2d_anisotropic_bandwidth(self):
        """2D KDE with different bandwidth per dimension (from MATLAB)."""
        # Unified with MATLAB: n=500, h=[[0.1,0.5],[0.2,1.0]], nh=10x10
        from fastlpr import cv_fastkde, get_hlist

        np.random.seed(42)
        n = 500
        # Anisotropic data: different scales in each dimension
        x = np.column_stack([np.random.randn(n) * 0.5, np.random.randn(n) * 2])

        # Different bandwidth ranges for each dimension
        hlist = get_hlist([10, 10], [[0.1, 0.5], [0.2, 1.0]])
        opt = {'order': 0}
        kde = cv_fastkde(x, hlist, opt)

        assert kde.fhat is not None
        assert len(kde.h) == 2

    def test_kde_2d_bandwidth_selection(self):
        """2D KDE LCV bandwidth selection works (from MATLAB)."""
        # Unified with MATLAB: n=200 (mixture), h=[0.1,1.0], nh=5x5
        from fastlpr import cv_fastkde, get_hlist

        np.random.seed(42)
        # Mixture of Gaussians
        n1, n2 = 100, 100
        x1 = np.column_stack([np.random.randn(n1) - 1, np.random.randn(n1) - 1]) * 0.3
        x2 = np.column_stack([np.random.randn(n2) + 1, np.random.randn(n2) + 1]) * 0.5
        x = np.vstack([x1, x2])

        hlist = get_hlist([5, 5], [[0.1, 1.0], [0.1, 1.0]])
        opt = {'order': 0}
        kde = cv_fastkde(x, hlist, opt)

        # Selected bandwidths should be in range
        assert 0.1 <= kde.h[0] <= 1.0
        assert 0.1 <= kde.h[1] <= 1.0
        assert kde.lcv is not None

    def test_kde_2d_density_integrates_to_1(self):
        """2D KDE density integrates to approximately 1 (from MATLAB)."""
        # Unified with MATLAB: n=500, h=[0.3,0.6], nh=3x3, N=[50,50]
        from fastlpr import cv_fastkde, get_hlist

        np.random.seed(42)
        n = 500
        x = np.random.randn(n, 2)

        hlist = get_hlist([3, 3], [[0.3, 0.6], [0.3, 0.6]])
        opt = {'order': 0, 'N': [50, 50]}
        kde = cv_fastkde(x, hlist, opt)

        # 2D numerical integration using trapezoidal rule
        grid_x, grid_y = kde.grid
        dx = grid_x[1] - grid_x[0]
        dy = grid_y[1] - grid_y[0]
        integral = np.trapezoid(np.trapezoid(kde.fhat, dx=dy, axis=1), dx=dx)

        # Integral should be close to 1 (within 15% tolerance for 2D)
        assert 0.85 <= integral <= 1.15

    def test_kde_2d_density_positive(self):
        """2D KDE density values should be non-negative (from MATLAB)."""
        # Unified with MATLAB: n=300, h=[0.3,0.3]
        from fastlpr import cv_fastkde

        np.random.seed(42)
        n = 300
        x = np.random.randn(n, 2)

        h = np.array([[0.3, 0.3]])
        kde = cv_fastkde(x, h)

        # Allow small negative values due to numerical precision (NUFFT grid artifacts)
        assert np.all(kde.fhat > -1e-9), "Density should be non-negative"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
