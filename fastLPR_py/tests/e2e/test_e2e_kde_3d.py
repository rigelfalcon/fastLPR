# ==============================================================================
# E2E Test: 3D Kernel Density Estimation
# ==============================================================================

import numpy as np
import pytest


class TestE2EKde3D:
    """End-to-end tests for 3D Kernel Density Estimation."""

    def test_kde_3d_normal(self):
        """3D KDE - single bandwidth, no LCV search."""
        from fastlpr import cv_fastkde

        np.random.seed(42)
        n = 300
        x = np.random.randn(n, 3)

        h = np.array([[0.5, 0.5, 0.5]])  # Single bandwidth
        opt = {'order': 0, 'N': [20, 20, 20]}
        kde = cv_fastkde(x, h, opt)

        assert kde.fhat is not None
        assert kde.grid is not None
        assert kde.fhat.shape == tuple(len(g) for g in kde.grid)
        assert np.all(kde.fhat >= 0)

    # ARCHIVED: 2026-01-10 - "test_kde_3d_single_bandwidth" (moved to dev/archive/tests-archive-20260110/python/e2e/archived_test_e2e_kde_3d.py)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
