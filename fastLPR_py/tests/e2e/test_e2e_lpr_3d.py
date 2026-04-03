# ==============================================================================
# E2E Test: 3D Local Polynomial Regression
# ==============================================================================

# ARCHIVED: 2026-01-10 - test_order0_3d (moved to dev/archive/tests-archive-20260110/python/e2e/archived_test_e2e_lpr_3d.py)
# ARCHIVED: 2026-01-10 - test_single_bandwidth_3d (moved to dev/archive/tests-archive-20260110/python/e2e/archived_test_e2e_lpr_3d.py)
# ARCHIVED: 2026-01-10 - test_small_sample_3d (moved to dev/archive/tests-archive-20260110/python/e2e/archived_test_e2e_lpr_3d.py)

import numpy as np
import pytest


class TestE2ELpr3D:
    """End-to-end tests for 3D Local Polynomial Regression."""

    def test_order1_3d(self):
        """3D Local Linear (order 1) - single bandwidth, no GCV."""
        from fastlpr import cv_fastlpr

        np.random.seed(42)
        n = 300
        x = np.random.rand(n, 3)  # [0, 1] range like MATLAB
        y_true = np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1]) * np.sin(np.pi * x[:, 2])
        y = y_true + 0.1 * np.random.randn(n)

        h = np.array([[0.3, 0.3, 0.3]])  # Single bandwidth
        opt = {'order': 1, 'N': [20, 20, 20], 'calc_dof': False}
        regs = cv_fastlpr(x, y, h, opt)

        assert regs.yhat is not None
        assert len(regs.yhat) == n


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
