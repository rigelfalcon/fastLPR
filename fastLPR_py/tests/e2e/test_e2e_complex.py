# ==============================================================================
# E2E Test: Complex-Valued Regression
# ==============================================================================

import numpy as np
import pytest


class TestE2EComplex:
    """End-to-end tests for complex-valued regression."""

    def test_complex_1d_order1(self):
        """1D complex-valued regression (order 1) works."""
        # Unified with MATLAB: n=500, h=[0.01,0.5], nh=10, noise=0.2
        from fastlpr import cv_fastlpr, get_hlist

        np.random.seed(42)
        n = 500
        x = np.sort(np.random.rand(n, 1), axis=0)
        y_real = np.sin(2 * np.pi * x).ravel()
        y_imag = np.cos(2 * np.pi * x).ravel()
        y_true = y_real + 1j * y_imag
        noise = 0.2 * (np.random.randn(n) + 1j * np.random.randn(n))
        y = y_true + noise

        hlist = get_hlist(10, [0.01, 0.5])
        opt = {'order': 1, 'calc_dof': True}
        regs = cv_fastlpr(x, y, hlist, opt)

        assert regs.yhat is not None
        assert len(regs.yhat) == n
        assert np.iscomplexobj(regs.yhat)

    def test_complex_1d_correlation(self):
        """1D complex regression preserves correlation."""
        # Unified with MATLAB: n=200, h=[0.05,0.5], nh=5, noise=0.1
        from fastlpr import cv_fastlpr, get_hlist

        np.random.seed(42)
        n = 200
        x = np.sort(np.random.rand(n, 1), axis=0)
        y_real = np.sin(2 * np.pi * x).ravel()
        y_imag = np.cos(2 * np.pi * x).ravel()
        y_true = y_real + 1j * y_imag
        noise = 0.1 * (np.random.randn(n) + 1j * np.random.randn(n))
        y = y_true + noise

        hlist = get_hlist(5, [0.05, 0.5])
        opt = {'order': 1}
        regs = cv_fastlpr(x, y, hlist, opt)

        # Check real and imaginary parts separately
        corr_real = np.corrcoef(np.real(regs.yhat).ravel(), y_real)[0, 1]
        corr_imag = np.corrcoef(np.imag(regs.yhat).ravel(), y_imag)[0, 1]

        assert corr_real > 0.9
        assert corr_imag > 0.9

    def test_complex_single_bandwidth(self):
        """Complex regression with single bandwidth works."""
        # Unified with MATLAB: n=100, h=0.15, noise=0.1
        from fastlpr import cv_fastlpr

        np.random.seed(42)
        n = 100
        x = np.sort(np.random.rand(n, 1), axis=0)
        y = np.sin(2 * np.pi * x).ravel() + 1j * np.cos(2 * np.pi * x).ravel()
        y = y + 0.1 * (np.random.randn(n) + 1j * np.random.randn(n))

        h = np.array([[0.15]])
        opt = {'order': 1}
        regs = cv_fastlpr(x, y, h, opt)

        assert regs.yhat is not None
        assert np.iscomplexobj(regs.yhat)

    def test_complex_bandwidth_selection(self):
        """Complex GCV bandwidth selection works (from MATLAB)."""
        # Unified with MATLAB: n=500, h=[0.01,0.5], nh=15, noise=0.1
        from fastlpr import cv_fastlpr, get_hlist

        np.random.seed(42)
        n = 500
        x = np.sort(np.random.rand(n, 1), axis=0)
        y = np.exp(1j * 2 * np.pi * x).ravel() + 0.1 * (np.random.randn(n) + 1j * np.random.randn(n))

        hlist = get_hlist(15, [0.01, 0.5])
        opt = {'order': 1}
        regs = cv_fastlpr(x, y, hlist, opt)

        # Selected bandwidth should be in range
        assert min(hlist) <= regs.h <= max(hlist)
        # GCV scores should exist
        assert regs.gcv_yhat is not None

    def test_complex_gcv_real(self):
        """Complex GCV scores are real-valued (from MATLAB)."""
        # Unified with MATLAB: n=500, h=[0.01,0.5], nh=10, noise=0.1
        from fastlpr import cv_fastlpr, get_hlist

        np.random.seed(42)
        n = 500
        x = np.sort(np.random.rand(n, 1), axis=0)
        y = np.sin(2 * np.pi * x).ravel() + 1j * np.cos(2 * np.pi * x).ravel()
        y = y + 0.1 * (np.random.randn(n) + 1j * np.random.randn(n))

        hlist = get_hlist(10, [0.01, 0.5])
        opt = {'order': 1, 'calc_dof': True}
        regs = cv_fastlpr(x, y, hlist, opt)

        # GCV scores should be real (not complex)
        assert regs.gcv_yhat is not None
        gcv_scores = regs.gcv_yhat['gcv_m']
        assert np.all(np.isreal(gcv_scores))
        # GCV scores should be positive
        assert np.all(gcv_scores >= 0)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
