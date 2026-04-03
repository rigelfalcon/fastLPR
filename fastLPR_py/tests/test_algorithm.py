"""
test_algorithm.py - Compare fastLPR with naive Nadaraya-Watson implementation.

Matches MATLAB test_fastlpr_vs_naive_nw.m test cases.

Tests:
1. 1D regression with varying sample sizes
2. 2D regression
3. Complex-valued data

Validates:
- Numerical accuracy (MSE between fastLPR and naive NW < threshold)
- Algorithm correctness

Author: fastLPR Development Team
Copyright (c) 2020-2025 fastLPR Development Team
License: GNU General Public License v3.0
"""

import numpy as np
import pytest
from scipy.special import j1 as besselj1

from fastlpr import cv_fastlpr


# =============================================================================
# Helper: Naive Nadaraya-Watson Implementation
# =============================================================================

def naive_nw_smooth(x_train, y_train, h, x_eval=None):
    """
    Naive O(N*M) Nadaraya-Watson kernel regression.

    Parameters
    ----------
    x_train : array (N, d)
        Training points
    y_train : array (N,) or (N, dy)
        Training values
    h : float or array
        Bandwidth (scalar for isotropic, array for per-dimension)
    x_eval : array (M, d), optional
        Evaluation points. Default: x_train

    Returns
    -------
    yhat : array (M,) or (M, dy)
        Predicted values
    """
    x_train = np.atleast_2d(x_train)
    if x_train.shape[0] == 1:
        x_train = x_train.T
    y_train = np.atleast_1d(y_train)

    if x_eval is None:
        x_eval = x_train
    else:
        x_eval = np.atleast_2d(x_eval)
        if x_eval.shape[0] == 1:
            x_eval = x_eval.T

    n_train = x_train.shape[0]
    n_eval = x_eval.shape[0]
    n_dim = x_train.shape[1]

    h = np.atleast_1d(h)
    if len(h) == 1:
        h = np.repeat(h, n_dim)

    # Output shape (preserve dtype for complex support)
    dtype = y_train.dtype
    if y_train.ndim == 1:
        yhat = np.zeros(n_eval, dtype=dtype)
    else:
        yhat = np.zeros((n_eval, y_train.shape[1]), dtype=dtype)

    for i in range(n_eval):
        # Compute distances
        diff = (x_train - x_eval[i]) / h
        u2 = np.sum(diff ** 2, axis=1)

        # Gaussian kernel weights
        weights = np.exp(-0.5 * u2)

        # Normalize
        sum_weights = np.sum(weights)
        if sum_weights > 0:
            result = np.sum(weights[:, np.newaxis] * y_train.reshape(n_train, -1), axis=0) / sum_weights
            # Extract scalar for 1D output to avoid DeprecationWarning
            yhat[i] = result.item() if result.size == 1 else result
        else:
            yhat[i] = np.nan if yhat.ndim == 1 else np.full(yhat.shape[1], np.nan)

    return yhat.squeeze()


# =============================================================================
# Test: 1D Regression Accuracy
# =============================================================================

class TestFastLPRvsNaiveNW1D:
    """Test fastLPR against naive NW for 1D data."""

    @pytest.fixture
    def accuracy(self):
        return 6

    def test_1d_small(self, accuracy):
        """UNIT: 1D regression with small sample size (n=500)."""
        # Matches: MATLAB test_1d_small
        np.random.seed(42)
        n = 500
        x = np.abs(2 * (np.random.rand(n, 1) - 0.5) * 15)
        y_true = besselj1(x.flatten())
        y = y_true + 0.5 * np.std(y_true) * np.random.randn(n)
        h = 0.5

        # Naive NW
        yhat_naive = naive_nw_smooth(x, y, h)

        # fastLPR (h must be 2D array)
        opt = {
            'order': 0,
            'accuracy': accuracy,
            'calc_dof': False,
            'verbose': False,
        }
        regs = cv_fastlpr(x, y.reshape(-1, 1), np.array([[h]]), opt)
        yhat_fast = regs.yhat.flatten()

        # Verify accuracy (MSE should be small relative to signal variance)
        mse = np.mean((yhat_fast - yhat_naive) ** 2)
        expected_mse = 0.05  # NUFFT approximation vs naive O(N*M) allows some difference
        assert mse < expected_mse, f"MSE={mse:.2e} exceeds threshold {expected_mse:.2e} for n={n}"

    def test_1d_medium(self, accuracy):
        """UNIT: 1D regression with medium sample size (n=5000)."""
        # Matches: MATLAB test_1d_medium
        np.random.seed(42)
        n = 5000
        x = np.abs(2 * (np.random.rand(n, 1) - 0.5) * 15)
        y_true = besselj1(x.flatten())
        y = y_true + 0.5 * np.std(y_true) * np.random.randn(n)
        h = 0.5

        # Naive NW
        yhat_naive = naive_nw_smooth(x, y, h)

        # fastLPR (h must be 2D array)
        opt = {
            'order': 0,
            'accuracy': accuracy,
            'calc_dof': False,
            'verbose': False,
        }
        regs = cv_fastlpr(x, y.reshape(-1, 1), np.array([[h]]), opt)
        yhat_fast = regs.yhat.flatten()

        # Verify accuracy
        mse = np.mean((yhat_fast - yhat_naive) ** 2)
        expected_mse = 0.05  # NUFFT approximation vs naive O(N*M) allows some difference
        assert mse < expected_mse, f"MSE={mse:.2e} exceeds threshold {expected_mse:.2e} for n={n}"

    @pytest.mark.slow
    def test_1d_large(self, accuracy):
        """UNIT: 1D regression with large sample size (n=30000)."""
        # Matches: MATLAB test_1d_large
        np.random.seed(42)
        n = 30000
        x = np.abs(2 * (np.random.rand(n, 1) - 0.5) * 15)
        y_true = besselj1(x.flatten())
        y = y_true + 0.5 * np.std(y_true) * np.random.randn(n)
        h = 0.5

        # Naive NW (slower for large n)
        yhat_naive = naive_nw_smooth(x, y, h)

        # fastLPR (h must be 2D array)
        opt = {
            'order': 0,
            'accuracy': accuracy,
            'calc_dof': False,
            'verbose': False,
        }
        regs = cv_fastlpr(x, y.reshape(-1, 1), np.array([[h]]), opt)
        yhat_fast = regs.yhat.flatten()

        # Verify accuracy
        mse = np.mean((yhat_fast - yhat_naive) ** 2)
        expected_mse = 0.05  # NUFFT approximation vs naive O(N*M) allows some difference
        assert mse < expected_mse, f"MSE={mse:.2e} exceeds threshold {expected_mse:.2e} for n={n}"


# =============================================================================
# Test: 2D Regression Accuracy
# =============================================================================

class TestFastLPRvsNaiveNW2D:
    """Test fastLPR against naive NW for 2D data."""

    def test_2d_small(self):
        """UNIT: 2D regression with small sample size."""
        # Matches: MATLAB test_2d_small pattern
        np.random.seed(42)
        n = 400
        x = np.random.rand(n, 2) * 4 - 2  # [-2, 2]^2
        y_true = np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
        y = y_true + 0.2 * np.random.randn(n)
        h = [0.3, 0.3]

        # Naive NW
        yhat_naive = naive_nw_smooth(x, y, h)

        # fastLPR
        opt = {
            'order': 0,
            'accuracy': 6,
            'calc_dof': False,
            'verbose': False,
        }
        regs = cv_fastlpr(x, y.reshape(-1, 1), np.array([h]), opt)
        yhat_fast = regs.yhat.flatten()

        # Verify accuracy
        mse = np.mean((yhat_fast - yhat_naive) ** 2)
        expected_mse = 0.05  # NUFFT approximation vs naive O(N*M) allows some difference
        assert mse < expected_mse, f"MSE={mse:.2e} exceeds threshold {expected_mse:.2e}"

    def test_2d_medium(self):
        """UNIT: 2D regression with medium sample size."""
        np.random.seed(42)
        n = 2000
        x = np.random.rand(n, 2) * 4 - 2
        y_true = np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
        y = y_true + 0.2 * np.random.randn(n)
        h = [0.3, 0.3]

        # Naive NW
        yhat_naive = naive_nw_smooth(x, y, h)

        # fastLPR
        opt = {
            'order': 0,
            'accuracy': 6,
            'calc_dof': False,
            'verbose': False,
        }
        regs = cv_fastlpr(x, y.reshape(-1, 1), np.array([h]), opt)
        yhat_fast = regs.yhat.flatten()

        # Verify accuracy
        mse = np.mean((yhat_fast - yhat_naive) ** 2)
        expected_mse = 0.05  # NUFFT approximation vs naive O(N*M) allows some difference
        assert mse < expected_mse, f"MSE={mse:.2e} exceeds threshold {expected_mse:.2e}"

    @pytest.mark.slow
    def test_2d_large(self):
        """UNIT: 2D regression with large sample size (n=5000)."""
        # Matches: MATLAB test_2d_large
        np.random.seed(42)
        n = 5000
        x = np.random.rand(n, 2) * 4 - 2
        y_true = np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
        y = y_true + 0.2 * np.random.randn(n)
        h = [0.3, 0.3]

        # Naive NW (slower for large n)
        yhat_naive = naive_nw_smooth(x, y, h)

        # fastLPR
        opt = {
            'order': 0,
            'accuracy': 6,
            'calc_dof': False,
            'verbose': False,
        }
        regs = cv_fastlpr(x, y.reshape(-1, 1), np.array([h]), opt)
        yhat_fast = regs.yhat.flatten()

        # Verify accuracy
        mse = np.mean((yhat_fast - yhat_naive) ** 2)
        expected_mse = 0.05  # NUFFT approximation vs naive O(N*M) allows some difference
        assert mse < expected_mse, f"MSE={mse:.2e} exceeds threshold {expected_mse:.2e}"


# =============================================================================
# Test: Complex-valued Data
# =============================================================================

class TestFastLPRvsNaiveComplex:
    """Test fastLPR against naive NW for complex data."""

    def test_complex_1d(self):
        """UNIT: Complex-valued 1D regression."""
        # Matches: MATLAB test_complex pattern
        np.random.seed(42)
        n = 500
        x = np.random.rand(n, 1) * 2 * np.pi
        y_true = np.exp(1j * x.flatten())  # Complex signal
        noise = 0.1 * (np.random.randn(n) + 1j * np.random.randn(n))
        y = y_true + noise
        h = 0.3

        # Naive NW (works with complex data)
        yhat_naive = naive_nw_smooth(x, y, h)

        # fastLPR (h must be 2D array, y must be 2D with real/imag split)
        opt = {
            'order': 0,
            'accuracy': 9,  # Higher accuracy for complex
            'calc_dof': False,
            'verbose': False,
        }
        # For complex data, stack real and imaginary parts
        y_stack = np.column_stack([y.real, y.imag])
        regs = cv_fastlpr(x, y_stack, np.array([[h]]), opt)
        yhat_fast = regs.yhat[:, 0] + 1j * regs.yhat[:, 1]

        # Verify accuracy
        mse = np.mean(np.abs(yhat_fast - yhat_naive) ** 2)
        expected_mse = 0.05
        assert mse < expected_mse, f"Complex MSE={mse:.2e} exceeds threshold {expected_mse:.2e}"


# =============================================================================
# Test: Local Linear (Order 1) Consistency
# =============================================================================

class TestFastLPRConsistency:
    """Test fastLPR internal consistency."""

    def test_order0_vs_order1_smooth(self):
        """UNIT: Order 0 and 1 should give similar results for smooth data."""
        from fastlpr import get_hlist
        np.random.seed(42)
        n = 500
        x = np.random.rand(n, 1)
        y_true = np.sin(2 * np.pi * x.flatten())
        y = y_true + 0.1 * np.random.randn(n)
        # Use bandwidth list (cv_fastlpr needs multiple h for proper GCV selection)
        hlist = get_hlist(10, [0.1, 0.5])

        opt0 = {'order': 0, 'accuracy': 6, 'calc_dof': False}
        opt1 = {'order': 1, 'accuracy': 6, 'calc_dof': False}

        regs0 = cv_fastlpr(x, y.reshape(-1, 1), hlist, opt0)
        regs1 = cv_fastlpr(x, y.reshape(-1, 1), hlist, opt1)

        # Both should give reasonable fits
        assert np.isfinite(regs0.yhat).all(), "Order 0 should give finite values"
        assert np.isfinite(regs1.yhat).all(), "Order 1 should give finite values"

        # Correlation should be high
        corr = np.corrcoef(regs0.yhat.flatten(), regs1.yhat.flatten())[0, 1]
        assert corr > 0.9, f"Order 0 and 1 correlation {corr:.3f} should be > 0.9"

    def test_deterministic_output(self):
        """UNIT: Same inputs should give same outputs."""
        from fastlpr import get_hlist
        np.random.seed(42)
        n = 200
        x = np.random.rand(n, 1)
        y = np.sin(2 * np.pi * x.flatten()) + 0.1 * np.random.randn(n)
        hlist = get_hlist(10, [0.1, 0.5])

        opt = {'order': 1, 'accuracy': 6, 'calc_dof': False}

        regs1 = cv_fastlpr(x, y.reshape(-1, 1), hlist, opt)
        regs2 = cv_fastlpr(x, y.reshape(-1, 1), hlist, opt)

        np.testing.assert_allclose(regs1.yhat, regs2.yhat, rtol=1e-12,
                                   err_msg="Output should be deterministic")

    def test_increasing_accuracy_improves_precision(self):
        """UNIT: Higher accuracy should give more precise results."""
        from fastlpr import get_hlist
        np.random.seed(42)
        n = 500
        x = np.random.rand(n, 1)
        y = np.sin(2 * np.pi * x.flatten())  # No noise for precision test
        hlist = get_hlist(10, [0.1, 0.5])

        # Compute with different accuracy levels
        yhat = {}
        for acc in [4, 6, 9]:
            opt = {'order': 0, 'accuracy': acc, 'calc_dof': False}
            regs = cv_fastlpr(x, y.reshape(-1, 1), hlist, opt)
            yhat[acc] = regs.yhat.flatten()

        # Higher accuracy should generally be closer to highest accuracy result
        # Allow tolerance since NUFFT approximation may not be strictly monotonic
        err_4 = np.max(np.abs(yhat[4] - yhat[9]))
        err_6 = np.max(np.abs(yhat[6] - yhat[9]))

        # Either err_6 < err_4, or both are very small (< 0.001)
        assert err_6 < err_4 or (err_4 < 0.001 and err_6 < 0.001), \
            f"acc=6 ({err_6:.2e}) should be <= acc=4 ({err_4:.2e}) or both < 0.001"
