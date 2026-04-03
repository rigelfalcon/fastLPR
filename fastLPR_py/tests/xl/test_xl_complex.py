# ==============================================================================
# Cross-Language Test: XL-08 - Complex-valued Regression vs MATLAB Reference
# ==============================================================================
# Test file: fastLPR_py/tests/xl/test_xl_complex.py
# Created: 2025-01-07
#
# This test validates the Python cv_fastlpr() implementation against MATLAB
# reference data for complex-valued regression.
#
# Reference file: fastLPR/tests/refs/crosslang_e2e/ref_complex.mat
# Contains:
#   - x: Input data (N x 1 matrix)
#   - y: Complex response data (N x 1 complex matrix)
#   - hlist: Bandwidth candidates (k x 1 matrix)
#   - gcv_m: GCV scores for all bandwidths (k x 1 matrix)
#   - h1se: Selected bandwidth using 1-SE rule (scalar)
#   - id1se: Index of selected bandwidth (1-indexed, MATLAB)
#   - yhat_mean: Fitted values at data points (N x 1 complex matrix)
#   - pdof_m: Pseudo-degrees of freedom (k x 1 matrix)
#   - elapsed: MATLAB execution time (scalar)
#
# Pass Criteria (per CLAUDE.md):
#   - BW MaxErr < 0.02 (bandwidth selection threshold)
#   - GCV MaxErr < 0.01 (GCV score absolute error)
#   - Mean MaxErr < 0.05 (prediction threshold)
#   - Real/Imag MaxErr < 0.05 (separate component errors)
#   - Speed ratio < 30x MATLAB time (relaxed for pure Python path)
# ==============================================================================

import time
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

# Import tolerance constants from conftest
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import TOL_CROSSLANG, assert_gcv_fields


def _get_project_root() -> Path:
    """Get project root (jss-code) from xl test directory."""
    return Path(__file__).parent.parent.parent.parent


def get_ref_path() -> Path:
    """Get path to MATLAB reference file."""
    return _get_project_root() / "fastLPR" / "tests" / "refs" / "crosslang_e2e" / "ref_complex.mat"


def load_matlab_ref():
    """Load MATLAB reference data for complex test."""
    ref_path = get_ref_path()
    if not ref_path.exists():
        return None
    return sio.loadmat(str(ref_path), squeeze_me=False)


class TestXLComplex:
    """Cross-language tests for Complex-valued Regression."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Skip if MATLAB reference data not available."""
        ref_path = get_ref_path()
        if not ref_path.exists():
            pytest.skip(f"MATLAB reference file not found: {ref_path}")

    def test_complex_matches_matlab(self):
        """Complex-valued regression matches MATLAB reference data."""
        from fastlpr import cv_fastlpr

        # ======================================================================
        # Load MATLAB reference data
        # ======================================================================
        ref = load_matlab_ref()
        if ref is None:
            pytest.skip("MATLAB reference file not found")

        # Extract reference data
        x_ref = np.asarray(ref['x']).reshape(-1, 1)  # (n, 1)
        y_ref = np.asarray(ref['y']).ravel()  # (n,) - should be complex
        hlist_ref = np.asarray(ref['hlist']).reshape(-1, 1)  # (dh, 1)
        gcv_m_ref = np.asarray(ref['gcv_m']).ravel()  # (dh,)
        h1se_ref = float(np.asarray(ref['h1se']).ravel()[0])
        id1se_ref = int(np.asarray(ref['id1se']).ravel()[0])  # 1-indexed
        yhat_mean_ref = np.asarray(ref['yhat_mean']).ravel()  # (n,) - complex
        matlab_time = float(np.asarray(ref['elapsed']).ravel()[0])

        n = x_ref.shape[0]
        dh = hlist_ref.shape[0]

        # Verify y is complex
        assert np.iscomplexobj(y_ref), "y should be complex-valued"

        print(f"\n=== XL-08: Complex-valued Regression vs MATLAB ===")
        print(f"Reference: ref_complex.mat")
        print(f"Sample size (N): {n}")
        print(f"Number of bandwidths: {dh}")
        print(f"MATLAB time: {matlab_time:.3f} s")
        print(f"MATLAB selected h: {h1se_ref:.6f} (id={id1se_ref})")

        # ======================================================================
        # Run Python implementation
        # ======================================================================
        options = {
            'order': 1,  # Local linear
            'calc_dof': True,
        }

        start_time = time.perf_counter()
        res = cv_fastlpr(x_ref, y_ref, h=hlist_ref, options=options)
        python_time = time.perf_counter() - start_time

        assert_gcv_fields(res.gcv)

        # Extract Python results
        h1se_py = float(np.asarray(res.h).item())
        id1se_py = int(res.gcv['id1se'])  # 0-indexed
        gcv_m_py = np.asarray(res.gcv['gcv_m']).ravel()
        yhat_py = np.asarray(res.yhat).ravel()

        print(f"\n--- Python Results ---")
        print(f"Python time: {python_time:.3f} s")
        print(f"Python selected h: {h1se_py:.6f} (id={id1se_py + 1})")

        # Verify predictions are complex
        assert np.iscomplexobj(yhat_py), "Python predictions should be complex"

        # ======================================================================
        # Compute error metrics
        # ======================================================================

        # Bandwidth selection error
        bw_maxerr = abs(h1se_py - h1se_ref)

        # Bandwidth index difference
        bw_idx_diff = abs(id1se_py - (id1se_ref - 1))

        # GCV absolute error (GCV should be real for complex data)
        gcv_m_py_real = np.real(gcv_m_py) if np.iscomplexobj(gcv_m_py) else gcv_m_py
        gcv_m_ref_real = np.real(gcv_m_ref) if np.iscomplexobj(gcv_m_ref) else gcv_m_ref
        gcv_maxerr = float(np.max(np.abs(gcv_m_py_real - gcv_m_ref_real)))

        # Complex prediction error (using modulus)
        yhat_diff = yhat_py - yhat_mean_ref
        mean_maxerr = float(np.max(np.abs(yhat_diff)))
        mean_mse = float(np.mean(np.abs(yhat_diff) ** 2))

        # Separate real and imaginary part errors
        real_maxerr = float(np.max(np.abs(np.real(yhat_py) - np.real(yhat_mean_ref))))
        imag_maxerr = float(np.max(np.abs(np.imag(yhat_py) - np.imag(yhat_mean_ref))))

        # Speed ratio (relaxed for pure Python path with large N)
        speed_ratio = python_time / matlab_time if matlab_time > 0 else float('inf')

        print(f"\n--- Error Metrics ---")
        print(f"BW MaxErr: {bw_maxerr:.2e} (threshold: {TOL_CROSSLANG['bw_maxerr']})")
        print(f"BW Index Diff: {bw_idx_diff}")
        print(f"GCV MaxErr: {gcv_maxerr:.2e} (threshold: 0.01)")
        print(f"Mean MaxErr: {mean_maxerr:.2e} (threshold: {TOL_CROSSLANG['mean_maxerr']})")
        print(f"Real MaxErr: {real_maxerr:.2e}")
        print(f"Imag MaxErr: {imag_maxerr:.2e}")
        print(f"Mean MSE: {mean_mse:.2e}")

        print(f"\n--- Performance ---")
        print(f"Speed ratio: {speed_ratio:.2f}x (threshold: 30x for pure Python)")

        # ======================================================================
        # Determine pass/fail
        # ======================================================================
        bw_pass = bw_maxerr < TOL_CROSSLANG['bw_maxerr']
        gcv_pass = gcv_maxerr < 0.01
        mean_pass = mean_maxerr < TOL_CROSSLANG['mean_maxerr']
        real_pass = real_maxerr < TOL_CROSSLANG['mean_maxerr']
        imag_pass = imag_maxerr < TOL_CROSSLANG['mean_maxerr']
        speed_pass = speed_ratio < 30  # Relaxed for pure Python with N=10000

        overall_pass = bw_pass and gcv_pass and mean_pass and real_pass and imag_pass and speed_pass

        print(f"\n--- Status ---")
        print(f"BW selection: {'PASS' if bw_pass else 'FAIL'}")
        print(f"GCV accuracy: {'PASS' if gcv_pass else 'FAIL'}")
        print(f"Mean accuracy: {'PASS' if mean_pass else 'FAIL'}")
        print(f"Real part: {'PASS' if real_pass else 'FAIL'}")
        print(f"Imag part: {'PASS' if imag_pass else 'FAIL'}")
        print(f"Speed: {'PASS' if speed_pass else 'FAIL'}")
        print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
        print("=" * 50)

        # ======================================================================
        # Assertions
        # ======================================================================

        # BW MaxErr
        assert bw_maxerr < TOL_CROSSLANG['bw_maxerr'], (
            f"BW MaxErr ({bw_maxerr:.2e}) should be < {TOL_CROSSLANG['bw_maxerr']}"
        )

        # BW index should be close (within 1)
        assert bw_idx_diff < 2, (
            f"BW index diff ({bw_idx_diff}) should be < 2"
        )

        # GCV MaxErr
        assert gcv_maxerr < 0.01, (
            f"GCV MaxErr ({gcv_maxerr:.2e}) should be < 0.01"
        )

        # Mean MaxErr (complex modulus)
        assert mean_maxerr < TOL_CROSSLANG['mean_maxerr'], (
            f"Mean MaxErr ({mean_maxerr:.2e}) should be < {TOL_CROSSLANG['mean_maxerr']}"
        )

        # Real part MaxErr
        assert real_maxerr < TOL_CROSSLANG['mean_maxerr'], (
            f"Real MaxErr ({real_maxerr:.2e}) should be < {TOL_CROSSLANG['mean_maxerr']}"
        )

        # Imag part MaxErr
        assert imag_maxerr < TOL_CROSSLANG['mean_maxerr'], (
            f"Imag MaxErr ({imag_maxerr:.2e}) should be < {TOL_CROSSLANG['mean_maxerr']}"
        )

        # Speed ratio (relaxed for pure Python path)
        assert speed_ratio < 30, (
            f"Speed ratio ({speed_ratio:.2f}x) should be < 30x (pure Python path)"
        )

    def test_complex_gcv_is_real(self):
        """GCV scores should be real for complex data."""
        from fastlpr import cv_fastlpr

        ref = load_matlab_ref()
        if ref is None:
            pytest.skip("MATLAB reference file not found")

        x_ref = np.asarray(ref['x']).reshape(-1, 1)
        y_ref = np.asarray(ref['y']).ravel()
        hlist_ref = np.asarray(ref['hlist']).reshape(-1, 1)

        options = {'order': 1, 'calc_dof': True}
        res = cv_fastlpr(x_ref, y_ref, h=hlist_ref, options=options)

        gcv_scores = res.gcv['gcv_m']

        # GCV should be real (or have negligible imaginary part)
        if np.iscomplexobj(gcv_scores):
            imag_part = np.max(np.abs(np.imag(gcv_scores)))
            assert imag_part < 1e-10, f"GCV imaginary part ({imag_part}) should be negligible"
        else:
            assert True, "GCV is real-valued"

        # GCV should be positive
        gcv_real = np.real(gcv_scores)
        assert np.all(gcv_real > 0), "GCV values should be positive"

    # =========================================================================
    # ARCHIVED: 2026-01-10
    # test_complex_reproducibility - "Complex regression should be reproducible"
    # Reason: Test unification - not in R reference (R archived this test on 2026-01-09)
    # Archive: dev/archive/tests-archive-20260110/python/xl/archived_test_xl_complex.py
    # =========================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
