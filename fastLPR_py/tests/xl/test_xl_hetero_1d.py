# ==============================================================================
# Cross-Language Integration Test: XL-09 Heteroscedastic 1D (Mean + Variance)
# ==============================================================================
# Test file: fastLPR_py/tests/xl/test_xl_hetero_1d.py
# Restored: 2026-01-10
#
# This test validates the Python cv_fastlpr() implementation for heteroscedastic
# regression (mean + variance estimation) against MATLAB reference data.
#
# TWO-STEP PROCESS:
#   1) Estimate mean: cv_fastlpr(x, y, hlist, opt_mean)
#   2) Estimate variance from residuals^2: cv_fastlpr(x, residuals^2, hlist, opt_var)
#
# Reference file: fastLPR/tests/refs/crosslang_e2e/ref_hetero_1d.mat
#
# Pass Criteria (per CLAUDE.md):
#   - BW MaxErr < 0.02 for both mean and var
#   - selected_idx matches id1se_* - 1 (Python 0-indexed)
#   - GCV RelErr < 0.05 for both mean and var
#   - Mean MaxErr < 0.05
#   - Var MaxErr < 0.05 (unified)
#   - Speed ratio (mean+var total) < 8x MATLAB time
# ==============================================================================

import time
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

# Import from conftest
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import TOL_CROSSLANG, assert_gcv_fields


def _get_project_root() -> Path:
    """Get the jss-code project root directory."""
    # xl/ -> tests/ -> fastLPR_py/ -> jss-code/
    return Path(__file__).parent.parent.parent.parent


def _get_e2e_refs_dir() -> Path:
    """Get the MATLAB E2E test reference directory."""
    return _get_project_root() / "fastLPR" / "tests" / "refs" / "crosslang_e2e"


def get_ref_path() -> Path:
    """Get path to MATLAB reference file."""
    return _get_e2e_refs_dir() / "ref_hetero_1d.mat"


def load_matlab_ref():
    """Load MATLAB reference data for heteroscedastic 1D test."""
    ref_path = get_ref_path()
    if not ref_path.exists():
        return None
    return sio.loadmat(str(ref_path), squeeze_me=False)


class TestXL09Hetero1D:
    """Cross-language tests for Heteroscedastic 1D (Mean + Variance) estimation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Skip if MATLAB reference data not available."""
        ref_path = get_ref_path()
        if not ref_path.exists():
            pytest.skip(f"MATLAB reference file not found: {ref_path}")

    def test_hetero_1d_matches_matlab(self):
        """Heteroscedastic 1D (mean + variance) matches MATLAB reference data."""
        # Import fastlpr here to avoid import errors if package not installed
        from fastlpr import cv_fastlpr

        # ======================================================================
        # Load MATLAB reference data
        # ======================================================================
        ref = load_matlab_ref()
        if ref is None:
            pytest.skip("MATLAB reference file not found")

        # Extract reference data with proper reshaping
        x = np.asarray(ref['x']).reshape(-1, 1)  # (n, 1)
        y = np.asarray(ref['y']).ravel()  # (n,)
        hlist = np.asarray(ref['hlist']).reshape(-1, 1)  # (dh, 1)
        elapsed = float(np.asarray(ref['elapsed']).ravel()[0])

        # Mean references
        gcv_m = np.asarray(ref['gcv_m']).ravel()  # (dh,)
        h1se_mean = float(np.asarray(ref['h1se_mean']).ravel()[0])
        id1se_mean = int(np.asarray(ref['id1se_mean']).ravel()[0])  # 1-indexed
        yhat_mean = np.asarray(ref['yhat_mean']).ravel()  # (n,)
        dof_random_vectors_mean = np.asarray(ref['dof_random_vectors_mean'])

        # Variance references
        gcv_var = np.asarray(ref['gcv_var']).ravel()  # (dh,)
        h1se_var = float(np.asarray(ref['h1se_var']).ravel()[0])
        id1se_var = int(np.asarray(ref['id1se_var']).ravel()[0])  # 1-indexed
        yhat_var = np.asarray(ref['yhat_var']).ravel()  # (n,)
        dof_random_vectors_var = np.asarray(ref['dof_random_vectors_var'])

        n = x.shape[0]
        dh = hlist.shape[0]

        print(f"\n=== XL-09: Heteroscedastic 1D (Mean + Variance) vs MATLAB ===")
        print(f"Reference: ref_hetero_1d.mat")
        print(f"Sample size (N): {n}")
        print(f"Number of bandwidths: {dh}")
        print(f"MATLAB time: {elapsed:.3f} s")
        print(f"MATLAB selected h (mean): {h1se_mean:.6f} (id={id1se_mean})")
        print(f"MATLAB selected h (var): {h1se_var:.6f} (id={id1se_var})")

        # ======================================================================
        # Run Python implementation: Step 1 - Mean estimation
        # ======================================================================
        options_mean = {
            'order': 1,
            'calc_dof': True,
            'dstd': 0,
            'random_matrix': dof_random_vectors_mean,
        }

        start_time = time.perf_counter()
        res_mean = cv_fastlpr(x, y, h=hlist, options=options_mean)
        time_mean = time.perf_counter() - start_time

        assert_gcv_fields(res_mean.gcv)

        # ======================================================================
        # Run Python implementation: Step 2 - Variance estimation
        # ======================================================================
        residuals = y - res_mean.yhat

        options_var = {
            'order': 1,
            'calc_dof': True,
            'y_type_out': 'variance',
            'dstd': 1,
            'random_matrix': dof_random_vectors_var,
        }

        start_var_time = time.perf_counter()
        res_var = cv_fastlpr(x, residuals**2, h=hlist, options=options_var)
        time_var = time.perf_counter() - start_var_time

        assert_gcv_fields(res_var.gcv)

        total_python_time = time_mean + time_var

        # Extract Python results
        h1se_mean_py = float(np.asarray(res_mean.h).item())
        id1se_mean_py = int(res_mean.gcv['id1se'])  # 0-indexed
        gcv_m_py = np.asarray(res_mean.gcv['gcv_m']).ravel()
        yhat_mean_py = np.asarray(res_mean.yhat).ravel()

        h1se_var_py = float(np.asarray(res_var.h).item())
        id1se_var_py = int(res_var.gcv['id1se'])  # 0-indexed
        gcv_var_py = np.asarray(res_var.gcv['gcv_m']).ravel()
        yhat_var_py = np.asarray(res_var.yhat).ravel()

        print(f"\n--- Python Results ---")
        print(f"Python time (mean): {time_mean:.3f} s")
        print(f"Python time (var): {time_var:.3f} s")
        print(f"Python total time: {total_python_time:.3f} s")
        print(f"Python selected h (mean): {h1se_mean_py:.6f} (id={id1se_mean_py + 1})")
        print(f"Python selected h (var): {h1se_var_py:.6f} (id={id1se_var_py + 1})")

        # ======================================================================
        # Compute error metrics
        # ======================================================================

        # Mean bandwidth selection error
        bw_mean_maxerr = abs(h1se_mean_py - h1se_mean)

        # Variance bandwidth selection error
        bw_var_maxerr = abs(h1se_var_py - h1se_var)

        # GCV relative error for mean: max(|scores - ref_gcv| / |ref_gcv|)
        gcv_mean_abs_err = np.abs(gcv_m_py - gcv_m)
        gcv_mean_denom = np.maximum(np.abs(gcv_m), 1e-10)
        gcv_mean_rel_err = np.max(gcv_mean_abs_err / gcv_mean_denom)

        # GCV relative error for variance
        gcv_var_abs_err = np.abs(gcv_var_py - gcv_var)
        gcv_var_denom = np.maximum(np.abs(gcv_var), 1e-10)
        gcv_var_rel_err = np.max(gcv_var_abs_err / gcv_var_denom)

        # Mean prediction error
        mean_maxerr = np.max(np.abs(yhat_mean_py - yhat_mean))

        # Variance prediction error
        var_maxerr = np.max(np.abs(yhat_var_py - yhat_var))

        # Speed ratio using total runtime
        speed_ratio = total_python_time / elapsed if elapsed > 0 else float('inf')

        print(f"\n--- Error Metrics ---")
        print(f"BW Mean MaxErr: {bw_mean_maxerr:.2e} (threshold: {TOL_CROSSLANG['bw_maxerr']})")
        print(f"BW Var MaxErr: {bw_var_maxerr:.2e} (threshold: {TOL_CROSSLANG['bw_maxerr']})")
        print(f"GCV Mean RelErr: {gcv_mean_rel_err:.2e} (threshold: {TOL_CROSSLANG['gcv_relerr']})")
        print(f"GCV Var RelErr: {gcv_var_rel_err:.2e} (threshold: {TOL_CROSSLANG['gcv_relerr']})")
        print(f"Mean MaxErr: {mean_maxerr:.2e} (threshold: {TOL_CROSSLANG['mean_maxerr']})")
        print(f"Var MaxErr: {var_maxerr:.2e} (threshold: {TOL_CROSSLANG['var_maxerr']})")

        print(f"\n--- Performance ---")
        print(f"Speed ratio: {speed_ratio:.2f}x (threshold: {TOL_CROSSLANG['speed_ratio']}x)")

        # ======================================================================
        # Determine pass/fail
        # ======================================================================
        bw_mean_pass = bw_mean_maxerr < TOL_CROSSLANG['bw_maxerr']
        bw_var_pass = bw_var_maxerr < TOL_CROSSLANG['bw_maxerr']
        idx_mean_pass = id1se_mean_py == id1se_mean - 1
        idx_var_pass = id1se_var_py == id1se_var - 1
        gcv_mean_pass = gcv_mean_rel_err < TOL_CROSSLANG['gcv_relerr']
        gcv_var_pass = gcv_var_rel_err < TOL_CROSSLANG['gcv_relerr']
        mean_pass = mean_maxerr < TOL_CROSSLANG['mean_maxerr']
        var_pass = var_maxerr < TOL_CROSSLANG['var_maxerr']
        speed_pass = speed_ratio < TOL_CROSSLANG['speed_ratio']

        overall_pass = (
            bw_mean_pass and bw_var_pass and
            idx_mean_pass and idx_var_pass and
            gcv_mean_pass and gcv_var_pass and
            mean_pass and var_pass and speed_pass
        )

        print(f"\n--- Status ---")
        print(f"BW Mean selection: {'PASS' if bw_mean_pass else 'FAIL'}")
        print(f"BW Var selection: {'PASS' if bw_var_pass else 'FAIL'}")
        print(f"Index Mean match: {'PASS' if idx_mean_pass else 'FAIL'} (py={id1se_mean_py}, matlab-1={id1se_mean - 1})")
        print(f"Index Var match: {'PASS' if idx_var_pass else 'FAIL'} (py={id1se_var_py}, matlab-1={id1se_var - 1})")
        print(f"GCV Mean accuracy: {'PASS' if gcv_mean_pass else 'FAIL'}")
        print(f"GCV Var accuracy: {'PASS' if gcv_var_pass else 'FAIL'}")
        print(f"Mean accuracy: {'PASS' if mean_pass else 'FAIL'}")
        print(f"Var accuracy: {'PASS' if var_pass else 'FAIL'}")
        print(f"Speed: {'PASS' if speed_pass else 'FAIL'}")
        print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
        print("=" * 60)

        # ======================================================================
        # Assertions
        # ======================================================================

        # BW errors for mean and var < TOL_CROSSLANG['bw_maxerr']
        assert bw_mean_maxerr < TOL_CROSSLANG['bw_maxerr'], (
            f"BW Mean MaxErr ({bw_mean_maxerr:.2e}) should be < {TOL_CROSSLANG['bw_maxerr']}"
        )
        assert bw_var_maxerr < TOL_CROSSLANG['bw_maxerr'], (
            f"BW Var MaxErr ({bw_var_maxerr:.2e}) should be < {TOL_CROSSLANG['bw_maxerr']}"
        )

        # selected_idx matches id1se_* - 1 (Python 0-indexed == MATLAB id1se - 1)
        assert id1se_mean_py == id1se_mean - 1, (
            f"Mean index mismatch: Python={id1se_mean_py} (0-indexed), "
            f"MATLAB={id1se_mean} (1-indexed)"
        )
        assert id1se_var_py == id1se_var - 1, (
            f"Var index mismatch: Python={id1se_var_py} (0-indexed), "
            f"MATLAB={id1se_var} (1-indexed)"
        )

        # GCV rel err for mean < TOL_CROSSLANG['gcv_relerr']
        assert gcv_mean_rel_err < TOL_CROSSLANG['gcv_relerr'], (
            f"GCV Mean RelErr ({gcv_mean_rel_err:.2e}) should be < {TOL_CROSSLANG['gcv_relerr']}"
        )

        # GCV rel err for var < TOL_CROSSLANG['gcv_relerr']
        assert gcv_var_rel_err < TOL_CROSSLANG['gcv_relerr'], (
            f"GCV Var RelErr ({gcv_var_rel_err:.2e}) should be < {TOL_CROSSLANG['gcv_relerr']}"
        )

        # Mean MaxErr < TOL_CROSSLANG['mean_maxerr']
        assert mean_maxerr < TOL_CROSSLANG['mean_maxerr'], (
            f"Mean MaxErr ({mean_maxerr:.2e}) should be < {TOL_CROSSLANG['mean_maxerr']}"
        )

        # Var MaxErr < TOL_CROSSLANG['var_maxerr']
        assert var_maxerr < TOL_CROSSLANG['var_maxerr'], (
            f"Var MaxErr ({var_maxerr:.2e}) should be < {TOL_CROSSLANG['var_maxerr']}"
        )

        # Speed ratio < TOL_CROSSLANG['speed_ratio']
        assert speed_ratio < TOL_CROSSLANG['speed_ratio'], (
            f"Speed ratio ({speed_ratio:.2f}x) should be < {TOL_CROSSLANG['speed_ratio']}x"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
