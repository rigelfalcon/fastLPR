# ==============================================================================
# Cross-Language Test: XL-06 - 1D LPR Order 2 (Local Quadratic) vs MATLAB Reference
# ==============================================================================
# Test file: fastLPR_py/tests/xl/test_xl_lpr_1d_o2.py
# Created: 2025-01-07
#
# This test validates the Python cv_fastlpr() implementation against MATLAB
# reference data for 1D local quadratic regression (order=2).
#
# Reference file: fastLPR/tests/refs/crosslang_e2e/ref_1d_order2.mat
# Contains:
#   - x: Input data (N x 1 matrix)
#   - y: Response data (N x 1 matrix)
#   - hlist: Bandwidth candidates (k x 1 matrix)
#   - gcv_m: GCV scores for all bandwidths (k x 1 matrix)
#   - h1se: Selected bandwidth using 1-SE rule (scalar)
#   - id1se: Index of selected bandwidth (1-indexed, MATLAB)
#   - yhat_mean: Fitted values at data points (N x 1 matrix)
#   - dof_random_vectors: Random vectors for DOF estimation
#   - elapsed: MATLAB execution time (scalar)
#
# Pass Criteria (per CLAUDE.md):
#   - BW MaxErr < 0.02 (bandwidth selection threshold)
#   - GCV RelErr < 0.05 (GCV score relative error)
#   - Mean MaxErr < 0.05 (prediction threshold)
#   - Speed ratio < 8x MATLAB time
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
    return _get_project_root() / "fastLPR" / "tests" / "refs" / "crosslang_e2e" / "ref_1d_order2.mat"


def load_matlab_ref():
    """Load MATLAB reference data for 1D order 2 test."""
    ref_path = get_ref_path()
    if not ref_path.exists():
        return None
    return sio.loadmat(str(ref_path), squeeze_me=False)


class TestXL1DLprOrder2:
    """Cross-language tests for 1D Local Quadratic Regression (order=2)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Skip if MATLAB reference data not available."""
        ref_path = get_ref_path()
        if not ref_path.exists():
            pytest.skip(f"MATLAB reference file not found: {ref_path}")

    def test_1d_order2_matches_matlab(self):
        """1D Order 2 (Local Quadratic) matches MATLAB reference data."""
        from fastlpr import cv_fastlpr

        # ======================================================================
        # Load MATLAB reference data
        # ======================================================================
        ref = load_matlab_ref()
        if ref is None:
            pytest.skip("MATLAB reference file not found")

        # Extract reference data with proper reshaping
        x_ref = np.asarray(ref['x']).reshape(-1, 1)  # (n, 1)
        y_ref = np.asarray(ref['y']).ravel()  # (n,)
        hlist_ref = np.asarray(ref['hlist']).reshape(-1, 1)  # (dh, 1)
        gcv_m_ref = np.asarray(ref['gcv_m']).ravel()  # (dh,)
        h1se_ref = float(np.asarray(ref['h1se']).ravel()[0])
        id1se_ref = int(np.asarray(ref['id1se']).ravel()[0])  # 1-indexed
        yhat_mean_ref = np.asarray(ref['yhat_mean']).ravel()  # (n,)
        dof_random_vectors = np.asarray(ref['dof_random_vectors'])
        matlab_time = float(np.asarray(ref['elapsed']).ravel()[0])

        n = x_ref.shape[0]
        dh = hlist_ref.shape[0]

        print(f"\n=== XL-06: 1D Order 2 (Local Quadratic) vs MATLAB ===")
        print(f"Reference: ref_1d_order2.mat")
        print(f"Sample size (N): {n}")
        print(f"Number of bandwidths: {dh}")
        print(f"MATLAB time: {matlab_time:.3f} s")
        print(f"MATLAB selected h: {h1se_ref:.6f} (id={id1se_ref})")

        # ======================================================================
        # Run Python implementation
        # ======================================================================
        options = {
            'order': 2,  # Local quadratic
            'calc_dof': True,
            'random_matrix': dof_random_vectors,
        }

        start_time = time.perf_counter()
        res = cv_fastlpr(x_ref, y_ref, h=hlist_ref, options=options)
        python_time = time.perf_counter() - start_time

        assert_gcv_fields(res.gcv_yhat)

        # Extract Python results
        h1se_py = float(np.asarray(res.h).item())
        id1se_py = int(res.gcv_yhat['id1se'])  # 0-indexed
        gcv_m_py = np.asarray(res.gcv_yhat['gcv_m']).ravel()
        yhat_py = np.asarray(res.yhat).ravel()

        print(f"\n--- Python Results ---")
        print(f"Python time: {python_time:.3f} s")
        print(f"Python selected h: {h1se_py:.6f} (id={id1se_py + 1})")

        # ======================================================================
        # Compute error metrics
        # ======================================================================

        # Bandwidth selection error
        bw_maxerr = abs(h1se_py - h1se_ref)

        # Bandwidth index difference (convert MATLAB 1-indexed to Python 0-indexed)
        bw_idx_diff = abs(id1se_py - (id1se_ref - 1))

        # GCV relative error: max(|scores - ref_gcv| / |ref_gcv|)
        gcv_abs_err = np.abs(gcv_m_py - gcv_m_ref)
        gcv_denom = np.maximum(np.abs(gcv_m_ref), 1e-10)
        gcv_rel_err = np.max(gcv_abs_err / gcv_denom)

        # Mean prediction error
        mean_maxerr = np.max(np.abs(yhat_py - yhat_mean_ref))
        mean_mse = np.mean((yhat_py - yhat_mean_ref) ** 2)

        # Speed ratio
        speed_ratio = python_time / matlab_time if matlab_time > 0 else float('inf')

        print(f"\n--- Error Metrics ---")
        print(f"BW MaxErr: {bw_maxerr:.2e} (threshold: {TOL_CROSSLANG['bw_maxerr']})")
        print(f"BW Index Diff: {bw_idx_diff}")
        print(f"GCV RelErr: {gcv_rel_err:.2e} (threshold: {TOL_CROSSLANG['gcv_relerr']})")
        print(f"Mean MaxErr: {mean_maxerr:.2e} (threshold: {TOL_CROSSLANG['mean_maxerr']})")
        print(f"Mean MSE: {mean_mse:.2e}")

        print(f"\n--- Performance ---")
        print(f"Speed ratio: {speed_ratio:.2f}x (threshold: {TOL_CROSSLANG['speed_ratio']}x)")

        # ======================================================================
        # Determine pass/fail
        # ======================================================================
        bw_pass = bw_maxerr < TOL_CROSSLANG['bw_maxerr']
        gcv_pass = gcv_rel_err < TOL_CROSSLANG['gcv_relerr']
        mean_pass = mean_maxerr < TOL_CROSSLANG['mean_maxerr']
        speed_pass = speed_ratio < TOL_CROSSLANG['speed_ratio']

        overall_pass = bw_pass and gcv_pass and mean_pass and speed_pass

        print(f"\n--- Status ---")
        print(f"BW selection: {'PASS' if bw_pass else 'FAIL'}")
        print(f"GCV accuracy: {'PASS' if gcv_pass else 'FAIL'}")
        print(f"Mean accuracy: {'PASS' if mean_pass else 'FAIL'}")
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

        # BW index must match exactly (after 1-indexed to 0-indexed conversion)
        assert id1se_py == id1se_ref - 1, (
            f"BW index mismatch: Python={id1se_py} (0-indexed), "
            f"MATLAB={id1se_ref} (1-indexed)"
        )

        # GCV RelErr
        assert gcv_rel_err < TOL_CROSSLANG['gcv_relerr'], (
            f"GCV RelErr ({gcv_rel_err:.2e}) should be < {TOL_CROSSLANG['gcv_relerr']}"
        )

        # Mean MaxErr
        assert mean_maxerr < TOL_CROSSLANG['mean_maxerr'], (
            f"Mean MaxErr ({mean_maxerr:.2e}) should be < {TOL_CROSSLANG['mean_maxerr']}"
        )

        # Speed ratio
        assert speed_ratio < TOL_CROSSLANG['speed_ratio'], (
            f"Speed ratio ({speed_ratio:.2f}x) should be < {TOL_CROSSLANG['speed_ratio']}x"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
