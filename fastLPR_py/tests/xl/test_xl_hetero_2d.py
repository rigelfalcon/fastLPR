# ==============================================================================
# Cross-Language Test: XL-10 - 2D Heteroscedastic Regression vs MATLAB Reference
# ==============================================================================
# Test file: fastLPR_py/tests/xl/test_xl_hetero_2d.py
# Restored: 2026-01-10
#
# This test validates the Python cv_fastlpr() heteroscedastic implementation
# (mean + variance) against MATLAB reference data for 2D inputs.
#
# TWO-STEP PROCESS (matching MATLAB):
#   1) Estimate mean: cv_fastlpr(x, y, h=hlist, options={order=1, calc_dof=True, dstd=0})
#   2) Estimate variance from residuals^2:
#      cv_fastlpr(x, residuals**2, h=hlist, options={order=1, calc_dof=True,
#                 y_type_out='variance', dstd=1})
#
# Reference file: fastLPR/tests/refs/crosslang_e2e/ref_hetero_2d.mat
#
# Pass Criteria (per CLAUDE.md - 2D heteroscedastic):
#   - BW MaxErr < 0.15 for both mean and var (2D exception)
#   - id1se matches (python 0-based == matlab id1se - 1)
#   - GCV RelErr < 0.10 for mean and var scores (2D exception)
#   - Mean MaxErr < 0.5 (2D exception)
#   - Var MaxErr < 0.2 (2D exception)
#   - Speed ratio < 8x
# ==============================================================================

import time
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

import fastlpr
from fastlpr import cv_fastlpr

# Import tolerances from conftest
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import TOL_CROSSLANG, assert_gcv_fields, get_e2e_refs_dir


class TestXL10Hetero2D:
    """Cross-language test: 2D Heteroscedastic regression vs MATLAB reference."""

    # 2D heteroscedastic tolerances (more lenient for higher-dimensional case)
    BW_MAXERR = 0.15      # 2D bandwidth selection threshold
    MEAN_MAXERR = 0.5     # 2D mean prediction threshold
    VAR_MAXERR = 0.2      # Variance estimation threshold
    GCV_RELERR = 0.10     # GCV relative error (10%)
    SPEED_RATIO = TOL_CROSSLANG['speed_ratio']  # 8x

    @pytest.fixture
    def ref_data(self):
        """Load MATLAB reference data for 2D heteroscedastic regression."""
        ref_path = get_e2e_refs_dir() / "ref_hetero_2d.mat"

        if not ref_path.exists():
            pytest.skip(f"MATLAB reference file not found: {ref_path}")

        # Load with squeeze_me=True for cleaner scalar handling
        ref = sio.loadmat(str(ref_path), squeeze_me=True, struct_as_record=False)

        return ref

    def test_hetero_2d_matches_matlab(self, ref_data):
        """Test that 2D heteroscedastic regression matches MATLAB reference."""
        ref = ref_data

        # ====================================================================
        # Extract reference data
        # ====================================================================
        x = np.asarray(ref['x'])  # (n, 2)
        y = np.asarray(ref['y']).ravel()  # (n,)
        hlist = np.asarray(ref['hlist'])  # (dh, 2)
        matlab_time = float(np.asarray(ref['elapsed']).ravel()[0])

        n = x.shape[0]
        dh = hlist.shape[0]

        # Mean reference values
        gcv_m_ref = np.asarray(ref['gcv_m']).ravel()
        h1se_mean_ref = np.asarray(ref['h1se_mean']).ravel()
        id1se_mean_ref = int(np.asarray(ref['id1se_mean']).ravel()[0])  # 1-based
        yhat_mean_ref = np.asarray(ref['yhat_mean']).ravel()
        dof_random_vectors_mean = np.asarray(ref['dof_random_vectors_mean'])

        # Variance reference values
        gcv_var_ref = np.asarray(ref['gcv_var']).ravel()
        h1se_var_ref = np.asarray(ref['h1se_var']).ravel()
        id1se_var_ref = int(np.asarray(ref['id1se_var']).ravel()[0])  # 1-based
        yhat_var_ref = np.asarray(ref['yhat_var']).ravel()
        dof_random_vectors_var = np.asarray(ref['dof_random_vectors_var'])

        # ====================================================================
        # Run Python implementation (TWO-STEP PROCESS)
        # ====================================================================
        start_time = time.perf_counter()

        # Step 1: Estimate mean
        res_mean = cv_fastlpr(
            x, y, h=hlist,
            options={
                'order': 1,
                'calc_dof': True,
                'dstd': 0,
                'random_matrix': dof_random_vectors_mean,
                'verbose': False
            }
        )

        # Step 2: Estimate variance from residuals^2
        residuals = y - res_mean.yhat
        res_var = cv_fastlpr(
            x, residuals**2, h=hlist,
            options={
                'order': 1,
                'calc_dof': True,
                'y_type_out': 'variance',
                'dstd': 1,
                'random_matrix': dof_random_vectors_var,
                'verbose': False
            }
        )

        py_time = time.perf_counter() - start_time

        assert_gcv_fields(res_mean.gcv)
        assert_gcv_fields(res_var.gcv)

        # ====================================================================
        # Extract Python results
        # ====================================================================

        # Mean results
        gcv_m_py = np.asarray(res_mean.gcv['gcv_m']).ravel()
        h1se_mean_py = np.asarray(res_mean.h).ravel()
        id1se_mean_py = res_mean.gcv['id1se']  # 0-based
        yhat_mean_py = np.asarray(res_mean.yhat).ravel()

        # Variance results
        gcv_var_py = np.asarray(res_var.gcv['gcv_m']).ravel()
        h1se_var_py = np.asarray(res_var.h).ravel()
        id1se_var_py = res_var.gcv['id1se']  # 0-based
        yhat_var_py = np.asarray(res_var.yhat).ravel()

        # ====================================================================
        # Compute error metrics
        # ====================================================================

        # --- Mean estimation metrics ---
        bw_mean_maxerr = float(np.max(np.abs(h1se_mean_py - h1se_mean_ref)))
        id1se_mean_match = (id1se_mean_py == id1se_mean_ref - 1)
        mean_maxerr = float(np.max(np.abs(yhat_mean_py - yhat_mean_ref)))

        # GCV relative error for mean (ignore NaN/Inf)
        valid_gcv_mean = np.isfinite(gcv_m_ref) & np.isfinite(gcv_m_py)
        if np.any(valid_gcv_mean):
            gcv_mean_abs_err = np.abs(gcv_m_py[valid_gcv_mean] - gcv_m_ref[valid_gcv_mean])
            gcv_mean_relerr = float(np.max(
                gcv_mean_abs_err / np.maximum(np.abs(gcv_m_ref[valid_gcv_mean]), 1e-10)
            ))
        else:
            gcv_mean_relerr = np.inf

        # --- Variance estimation metrics ---
        bw_var_maxerr = float(np.max(np.abs(h1se_var_py - h1se_var_ref)))
        id1se_var_match = (id1se_var_py == id1se_var_ref - 1)
        var_maxerr = float(np.max(np.abs(yhat_var_py - yhat_var_ref)))

        # GCV relative error for variance
        valid_gcv_var = np.isfinite(gcv_var_ref) & np.isfinite(gcv_var_py)
        if np.any(valid_gcv_var):
            gcv_var_abs_err = np.abs(gcv_var_py[valid_gcv_var] - gcv_var_ref[valid_gcv_var])
            gcv_var_relerr = float(np.max(
                gcv_var_abs_err / np.maximum(np.abs(gcv_var_ref[valid_gcv_var]), 1e-10)
            ))
        else:
            gcv_var_relerr = np.inf

        # Speed ratio
        speed_ratio = py_time / matlab_time

        # ====================================================================
        # Report results
        # ====================================================================
        print(f"\n=== XL-10: 2D Heteroscedastic Cross-Language Test ===")
        print(f"Sample size (N): {n}")
        print(f"Dimensions (dx): 2")
        print(f"Number of bandwidths: {dh}")
        print(f"Type: mean + variance (two-step)")
        print(f"MATLAB time: {matlab_time:.3f}s")
        print(f"Python time: {py_time:.3f}s")
        print(f"Speed ratio: {speed_ratio:.2f}x (threshold: {self.SPEED_RATIO}x)")

        print(f"\n--- Mean Estimation ---")
        print(f"MATLAB h1se_mean: [{h1se_mean_ref[0]:.4f}, {h1se_mean_ref[1]:.4f}]")
        print(f"Python h1se_mean: [{h1se_mean_py[0]:.4f}, {h1se_mean_py[1]:.4f}]")
        print(f"BW Mean MaxErr: {bw_mean_maxerr:.2e} (threshold: {self.BW_MAXERR})")
        print(f"id1se match: {id1se_mean_match} (py={id1se_mean_py}, matlab-1={id1se_mean_ref - 1})")
        print(f"GCV Mean RelErr: {gcv_mean_relerr:.4%} (threshold: {self.GCV_RELERR:.0%})")
        print(f"Mean MaxErr: {mean_maxerr:.2e} (threshold: {self.MEAN_MAXERR})")

        print(f"\n--- Variance Estimation ---")
        print(f"MATLAB h1se_var: [{h1se_var_ref[0]:.4f}, {h1se_var_ref[1]:.4f}]")
        print(f"Python h1se_var: [{h1se_var_py[0]:.4f}, {h1se_var_py[1]:.4f}]")
        print(f"BW Var MaxErr: {bw_var_maxerr:.2e} (threshold: {self.BW_MAXERR})")
        print(f"id1se match: {id1se_var_match} (py={id1se_var_py}, matlab-1={id1se_var_ref - 1})")
        print(f"GCV Var RelErr: {gcv_var_relerr:.4%} (threshold: {self.GCV_RELERR:.0%})")
        print(f"Var MaxErr: {var_maxerr:.2e} (threshold: {self.VAR_MAXERR})")

        # ====================================================================
        # Assertions
        # ====================================================================

        # 1. Bandwidth abs error for mean < BW_MAXERR
        assert bw_mean_maxerr < self.BW_MAXERR, (
            f"BW Mean MaxErr ({bw_mean_maxerr:.2e}) should be < {self.BW_MAXERR}"
        )

        # 2. Bandwidth abs error for var < BW_MAXERR
        assert bw_var_maxerr < self.BW_MAXERR, (
            f"BW Var MaxErr ({bw_var_maxerr:.2e}) should be < {self.BW_MAXERR}"
        )

        # 3. id1se_mean matches (python 0-based == matlab id1se - 1)
        assert id1se_mean_match, (
            f"id1se_mean mismatch: Python {id1se_mean_py} != MATLAB {id1se_mean_ref} - 1"
        )

        # 4. id1se_var matches
        assert id1se_var_match, (
            f"id1se_var mismatch: Python {id1se_var_py} != MATLAB {id1se_var_ref} - 1"
        )

        # 5. GCV rel err for mean < GCV_RELERR
        assert gcv_mean_relerr < self.GCV_RELERR, (
            f"GCV Mean RelErr ({gcv_mean_relerr:.4%}) should be < {self.GCV_RELERR:.0%}"
        )

        # 6. GCV rel err for var < GCV_RELERR
        assert gcv_var_relerr < self.GCV_RELERR, (
            f"GCV Var RelErr ({gcv_var_relerr:.4%}) should be < {self.GCV_RELERR:.0%}"
        )

        # 7. Mean MaxErr < MEAN_MAXERR
        assert mean_maxerr < self.MEAN_MAXERR, (
            f"Mean MaxErr ({mean_maxerr:.2e}) should be < {self.MEAN_MAXERR}"
        )

        # 8. Var MaxErr < VAR_MAXERR
        assert var_maxerr < self.VAR_MAXERR, (
            f"Var MaxErr ({var_maxerr:.2e}) should be < {self.VAR_MAXERR}"
        )

        # 9. Speed ratio < SPEED_RATIO
        assert speed_ratio < self.SPEED_RATIO, (
            f"Speed ratio ({speed_ratio:.2f}x) should be < {self.SPEED_RATIO}x"
        )

        # ====================================================================
        # Summary
        # ====================================================================
        bw_mean_pass = bw_mean_maxerr < self.BW_MAXERR
        bw_var_pass = bw_var_maxerr < self.BW_MAXERR
        gcv_mean_pass = gcv_mean_relerr < self.GCV_RELERR
        gcv_var_pass = gcv_var_relerr < self.GCV_RELERR
        mean_pass = mean_maxerr < self.MEAN_MAXERR
        var_pass = var_maxerr < self.VAR_MAXERR
        speed_pass = speed_ratio < self.SPEED_RATIO
        overall = all([
            bw_mean_pass, bw_var_pass, id1se_mean_match, id1se_var_match,
            gcv_mean_pass, gcv_var_pass, mean_pass, var_pass, speed_pass
        ])

        print(f"\n--- Status ---")
        print(f"BW Mean: {'PASS' if bw_mean_pass else 'FAIL'}")
        print(f"BW Var: {'PASS' if bw_var_pass else 'FAIL'}")
        print(f"id1se Mean: {'PASS' if id1se_mean_match else 'FAIL'}")
        print(f"id1se Var: {'PASS' if id1se_var_match else 'FAIL'}")
        print(f"GCV Mean: {'PASS' if gcv_mean_pass else 'FAIL'}")
        print(f"GCV Var: {'PASS' if gcv_var_pass else 'FAIL'}")
        print(f"Mean MaxErr: {'PASS' if mean_pass else 'FAIL'}")
        print(f"Var MaxErr: {'PASS' if var_pass else 'FAIL'}")
        print(f"Speed: {'PASS' if speed_pass else 'FAIL'}")
        print(f"Overall: {'PASS' if overall else 'FAIL'}")
        print("=" * 50)

        # Print summary table row (for documentation)
        print(f"\n| Hetero 2D | {n} | 2 | 1 | mean+var | {matlab_time:.3f}s | "
              f"{py_time:.3f}s | {speed_ratio:.2f}x | {gcv_mean_relerr:.2e} | "
              f"{bw_mean_maxerr:.2e} | {mean_maxerr:.2e} | {var_maxerr:.2e} | "
              f"{'PASS' if overall else 'FAIL'} |")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
