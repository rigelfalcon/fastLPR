# ==============================================================================
# Cross-Language Test: XL-06 - 2D LPR Order 1 vs MATLAB Reference
# ==============================================================================
# Test file: fastLPR_py/tests/xl/test_xl_lpr_2d_o1.py
# Created: 2025-01-07
#
# This test validates the Python cv_fastlpr() implementation for 2D local linear
# regression against MATLAB reference data to ensure numerical consistency.
#
# Reference file: fastLPR/tests/refs/crosslang_e2e/ref_2d_order1.mat
# Contains:
#   - x: Input data (n x 2 matrix)
#   - y: Response values (n x 1 vector)
#   - hlist: Bandwidth candidates (dh x 2 matrix)
#   - gcv_m: GCV scores for all bandwidths (dh x 1 vector)
#   - h1se: Selected bandwidth using 1-SE rule (1 x 2 vector)
#   - id1se: Index of selected bandwidth (1-based, MATLAB)
#   - yhat_mean: Fitted values at data points (n x 1 vector)
#   - dof_random_vectors: Random vectors for DOF estimation (n x num_samples)
#   - elapsed: MATLAB execution time (scalar)
#
# Pass Criteria (per CLAUDE.md):
#   - BW MaxErr < 0.02 (bandwidth selection threshold)
#   - id1se matches (python 0-based == matlab id1se - 1)
#   - GCV RelErr < 0.05 (5% relative error)
#   - Mean MaxErr < 0.05 (mean prediction max absolute error)
#   - Speed ratio < 8x MATLAB time
# ==============================================================================

import time
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from fastlpr import cv_fastlpr

# Import tolerances from conftest
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import TOL_CROSSLANG, assert_gcv_fields, get_e2e_refs_dir


class TestXL06LPR2DOrder1:
    """Cross-language test: 2D LPR Order 1 vs MATLAB reference."""

    @pytest.fixture
    def ref_data(self):
        """Load MATLAB reference data for 2D LPR Order 1."""
        ref_path = get_e2e_refs_dir() / "ref_2d_order1.mat"

        if not ref_path.exists():
            pytest.skip(f"MATLAB reference file not found: {ref_path}")

        # Load with squeeze_me=True to simplify array dimensions
        ref = sio.loadmat(str(ref_path), squeeze_me=True, struct_as_record=False)

        return ref

    def test_2d_lpr_order1_matches_matlab(self, ref_data):
        """Test that 2D LPR Order 1 matches MATLAB reference data."""
        ref = ref_data

        # ====================================================================
        # Extract reference data
        # ====================================================================
        x_ref = np.asarray(ref['x'])
        y_ref = np.asarray(ref['y'])
        hlist_ref = np.asarray(ref['hlist'])

        # Ensure x is (n, 2) for 2D data
        if x_ref.ndim == 1:
            x_ref = x_ref.reshape(-1, 1)
        assert x_ref.shape[1] == 2, f"Expected 2D data, got shape {x_ref.shape}"
        n = x_ref.shape[0]

        # Ensure y is column vector
        if y_ref.ndim == 1:
            y_ref = y_ref.reshape(-1, 1)

        # Ensure hlist is (dh, 2) for 2D bandwidths
        if hlist_ref.ndim == 1:
            hlist_ref = hlist_ref.reshape(-1, 1)
        assert hlist_ref.shape[1] == 2, f"Expected 2D hlist, got shape {hlist_ref.shape}"
        dh = hlist_ref.shape[0]

        # Extract MATLAB results
        gcv_m_ref = np.asarray(ref['gcv_m']).ravel()
        h1se_ref = np.asarray(ref['h1se']).ravel()
        id1se_ref = int(ref['id1se'])  # 1-based MATLAB index
        yhat_mean_ref = np.asarray(ref['yhat_mean']).ravel()
        matlab_time = float(ref['elapsed'])

        # Get DOF random vectors (for reproducible DOF estimation)
        dof_random_vectors = None
        if 'dof_random_vectors' in ref:
            dof_random_vectors = np.asarray(ref['dof_random_vectors'])

        # ====================================================================
        # Run Python implementation
        # ====================================================================
        options = {
            'order': 1,
            'calc_dof': True,
        }
        if dof_random_vectors is not None:
            options['random_matrix'] = dof_random_vectors

        start_time = time.perf_counter()
        result = cv_fastlpr(x_ref, y_ref, h=hlist_ref, options=options)
        py_time = time.perf_counter() - start_time

        assert_gcv_fields(result.gcv)

        # ====================================================================
        # Extract Python results
        # ====================================================================
        h_selected_py = np.asarray(result.h).ravel()
        id1se_py = result.gcv['id1se']  # 0-based Python index
        gcv_m_py = np.asarray(result.gcv['gcv_m']).ravel()
        yhat_py = np.asarray(result.yhat).ravel()

        # ====================================================================
        # Compute error metrics
        # ====================================================================

        # Bandwidth selection: max absolute error across dimensions
        bw_maxerr = float(np.max(np.abs(h_selected_py - h1se_ref)))

        # Index comparison (Python 0-based == MATLAB id1se - 1)
        id1se_match = (id1se_py == id1se_ref - 1)

        # GCV relative error: max(|py - ref| / |ref|)
        gcv_abs_err = np.abs(gcv_m_py - gcv_m_ref)
        # Use safe division (avoid /0 for near-zero GCV values)
        gcv_denom = np.maximum(np.abs(gcv_m_ref), 1e-10)
        gcv_rel_err = float(np.max(gcv_abs_err / gcv_denom))

        # Mean prediction: max absolute error
        mean_maxerr = float(np.max(np.abs(yhat_py - yhat_mean_ref)))

        # Speed ratio
        speed_ratio = py_time / matlab_time

        # ====================================================================
        # Report results
        # ====================================================================
        print(f"\n=== XL-06: 2D LPR Order 1 Cross-Language Test ===")
        print(f"Sample size (N): {n}")
        print(f"Dimensions (dx): 2")
        print(f"Number of bandwidths (dh): {dh}")
        print(f"MATLAB time: {matlab_time:.3f}s")
        print(f"Python time: {py_time:.3f}s")
        print(f"MATLAB selected h: [{h1se_ref[0]:.6f}, {h1se_ref[1]:.6f}] (id={id1se_ref})")
        print(f"Python selected h: [{h_selected_py[0]:.6f}, {h_selected_py[1]:.6f}] (id={id1se_py + 1})")
        print(f"\n--- Error Metrics ---")
        print(f"BW MaxErr: {bw_maxerr:.2e} (threshold: {TOL_CROSSLANG['bw_maxerr']})")
        print(f"id1se match: {id1se_match} (py={id1se_py}, matlab-1={id1se_ref - 1})")
        print(f"GCV RelErr: {gcv_rel_err:.4%} (threshold: {TOL_CROSSLANG['gcv_relerr']:.0%})")
        print(f"Mean MaxErr: {mean_maxerr:.2e} (threshold: {TOL_CROSSLANG['mean_maxerr']})")
        print(f"Speed ratio: {speed_ratio:.2f}x (threshold: {TOL_CROSSLANG['speed_ratio']}x)")

        # ====================================================================
        # Assertions
        # ====================================================================

        # 1. Bandwidth max abs error < TOL_CROSSLANG['bw_maxerr']
        assert bw_maxerr < TOL_CROSSLANG['bw_maxerr'], (
            f"BW MaxErr ({bw_maxerr:.2e}) should be < {TOL_CROSSLANG['bw_maxerr']}"
        )

        # 2. id1se matches (python 0-based == matlab id1se - 1)
        assert id1se_match, (
            f"id1se mismatch: Python {id1se_py} != MATLAB {id1se_ref} - 1"
        )

        # 3. GCV rel err < TOL_CROSSLANG['gcv_relerr']
        assert gcv_rel_err < TOL_CROSSLANG['gcv_relerr'], (
            f"GCV RelErr ({gcv_rel_err:.4%}) should be < {TOL_CROSSLANG['gcv_relerr']:.0%}"
        )

        # 4. Mean max abs err < TOL_CROSSLANG['mean_maxerr']
        assert mean_maxerr < TOL_CROSSLANG['mean_maxerr'], (
            f"Mean MaxErr ({mean_maxerr:.2e}) should be < {TOL_CROSSLANG['mean_maxerr']}"
        )

        # 5. speed_ratio < TOL_CROSSLANG['speed_ratio']
        assert speed_ratio < TOL_CROSSLANG['speed_ratio'], (
            f"Speed ratio ({speed_ratio:.2f}x) should be < {TOL_CROSSLANG['speed_ratio']}x"
        )

        # ====================================================================
        # Summary
        # ====================================================================
        bw_pass = bw_maxerr < TOL_CROSSLANG['bw_maxerr']
        gcv_pass = gcv_rel_err < TOL_CROSSLANG['gcv_relerr']
        mean_pass = mean_maxerr < TOL_CROSSLANG['mean_maxerr']
        speed_pass = speed_ratio < TOL_CROSSLANG['speed_ratio']
        overall = bw_pass and id1se_match and gcv_pass and mean_pass and speed_pass

        print(f"\n--- Status ---")
        print(f"BW selection: {'PASS' if bw_pass else 'FAIL'}")
        print(f"id1se match: {'PASS' if id1se_match else 'FAIL'}")
        print(f"GCV RelErr: {'PASS' if gcv_pass else 'FAIL'}")
        print(f"Mean MaxErr: {'PASS' if mean_pass else 'FAIL'}")
        print(f"Speed: {'PASS' if speed_pass else 'FAIL'}")
        print(f"Overall: {'PASS' if overall else 'FAIL'}")
        print("=" * 50)

    def test_2d_lpr_order1_predictions_valid(self, ref_data):
        """Test that 2D LPR Order 1 produces valid predictions."""
        ref = ref_data

        x_ref = np.asarray(ref['x'])
        y_ref = np.asarray(ref['y'])
        hlist_ref = np.asarray(ref['hlist'])

        # Ensure correct shapes
        if x_ref.ndim == 1:
            x_ref = x_ref.reshape(-1, 1)
        if y_ref.ndim == 1:
            y_ref = y_ref.reshape(-1, 1)
        if hlist_ref.ndim == 1:
            hlist_ref = hlist_ref.reshape(-1, 1)

        n = x_ref.shape[0]

        # Get DOF random vectors if available
        options = {'order': 1, 'calc_dof': True}
        if 'dof_random_vectors' in ref:
            options['random_matrix'] = np.asarray(ref['dof_random_vectors'])

        result = cv_fastlpr(x_ref, y_ref, h=hlist_ref, options=options)

        # Predictions should have same length as input
        yhat = np.asarray(result.yhat).ravel()
        assert len(yhat) == n, (
            f"Predictions length ({len(yhat)}) should equal input length ({n})"
        )

        # Predictions should be finite
        assert np.all(np.isfinite(yhat)), "All predictions should be finite"

        # Selected bandwidth should be in the search range
        h_selected = np.asarray(result.h).ravel()
        for d in range(2):
            h_min = float(np.min(hlist_ref[:, d]))
            h_max = float(np.max(hlist_ref[:, d]))
            assert h_selected[d] >= h_min * 0.99, (
                f"Selected bandwidth h[{d}]={h_selected[d]} should be >= min(hlist)={h_min}"
            )
            assert h_selected[d] <= h_max * 1.01, (
                f"Selected bandwidth h[{d}]={h_selected[d]} should be <= max(hlist)={h_max}"
            )

