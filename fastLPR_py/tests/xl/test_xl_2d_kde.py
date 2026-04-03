# ==============================================================================
# Cross-Language Test: XL-02 - 2D KDE vs MATLAB Reference
# ==============================================================================
# Test file: fastLPR_py/tests/xl/test_xl_kde_2d.py
# Created: 2025-01-07
#
# This test validates the Python cv_fastkde() implementation against MATLAB
# reference data for 2D kernel density estimation.
#
# Reference file: fastLPR/tests/refs/crosslang_e2e/ref_2d_kde.mat
# Contains:
#   - x: Input data (N x 2 matrix)
#   - hlist: Bandwidth candidates (k x 2 matrix)
#   - lcv.m: LCV scores for all bandwidths (k x 1 vector)
#   - h.selected: Selected bandwidth using 1-SE rule (1 x 2 vector)
#   - id1se: Index of selected bandwidth (1-based, MATLAB)
#   - elapsed: MATLAB execution time (scalar)
#
# Pass Criteria (per CLAUDE.md):
#   - BW MaxErr < 0.05 (bandwidth selection threshold for 2D)
#   - id1se matches (python 0-based == matlab id1se - 1)
#   - LCV RelErr < 0.01 (1% relative error)
#   - Speed ratio < 8x MATLAB time
# ==============================================================================

import time
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

import fastlpr
from fastlpr import cv_fastkde

# Import tolerances from conftest
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import TOL_CROSSLANG


def _get_project_root() -> Path:
    """Get project root (jss-code) from xl test directory."""
    return Path(__file__).parent.parent.parent.parent


def _get_ref_path() -> Path:
    """Get the path to MATLAB reference file for 2D KDE."""
    return _get_project_root() / "fastLPR" / "tests" / "refs" / "crosslang_e2e" / "ref_2d_kde.mat"


class TestXL02KDE2D:
    """Cross-language test: 2D KDE vs MATLAB reference."""

    @pytest.fixture
    def ref_data(self):
        """Load MATLAB reference data for 2D KDE."""
        ref_path = _get_ref_path()

        if not ref_path.exists():
            pytest.skip(f"MATLAB reference file not found: {ref_path}")

        ref = sio.loadmat(str(ref_path), squeeze_me=True, struct_as_record=False)
        return ref

    def test_2d_kde_matches_matlab(self, ref_data):
        """Test that 2D KDE matches MATLAB reference data."""
        ref = ref_data

        # Extract reference data
        x_ref = np.asarray(ref['x'])
        hlist_ref = np.asarray(ref['hlist'])

        # Ensure proper shapes
        if x_ref.ndim == 1:
            x_ref = x_ref.reshape(-1, 2)
        n = x_ref.shape[0]
        dx = x_ref.shape[1]

        if hlist_ref.ndim == 1:
            hlist_ref = hlist_ref.reshape(-1, dx)
        dh = hlist_ref.shape[0]

        # Extract MATLAB results
        lcv_m_ref = np.asarray(ref['lcv_m']).ravel()
        h_selected_ref = np.asarray(ref['h_selected']).ravel()
        id1se_ref = int(ref['id1se'])  # 1-based MATLAB index
        matlab_time = float(ref['elapsed'])

        # ====================================================================
        # Run Python implementation
        # ====================================================================
        start_time = time.perf_counter()
        kde = cv_fastkde(x_ref, h=hlist_ref, options={'verbose': False})
        py_time = time.perf_counter() - start_time

        # Extract Python results
        h_selected_py = np.asarray(kde.h).ravel()
        id1se_py = kde.lcv['id1se']  # 0-based Python index
        lcv_m_py = np.asarray(kde.lcv['lcv_m']).ravel()

        # ====================================================================
        # Compute error metrics
        # ====================================================================

        # Bandwidth selection error (max over dimensions)
        bw_maxerr = float(np.max(np.abs(h_selected_py - h_selected_ref)))

        # Index comparison (Python 0-based == MATLAB id1se - 1)
        id1se_match = (id1se_py == id1se_ref - 1)

        # LCV relative error: max(|py - ref| / |ref|)
        lcv_abs_err = np.abs(lcv_m_py - lcv_m_ref)
        lcv_rel_err = float(np.max(lcv_abs_err / np.abs(lcv_m_ref)))

        # Speed ratio
        speed_ratio = py_time / matlab_time

        # ====================================================================
        # Report results
        # ====================================================================
        print(f"\n=== XL-02: 2D KDE Cross-Language Test ===")
        print(f"Sample size (N): {n}")
        print(f"Dimensions: {dx}")
        print(f"Number of bandwidths: {dh}")
        print(f"MATLAB time: {matlab_time:.3f}s")
        print(f"Python time: {py_time:.3f}s")
        print(f"MATLAB selected h: [{h_selected_ref[0]:.6f}, {h_selected_ref[1]:.6f}] (id={id1se_ref})")
        print(f"Python selected h: [{h_selected_py[0]:.6f}, {h_selected_py[1]:.6f}] (id={id1se_py + 1})")
        print(f"\n--- Error Metrics ---")
        print(f"BW MaxErr: {bw_maxerr:.2e} (threshold: 0.05)")
        print(f"id1se match: {id1se_match} (py={id1se_py}, matlab-1={id1se_ref - 1})")
        print(f"LCV RelErr: {lcv_rel_err:.4%} (threshold: {TOL_CROSSLANG['lcv_relerr']:.0%})")
        print(f"Speed ratio: {speed_ratio:.2f}x (threshold: {TOL_CROSSLANG['speed_ratio']}x)")

        # ====================================================================
        # Assertions
        # ====================================================================

        # 1. Bandwidth abs error < 0.05 for 2D
        assert bw_maxerr < 0.05, (
            f"BW MaxErr ({bw_maxerr:.2e}) should be < 0.05"
        )

        # 2. id1se matches (python 0-based == matlab id1se - 1)
        assert id1se_match, (
            f"id1se mismatch: Python {id1se_py} != MATLAB {id1se_ref} - 1"
        )

        # 3. LCV rel err < TOL_CROSSLANG['lcv_relerr']
        assert lcv_rel_err < TOL_CROSSLANG['lcv_relerr'], (
            f"LCV RelErr ({lcv_rel_err:.4%}) should be < {TOL_CROSSLANG['lcv_relerr']:.0%}"
        )

        # 4. speed_ratio < TOL_CROSSLANG['speed_ratio']
        assert speed_ratio < TOL_CROSSLANG['speed_ratio'], (
            f"Speed ratio ({speed_ratio:.2f}x) should be < {TOL_CROSSLANG['speed_ratio']}x"
        )

        # ====================================================================
        # Summary
        # ====================================================================
        bw_pass = bw_maxerr < 0.05
        lcv_pass = lcv_rel_err < TOL_CROSSLANG['lcv_relerr']
        speed_pass = speed_ratio < TOL_CROSSLANG['speed_ratio']
        overall = bw_pass and id1se_match and lcv_pass and speed_pass

        print(f"\n--- Status ---")
        print(f"BW selection: {'PASS' if bw_pass else 'FAIL'}")
        print(f"id1se match: {'PASS' if id1se_match else 'FAIL'}")
        print(f"LCV RelErr: {'PASS' if lcv_pass else 'FAIL'}")
        print(f"Speed: {'PASS' if speed_pass else 'FAIL'}")
        print(f"Overall: {'PASS' if overall else 'FAIL'}")
        print("=" * 40)

    def test_2d_kde_produces_valid_density(self, ref_data):
        """Test that 2D KDE produces valid density estimates."""
        ref = ref_data

        x_ref = np.asarray(ref['x'])
        hlist_ref = np.asarray(ref['hlist'])

        if x_ref.ndim == 1:
            x_ref = x_ref.reshape(-1, 2)
        if hlist_ref.ndim == 1:
            hlist_ref = hlist_ref.reshape(-1, 2)

        kde = cv_fastkde(x_ref, h=hlist_ref, options={'verbose': False})

        # Density should be non-negative
        assert np.all(kde.fhat >= -1e-10), "All density values should be non-negative"

        # Density should integrate to approximately 1
        grid0 = kde.grid[0]
        grid1 = kde.grid[1]
        dx0 = grid0[1] - grid0[0]
        dx1 = grid1[1] - grid1[0]
        integral = float(np.sum(kde.fhat) * dx0 * dx1)

        assert abs(integral - 1.0) < 0.15, (
            f"Density integral ({integral:.4f}) should be close to 1"
        )

        # Selected bandwidth should be 2D
        assert len(kde.h) == 2, "Selected bandwidth should have 2 components"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
