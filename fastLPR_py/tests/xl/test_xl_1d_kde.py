# ==============================================================================
# Cross-Language Test: XL-01 - 1D KDE vs MATLAB Reference
# ==============================================================================
# Test file: fastLPR_py/tests/xl/test_xl_kde_1d.py
# Created: 2025-01-07
#
# This test validates the Python cv_fastkde() implementation against MATLAB
# reference data to ensure numerical consistency across implementations.
#
# Reference file: fastLPR/tests/refs/crosslang_e2e/ref_1d_kde.mat
# Contains:
#   - x: Input data (N x 1 matrix)
#   - hlist: Bandwidth candidates (k x 1 matrix)
#   - lcv.m: LCV scores for all bandwidths (k x 1 vector)
#   - h.selected: Selected bandwidth using 1-SE rule (scalar)
#   - id1se: Index of selected bandwidth (1-based, MATLAB)
#   - fhat.at.x: Density estimates at data points (N x 1 vector)
#   - elapsed: MATLAB execution time (scalar)
#
# Pass Criteria (per CLAUDE.md):
#   - BW MaxErr < 0.02 (bandwidth selection threshold)
#   - id1se matches (python 0-based == matlab id1se - 1)
#   - LCV RelErr < 0.01 (1% relative error)
#   - Speed ratio < 8x MATLAB time
#   - Density correlation > 0.95
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


def _get_ref_path() -> Path:
    """Get the path to MATLAB reference file for 1D KDE.

    Searches multiple possible locations for cross-language compatibility.
    """
    # This file is at: fastLPR_py/tests/xl/test_xl_kde_1d.py
    # Project root is: jss-code/
    # Reference file is at: jss-code/fastLPR/tests/refs/crosslang_e2e/ref_1d_kde.mat
    test_file = Path(__file__).resolve()
    # Go up from xl/ -> tests/ -> fastLPR_py/ -> jss-code/
    project_root = test_file.parent.parent.parent.parent

    ref_paths = [
        project_root / "fastLPR" / "tests" / "refs" / "crosslang_e2e" / "ref_1d_kde.mat",
        # Fallback: relative from test file
        test_file.parent.parent.parent.parent / "fastLPR" / "tests" / "refs" / "crosslang_e2e" / "ref_1d_kde.mat",
    ]

    for p in ref_paths:
        if p.exists():
            return p

    return ref_paths[0]  # Return first path for error message


class TestXL01KDE1D:
    """Cross-language test: 1D KDE vs MATLAB reference."""

    @pytest.fixture
    def ref_data(self):
        """Load MATLAB reference data for 1D KDE."""
        ref_path = _get_ref_path()

        if not ref_path.exists():
            pytest.skip(f"MATLAB reference file not found: {ref_path}")

        # Load with squeeze_me=True, struct_as_record=False per spec
        ref = sio.loadmat(str(ref_path), squeeze_me=True, struct_as_record=False)

        return ref

    def test_1d_kde_matches_matlab(self, ref_data):
        """Test that 1D KDE matches MATLAB reference data."""
        ref = ref_data

        # Extract reference data
        x_ref = np.asarray(ref['x'])
        hlist_ref = np.asarray(ref['hlist'])

        # Reshape x to (n, 1) and hlist to (dh, 1) per spec
        if x_ref.ndim == 1:
            x_ref = x_ref.reshape(-1, 1)
        n = x_ref.shape[0]

        if hlist_ref.ndim == 1:
            hlist_ref = hlist_ref.reshape(-1, 1)
        dh = hlist_ref.shape[0]

        # Extract MATLAB results
        lcv_m_ref = np.asarray(ref['lcv_m']).ravel()
        h_selected_ref = float(ref['h_selected'])
        id1se_ref = int(ref['id1se'])  # 1-based MATLAB index
        fhat_at_x_ref = np.asarray(ref['fhat_at_x']).ravel()
        matlab_time = float(ref['elapsed'])

        # ====================================================================
        # Run Python implementation
        # ====================================================================
        start_time = time.perf_counter()
        kde = cv_fastkde(x_ref, h=hlist_ref, options={'verbose': False})
        py_time = time.perf_counter() - start_time

        # Extract Python results
        h_selected_py = float(np.sum(kde.h))  # Sum for 1D (single value)
        id1se_py = kde.lcv['id1se']  # 0-based Python index
        lcv_m_py = np.asarray(kde.lcv['lcv_m']).ravel()

        # Evaluate density at data points
        fhat_at_x_py = kde.fpp(x_ref)
        if hasattr(fhat_at_x_py, 'ravel'):
            fhat_at_x_py = fhat_at_x_py.ravel()

        # ====================================================================
        # Compute error metrics
        # ====================================================================

        # Bandwidth selection error
        bw_maxerr = abs(h_selected_py - h_selected_ref)

        # Index comparison (Python 0-based == MATLAB id1se - 1)
        id1se_match = (id1se_py == id1se_ref - 1)

        # LCV relative error: max(|py - ref| / |ref|)
        lcv_abs_err = np.abs(lcv_m_py - lcv_m_ref)
        lcv_rel_err = float(np.max(lcv_abs_err / np.abs(lcv_m_ref)))

        # Speed ratio
        speed_ratio = py_time / matlab_time

        # Density correlation
        corr_matrix = np.corrcoef(fhat_at_x_py, fhat_at_x_ref)
        density_corr = float(corr_matrix[0, 1])

        # ====================================================================
        # Report results
        # ====================================================================
        print(f"\n=== XL-01: 1D KDE Cross-Language Test ===")
        print(f"Sample size (N): {n}")
        print(f"Number of bandwidths: {dh}")
        print(f"MATLAB time: {matlab_time:.3f}s")
        print(f"Python time: {py_time:.3f}s")
        print(f"MATLAB selected h: {h_selected_ref:.6f} (id={id1se_ref})")
        print(f"Python selected h: {h_selected_py:.6f} (id={id1se_py + 1})")
        print(f"\n--- Error Metrics ---")
        print(f"BW MaxErr: {bw_maxerr:.2e} (threshold: {TOL_CROSSLANG['bw_maxerr']})")
        print(f"id1se match: {id1se_match} (py={id1se_py}, matlab-1={id1se_ref - 1})")
        print(f"LCV RelErr: {lcv_rel_err:.4%} (threshold: {TOL_CROSSLANG['lcv_relerr']:.0%})")
        print(f"Speed ratio: {speed_ratio:.2f}x (threshold: {TOL_CROSSLANG['speed_ratio']}x)")
        print(f"Density correlation: {density_corr:.6f} (threshold: 0.95)")

        # ====================================================================
        # Assertions
        # ====================================================================

        # 1. Bandwidth abs error < TOL_CROSSLANG['bw_maxerr']
        assert bw_maxerr < TOL_CROSSLANG['bw_maxerr'], (
            f"BW MaxErr ({bw_maxerr:.2e}) should be < {TOL_CROSSLANG['bw_maxerr']}"
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

        # 5. Density correlation > 0.95
        assert density_corr > 0.95, (
            f"Density correlation ({density_corr:.6f}) should be > 0.95"
        )

        # ====================================================================
        # Summary
        # ====================================================================
        bw_pass = bw_maxerr < TOL_CROSSLANG['bw_maxerr']
        lcv_pass = lcv_rel_err < TOL_CROSSLANG['lcv_relerr']
        speed_pass = speed_ratio < TOL_CROSSLANG['speed_ratio']
        corr_pass = density_corr > 0.95
        overall = bw_pass and id1se_match and lcv_pass and speed_pass and corr_pass

        print(f"\n--- Status ---")
        print(f"BW selection: {'PASS' if bw_pass else 'FAIL'}")
        print(f"id1se match: {'PASS' if id1se_match else 'FAIL'}")
        print(f"LCV RelErr: {'PASS' if lcv_pass else 'FAIL'}")
        print(f"Speed: {'PASS' if speed_pass else 'FAIL'}")
        print(f"Density correlation: {'PASS' if corr_pass else 'FAIL'}")
        print(f"Overall: {'PASS' if overall else 'FAIL'}")
        print("=" * 40)

    def test_1d_kde_produces_valid_density(self, ref_data):
        """Test that 1D KDE produces valid density estimates."""
        ref = ref_data

        x_ref = np.asarray(ref['x'])
        hlist_ref = np.asarray(ref['hlist'])

        if x_ref.ndim == 1:
            x_ref = x_ref.reshape(-1, 1)
        if hlist_ref.ndim == 1:
            hlist_ref = hlist_ref.reshape(-1, 1)

        kde = cv_fastkde(x_ref, h=hlist_ref, options={'verbose': False})

        # Density should be non-negative
        assert np.all(kde.fhat >= -1e-10), "All density values should be non-negative"

        # Density should integrate to approximately 1
        grid = kde.grid[0]
        dx = grid[1] - grid[0]
        integral = float(np.sum(kde.fhat) * dx)

        assert abs(integral - 1.0) < 0.15, (
            f"Density integral ({integral:.4f}) should be close to 1"
        )

        # Selected bandwidth should be in the search range
        h_min = float(np.min(hlist_ref))
        h_max = float(np.max(hlist_ref))
        h_selected = float(np.sum(kde.h))

        assert h_selected >= h_min * 0.99, (
            f"Selected bandwidth ({h_selected}) should be >= min(hlist)"
        )
        assert h_selected <= h_max * 1.01, (
            f"Selected bandwidth ({h_selected}) should be <= max(hlist)"
        )
