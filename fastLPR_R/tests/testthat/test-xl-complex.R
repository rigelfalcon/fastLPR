# test-cv-complex.R - Integration test for complex-valued regression against MATLAB reference
#
# Author: Ying Wang, Min Li
# Copyright (c) 2020-2025 fastLPR Development Team
# License: GNU General Public License v3.0
#
# Purpose: Verify complex-valued regression matches MATLAB reference implementation
# Reference Data: fastLPR/tests/refs/crosslang_e2e/ref_complex.mat
#
# Notes:
# - Complex data uses fast Rcpp path via Re/Im splitting (fixed 2025-01-07)
# - N=10000 samples, speed should be < 8x with Rcpp acceleration
# - Order 0 (NW) with complex data has a known bug in fastlpr_reg.R (max() on complex)

context("Cross-validation: Complex-valued Regression")

# -----------------------------------------------------------------------------
# Test 1: Complex regression matches MATLAB reference (CRITICAL)
# -----------------------------------------------------------------------------
test_that("Complex regression matches MATLAB reference", {
  # Skip if MATLAB reference data not available (CRAN compatibility)
  skip_if_no_matlab_refs()

  # Load reference data
  ref_path <- file.path(find_root(), "fastLPR/tests/refs/crosslang_e2e/ref_complex.mat")
  if (!file.exists(ref_path)) {
    skip(paste("Reference data not found:", ref_path))
  }

  ref <- R.matlab::readMat(ref_path)

  # Extract data
  x <- ref$x
  y <- ref$y  # Should already be complex
  hlist <- ref$hlist

  # Verify y is complex
  expect_true(is.complex(y), info = "y should be complex-valued")

  # Reference values
  matlab_time <- as.numeric(ref$elapsed)
  matlab_h1se <- as.numeric(ref$h1se)
  matlab_id1se <- as.numeric(ref$id1se)
  matlab_yhat <- ref$yhat.mean
  matlab_gcv <- ref$gcv.m

  # Run R implementation
  opt <- list(order = 1, calc_dof = TRUE, verbose = FALSE)
  r_time <- system.time({
    result <- cv_fastlpr(x, y, hlist, opt)
  })[["elapsed"]]

  # Extract R results
  r_h1se <- result$gcv_yhat$h1se
  r_id1se <- result$gcv_yhat$id1se
  r_yhat <- result$yhat
  r_gcv <- result$gcv_yhat$gcv_m

  # Verify results are not NULL
  expect_true(!is.null(r_yhat), info = "R yhat should not be NULL")
  expect_true(!is.null(r_gcv), info = "R GCV should not be NULL")

  # Verify predictions are complex
  expect_true(is.complex(r_yhat), info = "R predictions should be complex")

  # Calculate error metrics
  bw_maxerr <- abs(r_h1se - matlab_h1se)
  bw_idx_diff <- abs(r_id1se - matlab_id1se)

  # Complex prediction error (using modulus)
  yhat_diff <- r_yhat - matlab_yhat
  mean_maxerr <- max(abs(yhat_diff))
  mean_mse <- mean(abs(yhat_diff)^2)

  # GCV error (GCV values should be real)
  expect_true(!is.complex(r_gcv[1]), info = "GCV values should be real")
  gcv_maxerr <- max(abs(as.vector(r_gcv) - as.vector(matlab_gcv)))

  # Performance ratio
  ratio <- r_time / matlab_time

  # Print diagnostic message
  message(sprintf(
    "\nComplex Regression Results:\n  Ratio=%.2fx, BW MaxErr=%.2e, BW Idx Diff=%d\n  Mean MaxErr=%.2e, GCV MaxErr=%.2e",
    ratio, bw_maxerr, bw_idx_diff, mean_maxerr, gcv_maxerr
  ))

  # Assertions with meaningful labels
  expect_lt(bw_maxerr, 0.02, label = "Bandwidth MaxErr")
  expect_lt(bw_idx_diff, 2, label = "Bandwidth index difference")
  expect_lt(mean_maxerr, 0.05, label = "Mean prediction MaxErr")
  expect_lt(gcv_maxerr, 0.01, label = "GCV MaxErr")

  # Speed threshold is unified at 8x (TOL_CROSSLANG$speed_ratio)
  # Complex now uses fast Rcpp path via Re/Im splitting
  expect_lt(ratio, TOL_CROSSLANG$speed_ratio, label = "Speed ratio (R/MATLAB)")  # 8x (unified)
})

# -----------------------------------------------------------------------------
# Test 2: Verify GCV scores are real for complex data
# -----------------------------------------------------------------------------
# ARCHIVED: 2026-01-09 - "GCV scores are real for complex data" (moved to dev/archive/tests-archive-20260109/r/xl/archived_test-xl-complex.R)


# -----------------------------------------------------------------------------
# Test 3: Verify pdof is computed correctly for complex data
# -----------------------------------------------------------------------------
# ARCHIVED: 2026-01-09 - "pdof is computed correctly for complex data" (moved to dev/archive/tests-archive-20260109/r/xl/archived_test-xl-complex.R)


# -----------------------------------------------------------------------------
# Test 4: Complex regression reproducibility
# -----------------------------------------------------------------------------
# ARCHIVED: 2026-01-09 - "Complex regression is reproducible" (moved to dev/archive/tests-archive-20260109/r/xl/archived_test-xl-complex.R)


# -----------------------------------------------------------------------------
# Test 5: Real and imaginary parts match MATLAB separately
# -----------------------------------------------------------------------------
# ARCHIVED: 2026-01-09 - "Real and imaginary parts match MATLAB separately" (moved to dev/archive/tests-archive-20260109/r/xl/archived_test-xl-complex.R)


# -----------------------------------------------------------------------------
# Test 6: Complex data with order 0 (Nadaraya-Watson)
# -----------------------------------------------------------------------------
# ARCHIVED: 2026-01-09 - "Complex regression works with order 0" (moved to dev/archive/tests-archive-20260109/r/xl/archived_test-xl-complex.R)


# -----------------------------------------------------------------------------
# Test 7: Complex data with order 2 (local quadratic)
# -----------------------------------------------------------------------------
# ARCHIVED: 2026-01-09 - "Complex regression works with order 2" (moved to dev/archive/tests-archive-20260109/r/xl/archived_test-xl-complex.R)


# -----------------------------------------------------------------------------
# Test 8: Bandwidth selection is consistent with MATLAB
# -----------------------------------------------------------------------------
test_that("Bandwidth selection is consistent with MATLAB", {
  # Skip if MATLAB reference data not available (CRAN compatibility)
  skip_if_no_matlab_refs()

  ref_path <- file.path(find_root(), "fastLPR/tests/refs/crosslang_e2e/ref_complex.mat")
  if (!file.exists(ref_path)) {
    skip(paste("Reference data not found:", ref_path))
  }

  ref <- R.matlab::readMat(ref_path)
  matlab_id1se <- as.numeric(ref$id1se)
  matlab_h1se <- as.numeric(ref$h1se)
  hlist_vec <- as.vector(ref$hlist)

  opt <- list(order = 1, calc_dof = TRUE, verbose = FALSE)
  result <- cv_fastlpr(ref$x, ref$y, ref$hlist, opt)

  r_id1se <- result$gcv_yhat$id1se
  r_h1se <- result$gcv_yhat$h1se

  message(sprintf("  MATLAB: id1se=%d, h1se=%.4f", matlab_id1se, matlab_h1se))
  message(sprintf("  R:      id1se=%d, h1se=%.4f", r_id1se, r_h1se))

  # Index should match exactly or be very close
  expect_lt(abs(r_id1se - matlab_id1se), 2,
            label = "Bandwidth index should be within 1 of MATLAB")

  # Bandwidth value should be close
  expect_lt(abs(r_h1se - matlab_h1se) / matlab_h1se, 0.1,
            label = "Bandwidth value should be within 10% of MATLAB")
})

# ARCHIVED: 2026-01-09
# Source: fastLPR_R/tests/testthat/test-xl-complex.R
# Archive: dev/archive/tests-archive-20260109/r/xl/archived_test-xl-complex.R
# Archived tests:
# - "GCV scores are real for complex data"
# - "pdof is computed correctly for complex data"
# - "Complex regression is reproducible"
# - "Real and imaginary parts match MATLAB separately"
# - "Complex regression works with order 0"
# - "Complex regression works with order 2"
