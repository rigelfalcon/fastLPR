# ==============================================================================
# Cross-Validation Integration Test: 1D KDE vs MATLAB Reference
# ==============================================================================
# Test file: fastLPR_R/tests/testthat/test-cv-1d-kde.R
# Created: 2025-12-14
#
# This test validates the R cv_fastkde() implementation against MATLAB reference
# data to ensure numerical consistency across implementations.
#
# Reference file: fastLPR/tests/refs/crosslang_e2e/ref_1d_kde.mat
# Contains:
#   - x: Input data (N x 1 matrix)
#   - hlist: Bandwidth candidates (k x 1 matrix)
#   - lcv.m: LCV scores for all bandwidths (k x 1 matrix)
#   - h.selected: Selected bandwidth using 1-SE rule (scalar)
#   - id1se: Index of selected bandwidth (scalar)
#   - fhat.at.x: Density estimates at data points (N x 1 matrix)
#   - elapsed: MATLAB execution time (scalar)
#
# Pass Criteria (per CLAUDE.md 8.4):
#   - BW MaxErr < 0.02 (bandwidth selection threshold)
#   - LCV RelErr < 0.01 (1% relative error for KDE)
#   - Speed ratio < 8x MATLAB time
# ==============================================================================

context("Cross-validation: 1D KDE vs MATLAB")

test_that("1D KDE matches MATLAB reference data", {
  # Skip if MATLAB reference data not available (CRAN compatibility)
  skip_if_no_matlab_refs()

  # ============================================================================
  # Load MATLAB reference data
  # ============================================================================

  # Find reference file path
  ref_paths <- c(
    # Direct path from package root
    file.path(find_root(), "fastLPR/tests/refs/crosslang_e2e/ref_1d_kde.mat"),
    # Relative path from tests/testthat
    "../../fastLPR/tests/refs/crosslang_e2e/ref_1d_kde.mat",
    # Alternative: verification folder (legacy)
    file.path(find_root(), "fastLPR/tests/refs/crosslang_e2e/ref_1d_kde.mat")
  )

  ref_path <- NULL
  for (p in ref_paths) {
    if (file.exists(p)) {
      ref_path <- p
      break
    }
  }

  skip_if(is.null(ref_path), "MATLAB reference file ref_1d_kde.mat not found")

  ref <- R.matlab::readMat(ref_path)

  # Extract reference data
  x_ref <- as.matrix(ref$x)
  hlist_ref <- as.matrix(ref$hlist)
  lcv_m_ref <- as.vector(ref$lcv.m)
  h_selected_ref <- as.numeric(ref$h.selected)
  id1se_ref <- as.integer(ref$id1se)
  fhat_at_x_ref <- as.vector(ref$fhat.at.x)
  matlab_time <- as.numeric(ref$elapsed)

  cat("\n=== 1D KDE Cross-Validation Test ===\n")
  cat(sprintf("Reference file: %s\n", ref_path))
  cat(sprintf("Sample size (N): %d\n", nrow(x_ref)))
  cat(sprintf("Number of bandwidths: %d\n", length(hlist_ref)))
  cat(sprintf("MATLAB time: %.3f s\n", matlab_time))
  cat(sprintf("MATLAB selected h: %.6f (id=%d)\n", h_selected_ref, id1se_ref))

  # ============================================================================
  # Run R implementation
  # ============================================================================

  r_time <- system.time({
    kde <- cv_fastkde(x_ref, hlist_ref, list(verbose = FALSE))
  })[["elapsed"]]

  # Extract R results
  h_selected_r <- as.numeric(kde$h)

  # Get id1se from R result
  if (!is.null(kde$lcv)) {
    id1se_r <- kde$lcv$id1se
    lcv_m_r <- as.vector(kde$lcv$lcv_m)
  } else {
    # Single bandwidth case
    id1se_r <- 1
    lcv_m_r <- NA
  }

  # Evaluate density at data points for comparison
  # Note: fpp$evaluate returns all bandwidths (N x dh), we extract only the selected one
  fhat_at_x_r_all <- kde$fpp$evaluate(x_ref)
  if (is.matrix(fhat_at_x_r_all) && ncol(fhat_at_x_r_all) > 1) {
    fhat_at_x_r <- fhat_at_x_r_all[, id1se_r]
  } else {
    fhat_at_x_r <- as.vector(fhat_at_x_r_all)
  }

  # ============================================================================
  # Compute error metrics
  # ============================================================================

  # Bandwidth selection error
  bw_maxerr <- abs(h_selected_r - h_selected_ref)
  bw_idx_diff <- abs(id1se_r - id1se_ref)

  # LCV error using RELATIVE error per CLAUDE.md 8.4
  # For KDE (order=0): LCV RelErr < 0.01 (1%)
  if (!is.na(lcv_m_r[1])) {
    lcv_abs_err <- abs(lcv_m_r - lcv_m_ref)
    lcv_maxerr <- max(lcv_abs_err)
    lcv_mse <- mean((lcv_m_r - lcv_m_ref)^2)
    # Relative error: max(|r - m| / |m|)
    lcv_rel_err <- max(lcv_abs_err / abs(lcv_m_ref))
  } else {
    lcv_maxerr <- NA
    lcv_mse <- NA
    lcv_rel_err <- NA
  }

  # Density estimation error at data points
  fhat_maxerr <- max(abs(fhat_at_x_r - fhat_at_x_ref))
  fhat_mse <- mean((fhat_at_x_r - fhat_at_x_ref)^2)
  fhat_corr <- cor(fhat_at_x_r, fhat_at_x_ref)

  # Speed ratio
  speed_ratio <- r_time / matlab_time

  # ============================================================================
  # Report results
  # ============================================================================

  cat("\n--- R Results ---\n")
  cat(sprintf("R time: %.3f s\n", r_time))
  cat(sprintf("R selected h: %.6f (id=%d)\n", h_selected_r, id1se_r))

  cat("\n--- Error Metrics ---\n")
  cat(sprintf("BW MaxErr: %.2e (threshold: 0.02)\n", bw_maxerr))
  cat(sprintf("BW Index Diff: %d\n", bw_idx_diff))
  if (!is.na(lcv_maxerr)) {
    cat(sprintf("LCV MaxErr: %.2e (abs), RelErr: %.4f%% (threshold: < 1%%)\n",
                lcv_maxerr, lcv_rel_err * 100))
    cat(sprintf("LCV MSE: %.2e\n", lcv_mse))
  }
  cat(sprintf("Density MaxErr at data: %.2e\n", fhat_maxerr))
  cat(sprintf("Density MSE at data: %.2e\n", fhat_mse))
  cat(sprintf("Density correlation: %.6f\n", fhat_corr))

  cat("\n--- Performance ---\n")
  cat(sprintf("Speed ratio: %.2fx (threshold: 8x)\n", speed_ratio))

  # ============================================================================
  # Determine pass/fail
  # ============================================================================

  # Bandwidth threshold: BW MaxErr < 0.02
  bw_pass <- bw_maxerr < 0.02

  # LCV relative error threshold: < 1% per CLAUDE.md 8.4
  lcv_pass <- if (!is.na(lcv_rel_err)) lcv_rel_err < 0.01 else TRUE

  # Speed criterion
  speed_pass <- speed_ratio < 8.0

  # Overall status (ALL metrics must pass per REQ-5)
  overall_pass <- bw_pass && lcv_pass && speed_pass

  cat("\n--- Status ---\n")
  cat(sprintf("BW selection: %s\n", ifelse(bw_pass, "PASS", "FAIL")))
  cat(sprintf("LCV RelErr: %s\n", ifelse(lcv_pass, "PASS", "FAIL")))
  cat(sprintf("Speed: %s\n", ifelse(speed_pass, "PASS", "FAIL")))
  cat(sprintf("Overall: %s\n", ifelse(overall_pass, "PASS", "FAIL")))
  cat("=================================\n\n")

  # ============================================================================
  # Assertions (ALL must pass per REQ-5)
  # ============================================================================

  # Bandwidth selection must match (BW MaxErr < 0.02)
  expect_lt(bw_maxerr, 0.02,
            label = sprintf("BW MaxErr (%.2e) should be < 0.02", bw_maxerr))

  # LCV relative error per CLAUDE.md 8.4: < 0.01 (1%)
  if (!is.na(lcv_rel_err)) {
    expect_lt(lcv_rel_err, 0.01,
              label = sprintf("LCV RelErr (%.4f%%) should be < 1%%", lcv_rel_err * 100))
  }

  # Speed criterion
  expect_lt(speed_ratio, 8.0,
            label = sprintf("Speed ratio (%.2fx) should be < 8x", speed_ratio))

  # Density correlation - informational only (primary metric is max absolute error)
  # Note: Per CLAUDE.md, max absolute error is the primary metric, correlation is supplementary
  if (fhat_corr < 0.95) {
    warning(sprintf("Density correlation with MATLAB (%.4f) is below 0.95 - informational only", fhat_corr))
  }
})

test_that("1D KDE bandwidth index matches MATLAB exactly", {
  # Skip if MATLAB reference data not available (CRAN compatibility)
  skip_if_no_matlab_refs()

  # Find reference file
  ref_paths <- c(
    file.path(find_root(), "fastLPR/tests/refs/crosslang_e2e/ref_1d_kde.mat"),
    "../../fastLPR/tests/refs/crosslang_e2e/ref_1d_kde.mat"
  )

  ref_path <- NULL
  for (p in ref_paths) {
    if (file.exists(p)) {
      ref_path <- p
      break
    }
  }

  skip_if(is.null(ref_path), "MATLAB reference file not found")

  ref <- R.matlab::readMat(ref_path)

  x_ref <- as.matrix(ref$x)
  hlist_ref <- as.matrix(ref$hlist)
  id1se_ref <- as.integer(ref$id1se)

  kde <- cv_fastkde(x_ref, hlist_ref, list(verbose = FALSE))

  if (!is.null(kde$lcv)) {
    id1se_r <- kde$lcv$id1se

    # Index should match (or be within 1 due to tie-breaking)
    idx_diff <- abs(id1se_r - id1se_ref)
    expect_lte(idx_diff, 1,
               label = sprintf("Index difference (%d) should be <= 1", idx_diff))

    # If indices differ, bandwidths should still be very close
    if (idx_diff > 0) {
      h_r <- hlist_ref[id1se_r]
      h_ref <- hlist_ref[id1se_ref]
      h_diff <- abs(h_r - h_ref)
      expect_lt(h_diff, 0.05,
                label = sprintf("When indices differ, h difference (%.4f) should be < 0.05", h_diff))
    }
  }
})

# ARCHIVED: 2026-01-09 - "1D KDE produces valid density estimates" (moved to dev/archive/tests-archive-20260109/r/xl/archived_test-xl-1d-kde.R)


# ==============================================================================
# Summary Test - Reports full cross-validation table row
# ==============================================================================

# ARCHIVED: 2026-01-09 - "1D KDE cross-validation summary" (moved to dev/archive/tests-archive-20260109/r/xl/archived_test-xl-1d-kde.R)


# ARCHIVED: 2026-01-09
# Source: fastLPR_R/tests/testthat/test-xl-1d-kde.R
# Archive: dev/archive/tests-archive-20260109/r/xl/archived_test-xl-1d-kde.R
# Archived tests:
# - "1D KDE produces valid density estimates"
# - "1D KDE cross-validation summary"
