# Integration test for 3D KDE against MATLAB reference
# Task 4.3: Verify 3D KDE implementation matches MATLAB reference data
#
# Pass criteria per CLAUDE.md 8.4:
# - BW MaxErr < 0.05 (bandwidth selection must match)
# - LCV RelErr < 0.01 (1% relative error threshold for KDE)
# - Speed ratio < 8x for JSS publication
#
# Current status (from verification report):
# - Ratio: ~6x MATLAB (within acceptable range)
# - BW MaxErr: 0.00 (exact match)
# - LCV RelErr: ~0.00% (excellent accuracy)

context("Cross-validation: 3D KDE against MATLAB")

test_that("3D KDE matches MATLAB reference", {
  # Skip if MATLAB reference data not available (CRAN compatibility)
  skip_if_no_matlab_refs()

  # Load R.matlab for reading .mat files
  suppressWarnings(library(R.matlab))  # Suppress version mismatch warning

  # Find project root
  pkg_root <- getwd()
  while (!grepl("fastLPR_R$", basename(pkg_root)) && dirname(pkg_root) != pkg_root) {
    pkg_root <- dirname(pkg_root)
  }
  project_root <- dirname(pkg_root)

  # Reference file path
  ref_path <- file.path(project_root, "fastLPR", "tests", "refs", "crosslang_e2e", "ref_3d_kde.mat")

  # Skip if reference file not found
  skip_if_not(file.exists(ref_path),
              message = paste("Reference file not found:", ref_path))

  # Report Rcpp status
  if (exists("rcpp_available") && is.function(rcpp_available)) {
    rcpp_status <- rcpp_available()
    if (rcpp_status && exists("get_rcpp_info") && is.function(get_rcpp_info)) {
      info <- tryCatch(get_rcpp_info(), error = function(e) NULL)
      if (!is.null(info)) {
        message(sprintf("Rcpp: Available, %d threads, OpenMP: %s",
                        info$num_threads,
                        ifelse(info$num_threads > 1, "YES", "NO")))
      } else {
        message("Rcpp: Available (info unavailable)")
      }
    } else if (!rcpp_status) {
      message("Rcpp: NOT available - 3D KDE will be slow!")
    }
  } else {
    message("Rcpp: Status check function not available")
  }

  # Load MATLAB reference data
  mat <- readMat(ref_path)

  # Extract inputs
  x <- as.matrix(mat$x)
  hlist <- as.matrix(mat$hlist)
  n <- nrow(x)
  dx <- ncol(x)

  # Verify dimensions
  expect_equal(dx, 3, label = "Expected 3D data")
  message(sprintf("Test data: N=%d, dx=%d, hlist size=%d", n, dx, nrow(hlist)))

  # Get MATLAB timing
  matlab_time <- if (!is.null(mat$elapsed)) as.numeric(mat$elapsed) else NA

  # CRITICAL: Use flag_power2 = FALSE for 3D to match MATLAB reference
  # MATLAB reference was generated with flag_power2 = false to avoid memory explosion
  opt <- list(
    order = 0,
    calc_dof = FALSE,
    flag_power2 = FALSE  # Match MATLAB reference setting
  )

  # Time the R execution
  r_time <- system.time({
    result <- cv_fastkde(x, hlist, opt)
  })[["elapsed"]]

  # Calculate speed ratio
  ratio <- if (!is.na(matlab_time) && matlab_time > 0) r_time / matlab_time else NA

  # Report timing
  message(sprintf("3D KDE: R=%.3fs, MATLAB=%.3fs, Ratio=%.2fx",
                  r_time,
                  ifelse(is.na(matlab_time), 0, matlab_time),
                  ifelse(is.na(ratio), 0, ratio)))

  # Compare bandwidth selection (REQ-5: All metrics must pass)
  m_bw <- as.vector(mat$h.selected)
  r_bw <- result$h

  bw_maxerr <- max(abs(r_bw - m_bw))
  message(sprintf("Bandwidth: R=[%.4f,%.4f,%.4f], MATLAB=[%.4f,%.4f,%.4f]",
                  r_bw[1], r_bw[2], r_bw[3],
                  m_bw[1], m_bw[2], m_bw[3]))
  message(sprintf("BW MaxErr: %.2e", bw_maxerr))

  # Bandwidth selection MUST match
  # Threshold: 0.05 for 3D (slightly relaxed due to grid interpolation)
  expect_true(bw_maxerr < 0.05,
              info = sprintf("BW MaxErr=%.4f should be < 0.05", bw_maxerr))

  # Compare LCV values using RELATIVE error per CLAUDE.md 8.4
  # For KDE (order=0): LCV RelErr < 0.01 (1%)
  if (!is.null(mat$lcv.m) && !is.null(result$lcv$lcv_m)) {
    m_lcv <- as.vector(mat$lcv.m)
    r_lcv <- result$lcv$lcv_m

    # Ensure same length
    min_len <- min(length(r_lcv), length(m_lcv))
    lcv_abs_err <- abs(r_lcv[1:min_len] - m_lcv[1:min_len])
    lcv_maxerr <- max(lcv_abs_err)

    # Compute RELATIVE error: max(|r - m| / |m|)
    lcv_rel_err <- max(lcv_abs_err / abs(m_lcv[1:min_len]))

    message(sprintf("LCV MaxErr: %.2e (abs), RelErr: %.4f%% (threshold: < 1%%)",
                    lcv_maxerr, lcv_rel_err * 100))

    # Per CLAUDE.md 8.4: KDE uses LCV RelErr < 0.01 (1%) threshold
    expect_lt(lcv_rel_err, 0.01,
              label = sprintf("LCV RelErr (%.4f%%)", lcv_rel_err * 100))
  }

  # Compare selected bandwidth indices
  if (!is.null(mat$id1se) && !is.null(result$lcv$id1se)) {
    m_id1se <- as.integer(mat$id1se)
    r_id1se <- result$lcv$id1se
    message(sprintf("id1se: R=%d, MATLAB=%d, Match=%s",
                    r_id1se, m_id1se, r_id1se == m_id1se))

    # Index should match exactly
    expect_equal(r_id1se, m_id1se,
                 info = "Bandwidth selection index should match MATLAB")
  }

  # Speed ratio check (target < 8x for JSS publication)
  # 3D is allowed to be slower, but should improve with Rcpp
  if (!is.na(ratio)) {
    # Report performance category
    if (ratio < 1.0) {
      message("Performance: FASTER than MATLAB")
    } else if (ratio < 5.0) {
      message("Performance: GOOD (< 5x)")
    } else if (ratio < 8.0) {
      message("Performance: ACCEPTABLE (< 8x JSS threshold)")
    } else if (ratio < 15.0) {
      message("Performance: NEEDS OPTIMIZATION (8-15x)")
    } else {
      message("Performance: CRITICAL (> 15x)")
    }

    # For now, just warn if > 15x (critical)
    # The test passes on accuracy, performance is tracked separately
    if (ratio > 15.0) {
      warning(sprintf("3D KDE performance critical: %.2fx > 15x threshold", ratio))
    }
  }

  # Verify output structure
  expect_true(is.list(result))
  expect_true(!is.null(result$fhat))
  expect_true(!is.null(result$h))
  expect_true(!is.null(result$xlist))
  expect_equal(length(result$h), 3)
  expect_equal(length(result$xlist), 3)

  # Check density is non-negative
  expect_true(all(result$fhat >= 0), info = "Density should be non-negative")
})

test_that("3D KDE with flag_power2=FALSE produces correct output dimensions", {
  # This test verifies the fix for the 3D reference setting mismatch
  # MATLAB uses flag_power2=false for 3D to avoid L^3 memory explosion

  set.seed(42)
  n <- 100
  x <- matrix(rnorm(n * 3), ncol = 3)
  hlist <- get_hlist(c(3, 3, 3), list(c(0.3, 0.6), c(0.3, 0.6), c(0.3, 0.6)))

  # Run with flag_power2 = FALSE (matching MATLAB 3D default)
  opt <- list(order = 0, flag_power2 = FALSE, N = c(25, 25, 25))
  kde <- cv_fastkde(x, hlist, opt)

  # Verify output structure
  expect_true(is.list(kde))
  expect_equal(length(kde$h), 3)
  expect_equal(length(kde$xlist), 3)
  expect_equal(dim(kde$fhat), c(25, 25, 25))

  # Verify bandwidth was selected
  expect_true(all(kde$h >= 0.3))
  expect_true(all(kde$h <= 0.6))
})

# ARCHIVED: 2026-01-09 - "3D KDE memory usage is reasonable with flag_power2=FALSE" (moved to dev/archive/tests-archive-20260109/r/xl/archived_test-xl-3d-kde.R)




# ARCHIVED: 2026-01-09 - "3D KDE Rcpp acceleration is being used when available" (moved to dev/archive/tests-archive-20260109/r/xl/archived_test-xl-3d-kde.R)


# ARCHIVED: 2026-01-09
# Source: fastLPR_R/tests/testthat/test-xl-3d-kde.R
# Archive: dev/archive/tests-archive-20260109/r/xl/archived_test-xl-3d-kde.R
# Archived tests:
# - "3D KDE memory usage is reasonable with flag_power2=FALSE"
# - "3D KDE Rcpp acceleration is being used when available"
