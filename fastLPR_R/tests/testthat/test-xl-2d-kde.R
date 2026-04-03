# Integration test for 2D KDE against MATLAB reference
# Task 4.2: Verify 2D KDE implementation matches MATLAB reference data
#
# 2D KDE performance (as of 2025-12-14):
# - Current ratio: ~2.88x MATLAB (EXCELLENT - target was < 3x!)
# - BW MaxErr: 0.00 (exact bandwidth match)
# - id1se: 17 (exact index match)
# - Status: PASS
#
# Note: This test monitors if Rcpp is being used, which is the key metric
# after optimization fixes. Even without Rcpp, pure R achieves < 3x ratio.
#
# Reference data structure (from ref_2d_kde.mat):
# - x: 2000 x 2 matrix (data points)
# - hlist: 20 x 2 matrix (bandwidth candidates)
# - h.selected: [0.6951928, 0.6951928]
# - id1se: 17
# - elapsed: 0.1249718 seconds (MATLAB time)
# - lcv.m: 20 values (LCV scores for each bandwidth)

context("Cross-validation: 2D KDE against MATLAB")

test_that("2D KDE matches MATLAB reference", {
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
  ref_path <- file.path(project_root, "fastLPR", "tests", "refs", "crosslang_e2e", "ref_2d_kde.mat")

  # Skip if reference file not found
  skip_if_not(file.exists(ref_path),
              message = paste("Reference file not found:", ref_path))

  # Report Rcpp status (KEY METRIC for optimization)
  rcpp_status <- FALSE
  if (exists("rcpp_available") && is.function(rcpp_available)) {
    rcpp_status <- rcpp_available()
    if (rcpp_status && exists("get_rcpp_info") && is.function(get_rcpp_info)) {
      info <- tryCatch(get_rcpp_info(), error = function(e) NULL)
      if (!is.null(info)) {
        message(sprintf("Rcpp: ENABLED, %d threads, OpenMP: %s",
                        info$num_threads,
                        ifelse(info$num_threads > 1, "YES", "NO")))
      } else {
        message("Rcpp: ENABLED (info unavailable)")
      }
    } else if (!rcpp_status) {
      message("Rcpp: NOT available - 2D KDE using pure R fallback")
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
  expect_equal(dx, 2, label = "Expected 2D data")
  message(sprintf("Test data: N=%d, dx=%d, hlist size=%d x %d",
                  n, dx, nrow(hlist), ncol(hlist)))

  # Get MATLAB timing
  matlab_time <- if (!is.null(mat$elapsed)) as.numeric(mat$elapsed) else NA
  message(sprintf("MATLAB reference time: %.3fs", matlab_time))

  # Configure options (matching MATLAB defaults)
  opt <- list(
    order = 0,       # KDE
    calc_dof = FALSE
    # Note: flag_power2 defaults to TRUE for 2D (unlike 3D where it's FALSE)
  )

  # Time the R execution
  r_time <- system.time({
    result <- cv_fastkde(x, hlist, opt)
  })[["elapsed"]]

  # Calculate speed ratio
  ratio <- if (!is.na(matlab_time) && matlab_time > 0) r_time / matlab_time else NA

  # Report timing
  message(sprintf("2D KDE: R=%.3fs, MATLAB=%.3fs, Ratio=%.2fx",
                  r_time,
                  ifelse(is.na(matlab_time), 0, matlab_time),
                  ifelse(is.na(ratio), 0, ratio)))

  # Compare bandwidth selection (REQ-5: All metrics must pass)
  m_bw <- as.vector(mat$h.selected)
  r_bw <- as.vector(result$h)

  bw_maxerr <- max(abs(r_bw - m_bw))
  message(sprintf("Bandwidth: R=[%.4f,%.4f], MATLAB=[%.4f,%.4f]",
                  r_bw[1], r_bw[2], m_bw[1], m_bw[2]))
  message(sprintf("BW MaxErr: %.2e", bw_maxerr))

  # Bandwidth selection MUST match
  # Threshold: 0.05 for 2D
  expect_lt(bw_maxerr, 0.05,
            label = sprintf("2D KDE BW MaxErr (%.4f)", bw_maxerr))

  # Compare selected bandwidth indices
  if (!is.null(mat$id1se) && !is.null(result$lcv$id1se)) {
    m_id1se <- as.integer(mat$id1se)
    r_id1se <- result$lcv$id1se
    message(sprintf("id1se: R=%d, MATLAB=%d, Match=%s",
                    r_id1se, m_id1se, ifelse(r_id1se == m_id1se, "YES", "NO")))

    # Index should match exactly
    expect_equal(r_id1se, m_id1se,
                 label = sprintf("id1se: R=%d, MATLAB=%d", r_id1se, m_id1se))
  }

  # Compare LCV values using RELATIVE error per CLAUDE.md 8.4
  # For KDE (order=0): LCV RelErr < 0.01 (1%)
  # Note: Absolute errors can be large (e.g., 0.184) when LCV values are large (e.g., -6228)
  # but relative error is what matters for log-likelihood comparisons
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

  # Speed ratio check (target < 8x for JSS, ideally < 3x after optimization)
  if (!is.na(ratio)) {
    # Report performance category with Rcpp status
    if (ratio < 1.0) {
      perf_status <- "FASTER than MATLAB"
    } else if (ratio < 3.0) {
      perf_status <- "EXCELLENT (< 3x - target achieved)"
    } else if (ratio < 5.0) {
      perf_status <- "GOOD (< 5x)"
    } else if (ratio < 8.0) {
      perf_status <- "ACCEPTABLE (< 8x JSS threshold)"
    } else if (ratio < 15.0) {
      perf_status <- "NEEDS OPTIMIZATION (8-15x)"
    } else {
      perf_status <- "CRITICAL (> 15x)"
    }

    message(sprintf("Performance: %s", perf_status))

    # For JSS publication, ratio must be < 8x
    expect_lt(ratio, 8,
              label = sprintf("Speed ratio (%.2fx)", ratio))

    # Additional warning if not meeting target
    if (ratio >= 3.0) {
      message(sprintf("NOTE: Ratio %.2fx > 3x target. Rcpp: %s",
                      ratio, ifelse(rcpp_status, "enabled", "disabled")))
    }
  }

  # Verify output structure
  expect_true(is.list(result))
  expect_true(!is.null(result$fhat))
  expect_true(!is.null(result$h))
  expect_true(!is.null(result$xlist))
  expect_equal(length(result$h), 2)
  expect_equal(length(result$xlist), 2)

  # Check density is non-negative
  expect_true(all(result$fhat >= 0), label = "Density non-negative")
})

# ARCHIVED: 2026-01-09 - "2D KDE basic functionality works with test data" (moved to dev/archive/tests-archive-20260109/r/xl/archived_test-xl-2d-kde.R)


test_that("2D KDE with mixture distribution selects appropriate bandwidth", {
  # Skip if functions not loaded
  skip_if_not(exists("cv_fastkde") && is.function(cv_fastkde),
              message = "cv_fastkde not available")
  skip_if_not(exists("get_hlist") && is.function(get_hlist),
              message = "get_hlist not available")

  # Test bandwidth selection on bimodal 2D data
  set.seed(42)
  n1 <- 100
  n2 <- 100

  # Two well-separated clusters
  x1 <- cbind(rnorm(n1, -2, 0.5), rnorm(n1, -2, 0.5))
  x2 <- cbind(rnorm(n2, 2, 0.5), rnorm(n2, 2, 0.5))
  x <- rbind(x1, x2)

  hlist <- get_hlist(c(7, 7), list(c(0.1, 1.5), c(0.1, 1.5)))

  kde <- cv_fastkde(x, hlist, list(verbose = FALSE, N = c(50, 50)))

  # Bandwidth should be selected
  expect_true(all(kde$h >= 0.1))
  expect_true(all(kde$h <= 1.5))

  # Should have LCV results
  expect_true(!is.null(kde$lcv))
  expect_true(!is.null(kde$lcv$lcv_m))
})

# ARCHIVED: 2026-01-09 - "2D KDE reproducibility check" (moved to dev/archive/tests-archive-20260109/r/xl/archived_test-xl-2d-kde.R)


# ARCHIVED: 2026-01-09 - "2D KDE Rcpp acceleration detection" (moved to dev/archive/tests-archive-20260109/r/xl/archived_test-xl-2d-kde.R)


# ARCHIVED: 2026-01-09 - "2D KDE density integrates to approximately 1" (moved to dev/archive/tests-archive-20260109/r/xl/archived_test-xl-2d-kde.R)


# ARCHIVED: 2026-01-09
# Source: fastLPR_R/tests/testthat/test-xl-2d-kde.R
# Archive: dev/archive/tests-archive-20260109/r/xl/archived_test-xl-2d-kde.R
# Archived tests:
# - "2D KDE basic functionality works with test data"
# - "2D KDE reproducibility check"
# - "2D KDE Rcpp acceleration detection"
# - "2D KDE density integrates to approximately 1"
