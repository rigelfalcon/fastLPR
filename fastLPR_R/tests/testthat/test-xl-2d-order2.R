# Integration Test: 2D Order 2 Local Quadratic Regression
# Cross-validation test comparing R implementation against MATLAB reference
#
# Test Case: 2D Local Quadratic (order=2)
# - More polynomial terms than Order 1, computationally heavier
# - Design matrix includes: 1, x1, x2, x1^2, x1*x2, x2^2 (6 terms)
# - Reference: fastLPR/tests/refs/crosslang_e2e/ref_2d_order2.mat

context("Cross-validation: 2D Order 2")

test_that("2D Order 2 matches MATLAB reference", {
  # Skip if MATLAB reference data not available (CRAN compatibility)
  skip_if_no_matlab_refs()

  # Load MATLAB reference data
  ref_path <- file.path(find_root(), "fastLPR", "tests", "refs", "crosslang_e2e", "ref_2d_order2.mat")

  if (!file.exists(ref_path)) {
    skip(paste("Reference file not found:", ref_path,
               "\nRun MATLAB first: matlab -batch \"run('fastLPR/tests/generate_refs.m')\""))
  }

  ref <- R.matlab::readMat(ref_path)

  # Extract reference data
  x <- ref$x
  y <- matrix(ref$y, ncol = 1)
  hlist <- ref$hlist

  # Extract MATLAB reference values
  matlab_time <- as.numeric(ref$elapsed)
  matlab_gcv <- as.vector(ref$gcv.m)
  matlab_h <- as.vector(ref$h1se)
  matlab_yhat <- as.vector(ref$yhat.mean)
  matlab_id1se <- as.integer(ref$id1se)

  # Run R implementation with timing
  opt <- list(order = 2, calc_dof = TRUE)

  # Use MATLAB random vectors for DOF reproducibility if available
  if (!is.null(ref$dof.random.vectors)) {
    opt$dof_random_vectors <- ref$dof.random.vectors
  }

  r_time <- system.time({
    result <- cv_fastlpr(x, y, hlist, opt)
  })[["elapsed"]]

  # Extract R results
  r_gcv <- result$gcv_yhat$gcv_m
  r_h <- result$gcv_yhat$h1se
  r_yhat <- as.vector(result$yhat)
  r_id1se <- result$gcv_yhat$id1se

  # Compute error metrics
  gcv_maxerr <- max(abs(r_gcv - matlab_gcv))
  bw_maxerr <- max(abs(r_h - matlab_h))
  mean_maxerr <- max(abs(r_yhat - matlab_yhat))
  id_diff <- abs(r_id1se - matlab_id1se)

  # Compute timing ratio
  ratio <- r_time / matlab_time

  # Print detailed diagnostics
  message(sprintf("\n=== 2D Order 2 Cross-Validation Results ==="))
  message(sprintf("  N = %d, dx = 2, order = 2", nrow(x)))
  message(sprintf("  MATLAB time: %.3f sec", matlab_time))
  message(sprintf("  R time: %.3f sec (ratio: %.2fx)", r_time, ratio))
  message(sprintf("  GCV MaxErr: %.2e", gcv_maxerr))
  message(sprintf("  BW MaxErr: %.2e (h_R = [%.4f, %.4f], h_MATLAB = [%.4f, %.4f])",
                  bw_maxerr, r_h[1], r_h[2], matlab_h[1], matlab_h[2]))
  message(sprintf("  Mean MaxErr: %.2e", mean_maxerr))
  message(sprintf("  Index diff: %d (R: %d, MATLAB: %d)", id_diff, r_id1se, matlab_id1se))

  # Thresholds based on TOL_CROSSLANG (unified across all platforms)
  # - GCV MaxErr < 0.01 for regression
  # - BW MaxErr < 0.05 for 2D (TOL_CROSSLANG excludes BW, but 0.02 works for 1D)
  # - Mean MaxErr < 0.05 (unified, actual error ~0.013)
  # - Speed ratio < 8x

  # Assertions
  expect_lt(gcv_maxerr, 0.01,
            label = sprintf("GCV MaxErr (%.2e) should be < 0.01", gcv_maxerr))

  expect_lt(bw_maxerr, 0.05,
            label = sprintf("BW MaxErr (%.2e) should be < 0.05 for 2D", bw_maxerr))

  expect_lt(mean_maxerr, TOL_CROSSLANG$mean_maxerr,  # 0.05 (unified)
            label = sprintf("Mean MaxErr (%.2e) should be < 0.05", mean_maxerr))

  expect_lt(ratio, TOL_CROSSLANG$speed_ratio,  # 8x (unified)
            label = sprintf("Speed ratio (%.2fx) should be < 8x", ratio))

  # Print final status
  all_pass <- gcv_maxerr < 0.01 && bw_maxerr < 0.05 && mean_maxerr < TOL_CROSSLANG$mean_maxerr && ratio < TOL_CROSSLANG$speed_ratio
  message(sprintf("  STATUS: %s", ifelse(all_pass, "PASS", "FAIL")))
})

# ARCHIVED: 2026-01-09 - "2D Order 2 handles quadratic surface well" (moved to dev/archive/tests-archive-20260109/r/xl/archived_test-xl-2d-order2.R)


# ARCHIVED: 2026-01-09 - "2D Order 2 reproducibility check" (moved to dev/archive/tests-archive-20260109/r/xl/archived_test-xl-2d-order2.R)


# ARCHIVED: 2026-01-10 - "2D Order 2 GCV bandwidth selection works" (moved to dev/archive/tests-archive-20260110/r/xl/archived_test-xl-2d-order2.R)
# Reason: Test unification - not in Python reference


# ARCHIVED: 2026-01-09 - "2D Order 2 with single bandwidth works" (moved to dev/archive/tests-archive-20260109/r/xl/archived_test-xl-2d-order2.R)


# ARCHIVED: 2026-01-09
# Source: fastLPR_R/tests/testthat/test-xl-2d-order2.R
# Archive: dev/archive/tests-archive-20260109/r/xl/archived_test-xl-2d-order2.R
# Archived tests:
# - "2D Order 2 handles quadratic surface well"
# - "2D Order 2 reproducibility check"
# - "2D Order 2 with single bandwidth works"

# ARCHIVED: 2026-01-10
# Archive: dev/archive/tests-archive-20260110/r/xl/archived_test-xl-2d-order2.R
# Archived tests:
# - "2D Order 2 GCV bandwidth selection works"
