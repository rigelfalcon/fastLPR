# ==============================================================================
# Cross-Validation Integration Test: 1D Order 1 (Local Linear) vs MATLAB Reference
# ==============================================================================
# Test file: fastLPR_R/tests/testthat/test-cv-1d-order1.R
# Created: 2025-12-29 (migrated from verify_r.R)
#
# This test validates the R cv_fastlpr() implementation against MATLAB reference
# data to ensure numerical consistency across implementations.
#
# Reference file: fastLPR/tests/refs/crosslang_e2e/ref_1d_order1.mat
# Contains:
#   - x: Input data (N x 1 matrix)
#   - y: Response data (N x 1 matrix)
#   - hlist: Bandwidth candidates (k x 1 matrix)
#   - gcv_m: GCV scores for all bandwidths (k x 1 matrix)
#   - h1se: Selected bandwidth using 1-SE rule (scalar)
#   - id1se: Index of selected bandwidth (scalar)
#   - yhat_mean: Fitted values at data points (N x 1 matrix)
#   - dof_random_vectors: Random vectors for DOF estimation
#   - elapsed: MATLAB execution time (scalar)
#
# Pass Criteria (per CLAUDE.md 8.4):
#   - BW MaxErr < 0.02 (bandwidth selection threshold)
#   - GCV MaxErr < 0.01 (GCV score threshold)
#   - Mean MaxErr < 0.05 (prediction threshold for 1D)
#   - Speed ratio < 8x MATLAB time
# ==============================================================================

context("Cross-validation: 1D Order 1 vs MATLAB")

# Helper function from verify_r.R
safe_extract <- function(mat, field) {
  if (!is.null(mat[[field]])) {
    return(as.vector(mat[[field]]))
  }
  # Try with dots replaced by underscores
  field_alt <- gsub("_", ".", field)
  if (!is.null(mat[[field_alt]])) {
    return(as.vector(mat[[field_alt]]))
  }
  return(NULL)
}

test_that("1D Order 1 matches MATLAB reference data", {
  # Skip if MATLAB reference data not available (CRAN compatibility)
  skip_if_no_matlab_refs()

  # ============================================================================
  # Load MATLAB reference data
  # ============================================================================

  # Find reference file path
  ref_paths <- c(
    file.path(find_root(), "fastLPR/tests/refs/crosslang_e2e/ref_1d_order1.mat"),
    "../../fastLPR/tests/refs/crosslang_e2e/ref_1d_order1.mat",
    file.path(find_root(), "fastLPR/tests/refs/crosslang_e2e/ref_1d_order1.mat")
  )

  ref_path <- NULL
  for (p in ref_paths) {
    if (file.exists(p)) {
      ref_path <- p
      break
    }
  }

  skip_if(is.null(ref_path), "MATLAB reference file ref_1d_order1.mat not found")

  ref <- R.matlab::readMat(ref_path)

  # Extract reference data
  x_ref <- matrix(ref$x, ncol = 1)
  y_ref <- matrix(ref$y, ncol = 1)
  hlist_ref <- matrix(as.vector(ref$hlist), ncol = 1)
  gcv_m_ref <- safe_extract(ref, "gcv_m")
  h1se_ref <- as.numeric(ref$h1se)
  id1se_ref <- as.integer(ref$id1se)
  yhat_mean_ref <- safe_extract(ref, "yhat_mean")
  matlab_time <- as.numeric(ref$elapsed)

  cat("\n=== 1D Order 1 Cross-Validation Test ===\n")
  cat(sprintf("Reference file: %s\n", ref_path))
  cat(sprintf("Sample size (N): %d\n", nrow(x_ref)))
  cat(sprintf("Number of bandwidths: %d\n", length(hlist_ref)))
  cat(sprintf("MATLAB time: %.3f s\n", matlab_time))
  cat(sprintf("MATLAB selected h: %.6f (id=%d)\n", h1se_ref, id1se_ref))

  # ============================================================================
  # Run R implementation
  # ============================================================================

  # FIX 2025-12-01: Use MATLAB random vectors for exact DOF reproducibility
  opt <- list(order = 1, calc_dof = TRUE)
  if (!is.null(ref$dof.random.vectors)) {
    opt$dof_random_vectors <- ref$dof.random.vectors
  }

  r_time <- system.time({
    res <- cv_fastlpr(x_ref, y_ref, hlist_ref, opt)
  })[["elapsed"]]

  # Extract R results
  h1se_r <- res$gcv_yhat$h1se
  id1se_r <- res$gcv_yhat$id1se
  gcv_m_r <- res$gcv_yhat$gcv_m
  yhat_r <- as.vector(res$yhat)

  # ============================================================================
  # Compute error metrics
  # ============================================================================

  # Bandwidth selection error
  bw_maxerr <- abs(h1se_r - h1se_ref)
  bw_idx_diff <- abs(id1se_r - id1se_ref)

  # GCV error
  gcv_maxerr <- max(abs(gcv_m_r - gcv_m_ref))
  gcv_mse <- mean((gcv_m_r - gcv_m_ref)^2)

  # Mean prediction error
  mean_maxerr <- max(abs(yhat_r - yhat_mean_ref))
  mean_mse <- mean((yhat_r - yhat_mean_ref)^2)

  # Speed ratio
  speed_ratio <- r_time / matlab_time

  # ============================================================================
  # Report results
  # ============================================================================

  cat("\n--- R Results ---\n")
  cat(sprintf("R time: %.3f s\n", r_time))
  cat(sprintf("R selected h: %.6f (id=%d)\n", h1se_r, id1se_r))

  cat("\n--- Error Metrics ---\n")
  cat(sprintf("BW MaxErr: %.2e (threshold: 0.02)\n", bw_maxerr))
  cat(sprintf("BW Index Diff: %d\n", bw_idx_diff))
  cat(sprintf("GCV MaxErr: %.2e (threshold: 0.01)\n", gcv_maxerr))
  cat(sprintf("GCV MSE: %.2e\n", gcv_mse))
  cat(sprintf("Mean MaxErr: %.2e (threshold: 0.05)\n", mean_maxerr))
  cat(sprintf("Mean MSE: %.2e\n", mean_mse))

  cat("\n--- Performance ---\n")
  cat(sprintf("Speed ratio: %.2fx (threshold: 8x)\n", speed_ratio))

  # ============================================================================
  # Determine pass/fail
  # ============================================================================

  bw_pass <- bw_maxerr < 0.02
  gcv_pass <- gcv_maxerr < 0.01
  mean_pass <- mean_maxerr < 0.05
  speed_pass <- speed_ratio < 8.0

  overall_pass <- bw_pass && gcv_pass && mean_pass && speed_pass

  cat("\n--- Status ---\n")
  cat(sprintf("BW selection: %s\n", ifelse(bw_pass, "PASS", "FAIL")))
  cat(sprintf("GCV accuracy: %s\n", ifelse(gcv_pass, "PASS", "FAIL")))
  cat(sprintf("Mean accuracy: %s\n", ifelse(mean_pass, "PASS", "FAIL")))
  cat(sprintf("Speed: %s\n", ifelse(speed_pass, "PASS", "FAIL")))
  cat(sprintf("Overall: %s\n", ifelse(overall_pass, "PASS", "FAIL")))
  cat("=================================\n\n")

  # ============================================================================
  # Assertions
  # ============================================================================

  expect_lt(bw_maxerr, 0.02,
            label = sprintf("BW MaxErr (%.2e) should be < 0.02", bw_maxerr))

  expect_lt(gcv_maxerr, 0.01,
            label = sprintf("GCV MaxErr (%.2e) should be < 0.01", gcv_maxerr))

  expect_lt(mean_maxerr, 0.05,
            label = sprintf("Mean MaxErr (%.2e) should be < 0.05", mean_maxerr))

  expect_lt(speed_ratio, 8.0,
            label = sprintf("Speed ratio (%.2fx) should be < 8x", speed_ratio))
})

test_that("1D Order 1 cross-validation summary", {
  # Skip if MATLAB reference data not available (CRAN compatibility)
  skip_if_no_matlab_refs()

  ref_paths <- c(
    file.path(find_root(), "fastLPR/tests/refs/crosslang_e2e/ref_1d_order1.mat"),
    "../../fastLPR/tests/refs/crosslang_e2e/ref_1d_order1.mat"
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

  x_ref <- matrix(ref$x, ncol = 1)
  y_ref <- matrix(ref$y, ncol = 1)
  hlist_ref <- matrix(as.vector(ref$hlist), ncol = 1)
  h1se_ref <- as.numeric(ref$h1se)
  matlab_time <- as.numeric(ref$elapsed)

  opt <- list(order = 1, calc_dof = TRUE)
  if (!is.null(ref$dof.random.vectors)) {
    opt$dof_random_vectors <- ref$dof.random.vectors
  }

  r_time <- system.time({
    res <- cv_fastlpr(x_ref, y_ref, hlist_ref, opt)
  })[["elapsed"]]

  bw_maxerr <- abs(res$gcv_yhat$h1se - h1se_ref)
  gcv_maxerr <- max(abs(res$gcv_yhat$gcv_m - as.vector(ref$gcv.m)))
  mean_maxerr <- max(abs(as.vector(res$yhat) - as.vector(ref$yhat.mean)))
  speed_ratio <- r_time / matlab_time

  status <- ifelse(bw_maxerr < 0.02 && gcv_maxerr < 0.01 &&
                   mean_maxerr < 0.05 && speed_ratio < 8.0, "PASS", "FAIL")

  cat("\n")
  cat("| Test | N | dx | order | Type | MATLAB | R Time | Ratio | GCV MaxErr | BW MaxErr | Mean MaxErr | Status |\n")
  cat("|------|-----|----|-------|------|--------|--------|-------|------------|-----------|-------------|--------|\n")
  cat(sprintf("| 1D Order 1 | %d | 1 | 1 | mean | %.3fs | %.3fs | %.2fx | %.2e | %.2e | %.2e | **%s** |\n",
              nrow(x_ref), matlab_time, r_time, speed_ratio,
              gcv_maxerr, bw_maxerr, mean_maxerr, status))
  cat("\n")

  expect_equal(status, "PASS", label = "1D Order 1 cross-validation should PASS")
})
