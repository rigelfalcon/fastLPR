# ==============================================================================
# Cross-Validation Integration Test: Heteroscedastic 1D vs MATLAB Reference
# ==============================================================================
# Test file: fastLPR_R/tests/testthat/test-xl-hetero1d.R
# Restored: 2026-01-10
#
# This test validates heteroscedastic (mean + variance) estimation in R
# against MATLAB reference data.
#
# TWO-STEP PROCESS:
#   1) Estimate mean: cv_fastlpr(x, y, hlist, opt_mean)
#   2) Estimate variance from residuals^2: cv_fastlpr(x, residuals^2, hlist, opt_var)
#
# Reference file: fastLPR/tests/refs/crosslang_e2e/ref_hetero_1d.mat
#
# Pass Criteria (per CLAUDE.md):
#   - BW MaxErr < 0.02 (bandwidth selection threshold)
#   - GCV MaxErr < 0.01 (GCV score threshold)
#   - Mean MaxErr < 0.05 (prediction threshold for 1D)
#   - Var MaxErr < 0.05 (variance estimation threshold, unified)
#   - Speed ratio < 8x MATLAB time
# ==============================================================================

context("Cross-validation: Hetero 1D vs MATLAB")

# Helper function for safe field extraction
safe_extract <- function(mat, field) {
  if (!is.null(mat[[field]])) {
    return(as.vector(mat[[field]]))
  }
  field_alt <- gsub("_", ".", field)
  if (!is.null(mat[[field_alt]])) {
    return(as.vector(mat[[field_alt]]))
  }
  return(NULL)
}

test_that("Hetero 1D matches MATLAB reference data", {
  # Require MATLAB reference tooling/data
  skip_if_not_installed("R.matlab")
  skip_if_no_matlab_refs()

  # Find reference file path
  ref_paths <- c(
    file.path(find_root(), "fastLPR/tests/refs/crosslang_e2e/ref_hetero_1d.mat"),
    "../../fastLPR/tests/refs/crosslang_e2e/ref_hetero_1d.mat",
    file.path(find_root(), "fastLPR/tests/refs/crosslang_e2e/ref_hetero_1d.mat")
  )

  ref_path <- NULL
  for (p in ref_paths) {
    if (file.exists(p)) {
      ref_path <- p
      break
    }
  }

  skip_if(is.null(ref_path), "MATLAB reference file ref_hetero_1d.mat not found")

  ref <- R.matlab::readMat(ref_path)

  # Extract reference data
  x_ref <- matrix(ref$x, ncol = 1)
  y_ref <- as.vector(ref$y)
  hlist_ref <- matrix(as.vector(ref$hlist), ncol = 1)
  gcv_m_ref <- safe_extract(ref, "gcv_m")
  h1se_mean_ref <- safe_extract(ref, "h1se_mean")
  yhat_mean_ref <- safe_extract(ref, "yhat_mean")
  yhat_var_ref <- safe_extract(ref, "yhat_var")
  matlab_time <- as.numeric(ref$elapsed)

  cat("\n=== Hetero 1D Cross-Validation Test ===\n")
  cat(sprintf("Reference file: %s\n", ref_path))
  cat(sprintf("Sample size (N): %d\n", nrow(x_ref)))
  cat(sprintf("Number of bandwidths: %d\n", length(hlist_ref)))
  cat(sprintf("MATLAB time: %.3f s\n", matlab_time))

  # ===========================================================================
  # Run R implementation (TWO-STEP PROCESS)
  # ===========================================================================

  r_time <- system.time({
    # Step 1: Estimate mean
    opt_mean <- list(order = 1, calc_dof = TRUE, dstd = 0)
    if (!is.null(ref$dof.random.vectors.mean)) {
      opt_mean$dof_random_vectors <- ref$dof.random.vectors.mean
    }
    res_mean <- cv_fastlpr(x_ref, y_ref, hlist_ref, opt_mean)

    # Step 2: Estimate variance from residuals^2
    residuals <- y_ref - res_mean$yhat
    opt_var <- list(order = 1, calc_dof = TRUE, y_type_out = "variance", dstd = 1)
    if (!is.null(ref$dof.random.vectors.var)) {
      opt_var$dof_random_vectors <- ref$dof.random.vectors.var
    }
    res_var <- cv_fastlpr(x_ref, residuals^2, hlist_ref, opt_var)
  })[["elapsed"]]

  # Extract R results
  gcv_m_r <- res_mean$gcv_yhat$gcv_m
  h1se_r <- res_mean$gcv_yhat$h1se
  hmin_r <- res_mean$gcv_yhat$hmin
  yhat_mean_r <- as.vector(res_mean$yhat)
  yhat_var_r <- as.vector(res_var$yhat)

  # ===========================================================================
  # Compute error metrics
  # ===========================================================================

  idmin_ref <- which.min(gcv_m_ref)
  hmin_ref <- as.numeric(hlist_ref[idmin_ref, 1])

  bw_maxerr <- abs(h1se_r - h1se_mean_ref)
  hmin_maxerr <- abs(as.numeric(hmin_r) - hmin_ref)
  gcv_maxerr <- max(abs(gcv_m_r - gcv_m_ref))
  mean_maxerr <- max(abs(yhat_mean_r - yhat_mean_ref))
  var_maxerr <- max(abs(yhat_var_r - yhat_var_ref))
  speed_ratio <- r_time / matlab_time

  # ===========================================================================
  # Report results
  # ===========================================================================

  cat("\n--- R Results ---\n")
  cat(sprintf("R time: %.3f s\n", r_time))
  cat(sprintf("R selected h (mean): %.6f\n", h1se_r))
  cat(sprintf("R hmin (mean): %.6f\n", as.numeric(hmin_r)))

  cat("\n--- Error Metrics ---\n")
  cat(sprintf("BW MaxErr: %.2e (threshold: 0.02)\n", bw_maxerr))
  cat(sprintf("Hmin MaxErr: %.2e (threshold: 0.02)\n", hmin_maxerr))
  cat(sprintf("GCV MaxErr: %.2e (threshold: 0.01)\n", gcv_maxerr))
  cat(sprintf("Mean MaxErr: %.2e (threshold: 0.05)\n", mean_maxerr))
  cat(sprintf("Var MaxErr: %.2e (threshold: 0.05)\n", var_maxerr))
  cat(sprintf("Speed ratio: %.2fx (threshold: 8x)\n", speed_ratio))

  # ===========================================================================
  # Determine pass/fail (ALL metrics must pass)
  # ===========================================================================

  bw_pass <- bw_maxerr < 0.02
  hmin_pass <- hmin_maxerr < 0.02
  gcv_pass <- gcv_maxerr < 0.01
  mean_pass <- mean_maxerr < 0.05
  var_pass <- var_maxerr < TOL_CROSSLANG$var_maxerr  # 0.05 (unified)
  speed_pass <- speed_ratio < 8.0

  overall_pass <- bw_pass && hmin_pass && gcv_pass && mean_pass && var_pass && speed_pass

  cat("\n--- Status ---\n")
  cat(sprintf("BW selection: %s\n", ifelse(bw_pass, "PASS", "FAIL")))
  cat(sprintf("Hmin selection: %s\n", ifelse(hmin_pass, "PASS", "FAIL")))
  cat(sprintf("GCV accuracy: %s\n", ifelse(gcv_pass, "PASS", "FAIL")))
  cat(sprintf("Mean accuracy: %s\n", ifelse(mean_pass, "PASS", "FAIL")))
  cat(sprintf("Var accuracy: %s\n", ifelse(var_pass, "PASS", "FAIL")))
  cat(sprintf("Speed: %s\n", ifelse(speed_pass, "PASS", "FAIL")))
  cat(sprintf("Overall: %s\n", ifelse(overall_pass, "PASS", "FAIL")))
  cat("=================================\n\n")

  # ===========================================================================
  # Assertions
  # ===========================================================================

  expect_lt(bw_maxerr, 0.02,
            label = sprintf("BW MaxErr (%.2e) should be < 0.02", bw_maxerr))

  expect_lt(hmin_maxerr, 0.02,
            label = sprintf("Hmin MaxErr (%.2e) should be < 0.02", hmin_maxerr))

  expect_lt(gcv_maxerr, 0.01,
            label = sprintf("GCV MaxErr (%.2e) should be < 0.01", gcv_maxerr))

  expect_lt(mean_maxerr, 0.05,
            label = sprintf("Mean MaxErr (%.2e) should be < 0.05", mean_maxerr))

  expect_lt(var_maxerr, TOL_CROSSLANG$var_maxerr,  # 0.05 (unified)
            label = sprintf("Var MaxErr (%.2e) should be < 0.05", var_maxerr))

  expect_lt(speed_ratio, 8.0,
            label = sprintf("Speed ratio (%.2fx) should be < 8x", speed_ratio))
})
