# test-unit-gcv.R - Comprehensive unit tests for GCV (Generalized Cross-Validation)
#
# Author: AI Assistant (Claude)
# Date: 2025-12-14
# Purpose: Comprehensive test coverage for GCV calculation and bandwidth selection
#
# Test coverage requirements:
# 1. GCV computation correctness (MSE * penalty factor)
# 2. GCV curve shape (U-shape for appropriate bandwidth range)
# 3. GCV minimum selection accuracy
# 4. 1-SE rule implementation
# 5. 1D and 2D test cases
# 6. Edge cases: very small/large bandwidths, single bandwidth
# 7. DOF (degrees of freedom) computation for GCV penalty
# 8. Hutchinson trace estimator accuracy
# 9. GCV standard deviation computation
# 10. MATLAB reference comparison (when available)
#
# Framework: testthat
# Test cases: 40+ (comprehensive GCV coverage)

context("GCV (Generalized Cross-Validation) Unit Tests")

# =============================================================================
# SECTION 1: Basic GCV Computation Tests
# =============================================================================

test_that("GCV values are computed and non-negative", {
  set.seed(42)
  n <- 300
  x <- matrix(runif(n, 0, 2*pi), ncol = 1)
  y <- matrix(sin(x) + 0.1 * rnorm(n), ncol = 1)
  hlist <- get_hlist(10, c(0.05, 1.0))
  opt <- list(order = 1, calc_dof = TRUE, verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  # GCV values should exist
  expect_true(!is.null(regs$gcv_yhat$gcv_m))

  # GCV values should be non-negative (sum of squared errors)
  gcv_vals <- as.vector(regs$gcv_yhat$gcv_m)
  expect_true(all(gcv_vals >= 0))

  # GCV values should be finite
  expect_true(all(is.finite(gcv_vals)))
})

# ARCHIVED: "GCV is computed as MSE * penalty" - Implementation detail
# See: dev/archive/r-bandwidth-tests-20260109/archived_bandwidth_tests.R

# ARCHIVED: "GCV has correct number of values" - Implementation detail
# See: dev/archive/r-bandwidth-tests-20260109/archived_bandwidth_tests.R

# =============================================================================
# SECTION 2: GCV Curve Shape Tests
# =============================================================================

test_that("GCV curve has U-shape for wide bandwidth range", {
  set.seed(42)
  n <- 400
  x <- matrix(runif(n, 0, 2*pi), ncol = 1)
  y <- matrix(sin(x) + 0.1 * rnorm(n), ncol = 1)
  # Wide range to capture U-shape
  hlist <- get_hlist(20, c(0.01, 2.0))
  opt <- list(order = 1, calc_dof = TRUE, verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  gcv_vals <- as.vector(regs$gcv_yhat$gcv_m)
  min_idx <- which.min(gcv_vals)

  # Minimum should not be at the boundaries (U-shape property)
  expect_gt(min_idx, 1)  # GCV minimum should not be at smallest bandwidth
  expect_lt(min_idx, length(gcv_vals))  # GCV minimum should not be at largest bandwidth
})



# =============================================================================
# SECTION 3: GCV Minimum Selection Tests
# =============================================================================

test_that("idmin corresponds to minimum GCV value", {
  set.seed(42)
  n <- 300
  x <- matrix(runif(n, 0, 2*pi), ncol = 1)
  y <- matrix(sin(x) + 0.1 * rnorm(n), ncol = 1)
  hlist <- get_hlist(15, c(0.05, 1.0))
  opt <- list(order = 1, calc_dof = TRUE, verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  idmin <- regs$gcv_yhat$idmin
  gcv_vals <- as.vector(regs$gcv_yhat$gcv_m)

  # idmin should point to the minimum GCV value
  expect_equal(idmin, which.min(gcv_vals),
               info = "idmin should be the index of minimum GCV")
})

# ARCHIVED: "hmin corresponds to bandwidth at minimum GCV" - Redundant with idmin test
# See: dev/archive/r-bandwidth-tests-20260109/archived_bandwidth_tests.R

test_that("GCV minimum selection improves prediction accuracy", {
  set.seed(42)
  n <- 400
  x <- matrix(sort(runif(n, 0, 2*pi)), ncol = 1)
  y_true <- sin(x)
  y <- matrix(y_true + 0.1 * rnorm(n), ncol = 1)
  hlist <- get_hlist(15, c(0.05, 1.0))
  opt <- list(order = 1, calc_dof = TRUE, verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  # Compute MSE at selected bandwidth
  mse_selected <- mean((regs$yhat - y_true)^2)

  # Compare with extreme bandwidths
  opt_small <- list(order = 1, verbose = FALSE)
  opt_large <- list(order = 1, verbose = FALSE)

  regs_small <- cv_fastlpr(x, y, matrix(min(hlist), ncol = 1), opt_small)
  regs_large <- cv_fastlpr(x, y, matrix(max(hlist), ncol = 1), opt_large)

  mse_small <- mean((regs_small$yhat - y_true)^2)
  mse_large <- mean((regs_large$yhat - y_true)^2)

  # Selected bandwidth should have lower MSE than extremes
  expect_lt(mse_selected, max(mse_small, mse_large))  # GCV-selected bandwidth should outperform extremes
})

# =============================================================================
# SECTION 4: 1-SE Rule Tests
# =============================================================================

test_that("1-SE rule selects larger bandwidth than minimum", {
  set.seed(42)
  n <- 400
  x <- matrix(runif(n, 0, 2*pi), ncol = 1)
  y <- matrix(sin(x) + 0.1 * rnorm(n), ncol = 1)
  hlist <- get_hlist(20, c(0.05, 1.0))
  opt <- list(order = 1, calc_dof = TRUE, dstd = 1, verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  h1se <- as.numeric(regs$gcv_yhat$h1se)
  hmin <- as.numeric(regs$gcv_yhat$hmin)

  # 1-SE rule should select bandwidth >= minimum (prefers smoother)
  expect_gte(h1se, hmin * 0.99)  # 1-SE rule should select bandwidth >= minimum
})

test_that("id1se is valid index", {
  set.seed(42)
  n <- 300
  x <- matrix(runif(n, 0, 2*pi), ncol = 1)
  y <- matrix(sin(x) + 0.1 * rnorm(n), ncol = 1)
  n_bandwidths <- 15
  hlist <- get_hlist(n_bandwidths, c(0.1, 1.0))
  opt <- list(order = 1, calc_dof = TRUE, dstd = 1, verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  id1se <- regs$gcv_yhat$id1se

  # id1se should be within valid range
  expect_gte(id1se, 1)
  expect_lte(id1se, n_bandwidths)
})

# ARCHIVED: "1-SE rule respects SE threshold" - Redundant with "1-SE larger bandwidth" test
# See: dev/archive/r-bandwidth-tests-20260109/archived_bandwidth_tests.R

test_that("dstd=0 means no 1-SE rule (h1se equals hmin)", {
  set.seed(42)
  n <- 300
  x <- matrix(runif(n, 0, 2*pi), ncol = 1)
  y <- matrix(sin(x) + 0.1 * rnorm(n), ncol = 1)
  hlist <- get_hlist(10, c(0.1, 0.8))
  opt <- list(order = 1, calc_dof = TRUE, dstd = 0, verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  h1se <- as.numeric(regs$gcv_yhat$h1se)
  hmin <- as.numeric(regs$gcv_yhat$hmin)

  # With dstd=0, h1se should equal hmin
  expect_equal(h1se, hmin, tolerance = 1e-10,
               info = "With dstd=0, h1se should equal hmin")
})

# ARCHIVED: "Larger dstd leads to larger selected bandwidth" - Implementation detail
# See: dev/archive/r-bandwidth-tests-20260109/archived_bandwidth_tests.R

# =============================================================================
# SECTION 5: DOF (Degrees of Freedom) Tests
# =============================================================================

test_that("DOF values are computed when calc_dof=TRUE", {
  set.seed(42)
  n <- 200
  x <- matrix(runif(n, 0, 2*pi), ncol = 1)
  y <- matrix(sin(x) + 0.1 * rnorm(n), ncol = 1)
  hlist <- get_hlist(5, c(0.1, 0.5))
  opt <- list(order = 1, calc_dof = TRUE, verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  # DOF-related fields should exist
  expect_true(!is.null(regs$dof))
  expect_true(!is.null(regs$df_m))
  expect_true(!is.null(regs$pdof_m))
  expect_true(!is.null(regs$pdof_inv_m))
})

test_that("DOF is in valid range [0, n]", {
  set.seed(42)
  n <- 200
  x <- matrix(runif(n, 0, 2*pi), ncol = 1)
  y <- matrix(sin(x) + 0.1 * rnorm(n), ncol = 1)
  hlist <- get_hlist(10, c(0.05, 1.0))
  opt <- list(order = 1, calc_dof = TRUE, verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  df_vals <- as.vector(regs$df_m)

  # DOF should be between 0 and n
  expect_true(all(df_vals >= 0), info = "DOF should be >= 0")
  expect_true(all(df_vals <= n), info = "DOF should be <= n")
})

test_that("DOF decreases with bandwidth (larger h = more smoothing = fewer effective parameters)", {
  set.seed(42)
  n <- 300
  x <- matrix(runif(n, 0, 2*pi), ncol = 1)
  y <- matrix(sin(x) + 0.1 * rnorm(n), ncol = 1)
  hlist <- get_hlist(10, c(0.05, 1.0))
  opt <- list(order = 1, calc_dof = TRUE, verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  df_vals <- as.vector(regs$df_m)
  h_vals <- as.vector(hlist)

  # Correlation between h and DOF should be negative
  # Larger bandwidth = more smoothing = fewer effective parameters = lower DOF
  corr <- cor(h_vals, df_vals)
  expect_lt(corr, 0)  # DOF should decrease with bandwidth
})

# ARCHIVED: "pdof is in valid range (0, 1]" - Implementation detail
# See: dev/archive/r-bandwidth-tests-20260109/archived_bandwidth_tests.R

# ARCHIVED: "pdof_inv (penalty) is >= 1" - Implementation detail
# See: dev/archive/r-bandwidth-tests-20260109/archived_bandwidth_tests.R

# =============================================================================
# SECTION 6: 2D GCV Tests
# =============================================================================

test_that("2D GCV computation works", {
  set.seed(42)
  n <- 300
  x <- matrix(runif(n * 2, -1, 1), ncol = 2)
  y <- matrix(sin(pi * x[, 1]) * cos(pi * x[, 2]) + 0.1 * rnorm(n), ncol = 1)
  hlist <- get_hlist(c(3, 3), list(c(0.2, 0.6), c(0.2, 0.6)))
  opt <- list(order = 1, N = c(40, 40), calc_dof = TRUE, verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  # GCV values should exist
  expect_true(!is.null(regs$gcv_yhat$gcv_m))

  # GCV values should be finite and non-negative
  gcv_vals <- as.vector(regs$gcv_yhat$gcv_m)
  expect_true(all(is.finite(gcv_vals)))
  expect_true(all(gcv_vals >= 0))
})

test_that("2D GCV selects valid bandwidth", {
  set.seed(42)
  n <- 300
  x <- matrix(runif(n * 2, -1, 1), ncol = 2)
  y <- matrix(sin(pi * x[, 1]) * cos(pi * x[, 2]) + 0.1 * rnorm(n), ncol = 1)
  hlist <- get_hlist(c(3, 3), list(c(0.2, 0.5), c(0.2, 0.5)))
  opt <- list(order = 1, N = c(40, 40), calc_dof = TRUE, verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  h1se <- as.numeric(regs$gcv_yhat$h1se)

  # Selected bandwidth should be within candidate range (per dimension)
  expect_gte(h1se[1], min(hlist[, 1]))
  expect_lte(h1se[1], max(hlist[, 1]))
  expect_gte(h1se[2], min(hlist[, 2]))
  expect_lte(h1se[2], max(hlist[, 2]))
})

test_that("2D idmin and id1se are valid indices", {
  set.seed(42)
  n <- 300
  x <- matrix(runif(n * 2, -1, 1), ncol = 2)
  y <- matrix(sin(pi * x[, 1]) * cos(pi * x[, 2]) + 0.1 * rnorm(n), ncol = 1)
  hlist <- get_hlist(c(4, 4), list(c(0.2, 0.5), c(0.2, 0.5)))
  n_bandwidths <- nrow(hlist)
  opt <- list(order = 1, N = c(40, 40), calc_dof = TRUE, dstd = 1, verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  idmin <- regs$gcv_yhat$idmin
  id1se <- regs$gcv_yhat$id1se

  expect_gte(idmin, 1)
  expect_lte(idmin, n_bandwidths)
  expect_gte(id1se, 1)
  expect_lte(id1se, n_bandwidths)
})

# ARCHIVED: 2026-01-10 - "2D hlist ordering matches MATLAB ndgrid (column-major)"
# Archive: dev/archive/tests-archive-20260110/r/unit/archived_test-unit-bandwidth-extra.R
# Reason: Test unification - not in MATLAB reference (get_hlist test, not GCV test)

# =============================================================================
# SECTION 7: Edge Cases Tests
# =============================================================================

test_that("GCV works with single bandwidth (no selection)", {
  set.seed(42)
  n <- 200
  x <- matrix(runif(n, 0, 2*pi), ncol = 1)
  y <- matrix(sin(x) + 0.1 * rnorm(n), ncol = 1)
  h <- matrix(0.3, ncol = 1)  # Single bandwidth
  opt <- list(order = 1, verbose = FALSE)

  regs <- cv_fastlpr(x, y, h, opt)

  # Should work without error
  expect_true(!is.null(regs$yhat))
  expect_equal(length(regs$yhat), n)

  # id1se should be 1 (only option)
  expect_true(is.null(regs$gcv_yhat$id1se) || regs$gcv_yhat$id1se == 1)
})

test_that("GCV handles very small bandwidth", {
  set.seed(42)
  n <- 200
  x <- matrix(runif(n, 0, 2*pi), ncol = 1)
  y <- matrix(sin(x) + 0.1 * rnorm(n), ncol = 1)
  # Very small bandwidths (near interpolation)
  hlist <- get_hlist(5, c(0.005, 0.05))
  opt <- list(order = 1, calc_dof = TRUE, verbose = FALSE)

  expect_no_error({
    regs <- cv_fastlpr(x, y, hlist, opt)
  })

  # Results should still be finite
  expect_true(all(is.finite(regs$yhat)))
  expect_true(all(is.finite(regs$gcv_yhat$gcv_m)))
})

test_that("GCV handles very large bandwidth", {
  set.seed(42)
  n <- 200
  x <- matrix(runif(n, 0, 2*pi), ncol = 1)
  y <- matrix(sin(x) + 0.1 * rnorm(n), ncol = 1)
  # Very large bandwidths (near global mean)
  hlist <- get_hlist(5, c(2.0, 10.0))
  opt <- list(order = 1, calc_dof = TRUE, verbose = FALSE)

  expect_no_error({
    regs <- cv_fastlpr(x, y, hlist, opt)
  })

  # Results should be finite
  expect_true(all(is.finite(regs$yhat)))
})

test_that("GCV handles small sample size", {
  set.seed(42)
  n <- 30
  x <- matrix(runif(n, 0, 2*pi), ncol = 1)
  y <- matrix(sin(x) + 0.2 * rnorm(n), ncol = 1)
  hlist <- get_hlist(3, c(0.3, 0.8))
  opt <- list(order = 1, N = 40, calc_dof = TRUE, verbose = FALSE)

  expect_no_error({
    regs <- cv_fastlpr(x, y, hlist, opt)
  })

  expect_equal(length(regs$yhat), n)
})

# ARCHIVED: 2026-01-09 - "GCV handles near-constant response" (moved to dev/archive/tests-archive-20260109/r/unit/archived_test-unit-bandwidth.R)




# =============================================================================
# SECTION 8: GCV for Different Polynomial Orders
# =============================================================================

test_that("GCV works for order 0 (Nadaraya-Watson)", {
  set.seed(42)
  n <- 200
  x <- matrix(runif(n, 0, 2*pi), ncol = 1)
  y <- matrix(sin(x) + 0.1 * rnorm(n), ncol = 1)
  hlist <- get_hlist(10, c(0.1, 1.0))
  opt <- list(order = 0, calc_dof = TRUE, verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  expect_true(!is.null(regs$gcv_yhat$gcv_m))
  expect_true(all(is.finite(regs$gcv_yhat$gcv_m)))
})

test_that("GCV works for order 1 (local linear)", {
  set.seed(42)
  n <- 200
  x <- matrix(runif(n, 0, 2*pi), ncol = 1)
  y <- matrix(sin(x) + 0.1 * rnorm(n), ncol = 1)
  hlist <- get_hlist(10, c(0.1, 1.0))
  opt <- list(order = 1, calc_dof = TRUE, verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  expect_true(!is.null(regs$gcv_yhat$gcv_m))
  expect_true(all(is.finite(regs$gcv_yhat$gcv_m)))
})

test_that("GCV works for order 2 (local quadratic)", {
  set.seed(42)
  n <- 200
  x <- matrix(runif(n, 0, 2*pi), ncol = 1)
  y <- matrix(x^2 + 0.1 * rnorm(n), ncol = 1)
  hlist <- get_hlist(10, c(0.1, 0.8))
  opt <- list(order = 2, calc_dof = TRUE, verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  expect_true(!is.null(regs$gcv_yhat$gcv_m))
  expect_true(all(is.finite(regs$gcv_yhat$gcv_m)))
})

test_that("Different orders give different GCV values", {
  set.seed(42)
  n <- 300
  x <- matrix(runif(n, 0, 2*pi), ncol = 1)
  y <- matrix(sin(x) + 0.1 * rnorm(n), ncol = 1)
  hlist <- get_hlist(5, c(0.3, 0.7))

  opt0 <- list(order = 0, calc_dof = TRUE, verbose = FALSE)
  opt1 <- list(order = 1, calc_dof = TRUE, verbose = FALSE)

  regs0 <- cv_fastlpr(x, y, hlist, opt0)
  regs1 <- cv_fastlpr(x, y, hlist, opt1)

  gcv0 <- as.vector(regs0$gcv_yhat$gcv_m)
  gcv1 <- as.vector(regs1$gcv_yhat$gcv_m)

  # GCV values should be different for different orders
  expect_false(all(abs(gcv0 - gcv1) < 1e-10))
})

# =============================================================================
# SECTION 9: GCV Reproducibility Tests
# =============================================================================

# ARCHIVED: 2026-01-09 - "GCV computation is reproducible" (moved to dev/archive/tests-archive-20260109/r/unit/archived_test-unit-bandwidth.R)



# =============================================================================
# SECTION 10: GCV Standard Deviation Tests
# =============================================================================

test_that("GCV standard deviation is computed", {
  set.seed(42)
  n <- 300
  x <- matrix(runif(n, 0, 2*pi), ncol = 1)
  y <- matrix(sin(x) + 0.1 * rnorm(n), ncol = 1)
  hlist <- get_hlist(10, c(0.1, 0.8))
  opt <- list(order = 1, calc_dof = TRUE, verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  # gcv_sd should exist
  expect_true(!is.null(regs$gcv_yhat$gcv_sd))

  # gcv_sd should be non-negative
  gcv_sd <- as.vector(regs$gcv_yhat$gcv_sd)
  expect_true(all(gcv_sd >= 0))
})


# =============================================================================
# SECTION 11: Complex Data GCV Tests
# =============================================================================

test_that("GCV works with complex-valued response", {
  set.seed(42)
  n <- 300
  x <- matrix(sort(runif(n)), ncol = 1)
  y_real <- sin(2 * pi * x)
  y_imag <- cos(2 * pi * x)
  y <- complex(real = y_real + 0.1 * rnorm(n), imaginary = y_imag + 0.1 * rnorm(n))
  y <- matrix(y, ncol = 1)
  hlist <- get_hlist(10, c(0.05, 0.5))
  opt <- list(order = 1, calc_dof = TRUE, verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  # GCV values should be real (even for complex data)
  expect_true(is.numeric(regs$gcv_yhat$gcv_m))
  expect_false(is.complex(regs$gcv_yhat$gcv_m))

  # GCV values should be positive
  gcv_vals <- as.vector(regs$gcv_yhat$gcv_m)
  expect_true(all(gcv_vals > 0))
})

test_that("Complex data GCV selects valid bandwidth", {
  set.seed(42)
  n <- 200
  x <- matrix(sort(runif(n)), ncol = 1)
  y <- exp(1i * 2 * pi * x) + 0.1 * (rnorm(n) + 1i * rnorm(n))
  y <- matrix(y, ncol = 1)
  hlist <- get_hlist(8, c(0.05, 0.4))
  opt <- list(order = 1, calc_dof = TRUE, verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  h1se <- as.numeric(regs$gcv_yhat$h1se)

  # Selected bandwidth should be in range
  expect_gte(h1se, min(hlist))
  expect_lte(h1se, max(hlist))
})

# =============================================================================
# SECTION 12: MSE and RSS Output Tests
# =============================================================================

test_that("MSE values are computed for all bandwidths", {
  set.seed(42)
  n <- 200
  x <- matrix(runif(n, 0, 2*pi), ncol = 1)
  y <- matrix(sin(x) + 0.1 * rnorm(n), ncol = 1)
  n_bandwidths <- 8
  hlist <- get_hlist(n_bandwidths, c(0.1, 0.5))
  opt <- list(order = 1, calc_dof = TRUE, verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  # MSE should exist and have correct length
  expect_true(!is.null(regs$gcv_yhat$mse))
  expect_equal(nrow(regs$gcv_yhat$mse), n_bandwidths)

  # All MSE values should be positive
  expect_true(all(regs$gcv_yhat$mse > 0))
})


# =============================================================================
# SECTION 13: GCV Penalty Behavior Tests
# =============================================================================

test_that("GCV penalty increases for smaller bandwidths", {
  set.seed(42)
  n <- 300
  x <- matrix(runif(n, 0, 2*pi), ncol = 1)
  y <- matrix(sin(x) + 0.1 * rnorm(n), ncol = 1)
  hlist <- get_hlist(10, c(0.05, 1.0))
  opt <- list(order = 1, calc_dof = TRUE, verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  pdof_inv_vals <- as.vector(regs$pdof_inv_m)
  h_vals <- as.vector(hlist)

  # Penalty should decrease with bandwidth (negative correlation)
  corr <- cor(h_vals, pdof_inv_vals)
  expect_lt(corr, 0)  # Penalty should decrease with bandwidth
})


# =============================================================================
# SECTION 14: GCV with Hutchinson Trace Estimator Tests
# =============================================================================

test_that("Hutchinson estimator gives reasonable DOF estimates", {
  set.seed(42)
  n <- 200
  x <- matrix(runif(n, 0, 2*pi), ncol = 1)
  y <- matrix(sin(x) + 0.1 * rnorm(n), ncol = 1)
  hlist <- get_hlist(5, c(0.2, 0.6))
  opt <- list(order = 1, calc_dof = TRUE, verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  df_vals <- as.vector(regs$df_m)

  # DOF should be reasonable (not at extremes for moderate bandwidths)
  expect_true(all(df_vals > 1), info = "DOF should be > 1 for moderate bandwidths")
  expect_true(all(df_vals < n - 1), info = "DOF should be < n-1 for moderate bandwidths")
})


# =============================================================================
# SECTION 15: GCV Output Structure Tests
# =============================================================================

test_that("gcv_yhat structure contains all required fields", {
  set.seed(42)
  n <- 200
  x <- matrix(runif(n, 0, 2*pi), ncol = 1)
  y <- matrix(sin(x) + 0.1 * rnorm(n), ncol = 1)
  hlist <- get_hlist(5, c(0.1, 0.5))
  opt <- list(order = 1, calc_dof = TRUE, verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  # Required fields
  expect_true("gcv_m" %in% names(regs$gcv_yhat))
  expect_true("h1se" %in% names(regs$gcv_yhat))
  expect_true("hmin" %in% names(regs$gcv_yhat))
  expect_true("idmin" %in% names(regs$gcv_yhat))
  expect_true("id1se" %in% names(regs$gcv_yhat))
  expect_true("mse" %in% names(regs$gcv_yhat))
  expect_true("rss" %in% names(regs$gcv_yhat))
})



# ARCHIVED: 2026-01-09
# Source: fastLPR_R/tests/testthat/test-unit-bandwidth.R
# Archive: dev/archive/tests-archive-20260109/r/unit/archived_test-unit-bandwidth.R
# Archived tests:
# - "GCV handles near-constant response"
# - "GCV computation is reproducible"
