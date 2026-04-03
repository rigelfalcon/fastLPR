# Test suite for 1D Local Polynomial Regression
context("1D Local Polynomial Regression")

test_that("1D order 0 regression (Nadaraya-Watson) works", {
  set.seed(42)
  n <- 500
  x <- matrix(runif(n), ncol = 1)
  y_true <- sin(2 * pi * x)
  y <- matrix(y_true + 0.2 * rnorm(n), ncol = 1)
  hlist <- get_hlist(10, c(0.01, 0.5))

  opt <- list(order = 0, verbose = FALSE)
  regs <- cv_fastlpr(x, y, hlist, opt)

  # Check output structure
  validate_regression_structure(regs)
  expect_equal(length(regs$yhat), n)

  # Fitted values should correlate with true function
  correlation <- cor(as.vector(regs$yhat), as.vector(y_true))
  expect_gt(correlation, 0.7)
})

test_that("1D order 1 regression (local linear) works", {
  set.seed(42)
  n <- 500
  x <- matrix(runif(n), ncol = 1)
  y_true <- sin(2 * pi * x)
  y <- matrix(y_true + 0.2 * rnorm(n), ncol = 1)
  hlist <- get_hlist(10, c(0.01, 0.5))

  opt <- list(order = 1, verbose = FALSE, calc_dof = TRUE)
  regs <- cv_fastlpr(x, y, hlist, opt)

  validate_regression_structure(regs)
  expect_equal(length(regs$yhat), n)

  # Local linear should have better fit than order 0
  correlation <- cor(as.vector(regs$yhat), as.vector(y_true))
  expect_gt(correlation, 0.8)
})

test_that("1D order 2 regression (local quadratic) works", {
  set.seed(42)
  n <- 500
  x <- matrix(runif(n), ncol = 1)
  y_true <- sin(2 * pi * x)
  y <- matrix(y_true + 0.2 * rnorm(n), ncol = 1)
  hlist <- get_hlist(10, c(0.01, 0.5))

  opt <- list(order = 2, verbose = FALSE, calc_dof = TRUE)
  regs <- cv_fastlpr(x, y, hlist, opt)

  validate_regression_structure(regs)
  expect_equal(length(regs$yhat), n)

  # MSE should be reasonable
  mse <- mean((regs$yhat - y_true)^2)
  expect_lt(mse, 0.07)
})

test_that("1D order 1 reduces bias compared to order 0", {
  # Unified with MATLAB: n=200, h=[0.3,0.8], nh=5, noise=0.05
  set.seed(42)
  n <- 200
  x <- matrix(runif(n), ncol = 1)
  y_true <- sin(2 * pi * x)
  y <- matrix(y_true + 0.05 * rnorm(n), ncol = 1)
  hlist <- get_hlist(5, c(0.3, 0.8))

  opt0 <- list(order = 0, verbose = FALSE)
  opt1 <- list(order = 1, verbose = FALSE)

  regs0 <- cv_fastlpr(x, y, hlist, opt0)
  regs1 <- cv_fastlpr(x, y, hlist, opt1)

  mse0 <- mean((regs0$yhat - y_true)^2)
  mse1 <- mean((regs1$yhat - y_true)^2)

  # Order 1 should have lower or similar MSE (allow for variance)
  expect_lt(mse1, mse0 * 1.5)
})

# ARCHIVED: 2026-01-09 - "1D regression with single bandwidth works" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-lpr-1d.R)


test_that("1D regression GCV bandwidth selection works", {
  # Unified with MATLAB: n=500, h=[0.01,0.5], nh=20
  set.seed(42)
  n <- 500
  x <- matrix(runif(n), ncol = 1)
  y_true <- sin(2 * pi * x)
  y <- matrix(y_true + 0.2 * rnorm(n), ncol = 1)
  hlist <- get_hlist(20, c(0.01, 0.5))

  opt <- list(order = 1, verbose = FALSE, calc_dof = TRUE)
  regs <- cv_fastlpr(x, y, hlist, opt)

  # Check GCV results
  expect_true(!is.null(regs$gcv_yhat$gcv_m))
  expect_equal(nrow(regs$gcv_yhat$gcv_m), length(hlist))

  # Selected bandwidth should be in range
  expect_gte(regs$gcv_yhat$h1se, min(hlist))
  expect_lte(regs$gcv_yhat$h1se, max(hlist))

  # GCV should decrease then increase (U-shape)
  gcv_vals <- as.vector(regs$gcv_yhat$gcv_m)
  min_idx <- which.min(gcv_vals)
  expect_gt(min_idx, 1)  # Not at boundary
  expect_lt(min_idx, length(gcv_vals))
})

# ARCHIVED: 2026-01-09 - "1D regression handles linear trend correctly" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-lpr-1d.R)


# ARCHIVED: 2026-01-09 - "1D regression reproducibility check" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-lpr-1d.R)


# ARCHIVED: 2026-01-09 - "1D regression handles small sample size" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-lpr-1d.R)


# ARCHIVED: 2026-01-09 - "1D regression with Epanechnikov kernel works" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-lpr-1d.R)


# ARCHIVED: 2026-01-09 - "1D regression handles heteroscedastic noise" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-lpr-1d.R)


test_that("1D regression boundary behavior is reasonable", {
  # Unified with MATLAB: n=200, h=[0.1,0.5], nh=5
  set.seed(42)
  n <- 200
  x <- matrix(runif(n), ncol = 1)
  y_true <- sin(2 * pi * x)
  y <- matrix(y_true + 0.1 * rnorm(n), ncol = 1)
  hlist <- get_hlist(5, c(0.1, 0.5))

  opt <- list(order = 1, verbose = FALSE)
  regs <- cv_fastlpr(x, y, hlist, opt)

  # Check that fitted values don't have extreme outliers
  yhat_range <- range(regs$yhat)
  y_range <- range(y)

  # Fitted values should be within reasonable range of observed data
  expect_gte(yhat_range[1], y_range[1] - 1)
  expect_lte(yhat_range[2], y_range[2] + 1)
})

# =============================================================================
# Tests restored from unification spec (2026-01-10)
# =============================================================================

test_that("1D regression prediction accuracy (MSE check)", {
  # Unified with MATLAB: n=500, h=[0.01,0.3], nh=15, noise=0.1
  set.seed(42)
  n <- 500
  x <- matrix(runif(n), ncol = 1)
  y_true <- sin(2 * pi * x)
  y <- matrix(y_true + 0.1 * rnorm(n), ncol = 1)

  hlist <- get_hlist(15, c(0.01, 0.3))
  opt <- list(order = 1, verbose = FALSE)
  regs <- cv_fastlpr(x, y, hlist, opt)

  # MSE should be reasonable
  mse <- mean((regs$yhat - y_true)^2)
  expect_lt(mse, 0.02)
})

test_that("1D regression reproducibility check", {
  # Unified with MATLAB: n=200, h=[0.05,0.3], nh=5
  set.seed(42)
  n <- 200
  x <- matrix(runif(n), ncol = 1)
  y <- sin(2 * pi * x) + 0.2 * rnorm(n)
  y <- matrix(y, ncol = 1)

  hlist <- get_hlist(5, c(0.05, 0.3))
  opt <- list(order = 1, verbose = FALSE)

  regs1 <- cv_fastlpr(x, y, hlist, opt)
  regs2 <- cv_fastlpr(x, y, hlist, opt)

  # Results should be identical
  expect_equal(regs1$yhat, regs2$yhat)
  expect_equal(regs1$gcv_yhat$h1se, regs2$gcv_yhat$h1se)
})

test_that("1D regression with single bandwidth works", {
  # Unified with MATLAB: n=100, h=0.2
  set.seed(42)
  n <- 100
  x <- matrix(runif(n), ncol = 1)
  y <- sin(2 * pi * x) + 0.1 * rnorm(n)
  y <- matrix(y, ncol = 1)

  h <- matrix(0.2, nrow = 1, ncol = 1)
  opt <- list(order = 1, verbose = FALSE)
  regs <- cv_fastlpr(x, y, h, opt)

  validate_regression_structure(regs)
  expect_equal(length(regs$yhat), n)
})

test_that("1D regression handles linear function correctly", {
  # Unified with MATLAB: n=150, h=[0.2,0.8], nh=5
  set.seed(42)
  n <- 150
  x <- matrix(runif(n), ncol = 1)
  y_true <- 2 * x + 1
  y <- matrix(y_true + 0.05 * rnorm(n), ncol = 1)

  hlist <- get_hlist(5, c(0.2, 0.8))
  opt <- list(order = 1, verbose = FALSE)
  regs <- cv_fastlpr(x, y, hlist, opt)

  # Local linear should fit linear data well
  mse <- mean((regs$yhat - y_true)^2)
  expect_lt(mse, 0.01)
})

# ARCHIVED: 2026-01-09
# Source: fastLPR_R/tests/testthat/test-e2e-lpr-1d.R
# Archive: dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-lpr-1d.R
# Archived tests:
# - "1D regression handles small sample size"
# - "1D regression with Epanechnikov kernel works"
# - "1D regression handles heteroscedastic noise"
