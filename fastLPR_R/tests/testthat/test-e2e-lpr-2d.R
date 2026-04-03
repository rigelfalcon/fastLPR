# Test suite for 2D Local Polynomial Regression
context("2D Local Polynomial Regression")

test_that("2D order 0 regression works", {
  # Unified with MATLAB: n=500, h=[0.05,0.5], nh=10x10, noise=0.2
  set.seed(42)
  n <- 500
  x <- matrix(runif(n * 2), ncol = 2)
  y_true <- sin(2 * pi * x[, 1]) * cos(2 * pi * x[, 2])
  y <- matrix(y_true + 0.2 * rnorm(n), ncol = 1)

  hlist <- get_hlist(c(10, 10), list(c(0.05, 0.5), c(0.05, 0.5)))
  opt <- list(order = 0, calc_dof = TRUE, verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  validate_regression_structure(regs)
  expect_equal(length(regs$yhat), n)
  expect_equal(length(regs$gcv_yhat$h1se), 2)
})

test_that("2D order 1 regression works", {
  # Unified with MATLAB: n=500, h=[0.05,0.5], nh=10x10, noise=0.2
  set.seed(42)
  n <- 500
  x <- matrix(runif(n * 2), ncol = 2)
  y_true <- sin(2 * pi * x[, 1]) * cos(2 * pi * x[, 2])
  y <- matrix(y_true + 0.2 * rnorm(n), ncol = 1)

  hlist <- get_hlist(c(10, 10), list(c(0.05, 0.5), c(0.05, 0.5)))
  opt <- list(order = 1, calc_dof = TRUE, verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  validate_regression_structure(regs)
  expect_equal(length(regs$yhat), n)
  expect_equal(length(regs$gcv_yhat$h1se), 2)
})

test_that("2D order 2 regression works", {
  # Unified with MATLAB: n=500, h=[0.05,0.5], nh=10x10, noise=0.2
  set.seed(42)
  n <- 500
  x <- matrix(runif(n * 2), ncol = 2)
  y_true <- sin(2 * pi * x[, 1]) * cos(2 * pi * x[, 2])
  y <- matrix(y_true + 0.2 * rnorm(n), ncol = 1)

  hlist <- get_hlist(c(10, 10), list(c(0.05, 0.5), c(0.05, 0.5)))
  opt <- list(order = 2, calc_dof = TRUE, verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  validate_regression_structure(regs)
  expect_equal(length(regs$yhat), n)
})

# ARCHIVED: 2026-01-09 - "2D regression with single bandwidth works" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-lpr-2d.R)


test_that("2D regression GCV bandwidth selection works", {
  # Unified with MATLAB: n=400, h=[0.1,0.6], nh=5x5, noise=0.1, N=[50,50]
  set.seed(42)
  n <- 400
  x <- matrix(2 * runif(n * 2) - 1, ncol = 2)  # [-1, 1]^2
  y_true <- sin(pi * x[, 1]) * cos(pi * x[, 2])
  y <- matrix(y_true + 0.1 * rnorm(n), ncol = 1)

  hlist <- get_hlist(c(5, 5), list(c(0.1, 0.6), c(0.1, 0.6)))
  opt <- list(order = 1, N = c(50, 50), verbose = FALSE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  # Check GCV results
  expect_true(!is.null(regs$gcv_yhat$gcv_m))
  expect_true(!is.null(regs$gcv_yhat$h1se))
  expect_equal(length(regs$gcv_yhat$h1se), 2)

  # Selected bandwidth should be in range
  expect_true(all(regs$gcv_yhat$h1se >= 0.1))
  expect_true(all(regs$gcv_yhat$h1se <= 0.6))
})

# ARCHIVED: 2026-01-09 - "2D regression handles anisotropic data" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-lpr-2d.R)


# ARCHIVED: 2026-01-09 - "2D regression with different grid sizes works" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-lpr-2d.R)


# ARCHIVED: 2026-01-09 - "2D regression handles linear surface correctly" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-lpr-2d.R)


# ARCHIVED: 2026-01-09 - "2D regression reproducibility check" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-lpr-2d.R)


# ARCHIVED: 2026-01-09 - "2D regression handles small sample size" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-lpr-2d.R)


# ARCHIVED: 2026-01-09 - "2D regression with Epanechnikov kernel works" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-lpr-2d.R)


# =============================================================================
# Tests restored from unification spec (2026-01-10)
# =============================================================================

test_that("2D regression handles anisotropic data", {
  # Unified with MATLAB: n=500, h=[[0.02,0.3],[0.1,0.8]], nh=10x10, noise=0.1
  set.seed(42)
  n <- 500
  x <- cbind(runif(n) * 0.5, runif(n) * 2)
  y_true <- x[, 1]^2 + x[, 2]
  y <- matrix(y_true + 0.1 * rnorm(n), ncol = 1)

  hlist <- get_hlist(c(10, 10), list(c(0.02, 0.3), c(0.1, 0.8)))
  opt <- list(order = 1, verbose = FALSE)
  regs <- cv_fastlpr(x, y, hlist, opt)

  validate_regression_structure(regs)
  expect_equal(length(regs$yhat), n)
})

# ARCHIVED: 2026-01-09
# Source: fastLPR_R/tests/testthat/test-e2e-lpr-2d.R
# Archive: dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-lpr-2d.R
# Archived tests:
# - "2D regression with single bandwidth works"
# - "2D regression with different grid sizes works"
# - "2D regression handles linear surface correctly"
# - "2D regression reproducibility check"
# - "2D regression handles small sample size"
# - "2D regression with Epanechnikov kernel works"
