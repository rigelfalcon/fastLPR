# test-lpr-hetero.R - Unit tests for heteroscedastic regression (mean + variance)
#
# Author: Ying Wang, Min Li
# Copyright (c) 2020-2025 fastLPR Development Team
# License: GNU General Public License v3.0

context("Heteroscedastic Regression (Mean + Variance)")

test_that("cv_fastlpr estimates mean and variance", {
  # Unified with MATLAB: n=500, h=[0.01,0.5], nh=10
  set.seed(42)
  n <- 500
  x <- matrix(sort(runif(n)), ncol = 1)
  y_true <- sin(2 * pi * x)
  var_true <- 0.1 + 0.3 * x^2
  y <- y_true + sqrt(var_true) * rnorm(n)
  y <- matrix(y, ncol = 1)
  hlist <- matrix(10^seq(log10(0.01), log10(0.5), length.out = 10), ncol = 1)
  opt <- list(order = 1, calc_dof = TRUE, y_type_out = "mean")

  regs <- cv_fastlpr(x, y, hlist, opt)

  expect_true(!is.null(regs$yhat))
  expect_equal(length(regs$yhat), n)
})

test_that("mean prediction is accurate", {
  # Unified with MATLAB/Python: n=500, h=[0.01,0.3], nh=15, heteroscedastic noise
  set.seed(42)
  n <- 500
  x <- matrix(sort(runif(n)), ncol = 1)
  y_true <- sin(2 * pi * x)
  var_true <- 0.1 + 0.3 * x^2
  y <- y_true + sqrt(var_true) * rnorm(n)
  y <- matrix(y, ncol = 1)
  hlist <- matrix(10^seq(log10(0.01), log10(0.3), length.out = 15), ncol = 1)
  opt <- list(order = 1, y_type_out = "mean")

  regs <- cv_fastlpr(x, y, hlist, opt)

  mse <- mean((regs$yhat - y_true)^2)
  expect_true(mse < 0.05)
})

test_that("variance estimates are non-negative", {
  # Unified with MATLAB: n=500, h=[0.02,0.4], nh=10
  # Uses two-step workflow: mean estimation -> squared residuals -> variance estimation
  set.seed(42)
  n <- 500
  x <- matrix(sort(runif(n)), ncol = 1)
  var_true <- 0.1 + 0.5 * x
  y <- x^2 + sqrt(var_true) * rnorm(n)
  y <- matrix(y, ncol = 1)
  hlist <- matrix(10^seq(log10(0.02), log10(0.4), length.out = 10), ncol = 1)

  # Step 1: Mean estimation
  opt_mean <- list(order = 1, y_type_out = "mean")
  regs_mean <- cv_fastlpr(x, y, hlist, opt_mean)

  # Step 2: Compute squared residuals
  residuals_sq <- (y - regs_mean$yhat)^2

  # Step 3: Variance estimation
  opt_var <- list(order = 1, y_type_out = "variance")
  regs_var <- cv_fastlpr(x, residuals_sq, hlist, opt_var)

  expect_true(all(regs_var$yhat >= 0))
})

test_that("2D heteroscedastic variance estimation works", {
  # Unified with MATLAB: n=400, h=[0.05,0.5], nh=5x5
  set.seed(42)
  n <- 400
  x <- matrix(runif(n * 2), ncol = 2)
  y_true <- sin(2 * pi * x[, 1]) * cos(2 * pi * x[, 2])
  var_true <- 0.1 + 0.2 * x[, 1]^2 + 0.2 * x[, 2]^2
  y <- y_true + sqrt(var_true) * rnorm(n)
  y <- matrix(y, ncol = 1)

  hlist <- get_hlist(c(5, 5), list(c(0.05, 0.5), c(0.05, 0.5)))

  # Step 1: Mean estimation
  opt_mean <- list(order = 1, y_type_out = "mean")
  regs_mean <- cv_fastlpr(x, y, hlist, opt_mean)

  # Step 2: Compute squared residuals
  residuals_sq <- (y - regs_mean$yhat)^2

  # Step 3: Variance estimation
  opt_var <- list(order = 1, y_type_out = "variance")
  regs_var <- cv_fastlpr(x, residuals_sq, hlist, opt_var)

  expect_true(!is.null(regs_var$yhat))
  expect_equal(length(regs_var$yhat), n)
  expect_true(all(regs_var$yhat >= 0))
})

test_that("heteroscedastic regression is reproducible", {
  # Unified with MATLAB: n=200, h=[0.1,0.5], nh=5
  set.seed(42)
  n <- 200
  x <- matrix(sort(runif(n)), ncol = 1)
  var_true <- 0.1 + 0.3 * x^2
  y <- sin(2 * pi * x) + sqrt(var_true) * rnorm(n)
  y <- matrix(y, ncol = 1)

  hlist <- matrix(10^seq(log10(0.1), log10(0.5), length.out = 5), ncol = 1)

  # Step 1: Mean estimation (run twice)
  opt_mean <- list(order = 1, y_type_out = "mean")
  regs_mean1 <- cv_fastlpr(x, y, hlist, opt_mean)
  regs_mean2 <- cv_fastlpr(x, y, hlist, opt_mean)

  # Verify mean estimation is reproducible
  expect_equal(regs_mean1$yhat, regs_mean2$yhat)

  # Step 2: Compute squared residuals
  residuals_sq <- (y - regs_mean1$yhat)^2

  # Step 3: Variance estimation (run twice)
  opt_var <- list(order = 1, y_type_out = "variance")
  regs_var1 <- cv_fastlpr(x, residuals_sq, hlist, opt_var)
  regs_var2 <- cv_fastlpr(x, residuals_sq, hlist, opt_var)

  # Verify variance estimation is reproducible
  expect_equal(regs_var1$yhat, regs_var2$yhat)
})
