# test-lpr-complex.R - Unit tests for complex-valued regression
#
# Author: Ying Wang, Min Li
# Copyright (c) 2020-2025 fastLPR Development Team
# License: GNU General Public License v3.0

context("Complex-valued Regression")

test_that("cv_fastlpr handles complex 1D data", {
  # Unified with MATLAB: n=500, h=[0.01,0.5], nh=10, noise=0.2
  set.seed(42)
  n <- 500
  x <- matrix(sort(runif(n)), ncol = 1)
  y_real <- sin(2 * pi * x)
  y_imag <- cos(2 * pi * x)
  y <- complex(real = y_real + 0.2 * rnorm(n), imaginary = y_imag + 0.2 * rnorm(n))
  y <- matrix(y, ncol = 1)
  hlist <- matrix(10^seq(log10(0.01), log10(0.5), length.out = 10), ncol = 1)
  opt <- list(order = 1, calc_dof = TRUE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  expect_true(!is.null(regs$yhat))
  expect_equal(length(regs$yhat), n)
})

test_that("complex regression selects valid bandwidth", {
  # Unified with MATLAB: n=500, h=[0.01,0.5], nh=15, noise=0.1
  set.seed(42)
  n <- 500
  x <- matrix(sort(runif(n)), ncol = 1)
  y <- exp(1i * 2 * pi * x) + 0.1 * (rnorm(n) + 1i * rnorm(n))
  y <- matrix(y, ncol = 1)
  hlist <- matrix(10^seq(log10(0.01), log10(0.5), length.out = 15), ncol = 1)
  opt <- list(order = 1)

  regs <- cv_fastlpr(x, y, hlist, opt)

  expect_true(regs$gcv_yhat$h1se > hlist[1])
  expect_true(regs$gcv_yhat$h1se < hlist[nrow(hlist)])
})

test_that("GCV scores are real for complex data", {
  # Unified with MATLAB: n=500, h=[0.01,0.5], nh=10, noise=0.1
  set.seed(42)
  n <- 500
  x <- matrix(sort(runif(n)), ncol = 1)
  y <- complex(real = sin(2 * pi * x), imaginary = cos(2 * pi * x))
  y <- y + 0.1 * (rnorm(n) + 1i * rnorm(n))
  y <- matrix(y, ncol = 1)
  hlist <- matrix(10^seq(log10(0.01), log10(0.5), length.out = 10), ncol = 1)
  opt <- list(order = 1, calc_dof = TRUE)

  regs <- cv_fastlpr(x, y, hlist, opt)

  expect_true(is.numeric(regs$gcv_yhat$gcv_m))
  expect_true(all(!is.complex(regs$gcv_yhat$gcv_m)))
})

# ARCHIVED: 2026-01-09 - "complex regression is reproducible" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-lpr-complex.R)


# =============================================================================
# Tests restored from unification spec (2026-01-10)
# =============================================================================

test_that("complex regression preserves real/imag correlation", {
  # Unified with MATLAB: n=200, h=[0.05,0.5], nh=5, noise=0.1
  set.seed(42)
  n <- 200
  x <- matrix(sort(runif(n)), ncol = 1)
  y_real <- sin(2 * pi * x)
  y_imag <- cos(2 * pi * x)
  y_true <- complex(real = y_real, imaginary = y_imag)
  noise <- 0.1 * (rnorm(n) + 1i * rnorm(n))
  y <- matrix(y_true + noise, ncol = 1)

  hlist <- matrix(10^seq(log10(0.05), log10(0.5), length.out = 5), ncol = 1)
  opt <- list(order = 1)

  regs <- cv_fastlpr(x, y, hlist, opt)

  # Check real and imaginary parts separately
  corr_real <- cor(Re(regs$yhat), as.vector(y_real))
  corr_imag <- cor(Im(regs$yhat), as.vector(y_imag))

  expect_gt(corr_real, 0.9)
  expect_gt(corr_imag, 0.9)
})

test_that("complex regression with single bandwidth works", {
  # Unified with MATLAB: n=100, h=0.15, noise=0.1
  set.seed(42)
  n <- 100
  x <- matrix(sort(runif(n)), ncol = 1)
  y <- sin(2 * pi * x) + 1i * cos(2 * pi * x)
  y <- y + 0.1 * (rnorm(n) + 1i * rnorm(n))
  y <- matrix(y, ncol = 1)

  h <- matrix(0.15, nrow = 1, ncol = 1)
  opt <- list(order = 1)
  regs <- cv_fastlpr(x, y, h, opt)

  expect_true(!is.null(regs$yhat))
  expect_true(is.complex(regs$yhat))
})

# ARCHIVED: 2026-01-09
# Source: fastLPR_R/tests/testthat/test-e2e-lpr-complex.R
# Archive: dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-lpr-complex.R
# Archived tests:
# - "complex regression is reproducible"
