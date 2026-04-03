# ==============================================================================
# Unit Tests for fastLPR vs Naive Nadaraya-Watson
# ==============================================================================
#
# This file validates that fastLPR (NUFFT-accelerated) produces results
# consistent with a naive O(N*M) Nadaraya-Watson implementation.
#
# Matches:
# - MATLAB: fastLPR/tests/selftest_unit/test_fastlpr_vs_naive_nw.m
# - Python: fastLPR_py/tests/test_algorithm.py
#
# Test cases:
# 1. 1D regression with varying sample sizes
# 2. 2D regression
# 3. Complex-valued data
#
# Author: fastLPR Development Team
# Copyright (c) 2020-2025 fastLPR Development Team
# License: GNU General Public License v3.0
# ==============================================================================

context("Unit Tests: fastLPR vs Naive Nadaraya-Watson")

# =============================================================================
# Naive Nadaraya-Watson Implementation
# =============================================================================

#' Naive O(N*M) Nadaraya-Watson kernel regression
#'
#' @param x_train Training points (N x d matrix)
#' @param y_train Training values (N x 1 or N-length vector)
#' @param h Bandwidth (scalar for isotropic, d-length vector for per-dimension)
#' @param x_eval Evaluation points (default: x_train)
#' @return Predicted values at x_eval
naive_nw_smooth <- function(x_train, y_train, h, x_eval = NULL) {
  # Ensure matrix format
  x_train <- as.matrix(x_train)
  y_train <- as.matrix(y_train)

  if (is.null(x_eval)) {
    x_eval <- x_train
  } else {
    x_eval <- as.matrix(x_eval)
  }

  n_train <- nrow(x_train)
  n_eval <- nrow(x_eval)
  n_dim <- ncol(x_train)

  # Broadcast bandwidth
  h <- rep_len(h, n_dim)

  # Output
  yhat <- matrix(0, nrow = n_eval, ncol = ncol(y_train))

  for (i in seq_len(n_eval)) {
    # Compute scaled differences
    diff <- sweep(x_train, 2, x_eval[i, , drop = FALSE], "-")
    diff_scaled <- sweep(diff, 2, h, "/")

    # Squared distances
    u2 <- rowSums(diff_scaled^2)

    # Gaussian kernel weights
    weights <- exp(-0.5 * u2)

    # Normalize
    sum_weights <- sum(weights)
    if (sum_weights > 0) {
      yhat[i, ] <- colSums(weights * y_train) / sum_weights
    } else {
      yhat[i, ] <- NA
    }
  }

  return(yhat)
}

# =============================================================================
# Test Constants
# =============================================================================

RANDOM_SEED <- 42
ACCURACY <- 6

# =============================================================================
# Section 1: 1D Regression Tests
# =============================================================================

test_that("1D small sample: fastLPR matches naive NW (n=500)", {
  # Matches: MATLAB test_1d_small, Python test_1d_small
  set.seed(RANDOM_SEED)
  n <- 500
  x <- matrix(abs(2 * (runif(n) - 0.5) * 15), ncol = 1)
  y_true <- besselJ(x, 1)  # Bessel function J1
  y <- y_true + 0.5 * sd(y_true) * rnorm(n)
  h <- 0.5

  # Naive NW
  yhat_naive <- naive_nw_smooth(x, y, h)

  # fastLPR
  opt <- list(order = 0, accuracy = ACCURACY, calc_dof = FALSE, verbose = FALSE)
  regs <- cv_fastlpr(x, y, h, opt)
  yhat_fast <- regs$yhat

  # Verify accuracy (NUFFT vs naive have small numerical differences)
  mse <- mean((yhat_fast - yhat_naive)^2)
  expected_mse <- 0.05  # NUFFT approximation vs naive O(N*M) allows some difference

  expect_lt(mse, expected_mse,
    label = sprintf("MSE=%.2e should be < %.2e for n=%d", mse, expected_mse, n))
})

test_that("1D medium sample: fastLPR matches naive NW (n=2000)", {
  # Matches: MATLAB test_1d_medium pattern (reduced n for speed)
  set.seed(RANDOM_SEED)
  n <- 2000
  x <- matrix(abs(2 * (runif(n) - 0.5) * 15), ncol = 1)
  y_true <- besselJ(x, 1)
  y <- y_true + 0.5 * sd(y_true) * rnorm(n)
  h <- 0.5

  # Naive NW
  yhat_naive <- naive_nw_smooth(x, y, h)

  # fastLPR
  opt <- list(order = 0, accuracy = ACCURACY, calc_dof = FALSE, verbose = FALSE)
  regs <- cv_fastlpr(x, y, h, opt)
  yhat_fast <- regs$yhat

  # Verify accuracy
  mse <- mean((yhat_fast - yhat_naive)^2)
  expected_mse <- 0.05  # NUFFT approximation vs naive O(N*M) allows some difference

  expect_lt(mse, expected_mse,
    label = sprintf("MSE=%.2e should be < %.2e for n=%d", mse, expected_mse, n))
})

test_that("1D large sample: fastLPR matches naive NW (n=10000)", {
  # Matches: MATLAB test_1d_large, Python test_1d_large
  set.seed(RANDOM_SEED)
  n <- 10000
  x <- matrix(abs(2 * (runif(n) - 0.5) * 15), ncol = 1)
  y_true <- besselJ(x, 1)
  y <- y_true + 0.5 * sd(y_true) * rnorm(n)
  h <- 0.5

  # Naive NW
  yhat_naive <- naive_nw_smooth(x, y, h)

  # fastLPR
  opt <- list(order = 0, accuracy = ACCURACY, calc_dof = FALSE, verbose = FALSE)
  regs <- cv_fastlpr(x, y, h, opt)
  yhat_fast <- regs$yhat

  # Verify accuracy
  mse <- mean((yhat_fast - yhat_naive)^2)
  expected_mse <- 0.05  # NUFFT approximation vs naive O(N*M) allows some difference

  expect_lt(mse, expected_mse,
    label = sprintf("MSE=%.2e should be < %.2e for n=%d", mse, expected_mse, n))
})

# =============================================================================
# Section 2: 2D Regression Tests
# =============================================================================

test_that("2D small sample: fastLPR matches naive NW (n=400)", {
  # Matches: MATLAB test_2d_small, Python test_2d_small
  set.seed(RANDOM_SEED)
  n <- 400
  x <- matrix(runif(n * 2) * 4 - 2, ncol = 2)  # [-2, 2]^2
  y_true <- sin(pi * x[, 1]) * cos(pi * x[, 2])
  y <- matrix(y_true + 0.2 * rnorm(n), ncol = 1)
  h <- c(0.3, 0.3)

  # Naive NW
  yhat_naive <- naive_nw_smooth(x, y, h)

  # fastLPR
  opt <- list(order = 0, accuracy = ACCURACY, calc_dof = FALSE, verbose = FALSE)
  regs <- cv_fastlpr(x, y, matrix(h, nrow = 1), opt)
  yhat_fast <- regs$yhat

  # Verify accuracy
  mse <- mean((yhat_fast - yhat_naive)^2)
  expected_mse <- 0.05  # NUFFT approximation vs naive O(N*M) allows some difference

  expect_lt(mse, expected_mse,
    label = sprintf("2D MSE=%.2e should be < %.2e", mse, expected_mse))
})

test_that("2D medium sample: fastLPR matches naive NW (n=1000)", {
  # Matches: MATLAB test_2d_medium, Python test_2d_medium
  set.seed(RANDOM_SEED)
  n <- 1000
  x <- matrix(runif(n * 2) * 4 - 2, ncol = 2)
  y_true <- sin(pi * x[, 1]) * cos(pi * x[, 2])
  y <- matrix(y_true + 0.2 * rnorm(n), ncol = 1)
  h <- c(0.3, 0.3)

  # Naive NW
  yhat_naive <- naive_nw_smooth(x, y, h)

  # fastLPR
  opt <- list(order = 0, accuracy = ACCURACY, calc_dof = FALSE, verbose = FALSE)
  regs <- cv_fastlpr(x, y, matrix(h, nrow = 1), opt)
  yhat_fast <- regs$yhat

  # Verify accuracy
  mse <- mean((yhat_fast - yhat_naive)^2)
  expected_mse <- 0.05  # NUFFT approximation vs naive O(N*M) allows some difference

  expect_lt(mse, expected_mse,
    label = sprintf("2D MSE=%.2e should be < %.2e", mse, expected_mse))
})

test_that("2D large sample: fastLPR matches naive NW (n=3000)", {
  # Matches: MATLAB test_2d_large, Python test_2d_large
  set.seed(RANDOM_SEED)
  n <- 3000
  x <- matrix(runif(n * 2) * 4 - 2, ncol = 2)
  y_true <- sin(pi * x[, 1]) * cos(pi * x[, 2])
  y <- matrix(y_true + 0.2 * rnorm(n), ncol = 1)
  h <- c(0.3, 0.3)

  # Naive NW
  yhat_naive <- naive_nw_smooth(x, y, h)

  # fastLPR
  opt <- list(order = 0, accuracy = ACCURACY, calc_dof = FALSE, verbose = FALSE)
  regs <- cv_fastlpr(x, y, matrix(h, nrow = 1), opt)
  yhat_fast <- regs$yhat

  # Verify accuracy
  mse <- mean((yhat_fast - yhat_naive)^2)
  expected_mse <- 0.05  # NUFFT approximation vs naive O(N*M) allows some difference

  expect_lt(mse, expected_mse,
    label = sprintf("2D MSE=%.2e should be < %.2e", mse, expected_mse))
})

# =============================================================================
# Section 3: Complex-valued Data Tests
# =============================================================================

test_that("Complex 1D: fastLPR matches naive NW for complex data", {
  # Matches: MATLAB test_complex, Python test_complex_1d
  set.seed(RANDOM_SEED)
  n <- 500
  x <- matrix(runif(n) * 2 * pi, ncol = 1)
  y_true <- exp(1i * x)  # Complex signal: e^(ix)
  noise <- 0.1 * (rnorm(n) + 1i * rnorm(n))
  y <- matrix(y_true + noise, ncol = 1)
  h <- 0.3

  # Naive NW (works with complex data)
  yhat_naive <- naive_nw_smooth(x, y, h)

  # fastLPR with higher accuracy for complex
  opt <- list(order = 0, accuracy = 9, calc_dof = FALSE, verbose = FALSE)
  regs <- cv_fastlpr(x, y, h, opt)
  yhat_fast <- regs$yhat

  # Verify accuracy (use Mod for complex error)
  mse <- mean(Mod(yhat_fast - yhat_naive)^2)
  expected_mse <- 0.05  # Complex data uses relaxed threshold

  expect_lt(mse, expected_mse,
    label = sprintf("Complex MSE=%.2e should be < %.2e", mse, expected_mse))
})

# =============================================================================
# Section 4: Consistency Tests
# =============================================================================

test_that("Order 0 and 1 give similar results for smooth data", {
  # Matches: Python test_order0_vs_order1_smooth
  set.seed(RANDOM_SEED)
  n <- 500
  x <- matrix(runif(n), ncol = 1)
  y_true <- sin(2 * pi * x)
  y <- matrix(y_true + 0.1 * rnorm(n), ncol = 1)
  h <- 0.15

  opt0 <- list(order = 0, accuracy = ACCURACY, calc_dof = FALSE)
  opt1 <- list(order = 1, accuracy = ACCURACY, calc_dof = FALSE)

  regs0 <- cv_fastlpr(x, y, h, opt0)
  regs1 <- cv_fastlpr(x, y, h, opt1)

  # Both should give reasonable fits
  expect_true(all(is.finite(regs0$yhat)), label = "Order 0 should give finite values")
  expect_true(all(is.finite(regs1$yhat)), label = "Order 1 should give finite values")

  # Correlation should be high
  corr <- cor(as.vector(regs0$yhat), as.vector(regs1$yhat))
  expect_gt(corr, 0.9, label = sprintf("Order 0 and 1 correlation %.3f should be > 0.9", corr))
})

test_that("fastLPR output is deterministic", {
  # Matches: Python test_deterministic_output
  set.seed(RANDOM_SEED)
  n <- 200
  x <- matrix(runif(n), ncol = 1)
  y <- matrix(sin(2 * pi * x) + 0.1 * rnorm(n), ncol = 1)
  h <- 0.2

  opt <- list(order = 1, accuracy = ACCURACY, calc_dof = FALSE)

  regs1 <- cv_fastlpr(x, y, h, opt)
  regs2 <- cv_fastlpr(x, y, h, opt)

  expect_equal(regs1$yhat, regs2$yhat, tolerance = 1e-12,
    label = "Output should be deterministic")
})

test_that("Higher accuracy improves precision", {
  # Matches: Python test_increasing_accuracy_improves_precision
  set.seed(RANDOM_SEED)
  n <- 500
  x <- matrix(runif(n), ncol = 1)
  y <- matrix(sin(2 * pi * x), ncol = 1)  # No noise for precision test
  h <- 0.2

  # Compute with different accuracy levels
  yhat <- list()
  for (acc in c(4, 6, 9)) {
    opt <- list(order = 0, accuracy = acc, calc_dof = FALSE)
    regs <- cv_fastlpr(x, y, h, opt)
    yhat[[as.character(acc)]] <- regs$yhat
  }

  # Higher accuracy should generally be closer to highest accuracy result
  # Allow some tolerance since NUFFT approximation is non-monotonic
  err_4 <- max(abs(yhat[["4"]] - yhat[["9"]]))
  err_6 <- max(abs(yhat[["6"]] - yhat[["9"]]))

  # Either err_6 < err_4, or both are very small (< 0.001)
  expect_true(err_6 < err_4 || (err_4 < 0.001 && err_6 < 0.001),
    label = sprintf("acc=6 (%.2e) should be <= acc=4 (%.2e) or both < 0.001", err_6, err_4))
})
