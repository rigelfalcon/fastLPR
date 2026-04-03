# ==============================================================================
# Unit Tests for Kernel Functions
# ==============================================================================
#
# Tests kernel_function for Gaussian and Epanechnikov kernels.
# Matches MATLAB/Python kernel tests for cross-language consistency.
#
# Matches:
# - MATLAB: fastLPR/tests/selftest_unit/test_kernel.m
# - Python: fastLPR_py/tests/test_kernel.py
#
# Author: fastLPR Development Team
# Copyright (c) 2020-2025 fastLPR Development Team
# License: GNU General Public License v3.0
# ==============================================================================

context("Unit Tests: Kernel Functions")

# =============================================================================
# Helper Functions
# =============================================================================

# Get kernel_function from package namespace
get_kernel_fn <- function() {
  fn <- get0("kernel_function", envir = asNamespace("fastlpr"), inherits = FALSE)
  if (!is.function(fn)) {
    skip("kernel_function not available")
  }
  return(fn)
}

# =============================================================================
# Test Constants
# =============================================================================

TOL_STRICT <- 1e-12
TOL_NUMERICAL <- 1e-6
TOL_NORMALIZATION <- 0.02  # 2% for numerical integration

# =============================================================================
# Section 1: Gaussian Kernel - Basic Values
# =============================================================================

test_that("Gaussian at origin 1D equals 1/sqrt(2*pi)/h", {
  kernel_fn <- get_kernel_fn()

  # Single point at origin
  xgrid <- array(0, dim = c(1, 1))
  h <- matrix(1.0, nrow = 1, ncol = 1)

  K <- kernel_fn(xgrid, NULL, h, "gaussian")
  expected <- 1 / sqrt(2 * pi)  # h=1, so no scaling

  expect_equal(as.numeric(K), expected, tolerance = TOL_STRICT)
})

test_that("Gaussian at origin 2D", {
  kernel_fn <- get_kernel_fn()

  # Single point at origin
  xgrid <- array(0, dim = c(1, 1, 2))
  h <- matrix(c(1.0, 1.0), nrow = 1, ncol = 2)

  K <- kernel_fn(xgrid, NULL, h, "gaussian")
  # For 2D: K(0,0) = 1/sqrt(2*pi) / (h1*h2)
  expected <- 1 / sqrt(2 * pi)

  expect_equal(as.numeric(K), expected, tolerance = TOL_STRICT)
})

test_that("Gaussian at origin 3D", {
  # Matches: MATLAB test_gaussian_at_origin_3d, Python test_gaussian_at_origin_3d
  kernel_fn <- get_kernel_fn()

  # Single point at origin in 3D
  xgrid <- array(0, dim = c(1, 1, 1, 3))
  h <- matrix(c(1.0, 1.0, 1.0), nrow = 1, ncol = 3)

  K <- kernel_fn(xgrid, NULL, h, "gaussian")
  # MATLAB uses 1/sqrt(2*pi) normalization for all dimensions
  expected <- 1 / sqrt(2 * pi)

  expect_equal(as.numeric(K), expected, tolerance = TOL_NUMERICAL)
})

test_that("Gaussian decay with distance", {
  kernel_fn <- get_kernel_fn()

  N <- 11
  x <- seq(0, 3, length.out = N)
  xgrid <- array(x, dim = c(N, 1))
  h <- matrix(1.0, nrow = 1, ncol = 1)

  K <- as.numeric(kernel_fn(xgrid, NULL, h, "gaussian"))

  # Kernel should decay monotonically
  for (i in 2:N) {
    expect_lt(K[i], K[i-1],
      label = sprintf("K should decay: K[%d]=%.4f < K[%d]=%.4f", i, K[i], i-1, K[i-1]))
  }
})

test_that("Gaussian tail behavior", {
  kernel_fn <- get_kernel_fn()

  # Far from origin
  xgrid <- array(c(5, 10), dim = c(2, 1))
  h <- matrix(1.0, nrow = 1, ncol = 1)

  K <- as.numeric(kernel_fn(xgrid, NULL, h, "gaussian"))

  # Tail values should be very small
  expect_lt(K[1], 1e-5)
  expect_lt(K[2], 1e-20)
})

test_that("Gaussian 2D shape: scattered data returns column vector", {
  # Matches: MATLAB test_gaussian_2d_shape, Python test_gaussian_2d_scattered
  kernel_fn <- get_kernel_fn()

  # Scattered 2D points (n=50)
  n <- 50
  x <- matrix(runif(n * 2), ncol = 2)
  xgrid <- array(x, dim = c(n, 1, 2))
  h <- matrix(c(1, 1), nrow = 1, ncol = 2)

  K <- kernel_fn(xgrid, NULL, h, "gaussian")

  # Output should be n x 1 x 1
  expect_equal(length(K), n)
  expect_true(all(K > 0))
})

test_that("Gaussian 3D shape: scattered data returns column vector", {
  # Matches: MATLAB test_gaussian_3d_shape, Python test_gaussian_3d_grid
  kernel_fn <- get_kernel_fn()

  # Scattered 3D points (n=50)
  n <- 50
  x <- matrix(runif(n * 3), ncol = 3)
  xgrid <- array(x, dim = c(n, 1, 1, 3))
  h <- matrix(c(1, 1, 1), nrow = 1, ncol = 3)

  K <- kernel_fn(xgrid, NULL, h, "gaussian")

  # Output should have n elements
  expect_equal(length(K), n)
  expect_true(all(K > 0))
})

# =============================================================================
# Section 2: Gaussian Kernel - Normalization
# =============================================================================

test_that("Gaussian 1D integrates to approximately 1", {
  kernel_fn <- get_kernel_fn()

  N <- 1001  # Unified: 1001 points for 1D integration (was 201)
  xmin <- -5
  xmax <- 5
  x <- seq(xmin, xmax, length.out = N)
  dx <- x[2] - x[1]
  xgrid <- array(x, dim = c(N, 1))
  h <- matrix(0.5, nrow = 1, ncol = 1)

  K <- as.numeric(kernel_fn(xgrid, NULL, h, "gaussian"))

  # Trapezoidal integration
  integral <- dx * (sum(K) - 0.5 * K[1] - 0.5 * K[N])

  expect_equal(integral, 1.0, tolerance = TOL_NORMALIZATION)
})

test_that("Gaussian with different bandwidth still integrates to 1", {
  kernel_fn <- get_kernel_fn()

  N <- 301
  x <- seq(-10, 10, length.out = N)
  dx <- x[2] - x[1]
  xgrid <- array(x, dim = c(N, 1))
  h <- matrix(2.0, nrow = 1, ncol = 1)  # Larger bandwidth

  K <- as.numeric(kernel_fn(xgrid, NULL, h, "gaussian"))
  integral <- dx * (sum(K) - 0.5 * K[1] - 0.5 * K[N])

  expect_equal(integral, 1.0, tolerance = TOL_NORMALIZATION)
})

# =============================================================================
# Section 3: Gaussian Kernel - Symmetry
# =============================================================================

test_that("Gaussian 1D is symmetric", {
  kernel_fn <- get_kernel_fn()

  N <- 21
  x <- seq(-2, 2, length.out = N)
  xgrid <- array(x, dim = c(N, 1))
  h <- matrix(0.5, nrow = 1, ncol = 1)

  K <- as.numeric(kernel_fn(xgrid, NULL, h, "gaussian"))

  # K(-x) = K(x)
  n_half <- (N - 1) / 2
  for (i in 1:n_half) {
    expect_equal(K[i], K[N - i + 1], tolerance = TOL_STRICT)
  }
})

test_that("Gaussian 2D is radially symmetric", {
  kernel_fn <- get_kernel_fn()

  N <- 21
  vals <- seq(-2, 2, length.out = N)
  xgrid <- array(0, dim = c(N, N, 2))
  for (j in 1:N) {
    for (i in 1:N) {
      xgrid[i, j, 1] <- vals[i]
      xgrid[i, j, 2] <- vals[j]
    }
  }
  h <- matrix(c(0.5, 0.5), nrow = 1, ncol = 2)  # Isotropic

  K <- kernel_fn(xgrid, NULL, h, "gaussian")[,,1]

  # Check radial symmetry: K(x,y) = K(y,x)
  expect_equal(K, t(K), tolerance = TOL_STRICT)
})

test_that("Gaussian 2D is symmetric: K(-x) = K(x)", {
  # Matches: MATLAB test_gaussian_symmetry_2d, Python test_gaussian_symmetry_2d
  kernel_fn <- get_kernel_fn()

  # Scattered points
  x_pos <- matrix(c(1, 0.5, 0.5, 1, 1, 1), ncol = 2, byrow = TRUE)
  x_neg <- -x_pos

  xgrid_pos <- array(x_pos, dim = c(3, 1, 2))
  xgrid_neg <- array(x_neg, dim = c(3, 1, 2))
  h <- matrix(c(1, 1), nrow = 1, ncol = 2)

  K_pos <- kernel_fn(xgrid_pos, NULL, h, "gaussian")
  K_neg <- kernel_fn(xgrid_neg, NULL, h, "gaussian")

  expect_equal(as.numeric(K_pos), as.numeric(K_neg), tolerance = TOL_STRICT)
})

test_that("Gaussian 3D is symmetric: K(-x) = K(x)", {
  # Matches: MATLAB test_gaussian_symmetry_3d, Python test_gaussian_symmetry_3d
  kernel_fn <- get_kernel_fn()

  # Scattered points in 3D
  x_pos <- matrix(c(1, 0.5, 0.3, 0.5, 1, 0.7, 1, 1, 1), ncol = 3, byrow = TRUE)
  x_neg <- -x_pos

  xgrid_pos <- array(x_pos, dim = c(3, 1, 1, 3))
  xgrid_neg <- array(x_neg, dim = c(3, 1, 1, 3))
  h <- matrix(c(1, 1, 1), nrow = 1, ncol = 3)

  K_pos <- kernel_fn(xgrid_pos, NULL, h, "gaussian")
  K_neg <- kernel_fn(xgrid_neg, NULL, h, "gaussian")

  expect_equal(as.numeric(K_pos), as.numeric(K_neg), tolerance = TOL_STRICT)
})

# =============================================================================
# Section 4: Gaussian Kernel - Positivity
# =============================================================================

test_that("Gaussian values are non-negative", {
  kernel_fn <- get_kernel_fn()

  N <- 51
  x <- seq(-5, 5, length.out = N)
  xgrid <- array(x, dim = c(N, 1))
  h <- matrix(0.5, nrow = 1, ncol = 1)

  K <- as.numeric(kernel_fn(xgrid, NULL, h, "gaussian"))

  expect_true(all(K >= 0))
})

test_that("Gaussian maximum is at origin", {
  kernel_fn <- get_kernel_fn()

  N <- 21
  x <- seq(-2, 2, length.out = N)
  xgrid <- array(x, dim = c(N, 1))
  h <- matrix(0.5, nrow = 1, ncol = 1)

  K <- as.numeric(kernel_fn(xgrid, NULL, h, "gaussian"))

  max_idx <- which.max(K)
  expect_equal(max_idx, (N + 1) / 2)  # Center index
})

# =============================================================================
# Section 5: Epanechnikov Kernel Tests
# =============================================================================

test_that("Epanechnikov has compact support", {
  kernel_fn <- get_kernel_fn()

  N <- 51
  x <- seq(-2, 2, length.out = N)
  xgrid <- array(x, dim = c(N, 1))
  h <- matrix(1.0, nrow = 1, ncol = 1)

  K <- as.numeric(kernel_fn(xgrid, NULL, h, "epanechnikov"))

  # Outside |u| > 1, should be nearly zero (clamped to eps)
  outside_mask <- abs(x) > 1
  expect_true(all(K[outside_mask] < 1e-10))
})

test_that("Epanechnikov is symmetric", {
  kernel_fn <- get_kernel_fn()

  N <- 21
  x <- seq(-2, 2, length.out = N)
  xgrid <- array(x, dim = c(N, 1))
  h <- matrix(1.0, nrow = 1, ncol = 1)

  K <- as.numeric(kernel_fn(xgrid, NULL, h, "epanechnikov"))

  # K(-x) = K(x)
  n_half <- (N - 1) / 2
  for (i in 1:n_half) {
    expect_equal(K[i], K[N - i + 1], tolerance = TOL_STRICT)
  }
})

test_that("Epanechnikov values are non-negative", {
  kernel_fn <- get_kernel_fn()

  N <- 51
  x <- seq(-2, 2, length.out = N)
  xgrid <- array(x, dim = c(N, 1))
  h <- matrix(1.0, nrow = 1, ncol = 1)

  K <- as.numeric(kernel_fn(xgrid, NULL, h, "epanechnikov"))

  expect_true(all(K >= 0))
})

test_that("Epanechnikov maximum is at origin", {
  kernel_fn <- get_kernel_fn()

  N <- 21
  x <- seq(-1, 1, length.out = N)
  xgrid <- array(x, dim = c(N, 1))
  h <- matrix(1.0, nrow = 1, ncol = 1)

  K <- as.numeric(kernel_fn(xgrid, NULL, h, "epanechnikov"))

  max_idx <- which.max(K)
  expect_equal(max_idx, (N + 1) / 2)  # Center index
})

test_that("Epanechnikov integrates to approximately 1", {
  # Matches: MATLAB test_epanechnikov_normalization_1d, Python test_epanechnikov_normalization_1d
  kernel_fn <- get_kernel_fn()

  # Unified: 1001 points, [-1.5, 1.5] range (Epanechnikov has compact support [-1, 1])
  N <- 1001
  xmin <- -1.5
  xmax <- 1.5
  x <- seq(xmin, xmax, length.out = N)
  dx <- x[2] - x[1]
  xgrid <- array(x, dim = c(N, 1))
  h <- matrix(1.0, nrow = 1, ncol = 1)

  K <- as.numeric(kernel_fn(xgrid, NULL, h, "epanechnikov"))

  # Trapezoidal integration
  integral <- dx * (sum(K) - 0.5 * K[1] - 0.5 * K[N])

  expect_equal(integral, 1.0, tolerance = TOL_NORMALIZATION)
})

# =============================================================================
# Section 6: Bandwidth Edge Cases
# =============================================================================

test_that("Very small bandwidth works without NaN", {
  kernel_fn <- get_kernel_fn()

  N <- 21
  x <- seq(-0.5, 0.5, length.out = N)
  xgrid <- array(x, dim = c(N, 1))
  h <- matrix(0.01, nrow = 1, ncol = 1)

  K <- as.numeric(kernel_fn(xgrid, NULL, h, "gaussian"))

  expect_false(any(is.nan(K)))
  expect_false(any(is.na(K)))
  expect_true(all(K >= 0))
})

test_that("Very large bandwidth produces flat kernel", {
  kernel_fn <- get_kernel_fn()

  N <- 21
  x <- seq(-1, 1, length.out = N)
  xgrid <- array(x, dim = c(N, 1))
  h <- matrix(10.0, nrow = 1, ncol = 1)

  K <- as.numeric(kernel_fn(xgrid, NULL, h, "gaussian"))

  # Large bandwidth = nearly flat
  cv <- sd(K) / mean(K)
  expect_lt(cv, 0.1)
})

test_that("Bandwidth scaling affects kernel width", {
  kernel_fn <- get_kernel_fn()

  N <- 101
  x <- seq(-5, 5, length.out = N)
  xgrid <- array(x, dim = c(N, 1))

  h_small <- matrix(0.5, nrow = 1, ncol = 1)
  h_large <- matrix(2.0, nrow = 1, ncol = 1)

  K_small <- as.numeric(kernel_fn(xgrid, NULL, h_small, "gaussian"))
  K_large <- as.numeric(kernel_fn(xgrid, NULL, h_large, "gaussian"))

  # Smaller bandwidth = higher peak at origin
  center_idx <- (N + 1) / 2
  expect_gt(K_small[center_idx], K_large[center_idx])
})

# =============================================================================
# Section 7: Multi-dimensional Tests
# =============================================================================

test_that("2D anisotropic bandwidth works", {
  kernel_fn <- get_kernel_fn()

  N <- 21
  vals <- seq(-2, 2, length.out = N)
  xgrid <- array(0, dim = c(N, N, 2))
  for (j in 1:N) {
    for (i in 1:N) {
      xgrid[i, j, 1] <- vals[i]
      xgrid[i, j, 2] <- vals[j]
    }
  }

  # Anisotropic: h1 < h2
  h <- matrix(c(0.3, 1.0), nrow = 1, ncol = 2)

  K <- kernel_fn(xgrid, NULL, h, "gaussian")[,,1]

  # Should vary more along dim 1 (smaller bandwidth)
  var_along_x1 <- mean(apply(K, 2, var))
  var_along_x2 <- mean(apply(K, 1, var))
  expect_gt(var_along_x1, var_along_x2)
})

test_that("3D kernel produces finite values", {
  kernel_fn <- get_kernel_fn()

  N <- 21  # Unified: 21³ = 9261 points (matches Python)
  vals <- seq(-1, 1, length.out = N)
  xgrid <- array(0, dim = c(N, N, N, 3))
  for (k in 1:N) {
    for (j in 1:N) {
      for (i in 1:N) {
        xgrid[i, j, k, 1] <- vals[i]
        xgrid[i, j, k, 2] <- vals[j]
        xgrid[i, j, k, 3] <- vals[k]
      }
    }
  }

  h <- matrix(c(0.5, 0.5, 0.5), nrow = 1, ncol = 3)

  K <- kernel_fn(xgrid, NULL, h, "gaussian")

  expect_true(all(is.finite(K)))
  expect_true(all(K >= 0))
})

# =============================================================================
# Section 8: Error Handling
# =============================================================================

test_that("Invalid kernel type raises error", {
  kernel_fn <- get_kernel_fn()

  xgrid <- array(0, dim = c(1, 1))
  h <- matrix(1.0, nrow = 1, ncol = 1)

  expect_error(kernel_fn(xgrid, NULL, h, "invalid_kernel"))
})

test_that("Polynomial kernel without order raises error", {
  # Matches: MATLAB test_polynomial_without_order, Python test_polynomial_without_order
  kernel_fn <- get_kernel_fn()

  xgrid <- array(c(0, 1, 2), dim = c(3, 1))
  h <- matrix(1.0, nrow = 1, ncol = 1)

  # Polynomial kernel requires 'order' parameter in opt
  expect_error(kernel_fn(xgrid, NULL, h, "polynomial"))
})

# =============================================================================
# Section 9: Determinism
# =============================================================================

test_that("Kernel output is deterministic", {
  kernel_fn <- get_kernel_fn()

  N <- 21
  x <- seq(-2, 2, length.out = N)
  xgrid <- array(x, dim = c(N, 1))
  h <- matrix(0.5, nrow = 1, ncol = 1)

  K1 <- kernel_fn(xgrid, NULL, h, "gaussian")
  K2 <- kernel_fn(xgrid, NULL, h, "gaussian")

  expect_equal(K1, K2, tolerance = TOL_STRICT)
})
