# Test suite for 2D Kernel Density Estimation
context("2D Kernel Density Estimation")

test_that("2D KDE basic functionality works", {
  # Unified with MATLAB: n=500, h=[0.1,1.0], nh=10x10
  set.seed(42)
  n <- 500
  x <- matrix(rnorm(n * 2), ncol = 2)
  hlist <- get_hlist(c(10, 10), list(c(0.1, 1.0), c(0.1, 1.0)))

  kde <- cv_fastkde(x, hlist, list(verbose = FALSE))

  # Check output structure
  validate_kde_structure(kde)
  expect_equal(length(kde$h), 2)
  expect_equal(length(kde$xlist), 2)
})

# ARCHIVED: 2026-01-09 - "2D KDE with single bandwidth works" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-kde-2d.R)


test_that("2D KDE bandwidth selection works", {
  # Unified with MATLAB: n=200 (mixture), h=[0.1,1.0], nh=5x5
  set.seed(42)
  # Mixture of Gaussians
  n1 <- 100
  n2 <- 100
  x1 <- cbind(rnorm(n1, -1, 1), rnorm(n1, -1, 1)) * 0.3
  x2 <- cbind(rnorm(n2, 1, 1), rnorm(n2, 1, 1)) * 0.5
  x <- rbind(x1, x2)

  hlist <- get_hlist(c(5, 5), list(c(0.1, 1.0), c(0.1, 1.0)))

  kde <- cv_fastkde(x, hlist, list(verbose = FALSE))

  # Check bandwidth selection
  expect_true(all(kde$h >= 0.1))
  expect_true(all(kde$h <= 1.0))
  expect_true(!is.null(kde$lcv))
})

test_that("2D KDE density integrates to 1", {
  # Unified with MATLAB: n=500, h=[0.3,0.6], nh=3x3, N=[50,50]
  set.seed(42)
  x <- matrix(rnorm(500 * 2), ncol = 2)
  hlist <- get_hlist(c(3, 3), list(c(0.3, 0.6), c(0.3, 0.6)))

  kde <- cv_fastkde(x, hlist, list(verbose = FALSE, N = c(50, 50)))

  # Check integral
  expect_true(check_density_integral(kde, tolerance = 0.15))
})

# ARCHIVED: 2026-01-09 - "2D KDE handles anisotropic data" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-kde-2d.R)


# ARCHIVED: 2026-01-09 - "2D KDE with small sample size works" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-kde-2d.R)


# ARCHIVED: 2026-01-09 - "2D KDE reproducibility check" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-kde-2d.R)


# ARCHIVED: 2026-01-09 - "2D KDE handles correlated data" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-kde-2d.R)


# ARCHIVED: 2026-01-09 - "2D KDE with different grid sizes per dimension works" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-kde-2d.R)


# =============================================================================
# Tests restored from unification spec (2026-01-10)
# =============================================================================

test_that("2D KDE anisotropic bandwidth", {
  # Unified with MATLAB: n=500, h=[[0.1,0.5],[0.2,1.0]], nh=10x10
  set.seed(42)
  n <- 500
  # Anisotropic data: different scales in each dimension
  x1 <- rnorm(n, 0, 0.5)
  x2 <- rnorm(n, 0, 2)
  x <- cbind(x1, x2)

  hlist <- get_hlist(c(10, 10), list(c(0.1, 0.5), c(0.2, 1.0)))
  kde <- cv_fastkde(x, hlist, list(verbose = FALSE))

  # Check output structure
  validate_kde_structure(kde)
  expect_equal(length(kde$h), 2)
})

test_that("2D KDE correlated data", {
  # Unified with MATLAB: n=300, h=[0.1,1.0], nh=5x5
  set.seed(42)
  n <- 300
  # Correlated 2D data
  x1 <- rnorm(n)
  x2 <- 0.8 * x1 + 0.6 * rnorm(n)
  x <- cbind(x1, x2)

  hlist <- get_hlist(c(5, 5), list(c(0.1, 1.0), c(0.1, 1.0)))
  kde <- cv_fastkde(x, hlist, list(verbose = FALSE))

  # Check output structure
  validate_kde_structure(kde)
  expect_true(all(kde$fhat >= -1e-9))
})

test_that("2D KDE with single bandwidth works", {
  # Unified with MATLAB: n=200, h=[0.5,0.5]
  set.seed(42)
  n <- 200
  x <- matrix(rnorm(n * 2), ncol = 2)

  h <- matrix(c(0.5, 0.5), nrow = 1, ncol = 2)
  kde <- cv_fastkde(x, h, list(verbose = FALSE))

  # Check output structure
  validate_kde_structure(kde)
  expect_equal(length(kde$xlist), 2)
})

test_that("2D KDE density is non-negative", {
  # Unified with MATLAB: n=300, h=[0.3,0.3]
  set.seed(42)
  n <- 300
  x <- matrix(rnorm(n * 2), ncol = 2)

  h <- matrix(c(0.3, 0.3), nrow = 1, ncol = 2)
  kde <- cv_fastkde(x, h, list(verbose = FALSE))

  # Allow small negative values due to numerical precision (NUFFT grid artifacts)
  expect_true(all(kde$fhat > -1e-9))
})

# ARCHIVED: 2026-01-09
# Source: fastLPR_R/tests/testthat/test-e2e-kde-2d.R
# Archive: dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-kde-2d.R
# Archived tests:
# - "2D KDE with small sample size works"
# - "2D KDE reproducibility check"
# - "2D KDE with different grid sizes per dimension works"
