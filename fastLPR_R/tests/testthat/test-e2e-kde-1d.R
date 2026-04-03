# Test suite for 1D Kernel Density Estimation
context("1D Kernel Density Estimation")

test_that("1D KDE basic functionality works", {
  # Unified with MATLAB: n=500, h=[0.05,1.0], nh=10
  set.seed(42)
  x <- matrix(rnorm(500), ncol = 1)
  hlist <- get_hlist(10, c(0.05, 1.0))

  kde <- cv_fastkde(x, hlist, list(verbose = FALSE))

  # Check output structure
  validate_kde_structure(kde)
  expect_equal(length(kde$h), 1)
  expect_true(kde$h > 0)
})

# ARCHIVED: 2026-01-09 - "1D KDE with single bandwidth works" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-kde-1d.R)


test_that("1D KDE bandwidth selection works", {
  # Unified with MATLAB: n=500, h=[0.05,1.0], nh=20
  set.seed(42)
  x <- matrix(rnorm(500), ncol = 1)
  hlist <- get_hlist(20, c(0.05, 1.0))

  kde <- cv_fastkde(x, hlist, list(verbose = FALSE))

  # Selected bandwidth should be in range
  expect_gte(kde$h, min(hlist))
  expect_lte(kde$h, max(hlist))

  # Should have LCV results
  expect_true(!is.null(kde$lcv))
  expect_true(!is.null(kde$lcv$lcv_m))
  expect_equal(NROW(kde$lcv$lcv_m), length(hlist))
})

test_that("1D KDE density integrates to 1", {
  # Unified with MATLAB: n=500, h=[0.2,0.5], nh=5, N=200
  set.seed(42)
  x <- matrix(rnorm(500), ncol = 1)
  hlist <- get_hlist(5, c(0.2, 0.5))

  kde <- cv_fastkde(x, hlist, list(verbose = FALSE, N = 200))

  # Check integral
  expect_true(check_density_integral(kde, tolerance = 0.1))
})

test_that("1D KDE handles uniform distribution", {
  # Unified with MATLAB: n=200, h=[0.05,0.2], nh=5
  set.seed(42)
  x <- matrix(runif(200, 0, 1), ncol = 1)
  hlist <- get_hlist(5, c(0.05, 0.2))

  kde <- cv_fastkde(x, hlist, list(verbose = FALSE))

  # Density should be roughly uniform (close to 1)
  fhat_mean <- mean(kde$fhat)
  expect_true(abs(fhat_mean - 1.0) < 0.3)  # Uniform density should be close to 1
})

# ARCHIVED: 2026-01-09 - "1D KDE handles small sample size" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-kde-1d.R)


# ARCHIVED: 2026-01-09 - "1D KDE with Epanechnikov kernel works" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-kde-1d.R)


# ARCHIVED: 2026-01-09 - "1D KDE reproducibility check" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-kde-1d.R)


# ARCHIVED: 2026-01-09 - "1D KDE handles negative values" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-kde-1d.R)


# ARCHIVED: 2026-01-09 - "1D KDE with custom grid size works" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-kde-1d.R)


# =============================================================================
# Tests restored from unification spec (2026-01-10)
# =============================================================================

test_that("1D KDE reproducibility check", {
  # Unified with MATLAB: n=200, h=[0.1,0.5], nh=5
  set.seed(42)
  x <- matrix(rnorm(200), ncol = 1)
  hlist <- get_hlist(5, c(0.1, 0.5))

  kde1 <- cv_fastkde(x, hlist, list(verbose = FALSE))
  kde2 <- cv_fastkde(x, hlist, list(verbose = FALSE))

  # Results should be identical
  expect_equal(kde1$fhat, kde2$fhat)
  expect_equal(kde1$h, kde2$h)
})

test_that("1D KDE bimodal distribution", {
  # Unified with MATLAB: n=400, h=[0.1,1.5], nh=10
  set.seed(42)
  n <- 400
  x1 <- matrix(rnorm(n / 2) - 2, ncol = 1)
  x2 <- matrix(rnorm(n / 2) + 2, ncol = 1)
  x <- rbind(x1, x2)

  hlist <- get_hlist(10, c(0.1, 1.5))
  kde <- cv_fastkde(x, hlist, list(verbose = FALSE))

  # Check output structure
  validate_kde_structure(kde)
  expect_true(all(kde$fhat >= 0))
})

test_that("1D KDE with single bandwidth works", {
  # Unified with MATLAB: n=200, h=0.5
  set.seed(42)
  n <- 200
  x <- matrix(rnorm(n), ncol = 1)

  h <- matrix(0.5, nrow = 1, ncol = 1)
  kde <- cv_fastkde(x, h, list(verbose = FALSE))

  # Check output structure
  validate_kde_structure(kde)
  expect_equal(length(kde$fhat), n)
})

test_that("1D KDE density is positive", {
  # Unified with MATLAB: n=500, h=0.2
  set.seed(42)
  n <- 500
  x <- matrix(rnorm(n), ncol = 1)

  h <- matrix(0.2, nrow = 1, ncol = 1)
  kde <- cv_fastkde(x, h, list(verbose = FALSE))

  # Density should be strictly positive
  expect_true(all(kde$fhat > 0))
})

# ARCHIVED: 2026-01-09
# Source: fastLPR_R/tests/testthat/test-e2e-kde-1d.R
# Archive: dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-kde-1d.R
# Archived tests:
# - "1D KDE handles small sample size"
# - "1D KDE with Epanechnikov kernel works"
# - "1D KDE handles negative values"
# - "1D KDE with custom grid size works"
