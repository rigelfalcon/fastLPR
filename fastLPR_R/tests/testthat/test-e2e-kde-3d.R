# Test suite for 3D Kernel Density Estimation
context("3D Kernel Density Estimation")

test_that("3D KDE basic functionality works", {
  # 3D KDE - single bandwidth, no LCV search
  set.seed(42)
  n <- 150
  x <- matrix(rnorm(n * 3), ncol = 3)
  h <- matrix(c(0.5, 0.5, 0.5), nrow = 1)  # Single bandwidth

  kde <- cv_fastkde(x, h, list(verbose = FALSE, N = c(30, 30, 30)))

  # Check output structure
  validate_kde_structure(kde)
  expect_equal(length(kde$h), 3)
  expect_equal(length(kde$xlist), 3)
  # R arrays may have trailing dimensions (30,30,30,1,1), check first 3
  expect_equal(dim(kde$fhat)[1:3], c(30, 30, 30))
})

# ARCHIVED: 2026-01-09 - "3D KDE with single bandwidth works" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-kde-3d.R)

# ARCHIVED: 2026-01-10 - "3D KDE bandwidth selection works" (moved to dev/archive/tests-archive-20260110/r/e2e/archived_test-e2e-kde-3d.R)

# ARCHIVED: 2026-01-09 - "3D KDE density integrates to 1" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-kde-3d.R)


# ARCHIVED: 2026-01-09 - "3D KDE handles anisotropic data" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-kde-3d.R)


# ARCHIVED: 2026-01-09 - "3D KDE with small sample size works" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-kde-3d.R)


# ARCHIVED: 2026-01-09 - "3D KDE reproducibility check" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-kde-3d.R)


# ARCHIVED: 2026-01-09 - "3D KDE handles mixture distribution" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-kde-3d.R)


# ARCHIVED: 2026-01-09 - "3D KDE with different grid sizes works" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-kde-3d.R)


# ARCHIVED: 2026-01-09
# Source: fastLPR_R/tests/testthat/test-e2e-kde-3d.R
# Archive: dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-kde-3d.R
# Archived tests:
# - "3D KDE with single bandwidth works"
# - "3D KDE density integrates to 1"
# - "3D KDE handles anisotropic data"
# - "3D KDE with small sample size works"
# - "3D KDE reproducibility check"
# - "3D KDE handles mixture distribution"
# - "3D KDE bandwidth selection works" (2026-01-10)
