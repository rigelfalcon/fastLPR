# Test suite for 3D Local Polynomial Regression
context("3D Local Polynomial Regression")

# =============================================================================
# Tests restored from unification spec (2026-01-10)
# =============================================================================

test_that("3D order 1 regression works", {
  # 3D Local Linear - single bandwidth, no GCV search
  set.seed(42)
  n <- 300
  x <- matrix(runif(n * 3, -1, 1), ncol = 3)
  y_true <- sin(pi * x[, 1]) * cos(pi * x[, 2]) * sin(pi * x[, 3])
  y <- matrix(y_true + 0.1 * rnorm(n), ncol = 1)

  h <- matrix(c(0.4, 0.4, 0.4), nrow = 1)  # Single bandwidth
  opt <- list(order = 1, N = c(20, 20, 20), verbose = FALSE, calc_dof = FALSE)

  regs <- cv_fastlpr(x, y, h, opt)

  expect_true(!is.null(regs$yhat))
  expect_equal(length(regs$yhat), n)

  # Correlation should be reasonable
  correlation <- cor(as.vector(regs$yhat), as.vector(y_true))
  expect_gt(correlation, 0.5)
})

# ARCHIVED: 2026-01-10 - "3D order 0 regression works" (moved to dev/archive/tests-archive-20260110/r/e2e/archived_test-e2e-lpr-3d.R)

# ARCHIVED: 2026-01-10 - "3D order 2 regression works for small datasets" (moved to dev/archive/tests-archive-20260110/r/e2e/archived_test-e2e-lpr-3d.R)

# ARCHIVED: 2026-01-09 - "3D regression with single bandwidth works" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-lpr-3d.R)

# ARCHIVED: 2026-01-10 - "3D regression GCV bandwidth selection works" (moved to dev/archive/tests-archive-20260110/r/e2e/archived_test-e2e-lpr-3d.R)

# ARCHIVED: 2026-01-09 - "3D regression handles linear function correctly" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-lpr-3d.R)


# ARCHIVED: 2026-01-09 - "3D regression handles anisotropic data" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-lpr-3d.R)


# ARCHIVED: 2026-01-09 - "3D regression with different grid sizes works" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-lpr-3d.R)


# ARCHIVED: 2026-01-09 - "3D regression reproducibility check" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-lpr-3d.R)


# ARCHIVED: 2026-01-09 - "3D regression handles small sample size" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-lpr-3d.R)


# ARCHIVED: 2026-01-09 - "3D regression with Epanechnikov kernel works" (moved to dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-lpr-3d.R)


# ARCHIVED: 2026-01-09
# Source: fastLPR_R/tests/testthat/test-e2e-lpr-3d.R
# Archive: dev/archive/tests-archive-20260109/r/e2e/archived_test-e2e-lpr-3d.R
# Archived tests:
# - "3D regression with single bandwidth works"
# - "3D regression handles linear function correctly"
# - "3D regression handles anisotropic data"
# - "3D regression with different grid sizes works"
# - "3D regression reproducibility check"
# - "3D regression handles small sample size"
# - "3D regression with Epanechnikov kernel works"
# - "3D order 0 regression works" (2026-01-10)
# - "3D order 1 regression works" (2026-01-10)
# - "3D regression GCV bandwidth selection works" (2026-01-10)
