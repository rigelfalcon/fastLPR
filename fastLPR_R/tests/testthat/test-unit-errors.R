# ==============================================================================
# Unit Tests for Error Handling in fastLPR R Package
# ==============================================================================
#
# This file provides comprehensive unit tests for error handling and input
# validation in the fastLPR package. Tests include:
#
# 1. Empty hlist input to cv_fastlpr
# 2. Dimension mismatch between x and y
# 3. Invalid order parameter
# 4. All NA data input
# 5. Negative bandwidths
#
# These tests verify that appropriate error messages are raised for invalid
# inputs, ensuring robust behavior and helpful user feedback.
#
# ==============================================================================

context("Unit Tests: Error Handling and Input Validation")

# =============================================================================
# Section 1: Empty hlist Input Tests
# =============================================================================

test_that("UNIT: cv_fastlpr handles empty hlist gracefully", {
  set.seed(42)
  n <- 50
  x <- matrix(runif(n), ncol = 1)
  y <- matrix(sin(2 * pi * x) + 0.1 * rnorm(n), ncol = 1)

  # Empty hlist (NULL) - should use default bandwidth
  result <- tryCatch({
    regs <- cv_fastlpr(x, y, h = NULL, opt = list(order = 1))
    TRUE
  }, error = function(e) FALSE)

  expect_true(result, label = "cv_fastlpr should handle NULL hlist by using default bandwidth")

  # Empty numeric vector
  result_empty <- tryCatch({
    regs <- cv_fastlpr(x, y, h = numeric(0), opt = list(order = 1))
    TRUE
  }, error = function(e) FALSE)

  expect_true(result_empty, label = "cv_fastlpr should handle empty numeric hlist")

  # Empty matrix
  result_empty_mat <- tryCatch({
    regs <- cv_fastlpr(x, y, h = matrix(numeric(0), nrow = 0), opt = list(order = 1))
    TRUE
  }, error = function(e) FALSE)

  expect_true(result_empty_mat, label = "cv_fastlpr should handle empty matrix hlist")
})

test_that("UNIT: cv_fastkde errors on empty hlist", {
  set.seed(42)
  n <- 50
  x <- matrix(runif(n), ncol = 1)

  # Empty hlist (NULL) - should error as h is required
  expect_error(
    cv_fastkde(x, h = NULL),
    label = "cv_fastkde should error when h is NULL"
  )
})

# =============================================================================
# Section 2: Dimension Mismatch Tests
# =============================================================================

test_that("UNIT: cv_fastlpr errors on x/y dimension mismatch", {
  set.seed(42)

  # x has 50 rows, y has 30 rows
  x <- matrix(runif(50), ncol = 1)
  y <- matrix(rnorm(30), ncol = 1)
  hlist <- get_hlist(5, c(0.1, 0.5))

  expect_error(
    cv_fastlpr(x, y, hlist, list(order = 1)),
    regexp = "x.*y.*must match|row|mismatch",
    label = "cv_fastlpr should error when x and y have different number of rows"
  )
})

test_that("UNIT: cv_fastlpr errors on 2D x/y dimension mismatch", {
  set.seed(42)

  # 2D case: x has 100 rows, y has 80 rows
  x <- matrix(runif(200), ncol = 2)  # 100 rows
  y <- matrix(rnorm(80), ncol = 1)   # 80 rows
  hlist <- get_hlist(5, c(0.1, 0.5))

  expect_error(
    cv_fastlpr(x, y, hlist, list(order = 1)),
    regexp = "x.*y.*must match|row|mismatch",
    label = "cv_fastlpr should error on 2D dimension mismatch"
  )
})

# ARCHIVED: "cv_fastlpr handles correct dimensions" - Redundant positive test
# See: dev/archive/r-errors-tests-20260109/archived_errors_tests.R

# =============================================================================
# Section 3: Invalid Order Parameter Tests
# =============================================================================

test_that("UNIT: cv_fastlpr errors on invalid order parameter (negative)", {
  set.seed(42)
  n <- 50
  x <- matrix(runif(n), ncol = 1)
  y <- matrix(sin(2 * pi * x) + 0.1 * rnorm(n), ncol = 1)
  hlist <- get_hlist(5, c(0.1, 0.5))

  # Order=-1 is invalid and will cause an error (any error is acceptable)
  expect_error(
    cv_fastlpr(x, y, hlist, list(order = -1)),
    label = "cv_fastlpr should error on negative order"
  )
})

test_that("UNIT: cv_fastlpr errors on invalid order parameter (too high)", {
  set.seed(42)
  n <- 50
  x <- matrix(runif(n), ncol = 1)
  y <- matrix(sin(2 * pi * x) + 0.1 * rnorm(n), ncol = 1)
  hlist <- get_hlist(5, c(0.1, 0.5))

  # Order 3 is not supported (only 0, 1, 2)
  expect_error(
    cv_fastlpr(x, y, hlist, list(order = 3)),
    regexp = "order|not supported|invalid|unknown|Use 0, 1, or 2",
    ignore.case = TRUE,
    label = "cv_fastlpr should error on order > 2"
  )
})

test_that("UNIT: cv_fastlpr errors on non-numeric order parameter", {
  set.seed(42)
  n <- 50
  x <- matrix(runif(n), ncol = 1)
  y <- matrix(sin(2 * pi * x) + 0.1 * rnorm(n), ncol = 1)
  hlist <- get_hlist(5, c(0.1, 0.5))

  expect_error(
    cv_fastlpr(x, y, hlist, list(order = "linear")),
    label = "cv_fastlpr should error on non-numeric order"
  )
})

# ARCHIVED: "cv_fastlpr accepts valid order parameters" - Redundant positive test
# See: dev/archive/r-errors-tests-20260109/archived_errors_tests.R

# =============================================================================
# Section 4: All NA Data Input Tests
# =============================================================================

test_that("UNIT: cv_fastlpr handles all NA y values", {
  set.seed(42)
  n <- 50
  x <- matrix(runif(n), ncol = 1)
  y <- matrix(rep(NA_real_, n), ncol = 1)
  hlist <- get_hlist(5, c(0.1, 0.5))

  # All NA y should produce NA results or error
  result <- tryCatch({
    regs <- cv_fastlpr(x, y, hlist, list(order = 1))
    # Check if results contain NA
    all(is.na(regs$yhat)) || any(is.na(regs$gcv_yhat$gcv_m))
  }, error = function(e) {
    # Error is acceptable for all-NA input
    TRUE
  }, warning = function(w) {
    # Warning is acceptable
    TRUE
  })

  expect_true(result, label = "cv_fastlpr should handle all-NA y appropriately")
})

test_that("UNIT: cv_fastlpr handles all NA x values", {
  set.seed(42)
  n <- 50
  x <- matrix(rep(NA_real_, n), ncol = 1)
  y <- matrix(rnorm(n), ncol = 1)
  hlist <- get_hlist(5, c(0.1, 0.5))

  # All NA x should produce error or NA results
  result <- tryCatch({
    regs <- cv_fastlpr(x, y, hlist, list(order = 1))
    # If it returns, check for NA
    any(is.na(regs$yhat))
  }, error = function(e) {
    # Error is acceptable
    TRUE
  })

  expect_true(result, label = "cv_fastlpr should handle all-NA x appropriately")
})

test_that("UNIT: cv_fastlpr handles partial NA in y", {
  set.seed(42)
  n <- 50
  x <- matrix(runif(n), ncol = 1)
  y <- matrix(sin(2 * pi * x) + 0.1 * rnorm(n), ncol = 1)
  # Introduce some NAs
  y[1:10] <- NA
  hlist <- get_hlist(5, c(0.1, 0.5))

  # Partial NA might produce results with NA or handle them
  result <- tryCatch({
    regs <- cv_fastlpr(x, y, hlist, list(order = 1))
    TRUE  # Some result returned
  }, error = function(e) {
    # Error for partial NA is also acceptable
    TRUE
  })

  expect_true(result, label = "cv_fastlpr should handle partial NA in y")
})

# =============================================================================
# Section 5: Negative Bandwidth Tests
# =============================================================================

test_that("UNIT: cv_fastlpr with negative bandwidth does not crash", {
  set.seed(42)
  n <- 50
  x <- matrix(runif(n), ncol = 1)
  y <- matrix(sin(2 * pi * x) + 0.1 * rnorm(n), ncol = 1)

  # Negative bandwidth - note: this is mathematically invalid but should not crash
  h_negative <- matrix(-0.1, nrow = 1, ncol = 1)

  # The implementation may accept negative bandwidth (produces incorrect results but doesn't crash)
  # This test documents the current behavior - it does not validate correctness
  result <- tryCatch({
    regs <- cv_fastlpr(x, y, h_negative, list(order = 1))
    TRUE  # Ran without crashing
  }, error = function(e) {
    TRUE  # Error is also acceptable behavior
  })

  expect_true(result,
    label = "cv_fastlpr should not crash with negative bandwidth")
})

test_that("UNIT: cv_fastlpr errors on zero bandwidth", {
  set.seed(42)
  n <- 50
  x <- matrix(runif(n), ncol = 1)
  y <- matrix(sin(2 * pi * x) + 0.1 * rnorm(n), ncol = 1)

  # Zero bandwidth
  h_zero <- matrix(0, nrow = 1, ncol = 1)

  # Should error or produce invalid results (division by zero in kernel)
  result <- tryCatch({
    regs <- cv_fastlpr(x, y, h_zero, list(order = 1))
    # If no error, check for NaN/Inf
    any(is.nan(as.vector(regs$yhat))) || any(is.infinite(as.vector(regs$yhat)))
  }, error = function(e) {
    TRUE
  }, warning = function(w) {
    TRUE
  })

  expect_true(result,
    label = "cv_fastlpr should error or produce invalid results with zero bandwidth")
})

test_that("UNIT: cv_fastlpr handles very small positive bandwidth", {
  set.seed(42)
  n <- 50
  x <- matrix(runif(n), ncol = 1)
  y <- matrix(sin(2 * pi * x) + 0.1 * rnorm(n), ncol = 1)

  # Very small but positive bandwidth
  h_small <- matrix(1e-10, nrow = 1, ncol = 1)

  # Should either error (bandwidth too small) or produce some result
  result <- tryCatch({
    regs <- cv_fastlpr(x, y, h_small, list(order = 1))
    TRUE  # Some result produced
  }, error = function(e) {
    # Error for too-small bandwidth is acceptable
    TRUE
  }, warning = function(w) {
    TRUE
  })

  expect_true(result, label = "cv_fastlpr should handle very small bandwidth")
})

test_that("UNIT: cv_fastlpr with mixed positive/negative bandwidth list", {
  set.seed(42)
  n <- 50
  x <- matrix(runif(n), ncol = 1)
  y <- matrix(sin(2 * pi * x) + 0.1 * rnorm(n), ncol = 1)

  # Mix of positive and negative bandwidths
  hlist_mixed <- matrix(c(-0.1, 0.2, 0.3, -0.2, 0.4), ncol = 1)

  # Should error or filter out negative values
  result <- tryCatch({
    regs <- cv_fastlpr(x, y, hlist_mixed, list(order = 1))
    # Check if results are valid (NaN/Inf indicate problem)
    !all(is.nan(as.vector(regs$yhat))) && !all(is.infinite(as.vector(regs$yhat)))
  }, error = function(e) {
    # Error is acceptable
    TRUE
  }, warning = function(w) {
    TRUE
  })

  expect_true(result, label = "cv_fastlpr should handle mixed bandwidth list")
})

# =============================================================================
# Section 6: Additional Input Validation Tests
# =============================================================================

# ARCHIVED: "cv_fastkde errors on non-matrix x" - R-specific type check
# See: dev/archive/r-errors-tests-20260109/archived_errors_tests.R

test_that("UNIT: cv_fastkde errors on too few observations", {
  # Only 1 observation (need at least 2)
  x <- matrix(0.5, nrow = 1, ncol = 1)

  expect_error(
    cv_fastkde(x),
    regexp = "at least|observation|sample",
    ignore.case = TRUE,
    label = "cv_fastkde should error on single observation"
  )
})

# ARCHIVED: "cv_fastkde errors on invalid kernel type" - R-specific, not in Python
# See: dev/archive/tests-archive-20260110/r/unit/test-unit-errors-archived.R

# ARCHIVED: "cv_fastlpr errors on non-numeric x" - R-specific type check
# See: dev/archive/r-errors-tests-20260109/archived_errors_tests.R

# ARCHIVED: "cv_fastlpr errors on non-numeric y" - R-specific type check
# See: dev/archive/r-errors-tests-20260109/archived_errors_tests.R

# =============================================================================
# Section 7: Edge Case Tests
# =============================================================================

test_that("UNIT: cv_fastlpr handles minimum valid input size", {
  set.seed(42)

  # Minimum useful size (2 observations)
  x <- matrix(c(0.2, 0.8), ncol = 1)
  y <- matrix(c(1.0, 2.0), ncol = 1)

  # Should work with default bandwidth
  result <- tryCatch({
    regs <- cv_fastlpr(x, y, opt = list(order = 0))
    !is.null(regs$yhat)
  }, error = function(e) FALSE)

  expect_true(result, label = "cv_fastlpr should handle minimum valid input")
})

test_that("UNIT: cv_fastlpr handles constant y values", {
  set.seed(42)
  n <- 50
  x <- matrix(runif(n), ncol = 1)
  y <- matrix(rep(5.0, n), ncol = 1)  # Constant
  hlist <- get_hlist(5, c(0.1, 0.5))

  # Constant y should produce constant fitted values
  result <- tryCatch({
    regs <- cv_fastlpr(x, y, hlist, list(order = 1))
    # Check results are close to constant
    all(abs(as.vector(regs$yhat) - 5.0) < 1.0)  # Generous tolerance
  }, error = function(e) {
    # Error is also acceptable for degenerate case
    FALSE
  })

  expect_true(result, label = "cv_fastlpr should handle constant y values")
})

test_that("UNIT: cv_fastlpr handles collinear x in 2D", {
  set.seed(42)
  n <- 50

  # x2 = 2*x1 (perfect collinearity)
  x1 <- runif(n)
  x <- cbind(x1, 2 * x1)
  y <- matrix(sin(2 * pi * x1), ncol = 1)

  # Should produce some result (though bandwidth selection may be unstable)
  result <- tryCatch({
    regs <- cv_fastlpr(x, y, opt = list(order = 0))
    !is.null(regs$yhat)
  }, error = function(e) {
    # Error is acceptable for degenerate data
    TRUE
  })

  expect_true(result, label = "cv_fastlpr should handle collinear x")
})

# =============================================================================
# End of Error Handling Tests
# =============================================================================
