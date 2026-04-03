# Helper utilities for fastLPR test suite
# Shared functions for setting up tests and validating results

# -----------------------------------------------------------------------------
# Load fastlpr package or source files
# -----------------------------------------------------------------------------
# Strategy:
#   1. Try installed package first (proper Rcpp symbol registration)
#   2. Fall back to source mode if package not installed
#      Note: Rcpp acceleration unavailable in source mode

.fastlpr_loaded <- FALSE

# Method 1: Use installed package (preferred - full Rcpp support)
if (requireNamespace("fastlpr", quietly = TRUE)) {
  tryCatch({
    library(fastlpr)
    .fastlpr_loaded <- TRUE
  }, error = function(e) NULL)
}

# Method 2: Fall back to sourcing R files (no Rcpp acceleration)
if (!.fastlpr_loaded) {
  pkg_root <- getwd()
  while (!grepl("fastLPR_R$", basename(pkg_root)) && dirname(pkg_root) != pkg_root) {
    pkg_root <- dirname(pkg_root)
  }
  if (grepl("fastLPR_R$", basename(pkg_root))) {
    r_dir <- file.path(pkg_root, "R")
    if (dir.exists(r_dir)) {
      r_files <- list.files(r_dir, pattern = "\\.R$", full.names = TRUE)
      for (f in r_files) {
        source(f, local = FALSE)
      }
    }
    # Note: Rcpp DLL not loaded in source mode - symbol registration not possible
    # Tests will use pure R fallback (slower but still correct)
  }
}
rm(.fastlpr_loaded)

# -----------------------------------------------------------------------------
# Global Tolerance Constants for Cross-Language Tests (Phase 2.2)
# -----------------------------------------------------------------------------
# These tolerances are used across all cross-language (xl) tests to ensure
# consistent error thresholds. See CLAUDE.md for pass criteria.

#' Strict tolerance for exact numerical matching (machine precision)
#' Use for: Pure R internal consistency tests
TOL_STRICT <- 1e-12

#' Numerical tolerance for algorithm comparisons
#' Use for: R-to-R comparisons, self-consistency tests
TOL_NUMERICAL <- 1e-6

#' Cross-language tolerance for R vs MATLAB comparisons
#' Use for: test-xl-*.R files, cross-validation tests
#' Per CLAUDE.md: BW MaxErr < 0.02, Mean MaxErr < 0.05
TOL_CROSSLANG <- list(
  bw_maxerr = 0.02,        # Bandwidth selection threshold
  mean_maxerr = 0.05,      # Mean estimate max error
  var_maxerr = 0.05,       # Variance estimate max error
  lcv_relerr = 0.01,       # LCV relative error (1%)
  gcv_relerr = 0.05,       # GCV relative error (5%)
  speed_ratio = 8.0        # Max allowed speed ratio vs MATLAB
)

#' NUFFT accuracy level tolerances (expected max error for each accuracy level)
ACCURACY_TOLERANCES <- list(
  `4` = 1e-3,
  `6` = 1e-5,
  `8` = 1e-7,
  `9` = 1e-9,
  `12` = 1e-11
)

# -----------------------------------------------------------------------------
# Skip Functions for CRAN Compatibility (Phase 2.1)
# -----------------------------------------------------------------------------

#' Skip test if MATLAB reference data directory is not available
#'
#' This function checks if the MATLAB reference data files exist and
#' skips the test if they are not available (e.g., on CRAN where
#' reference .mat files are not included in the package).
#'
#' @param ref_subdir Subdirectory under fastLPR/tests/refs/ (default: "crosslang_e2e")
#' @return NULL invisibly if refs available, otherwise skips the test
#'
#' @examples
#' test_that("XL test", {
#'   skip_if_no_matlab_refs()
#'   # ... test code that requires MATLAB reference data
#' })
skip_if_no_matlab_refs <- function(ref_subdir = "crosslang_e2e") {
  # Check if R.matlab package is available
  if (!requireNamespace("R.matlab", quietly = TRUE)) {
    skip("R.matlab package not available - required for cross-language tests")
  }

  # Construct reference directory path
  ref_dir <- tryCatch({
    file.path(find_root(), "fastLPR", "tests", "refs", ref_subdir)
  }, error = function(e) {
    skip(paste("Cannot locate reference directory:", e$message))
    return(NULL)
  })

  # Check if directory exists
  if (is.null(ref_dir) || !dir.exists(ref_dir)) {
    skip(paste("MATLAB reference directory not found:", ref_subdir,
               "- skipping cross-language test (expected on CRAN)"))
  }

  # Check if at least one .mat file exists
  mat_files <- tryCatch({
    list.files(ref_dir, pattern = "\\.mat$", full.names = FALSE)
  }, error = function(e) {
    character(0)
  })

  if (length(mat_files) == 0) {
    skip(paste("No .mat reference files found in:", ref_subdir,
               "- skipping cross-language test"))
  }

  invisible(NULL)
}

#' Skip test if running on CRAN (NOT_CRAN != "true")
#'
#' Convenience wrapper that combines testthat::skip_on_cran() with
#' custom messaging for cross-language tests.
#'
#' @return NULL invisibly if not on CRAN, otherwise skips
skip_xl_on_cran <- function() {
  # Check NOT_CRAN environment variable
  not_cran <- Sys.getenv("NOT_CRAN", unset = "")
  if (tolower(not_cran) != "true") {
    skip("Skipping cross-language test on CRAN (set NOT_CRAN=true to run)")
  }
  invisible(NULL)
}

#' Load MATLAB reference data from .mat files
#'
#' @param filename Path to .mat file (relative to project root)
#' @return List containing MATLAB data
load_matlab_reference <- function(filename) {
  if (!requireNamespace("R.matlab", quietly = TRUE)) {
    skip("R.matlab package not available")
  }

  full_path <- file.path(find_root(), filename)
  if (!file.exists(full_path)) {
    skip(paste("Reference data not found:", filename))
  }

  R.matlab::readMat(full_path)
}

#' Find project root directory
#'
#' @return Path to project root
find_root <- function() {
  # Start from current working directory
  path <- getwd()

  # Look for fastLPR_R directory
  while (!grepl("fastLPR_R$", path) && dirname(path) != path) {
    path <- dirname(path)
  }

  # Navigate to jss-code root
  if (grepl("fastLPR_R", path)) {
    return(dirname(path))
  }

  # Fallback: assume we're already in tests/testthat
  return(file.path(dirname(dirname(dirname(getwd())))))
}

#' Compute error metrics between R and reference values
#'
#' @param r_val R implementation output
#' @param ref_val Reference (MATLAB/Python) output
#' @return List with error metrics
compute_error_metrics <- function(r_val, ref_val) {
  r_vec <- as.vector(r_val)
  ref_vec <- as.vector(ref_val)

  abs_err <- abs(r_vec - ref_vec)
  max_abs_err <- max(abs_err, na.rm = TRUE)
  mean_abs_err <- mean(abs_err, na.rm = TRUE)

  ref_mean <- mean(abs(ref_vec), na.rm = TRUE)
  max_err_pct <- 100 * max_abs_err / (ref_mean + 1e-10)

  list(
    max_abs_err = max_abs_err,
    mean_abs_err = mean_abs_err,
    max_err_pct = max_err_pct,
    ref_mean = ref_mean,
    rmse = sqrt(mean((r_vec - ref_vec)^2, na.rm = TRUE))
  )
}

#' Check if values match within tolerance
#'
#' @param r_val R implementation output
#' @param ref_val Reference output
#' @param tolerance Maximum allowed percentage error (default: 1.0%)
#' @return TRUE if values match within tolerance
values_match <- function(r_val, ref_val, tolerance = 1.0) {
  metrics <- compute_error_metrics(r_val, ref_val)
  metrics$max_err_pct <= tolerance
}

#' Generate synthetic test data
#'
#' @param n Number of samples
#' @param d Number of dimensions
#' @param fun Function to generate true values (default: sin)
#' @param noise_sd Standard deviation of noise (default: 0.1)
#' @return List with x, y_true, y_noisy
generate_test_data <- function(n = 100, d = 1, fun = NULL, noise_sd = 0.1) {
  set.seed(42)  # For reproducibility

  if (d == 1) {
    x <- matrix(runif(n, 0, 2*pi), ncol = 1)
    if (is.null(fun)) fun <- sin
    y_true <- fun(x)
  } else if (d == 2) {
    x <- matrix(runif(n * 2, -1, 1), ncol = 2)
    if (is.null(fun)) {
      fun <- function(x) sin(pi * x[, 1]) * cos(pi * x[, 2])
    }
    y_true <- fun(x)
  } else if (d == 3) {
    x <- matrix(runif(n * 3, -1, 1), ncol = 3)
    if (is.null(fun)) {
      fun <- function(x) sin(pi * x[, 1]) * cos(pi * x[, 2]) * sin(pi * x[, 3])
    }
    y_true <- fun(x)
  } else {
    stop("Only d = 1, 2, 3 supported")
  }

  y_noisy <- matrix(y_true + rnorm(n, 0, noise_sd), ncol = 1)

  list(
    x = x,
    y_true = y_true,
    y = y_noisy,
    n = n,
    d = d
  )
}

#' Validate structure of KDE result
#'
#' @param kde KDE result object
#' @param expected_fields Expected field names
#' @return TRUE if valid, otherwise throws error
validate_kde_structure <- function(kde, expected_fields = c("fhat", "h", "xlist")) {
  expect_true(is.list(kde))

  for (field in expected_fields) {
    expect_true(field %in% names(kde),
                info = paste("Missing field:", field))
  }

  expect_true(!is.null(kde$fhat))
  expect_true(!is.null(kde$h))
  expect_true(length(kde$h) >= 1)

  TRUE
}

#' Validate structure of regression result
#'
#' @param regs Regression result object
#' @param expected_fields Expected field names
#' @return TRUE if valid, otherwise throws error
validate_regression_structure <- function(regs, expected_fields = c("yhat", "gcv_yhat")) {
  expect_true(is.list(regs))

  for (field in expected_fields) {
    expect_true(field %in% names(regs),
                info = paste("Missing field:", field))
  }

  expect_true(!is.null(regs$yhat))

  if ("gcv_yhat" %in% names(regs)) {
    expect_true(!is.null(regs$gcv_yhat$gcv_m))
    expect_true(!is.null(regs$gcv_yhat$mse))
  }

  TRUE
}

#' Check if density integrates to approximately 1
#'
#' @param kde KDE result with fhat and xlist
#' @param tolerance Tolerance for integration (default: 0.1)
#' @return TRUE if integral is close to 1
check_density_integral <- function(kde, tolerance = 0.1) {
  d <- length(kde$xlist)

  if (d == 1) {
    grid <- kde$xlist[[1]]
    dx <- diff(grid)[1]
    integral <- sum(kde$fhat) * dx
  } else if (d == 2) {
    dx1 <- diff(kde$xlist[[1]])[1]
    dx2 <- diff(kde$xlist[[2]])[1]
    integral <- sum(kde$fhat) * dx1 * dx2
  } else if (d == 3) {
    dx1 <- diff(kde$xlist[[1]])[1]
    dx2 <- diff(kde$xlist[[2]])[1]
    dx3 <- diff(kde$xlist[[3]])[1]
    integral <- sum(kde$fhat) * dx1 * dx2 * dx3
  } else {
    stop("Only 1D, 2D, 3D supported")
  }

  abs(integral - 1.0) < tolerance
}
