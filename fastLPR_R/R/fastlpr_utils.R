# Copyright (c) 2024-2025 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later

#' Column-wise complex indicator mirroring MATLAB iscomplex.m
#'
#' fastLPR's MATLAB implementation splits complex predictor columns into
#' real and imaginary parts using `iscomplex(x, 1)`, which returns TRUE
#' when *all* entries in a column have a non-zero imaginary part. This
#' helper replicates that behaviour.
#'
#' @param x Numeric matrix or vector.
#' @param dim Dimension specifier. Use `1` for column-wise checks,
#'   `2` for row-wise checks, `"all"` (default) to test the entire array.
#'
#' @return Logical vector/scalar indicating which slices are complex.
#' @keywords internal
fastlpr_is_complex <- function(x, dim = "all") {
  if (!is.numeric(x) && !is.complex(x)) {
    stop("fastlpr_is_complex: input must be numeric or complex.")
  }

  imag_nonzero <- Im(x) != 0

  if (identical(dim, "all")) {
    # FIX: For "all" mode, return TRUE if ANY element has imaginary part
    # This is used to detect if y is complex-valued for regression
    return(any(imag_nonzero))
  }

  # Handle vector case
  if (is.null(dim(imag_nonzero))) {
    # For vectors, return single value - use any() for complex detection
    return(any(imag_nonzero))
  }

  if (dim == 1) {
    return(apply(imag_nonzero, 2, all))
  }

  if (dim == 2) {
    return(apply(imag_nonzero, 1, all))
  }

  stop("fastlpr_is_complex: unsupported dim argument.")
}

#' Precision conversion shim for MATLAB compatibility
#'
#' MATLAB's `bit_convert` conditionally casts arrays to single or double
#' precision depending on the requested accuracy. R stores numeric values
#' as double precision by default and lacks an exact single-precision
#' analogue, so this helper simply returns the inputs unchanged while
#' preserving the call signature.
#'
#' @param accuracy Numeric accuracy (ignored) or character scalar
#'   `'single'`/`'double'`.
#' @param ... Arrays to "convert".
#'
#' @return List of inputs (unchanged).
#' @keywords internal
fastlpr_bit_convert <- function(accuracy, ...) {
  args <- list(...)
  # No-op: R's numeric type is double; retain for interface parity.
  return(args)
}

#' Compute NUFFT grid parameters (port of compute_grid_params.m)
#'
#' @param N Grid size (scalar or numeric vector).
#' @param accuracy Desired decimal-digit accuracy (default 8).
#'
#' @return List with components `Msp`, `Mr`, `tau`, and `ratio`.
#' @keywords internal
fastlpr_compute_grid_params <- function(N, accuracy = 8) {
  if (length(accuracy) != 1L || !is.numeric(accuracy)) {
    stop("fastlpr_compute_grid_params: accuracy must be a numeric scalar.")
  }
  N <- as.numeric(N)

  if (accuracy <= 6) {
    ratio <- 1
  } else if (accuracy <= 11) {
    ratio <- 2
  } else {
    ratio <- 3
  }

  Msp <- rep(accuracy, length(N))
  tau <- (pi * Msp) / (N * N * ratio * (ratio - 0.5))
  Mr <- ratio * N

  list(Msp = Msp, Mr = Mr, tau = tau, ratio = ratio)
}
