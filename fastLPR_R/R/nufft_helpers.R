# Copyright (c) 2024-2025 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later

#' Compute NUFFT grid parameters
#'
#' Calculates spreading width, oversampling ratio, and Gaussian width
#' for Non-Uniform FFT based on desired accuracy.
#' This is a direct translation of MATLAB's compute_grid_params.m
#'
#' @param N Original grid size (scalar or vector)
#' @param accuracy Desired accuracy in decimal digits (default: 8)
#'              6: ratio=1 (no oversampling)
#'              7-11: ratio=2 (2x oversampling)
#'              >11: ratio=3 (3x oversampling)
#'
#' @return List containing:
#'   \itemize{
#'     \item Msp: Spreading width
#'     \item Mr: Oversampled grid size = ratio * N
#'     \item tau: Gaussian spreading parameter
#'     \item ratio: Oversampling ratio (1, 2, or 3)
#'   }
#'
#' @noRd
compute_grid_params <- function(N, accuracy = 8) {
  # Determine oversampling ratio based on accuracy
  if (accuracy <= 6) {
    ratio <- 1  # No oversampling
  } else if (accuracy > 6 && accuracy <= 11) {
    ratio <- 2  # 2x oversampling
  } else {
    ratio <- 3  # 3x oversampling
  }

  # Compute grid parameters
  Msp <- rep(accuracy, length(N))  # Spreading width
  tau <- (pi * Msp) / (N * N * ratio * (ratio - 0.5))
  Mr <- ratio * N  # Oversampled grid size

  return(list(Msp = Msp, Mr = Mr, tau = tau, ratio = ratio))
}

#' Scale and shift knot locations for NUFFT
#'
#' Transforms knot locations to [0, 2pi] range required by Greengard's NUFFT.
#' Applies frequency scaling, centering, and modulo wrapping.
#' This is a direct translation of MATLAB's scale_knots.m
#'
#' @param knots Original knot locations (any range)
#' @param N Grid size
#' @param Fs Sampling frequency
#'
#' @return knots Scaled knot locations in [0, 2pi]
#'
#' @noRd
scale_knots <- function(knots, N, Fs) {
  knots <- as.matrix(knots)
  dx <- ncol(knots)
  N <- rep_len(as.numeric(N), dx)
  Fs <- rep_len(as.numeric(Fs), dx)

  # Try Rcpp version first (5-10x faster for large N)
  rcpp_fn <- tryCatch({
    fn <- get0("rcpp_scale_knots", envir = globalenv(), inherits = FALSE)
    if (is.null(fn)) {
      fn <- get0("rcpp_scale_knots", envir = asNamespace("fastlpr"), inherits = FALSE)
    }
    fn
  }, error = function(e) NULL)

  if (is.function(rcpp_fn)) {
    result <- tryCatch({
      rcpp_fn(knots, N, Fs)
    }, error = function(e) NULL)
    if (!is.null(result)) {
      return(result)
    }
  }

  # Fallback to pure R implementation
  # VECTORIZED: Scale all columns at once using sweep
  # knots[,i] * Fs[i] / N[i] for each column
  scale_factors <- Fs / N
  scaled <- sweep(knots, 2, scale_factors, "*")

  # Compute shift for each column: -0.5 - ((min + max)/2 - 0.5) = -min/2 - max/2
  col_mins <- apply(scaled, 2, min)
  col_maxs <- apply(scaled, 2, max)
  shifts <- -0.5 - ((col_mins + col_maxs) / 2 - 0.5)

  # Apply shift and convert to [0, 2pi]
  scaled <- sweep(scaled, 2, shifts, "+")
  scaled <- (2 * pi * scaled) %% (2 * pi)

  scaled
}

#' Generate frequency grid for NUFFT
#'
#' Creates a frequency grid used in Non-Uniform Fast Fourier Transform.
#' Frequencies are centered around zero with proper spacing.
#' This is a direct translation of MATLAB's nufftfreqs.m
#'
#' @param N Grid size in each dimension (1 x dx vector)
#' @param df Frequency spacing in each dimension (default: ones(1,dx))
#'
#' @return List containing:
#'   \itemize{
#'     \item grid: Frequency grid (prod(N) x dx matrix)
#'     \item df: Frequency spacing (1 x dx vector)
#'   }
#'
#' @noRd
nufftfreqs <- function(N, df = numeric()) {
  dx <- length(N)
  if (length(df) == 0) {
    df <- rep(1, dx)
  }

  # Compute frequency range for each dimension
  # Frequencies: trunc(-N/2) to N-trunc(N/2)-1
  # NOTE: MATLAB uses fix() which truncates toward zero, not floor()
  # R's floor() rounds toward -Inf, so floor(-1.5) = -2, but MATLAB fix(-1.5) = -1
  # R's trunc() matches MATLAB's fix() behavior (truncate toward zero)
  grid_range <- multispace(trunc(-N / 2), N - trunc(N / 2) - 1, N)
  if (!is.list(grid_range)) {
    grid_range <- lapply(seq_len(dx), function(i) grid_range[i, ])
  }

  for (i in seq_len(dx)) {
    grid_range[[i]] <- df[i] * grid_range[[i]]
  }

  grid_cells <- do.call(expand.grid, grid_range)
  grid <- as.matrix(grid_cells)

  list(grid = grid, df = df)
}

# get_ndgrid is now in utils.R - removed duplicate to avoid conflicts

#' Compute linear indices for a sub-patch of an array
#'
#' Converts multi-dimensional sub-patch bounds to linear indices for efficient
#' array indexing. Used for extracting sub-regions from padded arrays.
#' This is a direct translation of MATLAB's get_patch_index.m
#'
#' @param q Sub-patch bounds (dx x 2 matrix)
#'       q[,1] = start indices in each dimension
#'       q[,2] = end indices in each dimension
#' @param M Full array size (dM x 1 vector)
#'       Size of the full array from which to extract the patch
#'
#' @return idx Linear indices of all points in the sub-patch (column vector)
#'
#' @noRd
get_patch_index <- function(q, M) {
  # Get dimensions
  dx <- nrow(q)  # Number of patch dimensions
  dM <- length(M)  # Number of array dimensions

  # OPTIMIZATION: Special case for 2D (most common)
  if (dx == 2 && dM >= 2) {
    rows <- q[1, 1]:q[1, 2]  # Row indices
    cols <- q[2, 1]:q[2, 2]  # Column indices
    nr <- length(rows)
    nc <- length(cols)

    # Linear index = row + (col-1) * M[1]
    # Vectorized: rep rows nc times, add column offsets
    row_base <- rep(rows, times = nc)
    col_offsets <- rep((cols - 1) * M[1], each = nr)
    idx_patch <- row_base + col_offsets

    # Handle extra dimensions (e.g., multiple response columns)
    if (dM > 2) {
      extra_offset <- prod(M[1:2]) * 0:(prod(M[3:dM]) - 1)
      idx_patch <- rep(idx_patch, times = prod(M[3:dM])) + rep(extra_offset, each = length(idx_patch))
    }

    return(as.matrix(idx_patch))
  }

  # General case for 1D or 3D+
  # Compute linear indices for first dimension
  idx_patch <- q[1, 1]:q[1, 2]

  # Add offsets for additional dimensions
  if (dx > 1) {
    for (i in 2:dx) {
      # Compute offset for dimension i
      offset <- ((q[i, 1]:q[i, 2]) - 1) * prod(M[1:(i-1)])
      # OPTIMIZATION: Use rep instead of outer for the Cartesian product
      n_idx <- length(idx_patch)
      n_off <- length(offset)
      idx_patch <- rep(idx_patch, times = n_off) + rep(offset, each = n_idx)
    }
  }

  # Handle extra dimensions beyond patch dimensions
  if (dM > dx) {
    # Compute offsets for extra dimensions
    extra_offset <- prod(M[1:dx]) * 0:(prod(M[(dx+1):dM]) - 1)
    idx_patch <- rep(idx_patch, times = prod(M[(dx+1):dM])) + rep(extra_offset, each = length(idx_patch))
  }

  # Ensure output is column vector
  return(as.matrix(idx_patch))
}

#' Convert arrays to appropriate precision based on accuracy
#'
#' Converts input arrays to single or double precision based on required
#' accuracy (in decimal digits). Used to optimize memory and speed.
#' This is a direct translation of MATLAB's bit_convert.m
#'
#' @param accuracy Required accuracy in decimal digits, or 'single'/'double'
#'              < 5: convert to single (23-bit mantissa, ~7 digits)
#'              >= 5: convert to double (52-bit mantissa, ~15 digits)
#' @param ... Input arrays to convert
#'
#' @return List of converted arrays in single or double precision
#'
#' @noRd
bit_convert <- function(accuracy, ...) {
  arrays <- list(...)
  result <- vector("list", length(arrays))

  # Convert based on accuracy requirement
  # NOTE: R doesn't have native single precision, so we always use double
  # This is a no-op for interface compatibility with MATLAB
  if (is.numeric(accuracy)) {
    # Numeric accuracy: threshold at 5 digits
    # In MATLAB: accuracy < 5 uses single, >= 5 uses double
    # In R: always use double (R's default numeric type)
    for (i in seq_along(arrays)) {
      storage.mode(arrays[[i]]) <- "double"
      result[[i]] <- arrays[[i]]
    }
  } else if (is.character(accuracy)) {
    # Explicit precision specification
    # In R: always use double regardless of specification
    for (i in seq_along(arrays)) {
      storage.mode(arrays[[i]]) <- "double"
      result[[i]] <- arrays[[i]]
    }
  }

  return(result)
}

#' Find smallest numeric type that can represent a range
#'
#' Determines most memory-efficient numeric type (uint8, int16, single, etc.)
#' that can represent values in range [minval, maxval].
#' This is a direct translation of MATLAB's min_numerical_type.m
#'
#' @param minval Minimum value to represent (default: 0)
#' @param maxval Maximum value to represent (required)
#' @param redundancy Safety factor for max value (default: 1)
#' @param isunsign Force unsigned type (default: auto from minval>=0)
#' @param isfloating Force floating-point type (default: auto from integer check)
#'
#' @return List containing:
#'   \itemize{
#'     \item type: String name of optimal type ('uint8', 'int16', 'single', etc.)
#'     \item byte: Bytes per element for this type
#'   }
#'
#' @noRd
min_numerical_type <- function(minval = 0, maxval, redundancy = 1,
                                isunsign = NULL, isfloating = NULL) {
  if (is.null(isunsign)) {
    isunsign <- minval >= 0  # Auto-detect: unsigned if minval >= 0
  }
  if (is.null(isfloating)) {
    isfloating <- !all(c(minval, maxval) %% 1 == 0)  # Auto-detect: float if non-integer
  }

  # R doesn't have same integer types as MATLAB, so we'll use double or single
  # In practice, R integers are typically 32-bit or 64-bit

  if (isfloating) {
    # Floating-point types
    if (maxval < .Machine$single.xmax / redundancy) {
      type <- "single"
      byte_size <- 4
    } else {
      type <- "double"
      byte_size <- 8
    }
  } else {
    # Integer types - R uses double for integers, but we can optimize
    if (isunsign && maxval < 2^8 / redundancy) {
      type <- "integer"  # Could represent as 8-bit in theory
      byte_size <- 4  # But R still uses 32-bit minimum
    } else if (isunsign && maxval < 2^16 / redundancy) {
      type <- "integer"  # Could represent as 16-bit
      byte_size <- 4
    } else if (!isunsign && maxval < 2^7 / redundancy && minval >= -2^7 / redundancy) {
      type <- "integer"  # Could represent as 8-bit signed
      byte_size <- 4
    } else if (!isunsign && maxval < 2^15 / redundancy && minval >= -2^15 / redundancy) {
      type <- "integer"  # Could represent as 16-bit signed
      byte_size <- 4
    } else {
      type <- "integer"  # Use 32-bit or 64-bit
      byte_size <- 8
    }
  }

  return(list(type = type, byte = byte_size))
}
