# Copyright (c) 2024-2025 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later

# ============================================================================
# JIT COMPILATION NOTE
# ============================================================================
# JIT is disabled in zzz.R::.onLoad() to avoid 15+ second compilation overhead
# with complex closures in get_lwp_estimator(). Do NOT duplicate here.
# See zzz.R for details.
# ============================================================================


#' Complex-aware approxfun wrapper
#'
#' Creates a linear interpolation function that supports complex values.
#' MATLAB's griddedInterpolant and Python's scipy.interp1d support complex natively.
#' R's approxfun does not, so we interpolate real and imaginary parts separately.
#'
#' @param x Numeric vector of x coordinates (must be real)
#' @param y Numeric or complex vector of y values
#' @param method Interpolation method (default: "linear")
#' @param rule Extrapolation rule (default: 2 = use nearest value)
#' @param ... Additional arguments passed to approxfun
#'
#' @return A function that interpolates y at given x values
#' @keywords internal
complex_approxfun <- function(x, y, method = "linear", rule = 2, ...) {
  if (is.complex(y)) {
    # Complex case: separate interpolators for real and imaginary parts
    interp_re <- stats::approxfun(x = x, y = Re(y), method = method, rule = rule, ...)
    interp_im <- stats::approxfun(x = x, y = Im(y), method = method, rule = rule, ...)

    # Return combined interpolator
    function(xnew) {
      complex(real = interp_re(xnew), imaginary = interp_im(xnew))
    }
  } else {
    # Real case: delegate directly to approxfun
    stats::approxfun(x = x, y = y, method = method, rule = rule, ...)
  }
}


#' Generate multi-dimensional grid vectors
#'
#' Creates linearly or logarithmically spaced vectors for multiple dimensions.
#' Similar to MATLAB's multispace function.
#'
#' @param x_min Minimum values (scalar or m x 1 vector)
#' @param x_max Maximum values (scalar or m x 1 vector)
#' @param N Number of points (scalar or m x 1 vector)
#' @param space Spacing function (default: "linear")
#'   Options: "linear", "logspace"
#' @param type Output type (default: "array")
#'   * "array" - Return m x N matrix (if all N are equal)
#'   * "cell"  - Return m x 1 cell array
#' @param transpose Transpose output (default: FALSE)
#'
#' @return Grid vectors, either:
#'   * (m x N) matrix if type="array" and all N are equal
#'   * {m x 1} cell array otherwise
#'
#' @examples
#' # Create 2D grid: x in [0,1] with 10 points, y in [0,2] with 20 points
#' x <- multispace(c(0, 0), c(1, 2), c(10, 20))
#'
#' # Create with different N per dimension
#' x <- multispace(c(0, 0), c(1, 2), c(10, 20), type = "cell")
#'
#' @export
multispace <- function(x_min, x_max, N, space = "linear", type = "array", transpose = FALSE) {
  # Handle logspace
  if (space == "logspace") {
    x_min <- log10(x_min)
    x_max <- log10(x_max)
  }

  # Broadcast inputs to same size
  m <- max(length(x_min), length(x_max), length(N))

  # Broadcast scalars to vectors
  if (length(x_min) == 1) {
    x_min <- rep(x_min, m)
  }
  if (length(x_max) == 1) {
    x_max <- rep(x_max, m)
  }
  if (length(N) == 1) {
    N <- rep(N, m)
  }

  # Generate grid vectors
  if (type == "cell") {
    # Return list of vectors
    x <- vector("list", m)
    for (i in 1:m) {
      if (space == "logspace") {
        x[[i]] <- 10^seq(x_min[i], x_max[i], length.out = N[i])
      } else {
        x[[i]] <- seq(x_min[i], x_max[i], length.out = N[i])
      }
      if (transpose) {
        x[[i]] <- matrix(x[[i]], ncol = 1)  # Column vector
      }
    }
  } else if (all(N == N[1]) && type == "array") {
    # All dimensions have same N: return matrix
    x <- matrix(1, nrow = m, ncol = N[1])
    for (i in 1:m) {
      if (space == "logspace") {
        x[i, ] <- 10^seq(x_min[i], x_max[i], length.out = N[i])
      } else {
        x[i, ] <- seq(x_min[i], x_max[i], length.out = N[i])
      }
    }
    if (transpose) {
      x <- t(x)  # Transpose to (N x m)
    }
  } else {
    # Different N per dimension: return list
    x <- vector("list", m)
    for (i in 1:m) {
      if (space == "logspace") {
        x[[i]] <- 10^seq(x_min[i], x_max[i], length.out = N[i])
      } else {
        x[[i]] <- seq(x_min[i], x_max[i], length.out = N[i])
      }
      if (transpose) {
        x[[i]] <- matrix(x[[i]], ncol = 1)
      }
    }
  }

  return(x)
}

#' Set default values in options list
#'
#' Sets default values for fields in an options list if they don't exist.
#' Similar to MATLAB's set_defaults function.
#'
#' @param opt Options list (can be NULL)
#' @param field Field name to set
#' @param default Default value
#'
#' @return Updated options list
#'
#' @examples
#' opt <- list(order = 1)
#' opt <- set_defaults(opt, "kernel_type", "gaussian")
#' opt <- set_defaults(opt, "verbose", FALSE)
#'
#' @export
set_defaults <- function(opt, field, default) {
  if (is.null(opt)) {
    opt <- list()
  }

  if (!field %in% names(opt)) {
    opt[[field]] <- default
  }

  return(opt)
}

#' Standardize data (z-score normalization)
#'
#' Computes z-scores for each column of the input matrix.
#' Similar to MATLAB's zscore function.
#'
#' @param x Input data matrix
#'
#' @return List containing:
#'   \itemize{
#'     \item z: Standardized data
#'     \item mu: Column means
#'     \item sigma: Column standard deviations
#'   }
#'
#' @examples
#' x <- matrix(rnorm(100), ncol = 2)
#' result <- zscore(x)
#' standardized <- result$z
#' means <- result$mu
#' sds <- result$sigma
#'
#' @export
zscore <- function(x) {
  if (!is.matrix(x)) {
    x <- as.matrix(x)
  }

  mu <- colMeans(x)

  # Use R's default sd() which has ddof=1 (normalize by N-1)
  # This matches MATLAB's std(x, 0) which also uses ddof=1
  sigma <- apply(x, 2, sd)

  # Handle constant columns (sigma = 0)
  sigma[sigma == 0] <- 1

  z <- sweep(x, 2, mu, "-")
  z <- sweep(z, 2, sigma, "/")

  return(list(z = z, mu = mu, sigma = sigma))
}


#' Generate bandwidth candidates
#'
#' Creates a grid of bandwidth candidates for cross-validation.
#' Similar to MATLAB's get_hList function.
#'
#' @param n Number of bandwidth candidates per dimension
#' @param range Range of bandwidths (min, max) for each dimension
#' @param type Type of spacing ("linear" or "logspace")
#'
#' @return Matrix of bandwidth candidates (n x d)
#'
#' @examples
#' # 1D: 20 bandwidths from 0.01 to 1 (log scale)
#' hlist <- get_hlist(20, c(0.01, 1), "logspace")
#'
#' # 2D: 10x10 grid of bandwidths
#' hlist <- get_hlist(10, rbind(c(0.01, 1), c(0.01, 1)), "logspace")
#'
#' @export
#' Create multi-dimensional grid from vectors
#'
#' Generates an N-dimensional grid from coordinate vectors, similar to ndgrid
#' but with flexible output formats.
#'
#' @param x Grid vectors, either a matrix (N x d) or list of vectors
#' @param type Output format: 'cell', 'array', 'list', or 'cellList'
#' @return Grid in requested format
#' @keywords internal
get_ndgrid <- function(x, type = "cell") {
  # Convert input to list if needed
  if (is.matrix(x) && !is.list(x)) {
    # Matrix input: (N x d) -> list of d vectors
    dx <- ncol(x)
    x <- lapply(seq_len(dx), function(i) x[, i])
  } else if (!is.list(x)) {
    # Single vector -> list
    x <- list(as.vector(x))
  } else {
    # Already a list - ensure each element is a vector
    x <- lapply(x, as.vector)
  }

  dx <- length(x)
  dims <- sapply(x, length)

  # Grid size check - allow large 1D grids for benchmark scaling
  # For 1D with N=50M, grid size is also 50M (1D grid = N samples)
  # For multi-D, grid grows as N^(1/dx) which is smaller
  # Max grid size: 100M per dimension for 1D (allows benchmark up to 50M with 2x oversampling)
  #                10M per dimension for 2D (allows 10M x 10M = 100M total)
  #                1M per dimension for 3D (allows 1M x 1M x 1M = 1T - too large anyway)
  max_dim <- if (dx == 1) 1e8 else if (dx == 2) 1e7 else 1e6
  if (any(dims > max_dim)) {
    stop(sprintf("get_ndgrid: grid dimension too large (max=%d for %dD). This would cause memory issues.",
                 max_dim, dx))
  }

  # Use meshgrid-like approach
  if (dx == 1) {
    # 1D case
    grid_vec <- as.vector(x[[1]])  # Ensure it's a vector
    if (type == "cell") {
      return(list(matrix(grid_vec, ncol = 1)))
    } else if (type == "array") {
      return(array(grid_vec, dim = c(length(grid_vec), 1)))
    } else if (type == "list") {
      return(matrix(grid_vec, ncol = 1))
    } else if (type == "cellList") {
      return(list(matrix(grid_vec, ncol = 1)))
    }
  }

  # Multi-dimensional case
  # Check if grid would be too large (100M total for 1D, 100M for 2D, 10M for 3D)
  total_grid_size <- prod(dims)
  max_total <- if (dx == 1) 1e8 else if (dx == 2) 1e8 else 1e7
  if (total_grid_size > max_total) {
    stop(sprintf("get_ndgrid: grid too large (%d points, max=%d for %dD). Would cause memory overflow.",
                 total_grid_size, max_total, dx))
  }

  # Create grid using expand.grid
  grid_list <- expand.grid(x, KEEP.OUT.ATTRS = FALSE)

  # Convert to requested output format
  if (type == "cell") {
    # Cell format: list of grid matrices
    grid <- lapply(seq_len(dx), function(i) {
      array(grid_list[[i]], dim = dims)
    })
    return(grid)

  } else if (type == "array") {
    # Array format: (N1 x N2 x ... x Nd x dx)
    if (dx == 1) {
      # 1D case: return (N x 1) array
      return(array(grid_list[[1]], dim = c(dims, 1)))
    } else {
      # Multi-D case - use generalized dynamic indexing for any dx
      grid_array <- array(0, dim = c(dims, dx))
      for (i in seq_len(dx)) {
        # Build index list for assignment: all indices in spatial dims, plus i for last dim
        idx_list <- c(
          lapply(1:dx, function(d) seq_len(dims[d])),
          list(i)
        )
        # Assign using do.call
        grid_array <- do.call(`[<-`, c(list(grid_array), idx_list,
                                        list(value = array(grid_list[[i]], dim = dims))))
      }
      return(grid_array)
    }

  } else if (type == "list") {
    # List format: (prod(N) x d) matrix
    return(as.matrix(grid_list))

  } else if (type == "cellList") {
    # Cell list format: list of column vectors
    return(lapply(grid_list, function(col) matrix(col, ncol = 1)))
  }

  stop(sprintf("Unknown output format: %s", type))
}

#' Create grid from scattered data bounds
#'
#' Generates a regular grid covering the range of scattered data points.
#' Useful for creating evaluation grids for regression.
#'
#' @param x Scattered data points (T x d matrix)
#' @param type Output format (default: 'cell')
#' @param N Number of grid points per dimension (default: nrow(x))
#' @param space Spacing function: 'linear' or 'logspace' (default: 'linear')
#' @return Regular grid covering range of x
#' @keywords internal
get_ndgrid_scatter <- function(x, type = "cell", N = NULL, space = "linear") {
  if (!is.matrix(x)) {
    x <- as.matrix(x)
  }

  if (is.null(N)) {
    N <- nrow(x)
  }

  # Create grid vectors covering data range
  x_min <- apply(x, 2, min)
  x_max <- apply(x, 2, max)

  # Use multispace to create grid vectors
  # In MATLAB: multispace(...)'  returns (N x dx) matrix, then passed to get_ndgrid
  # We want the same: (N x dx) matrix
  grid_vectors <- multispace(x_min, x_max, N, space, type = "array", transpose = TRUE)

  # Generate grid
  get_ndgrid(grid_vectors, type)
}

#' Generate bandwidth candidates for cross-validation
#'
#' Creates a grid of bandwidth candidates for cross-validation.
#' Port from MATLAB's get_hlist.m (unified API v2.0).
#'
#' @param n Number of bandwidth candidates per dimension.
#'   Scalar: same number of points for all dimensions.
#'   Vector: specify number of points per dimension.
#'   Typical values: 20 for 1D, c(15, 15) for 2D.
#' @param range Range of bandwidths (min, max) for each dimension.
#'   1D: c(h_min, h_max) or matrix with 1 row.
#'   Multi-D: matrix with one row per dimension.
#'   Rule of thumb: h_min = 0.1 * sd(x), h_max = 1.0 * sd(x).
#' @param spacing Spacing function type (default: "logspace").
#'   "logspace": logarithmic spacing (better for exploring bandwidth scales).
#'   "linear": linear spacing.
#'
#' @return Matrix of bandwidth candidates (n_total x d).
#'   Each row is one bandwidth combination.
#'
#' @examples
#' # 1D: 20 bandwidths from 0.01 to 1 (log scale, default)
#' hlist <- get_hlist(20, c(0.01, 1))
#'
#' # 2D: 10x10 grid of bandwidths
#' hlist <- get_hlist(10, rbind(c(0.01, 1), c(0.01, 1)))
#'
#' # Linear spacing
#' hlist <- get_hlist(10, c(0.5, 0.6), "linear")
#'
#' @export
get_hlist <- function(n, range, spacing = "logspace") {
  # ============================================================
  # Main implementation (Unified API v2.0)
  # ============================================================

  # Convert list input to matrix
  if (is.list(range)) {
    # Convert list of vectors to matrix
    range <- do.call(rbind, range)
  }

  if (!is.matrix(range)) {
    range <- matrix(range, nrow = 1)
  }

  dims <- nrow(range)

  # Broadcast n to match dims if scalar
  if (length(n) == 1) {
    n <- rep(n, dims)
  }

  if (dims == 1) {
    # 1D case
    if (spacing == "logspace") {
      h <- 10^seq(log10(range[1]), log10(range[2]), length.out = n[1])
    } else {
      h <- seq(range[1], range[2], length.out = n[1])
    }
    h <- matrix(h, ncol = 1)
  } else {
    # Multi-dimensional case
    h_candidates <- list()
    for (d in 1:dims) {
      if (spacing == "logspace") {
        h_candidates[[d]] <- 10^seq(log10(range[d, 1]), log10(range[d, 2]), length.out = n[d])
      } else {
        h_candidates[[d]] <- seq(range[d, 1], range[d, 2], length.out = n[d])
      }
    }

    # Create all combinations
    h <- expand.grid(h_candidates)
    colnames(h) <- NULL
    h <- as.matrix(h)
  }

  return(h)
}

#' Create grid interpolator (MATLAB griddedInterpolant equivalent)
#'
#' @param regs Regression structure with xlist, xraw, N, dx, Tx
#' @param values Grid values (N1 x N2 x ... x Nd x ...) array
#' @param xlist Grid vectors (default: regs$xlist)
#' @param method Interpolation method (default: "griddedInterpolant")
#' @param opt Interpolation options (Method, ExtrapolationMethod)
#' @return List with GridVectors, Values, Method, and evaluate() function
#' @noRd
fastlpr_gridinterp <- function(regs, values, xlist = NULL, method = "griddedInterpolant", opt = NULL) {
  if (is.null(xlist)) xlist <- regs$xlist
  if (is.null(opt)) {
    # Default to "linear" to match MATLAB's griddedInterpolant('linear')
    # Previously defaulted to "spline" which caused 2D regression divergence (287% GCV error)
    opt <- list(
      Method = "linear",
      ExtrapolationMethod = "linear"
    )
  }

  dx <- regs$dx
  N <- regs$N

  # Reshape values to (N1 x N2 x ... x Nd x prod(remaining dims))
  sz <- dim(values)
  if (is.null(sz)) sz <- length(values)
  if (length(sz) > dx) {
    # Multiple responses/bandwidths: reshape
    values_reshaped <- array(values, dim = c(N, prod(sz[(dx+1):length(sz)])))
  } else {
    values_reshaped <- values
  }

  # Create interpolator object
  interpolator <- list(
    GridVectors = xlist,
    Values = values_reshaped,
    Method = opt$Method,
    ExtrapolationMethod = opt$ExtrapolationMethod,
    dx = dx,
    N = N,
    sz_original = sz
  )
  class(interpolator) <- "fastlpr_interpolator"

  # Add evaluate method
  interpolator$evaluate <- function(x_new) {
    fastlpr_interpolator_eval(interpolator, x_new)
  }

  return(interpolator)
}


#' Batch interpolation using Rcpp (FAST PATH)
#'
#' Interpolates grid values at query points for ALL bandwidths simultaneously.
#' This is 50-100x faster than looping over approxfun() in R.
#'
#' @param x_query Query points (Tx x dx matrix)
#' @param grid Grid vectors (list of dx vectors)
#' @param values Grid values with bandwidths in last dimension
#' @param dx Dimensionality (1, 2, or 3)
#' @return Interpolated values (Tx x dh matrix)
#' @keywords internal
#' @keywords internal
interp_batch_fast <- function(x_query, grid, values, dx) {
  # ==========================================================================
  # UNIFIED N-DIMENSIONAL BATCH INTERPOLATION (like MATLAB)
  # Uses rcpp_interp_batch_nd for all dimensions - no dimension-specific code
  # O(N) complexity, OpenMP parallelized over bandwidths
  # ==========================================================================

  x_query <- as.matrix(x_query)
  n_query <- nrow(x_query)
  grid_sizes <- sapply(grid, length)
  values_dim <- dim(values)

  # Determine number of bandwidths
  if (is.null(values_dim)) {
    # 1D values vector - single bandwidth
    n_bandwidth <- 1
    values_dim <- c(length(values))
  } else if (length(values_dim) == dx) {
    # No bandwidth dimension
    n_bandwidth <- 1
  } else {
    # Bandwidth is last dimension
    n_bandwidth <- values_dim[dx + 1]
  }

  # Handle complex values (Rcpp doesn't support complex batch directly)
  if (is.complex(values)) {
    if (n_bandwidth == 1) {
      result_re <- rcpp_interp_nd(lapply(grid, as.numeric), as.numeric(Re(values)), x_query)
      result_im <- rcpp_interp_nd(lapply(grid, as.numeric), as.numeric(Im(values)), x_query)
      return(result_re + 1i * result_im)
    } else {
      result <- matrix(complex(1), nrow = n_query, ncol = n_bandwidth)
      for (k in seq_len(n_bandwidth)) {
        # Extract k-th bandwidth slice
        idx_list <- c(lapply(seq_len(dx), function(d) seq_len(values_dim[d])), list(k))
        values_slice <- do.call(`[`, c(list(values), idx_list, list(drop = TRUE)))
        result_re <- rcpp_interp_nd(lapply(grid, as.numeric), as.numeric(Re(values_slice)), x_query)
        result_im <- rcpp_interp_nd(lapply(grid, as.numeric), as.numeric(Im(values_slice)), x_query)
        result[, k] <- result_re + 1i * result_im
      }
      return(result)
    }
  }

  # Real values: unified batch interpolation for ALL dimensions
  if (n_bandwidth == 1) {
    # Single bandwidth: use rcpp_interp_nd
    return(rcpp_interp_nd(lapply(grid, as.numeric), as.numeric(values), x_query))
  } else {
    # Multiple bandwidths: use unified rcpp_interp_batch_nd
    dims <- as.integer(c(grid_sizes, n_bandwidth))
    return(rcpp_interp_batch_nd(
      lapply(grid, as.numeric),
      as.numeric(values),
      dims,
      x_query
    ))
  }
}

#' Evaluate fastlpr_interpolator at new points
#'
#' UNIFIED N-DIMENSIONAL IMPLEMENTATION (like MATLAB griddedInterpolant)
#' Single code path for all dimensions using loops, not dimension-specific branches.
#'
#' @param interp Interpolator object from fastlpr_gridinterp
#' @param x_new Matrix of evaluation points (n x dx)
#' @return Interpolated values at x_new
#' @noRd
fastlpr_interpolator_eval <- function(interp, x_new) {
  x_new <- as.matrix(x_new)
  n_new <- nrow(x_new)
  dx <- interp$dx
  grid_list <- interp$GridVectors
  values <- interp$Values
  values_dim <- dim(values)

  # Check if spline interpolation is requested (matches MATLAB griddedInterpolant('spline'))
  use_spline <- (!is.null(interp$Method) && interp$Method == "spline")

  # Determine number of bandwidths from values shape
  # values has shape (N1, N2, ..., Nd) or (N1, N2, ..., Nd, n_bandwidth)
  n_bandwidth <- 1
  if (!is.null(values_dim) && length(values_dim) > dx) {
    n_bandwidth <- values_dim[dx + 1]
  }

  # Get grid sizes
  grid_sizes <- sapply(grid_list, length)

  # ==========================================================================
  # SPLINE INTERPOLATION: Dimension-specific (requires different libraries)
  # ==========================================================================
  if (use_spline) {
    # Spline interpolation requires dimension-specific implementations
    # because different R packages handle different dimensions
    return(fastlpr_interpolator_eval_spline(interp, x_new, n_bandwidth, grid_sizes))
  }

  # ==========================================================================
  # LINEAR INTERPOLATION: UNIFIED CODE PATH FOR ALL DIMENSIONS
  # Uses rcpp_interp_batch_nd which handles any dx with loops (like MATLAB)
  # ==========================================================================

  if (n_bandwidth == 1) {
    # ----- Single bandwidth: use dimension-agnostic rcpp_interp_nd -----
    # Flatten values to vector and reshape if needed
    if (!is.null(values_dim) && length(values_dim) > dx) {
      # Extract first bandwidth slice
      idx_list <- c(lapply(seq_len(dx), function(d) seq_len(values_dim[d])), list(1))
      values <- do.call(`[`, c(list(values), idx_list, list(drop = TRUE)))
    }

    # Handle complex values
    if (is.complex(values)) {
      result_re <- rcpp_interp_nd(lapply(grid_list, as.numeric), as.numeric(Re(values)), as.matrix(x_new))
      result_im <- rcpp_interp_nd(lapply(grid_list, as.numeric), as.numeric(Im(values)), as.matrix(x_new))
      return(result_re + 1i * result_im)
    }

    return(rcpp_interp_nd(lapply(grid_list, as.numeric), as.numeric(values), as.matrix(x_new)))

  } else {
    # ----- Multiple bandwidths: use unified rcpp_interp_batch_nd -----
    # Complex values: fall back to loop (no complex batch support in Rcpp)
    if (is.complex(values)) {
      result <- matrix(complex(1), nrow = n_new, ncol = n_bandwidth)
      for (k in seq_len(n_bandwidth)) {
        # Extract k-th bandwidth slice using dynamic indexing
        idx_list <- c(lapply(seq_len(dx), function(d) seq_len(values_dim[d])), list(k))
        values_slice <- do.call(`[`, c(list(values), idx_list, list(drop = TRUE)))
        result_re <- rcpp_interp_nd(lapply(grid_list, as.numeric), as.numeric(Re(values_slice)), as.matrix(x_new))
        result_im <- rcpp_interp_nd(lapply(grid_list, as.numeric), as.numeric(Im(values_slice)), as.matrix(x_new))
        result[, k] <- result_re + 1i * result_im
      }
      return(result)
    }

    # Real values: unified batch interpolation for ALL dimensions
    dims <- as.integer(c(grid_sizes, n_bandwidth))
    return(rcpp_interp_batch_nd(
      lapply(grid_list, as.numeric),
      as.numeric(values),
      dims,
      as.matrix(x_new)
    ))
  }
}

#' Spline interpolation (dimension-specific, for use_spline=TRUE only)
#' @noRd
fastlpr_interpolator_eval_spline <- function(interp, x_new, n_bandwidth, grid_sizes) {
  dx <- interp$dx
  grid_list <- interp$GridVectors
  values <- interp$Values
  values_dim <- dim(values)
  n_new <- nrow(x_new)

  if (dx == 1) {
    # 1D spline: stats::spline
    grid <- grid_list[[1]]
    rule <- if (interp$ExtrapolationMethod == "linear") 2 else 1

    interp_1d_spline <- function(vals_vec) {
      result <- stats::spline(x = grid, y = vals_vec, xout = x_new[, 1], method = "natural")$y
      if (rule == 1) result[x_new[, 1] < min(grid) | x_new[, 1] > max(grid)] <- NA
      return(result)
    }

    if (n_bandwidth == 1) {
      vals <- if (!is.null(values_dim) && length(values_dim) > 1) values[, 1] else as.vector(values)
      if (is.complex(vals)) return(interp_1d_spline(Re(vals)) + 1i * interp_1d_spline(Im(vals)))
      return(interp_1d_spline(vals))
    } else {
      result <- matrix(0, nrow = n_new, ncol = n_bandwidth)
      for (k in seq_len(n_bandwidth)) {
        vals <- values[, k]
        if (is.complex(vals)) {
          result[, k] <- interp_1d_spline(Re(vals)) + 1i * interp_1d_spline(Im(vals))
        } else {
          result[, k] <- interp_1d_spline(vals)
        }
      }
      return(result)
    }

  } else if (dx == 2) {
    # 2D spline: akima::bicubic
    if (!requireNamespace("akima", quietly = TRUE)) {
      stop("Package 'akima' is required for 2D spline interpolation. Install with: install.packages('akima')")
    }
    interp_2d_spline <- function(vals_mat) {
      akima::bicubic(x = grid_list[[1]], y = grid_list[[2]], z = vals_mat,
                     x0 = x_new[, 1], y0 = x_new[, 2])$z
    }

    if (n_bandwidth == 1) {
      vals <- if (!is.null(values_dim) && length(values_dim) > 2) values[, , 1] else values
      if (is.complex(vals)) return(interp_2d_spline(Re(vals)) + 1i * interp_2d_spline(Im(vals)))
      return(interp_2d_spline(vals))
    } else {
      result <- matrix(0, nrow = n_new, ncol = n_bandwidth)
      for (k in seq_len(n_bandwidth)) {
        vals <- values[, , k]
        if (is.complex(vals)) {
          result[, k] <- interp_2d_spline(Re(vals)) + 1i * interp_2d_spline(Im(vals))
        } else {
          result[, k] <- interp_2d_spline(vals)
        }
      }
      return(result)
    }

  } else {
    # dx >= 3: No spline support, fall back to linear with warning
    warning("Spline interpolation not supported for dx >= 3, using linear interpolation")
    interp$Method <- "linear"
    return(fastlpr_interpolator_eval(interp, x_new))
  }
}

#' Predict response values at new locations
#'
#' Predicts response values at new predictor locations using the fitted
#' regression model from cv_fastlpr. Uses fast spline interpolation for
#' efficient prediction at arbitrary points.
#'
#' Port from MATLAB: fastLPR/utility/fastLPR_predict.m
#'
#' @param regs Regression structure from cv_fastlpr containing:
#'             - fpp_yhat: Interpolator object with fitted values
#'             - opt: Options structure
#'             - d: Normalization factor (for variance estimation)
#'             - dx: Number of predictor dimensions
#' @param x_new New predictor locations (Tnew x dx matrix)
#'              Can be complex-valued (will be split into real/imaginary parts)
#'
#' @return yhat Predicted response values (Tnew x dy matrix/vector)
#'         For mean estimation: predicted mean values
#'         For variance estimation: predicted variance values
#'
#' @examples
#' \dontrun{
#' # Fit model
#' regs <- cv_fastlpr(x, y, hlist, opt)
#'
#' # Predict at new locations
#' x_new <- matrix(seq(min(x), max(x), length.out = 100), ncol = 1)
#' y_pred <- fastlpr_predict(regs, x_new)
#'
#' # Plot results
#' plot(x, y, pch = 20)
#' lines(x_new, y_pred, col = "blue", lwd = 2)
#' }
#'
#' @export
fastlpr_predict <- function(regs, x_new) {
  # Validate inputs
  if (!is.list(regs)) {
    stop("fastlpr_predict: regs must be a list from cv_fastlpr")
  }
  if (is.null(regs$fpp_yhat)) {
    stop("fastlpr_predict: regs must contain fpp_yhat field. Run cv_fastlpr first.")
  }

  # Ensure x_new is a matrix
  x_new <- as.matrix(x_new)
  n_new <- nrow(x_new)

  # Get interpolator and options
  fpp <- regs$fpp_yhat
  opt <- regs$opt

  # Handle complex-valued predictors
  # Split complex predictors into [real, imag] columns
  if (is.complex(x_new)) {
    x_iscomplex <- apply(x_new, 2, function(col) any(Im(col) != 0))
    x_new <- cbind(Re(x_new), Im(x_new[, x_iscomplex, drop = FALSE]))
  }

  # Predict using interpolator
  # Check if fpp is a structured interpolator or a function
  if (is.list(fpp) && "GridVectors" %in% names(fpp) && "Values" %in% names(fpp)) {
    # Structured interpolator: use fastlpr_interpolator_eval
    # Add dx if not present
    if (is.null(fpp$dx)) {
      fpp$dx <- length(fpp$GridVectors)
    }
    yhat <- fastlpr_interpolator_eval(fpp, x_new)
  } else if (is.function(fpp)) {
    # Simple function interpolator (1D approxfun case)
    yhat <- fpp(x_new[, 1])
  } else {
    stop("fastlpr_predict: Unknown fpp_yhat type")
  }

  # Handle variance estimation
  # For variance estimation, the model fits log(variance)
  # We need to transform back to original scale and apply normalization
  if (!is.null(opt$y_type_out) && opt$y_type_out == "variance") {
    # Get normalization factor d from regression structure
    if (!is.null(regs$d) && length(regs$d) > 0) {
      d <- regs$d
      # Transform back from log scale and apply normalization
      # Formula: variance = exp(log_variance) / d
      yhat <- exp(yhat) / d
    } else {
      # If d is not available, just use exp transform
      warning("fastlpr_predict: Normalization factor d not found. Variance estimates may be unnormalized.")
      yhat <- exp(yhat)
    }
  }

  return(yhat)
}

# NOTE: fastlpr_interval is now in fastlpr_interval.R (removed duplicate)

#' Check if data has complex values
#'
#' Utility function to check if a matrix or array has imaginary components.
#' Similar to MATLAB's fastlpr_is_complex function.
#'
#' @param x Data matrix or array
#' @param dim Dimension to check (1 for columns, "all" for any element)
#' @return Logical vector (if dim=1) or logical scalar (if dim="all")
#' @noRd
fastlpr_is_complex <- function(x, dim = "all") {
  if (is.character(dim) && dim == "all") {
    # Check if ANY element has imaginary part
    return(any(Im(x) != 0))
  } else if (dim == 1) {
    # Check each column for imaginary parts
    if (is.matrix(x)) {
      return(apply(x, 2, function(col) any(Im(col) != 0)))
    } else {
      # Single column
      return(any(Im(x) != 0))
    }
  } else {
    stop("fastlpr_is_complex: dim must be 1 or 'all'")
  }
}

#' Convert arrays to appropriate precision based on accuracy
#'
#' Simplified version for R (R uses double by default)
#' In MATLAB, this converts to single for accuracy < 5
#' In R, we keep everything as double
#'
#' @param accuracy Required accuracy in decimal digits
#' @param ... Arrays to convert
#' @return List of converted arrays (as double)
#' @noRd
fastlpr_bit_convert <- function(accuracy, ...) {
  # Get all input arrays
  args <- list(...)

  # In R, just ensure everything is numeric (double precision)
  # R doesn't have true single precision like MATLAB
  # Preserve matrix/array structure (as.double drops dimensions)
  result <- lapply(args, function(x) {
    if (is.numeric(x)) {
      if (is.matrix(x) || is.array(x)) {
        # Preserve dimensions when converting to double
        array(as.double(x), dim = dim(x))
      } else {
        as.double(x)
      }
    } else {
      x
    }
  })

  return(result)
}

# =============================================================================
# Dynamic N-D Array Slicing Functions (for dx > 3 support)
# =============================================================================

#' Extract slice from N-D array with dynamic indexing
#'
#' Generalizes array slicing to arbitrary dimensions.
#' For dx-dimensional data with extra trailing dimensions (ih, iy, etc.),
#' extracts arr[..., ih, iy] where ... spans dx dimensions.
#'
#' @param arr N-dimensional array
#' @param dx Number of spatial dimensions
#' @param ih Bandwidth index (or NULL to get all)
#' @param iy Response index (or NULL to get all)
#' @return Extracted slice as array with dx dimensions
#' @noRd
extract_slice_nd <- function(arr, dx, ih = NULL, iy = NULL) {
  ndims <- length(dim(arr))

  # Build index list: [1:n1, 1:n2, ..., ih, iy]
  idx_list <- vector("list", ndims)

  # First dx dimensions: take all
  for (d in 1:dx) {
    idx_list[[d]] <- seq_len(dim(arr)[d])
  }

  # Trailing dimensions: use provided indices or take all
  trailing_pos <- dx + 1
  if (trailing_pos <= ndims) {
    if (!is.null(ih)) {
      idx_list[[trailing_pos]] <- ih
      trailing_pos <- trailing_pos + 1
    } else {
      idx_list[[trailing_pos]] <- seq_len(dim(arr)[trailing_pos])
      trailing_pos <- trailing_pos + 1
    }
  }

  if (trailing_pos <= ndims) {
    if (!is.null(iy)) {
      idx_list[[trailing_pos]] <- iy
    } else {
      idx_list[[trailing_pos]] <- seq_len(dim(arr)[trailing_pos])
    }
  }

  # Extract using do.call
  result <- do.call(`[`, c(list(arr), idx_list, list(drop = FALSE)))

  # If single ih and iy, drop those dimensions to get dx-D array
  if (!is.null(ih) && !is.null(iy)) {
    result <- drop(result)
    # Ensure at least 1D (in-place reshape)
    if (is.null(dim(result))) {
      dim(result) <- dim(arr)[1:dx]
    }
  }

  return(result)
}

#' Assign value to slice of N-D array with dynamic indexing
#'
#' Generalizes array assignment to arbitrary dimensions.
#' Sets arr[..., ih, iy] <- value where ... spans dx dimensions.
#'
#' @param arr N-dimensional array (modified in place conceptually)
#' @param dx Number of spatial dimensions
#' @param ih Bandwidth index
#' @param iy Response index (or NULL)
#' @param value Value to assign (same shape as spatial dimensions)
#' @return Modified array
#' @noRd
assign_slice_nd <- function(arr, dx, ih, iy = NULL, value) {
  ndims <- length(dim(arr))

  # Build index list
  idx_list <- vector("list", ndims)

  # First dx dimensions: take all
  for (d in 1:dx) {
    idx_list[[d]] <- seq_len(dim(arr)[d])
  }

  # Trailing dimensions
  trailing_pos <- dx + 1
  if (trailing_pos <= ndims) {
    idx_list[[trailing_pos]] <- ih
    trailing_pos <- trailing_pos + 1
  }

  if (trailing_pos <= ndims && !is.null(iy)) {
    idx_list[[trailing_pos]] <- iy
  } else if (trailing_pos <= ndims) {
    idx_list[[trailing_pos]] <- seq_len(dim(arr)[trailing_pos])
  }

  # Assign using do.call
  arr <- do.call(`[<-`, c(list(arr), idx_list, list(value = value)))

  return(arr)
}

#' Create dynamic array dimensions for N-D + trailing dimensions
#'
#' Creates dimension vector for arrays with dx spatial dims plus trailing dims.
#'
#' @param N Grid sizes per dimension (length dx vector or scalar)
#' @param dx Number of spatial dimensions
#' @param dh Number of bandwidths
#' @param dy Number of responses (default 1)
#' @return Dimension vector
#' @noRd
make_nd_dims <- function(N, dx, dh, dy = 1) {
  # Ensure N is a vector of length dx
  if (length(N) == 1) {
    N <- rep(N, dx)
  }

  # Build dimension vector: [N1, N2, ..., Ndx, dh, dy]
  dims <- c(N[1:dx], dh, dy)

  return(dims)
}

#' Pad kernel array for FFT with dynamic dimensions
#'
#' Generalizes FFT padding to arbitrary dx dimensions.
#' Places kernel data at correct position based on qin indices.
#'
#' @param kd Kernel density array (dx-dimensional)
#' @param qin 2 x dx matrix with [start; end] indices per dimension
#' @param pad_size Total padded size per dimension
#' @return Padded array
#' @noRd
pad_kernel_nd <- function(kd, qin, pad_size) {
  dx <- ncol(qin)

  # Create zero-padded array
  kd_padded <- array(0, dim = pad_size)

  # Build index list for assignment: (qin[1,d]+1):(qin[2,d]+1) per dimension
  idx_list <- lapply(1:dx, function(d) (qin[1,d]+1):(qin[2,d]+1))

  # Assign kernel data to padded array
  kd_padded <- do.call(`[<-`, c(list(kd_padded), idx_list, list(value = kd)))

  return(kd_padded)
}

#' Extract FFT result with dynamic dimensions
#'
#' Generalizes FFT result extraction to arbitrary dx dimensions.
#' Extracts fft_result[q_start[1]:q_end[1], ..., :] for dx dimensions.
#'
#' @param fft_result FFT result array
#' @param q_start Start indices per dimension (1-based)
#' @param q_end End indices per dimension (1-based)
#' @param dx Number of spatial dimensions
#' @return Extracted array
#' @noRd
extract_fft_nd <- function(fft_result, q_start, q_end, dx) {
  ndims <- length(dim(fft_result))

  # Build index list
  idx_list <- vector("list", ndims)

  # First dx dimensions: use q_start:q_end
  for (d in 1:dx) {
    idx_list[[d]] <- q_start[d]:q_end[d]
  }

  # Trailing dimensions: take all
  if (ndims > dx) {
    for (d in (dx+1):ndims) {
      idx_list[[d]] <- seq_len(dim(fft_result)[d])
    }
  }

  # Extract
  result <- do.call(`[`, c(list(fft_result), idx_list, list(drop = FALSE)))

  return(result)
}

#' N-dimensional tensor product linear interpolation
#'
#' For dx > 3, uses tensor product of 1D linear interpolations.
#' Less accurate than specialized methods but works for any dimension.
#'
#' @param grid_list List of 1D grid vectors (length dx)
#' @param values Array of values on grid (dx-dimensional)
#' @param x_query Query points matrix (n_query x dx)
#' @return Interpolated values at query points
#' @noRd
interp_nd_tensor <- function(grid_list, values, x_query) {
  dx <- length(grid_list)
  n_query <- nrow(x_query)

  # Removed dx<=3 special case that called interp_batch_fast
  # This was causing infinite mutual recursion:
  #   interp_batch_fast -> interp_nd_tensor -> interp_batch_fast -> ...
  # Now always use tensor product interpolation via interp_point_nd

  # Handle multi-column values (e.g., matrix with multiple bandwidths)
  values_dims <- dim(values)
  if (is.null(values_dims)) {
    # 1D case: values is a vector
    n_cols <- 1
    values_is_matrix <- FALSE
  } else if (length(values_dims) == 2 && dx == 1) {
    # 1D with multiple columns (Ng x dh matrix)
    n_cols <- values_dims[2]
    values_is_matrix <- TRUE
  } else {
    # Multi-D case or single-column: treat as single set
    n_cols <- 1
    values_is_matrix <- FALSE
  }

  if (n_cols > 1) {
    # Multi-column case: interpolate each column separately
    result <- matrix(NA, nrow = n_query, ncol = n_cols)
    for (col in 1:n_cols) {
      col_values <- values[, col]
      for (i in 1:n_query) {
        result[i, col] <- interp_point_nd(grid_list, col_values, x_query[i, ])
      }
    }
    return(result)
  }

  # Single-column case: tensor product interpolation
  result <- numeric(n_query)
  for (i in 1:n_query) {
    result[i] <- interp_point_nd(grid_list, values, x_query[i, ])
  }

  return(result)
}

#' Single point N-D interpolation via tensor product
#'
#' Multilinear interpolation at a single point using tensor product.
#'
#' @param grid_list List of 1D grid vectors
#' @param values dx-dimensional array of values
#' @param x_point Single query point (length dx vector)
#' @return Interpolated value
#' @noRd
interp_point_nd <- function(grid_list, values, x_point) {
  dx <- length(grid_list)

  # Find bracketing indices and weights for each dimension
  idx_lo <- integer(dx)
  idx_hi <- integer(dx)
  weights <- numeric(dx)

  for (d in 1:dx) {
    grid <- grid_list[[d]]
    n <- length(grid)
    x <- x_point[d]

    # Find bracketing interval
    if (x <= grid[1]) {
      idx_lo[d] <- 1
      idx_hi[d] <- 1
      weights[d] <- 0
    } else if (x >= grid[n]) {
      idx_lo[d] <- n
      idx_hi[d] <- n
      weights[d] <- 0
    } else {
      # Binary search
      lo <- findInterval(x, grid)
      lo <- max(1, min(lo, n-1))
      idx_lo[d] <- lo
      idx_hi[d] <- lo + 1
      weights[d] <- (x - grid[lo]) / (grid[lo+1] - grid[lo])
    }
  }

  # Tensor product: sum over all 2^dx corners
  result <- 0
  for (corner in 0:(2^dx - 1)) {
    w <- 1
    idx <- integer(dx)
    for (d in 1:dx) {
      if (bitwAnd(corner, 2^(d-1)) > 0) {
        idx[d] <- idx_hi[d]
        w <- w * weights[d]
      } else {
        idx[d] <- idx_lo[d]
        w <- w * (1 - weights[d])
      }
    }
    # Get value at this corner
    val <- do.call(`[`, c(list(values), as.list(idx)))
    result <- result + w * val
  }

  return(result)
}


#' Convert subscript indices to linear indices
#'
#' Converts multi-dimensional array subscripts to linear (1D) indices.
#' This is the R equivalent of MATLAB's sub2ind function.
#'
#' @param dims Integer vector of array dimensions
#' @param indices Matrix where each row is a set of subscript indices,
#'   or a vector of subscript indices for a single element.
#'   Indices should be 1-based (R convention).
#'
#' @return Integer vector of linear indices (1-based)
#'
#' @examples
#' # 3x4 matrix: convert (2,3) to linear index
#' sub2ind(c(3, 4), c(2, 3))  # Returns 8
#'
#' # Multiple subscripts at once
#' sub2ind(c(3, 4), rbind(c(1, 1), c(2, 3), c(3, 4)))  # Returns c(1, 8, 12)
#'
#' @noRd
sub2ind <- function(dims, indices) {
  # Handle single set of indices (vector input)
  if (!is.matrix(indices)) {
    indices <- matrix(indices, nrow = 1)
  }

  # Check dimensions
  if (ncol(indices) != length(dims)) {
    stop("Number of columns in indices must match length of dims")
  }

  # Compute linear indices using column-major ordering (R convention)
  # Linear index = i1 + (i2-1)*d1 + (i3-1)*d1*d2 + ...
  n_indices <- nrow(indices)
  linear_idx <- indices[, 1]

  if (length(dims) > 1) {
    multiplier <- 1
    for (d in 2:length(dims)) {
      multiplier <- multiplier * dims[d - 1]
      linear_idx <- linear_idx + (indices[, d] - 1) * multiplier
    }
  }

  return(as.integer(linear_idx))
}


#' Convert linear indices to subscript indices
#'
#' Converts linear (1D) indices to multi-dimensional array subscripts.
#' This is the R equivalent of MATLAB's ind2sub function.
#'
#' @param dims Integer vector of array dimensions
#' @param linear_idx Integer vector of linear indices (1-based)
#'
#' @return Matrix where each row is a set of subscript indices (1-based)
#'
#' @examples
#' # 3x4 matrix: convert linear index 8 to subscripts
#' ind2sub(c(3, 4), 8)  # Returns matrix(c(2, 3), nrow=1)
#'
#' @noRd
ind2sub <- function(dims, linear_idx) {
  n_indices <- length(linear_idx)
  n_dims <- length(dims)

  # Initialize output matrix
  indices <- matrix(0L, nrow = n_indices, ncol = n_dims)

  # Convert each linear index to subscripts
  remaining <- linear_idx - 1  # Convert to 0-based for computation

  for (d in 1:n_dims) {
    indices[, d] <- (remaining %% dims[d]) + 1  # Convert back to 1-based
    remaining <- remaining %/% dims[d]
  }

  return(indices)
}

