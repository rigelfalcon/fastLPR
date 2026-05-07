# Copyright (c) 2024-2025 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later

#' Local Weighted Polynomial (LWP) Estimator Functions
#'
#' Provides functions for computing design matrices and solving local polynomial
#' regression systems for orders 0, 1, and 2.
#'
#' Mirrors: fastLPR/utility/core/get_lwp_estimator.m
#' Python reference: fastLPR_py/src/fastlpr/lwp_estimator.py
#'
#' Instead of using symbolic math (as in MATLAB), we use direct numerical
#' computation with efficient R operations (following Python approach).
#'
#' @keywords internal

#' Get polynomial design matrix terms
#'
#' @param x Matrix of normalized predictor variables (x - x0) / h, shape (N, d)
#' @param order Polynomial order (0, 1, or 2)
#' @return Design matrix with polynomial terms, shape (N, n_terms)
#'
#' @details
#' For order 0: X = [1]
#' For order 1: X = [1, x1, x2, ..., xd]
#' For order 2: X = [1, x1, x2, ..., xd, x1^2, x1*x2, ..., xd^2]
#'
#' @keywords internal
get_polynomial_terms <- function(x, order) {
  x <- as.matrix(x)
  N <- nrow(x)
  dims <- ncol(x)

  # Check for unsupported high-dimensional, high-order cases
  if (dims > 2 && order > 2) {
    stop(sprintf(
      "Polynomial order %d for %dD data is not supported. High-dimensional (dims > 2) high-order (order > 2) regression requires large computational power. Please use order <= 2 for dims > 2.",
      order, dims
    ))
  }

  if (order == 0) {
    # Order 0: Nadaraya-Watson (local constant)
    return(matrix(1, nrow = N, ncol = 1))

  } else if (order == 1) {
    # Order 1: Local linear
    return(cbind(rep(1, N), x))

  } else if (order == 2) {
    # Order 2: Local quadratic
    # Include all quadratic terms (lower triangular of x*x')
    terms <- list(rep(1, N))

    # Linear terms
    for (d in seq_len(dims)) {
      terms[[length(terms) + 1]] <- x[, d]
    }

    # Quadratic terms (lower triangular)
    for (i in seq_len(dims)) {
      for (j in seq(i, dims)) {
        terms[[length(terms) + 1]] <- x[, i] * x[, j]
      }
    }

    return(do.call(cbind, terms))

  } else {
    stop(sprintf("Order %d not supported. Use 0, 1, or 2.", order))
  }
}

#' Extract lower triangular elements of a matrix (column-wise)
#'
#' @param A Symmetric matrix, shape (n, n)
#' @return Lower triangular elements as vector
#' @keywords internal
mat2tril <- function(A) {
  n <- nrow(A)
  indices <- which(lower.tri(A, diag = TRUE), arr.ind = TRUE)
  # Sort by column first, then row (column-major order)
  indices <- indices[order(indices[, 2], indices[, 1]), ]
  A[indices]
}

#' Reconstruct symmetric matrix from lower triangular elements
#'
#' @param v Lower triangular elements (column-wise)
#' @return Symmetric matrix
#' @keywords internal
vec2tril <- function(v) {
  # Determine matrix size from vector length
  # n*(n+1)/2 = length(v)
  # n^2 + n - 2*length(v) = 0
  # n = (-1 + sqrt(1 + 8*length(v))) / 2
  n <- (-1 + sqrt(1 + 8 * length(v))) / 2
  if (abs(n - round(n)) > 1e-10) {
    stop("Invalid vector length for lower triangular matrix")
  }
  n <- round(n)

  # Create symmetric matrix
  A <- matrix(0, nrow = n, ncol = n)

  # Fill lower triangular part
  indices <- which(lower.tri(A, diag = TRUE), arr.ind = TRUE)
  indices <- indices[order(indices[, 2], indices[, 1]), ]
  A[indices] <- v

  # Mirror to upper triangular
  A[upper.tri(A)] <- t(A)[upper.tri(A)]

  A
}

#' Generate polynomial terms from grid coordinates
#'
#' Helper function to compute polynomial terms for design matrix.
#' Mirrors the symbolic computation in MATLAB's get_lwp_estimator.m
#'
#' MATLAB COMPATIBILITY NOTE:
#' MATLAB's Sxfun uses xgrid(:,:,d) indexing on a 4D array [N1, N2, dx, dh],
#' which extracts only the FIRST bandwidth's coordinates [N1, N2], NOT all bandwidths.
#' The result is then broadcast across all bandwidths when multiplied with kd.
#' R must replicate this behavior: extract first bandwidth coords only.
#'
#' @param xgrid Grid coordinates (array with last dimension = dx, possibly dh after)
#' @param order Polynomial order
#' @param dx Number of dimensions
#'
#' @return List of polynomial terms (spatial dimensions only, broadcast via R recycling)
#' @keywords internal
get_polynomial_terms_from_grid <- function(xgrid, order, dx) {
  # Extract grid dimensions
  grid_dims <- dim(xgrid)

  # Check if xgrid has an extra dh dimension (from 2D+ broadcasting)
  # Expected: (N1, N2, ..., dx) for normal case
  # Or: (N1, N2, ..., dx, dh) for multi-bandwidth case
  has_dh_dim <- length(grid_dims) > (dx + 1)

  if (order == 0) {
    # Order 0: Just return constant term (1)
    if (has_dh_dim) {
      # Remove dx dimension, keep dh: (N1, N2, ..., dh)
      ones_dims <- c(grid_dims[1:(length(grid_dims)-2)], grid_dims[length(grid_dims)])
    } else {
      # Remove dx dimension: (N1, N2, ...)
      ones_dims <- grid_dims[-length(grid_dims)]
    }
    ones <- array(1, dim = ones_dims)
    return(list(ones))
  }

  # Extract coordinate arrays for each dimension
  x_coords <- list()

  for (d in 1:dx) {
    # Extract d-th coordinate
    if (dx == 1) {
      # For 1D: xgrid can be (N x 1) or (N x dh)
      #
      # For multi-bandwidth case, we need to keep ALL columns!
      # Each column has coordinates normalized by a different bandwidth.
      # kd[i,j] must be multiplied by xgrid_norm[i,j] (same j), not xgrid_norm[i,1]!
      #
      # Previous bug: extracted only first column and broadcast it to all bandwidths,
      # causing S[[2]] = kd * xgrid_norm[:,1] instead of S[[2]][,j] = kd[,j] * xgrid_norm[,j]
      #
      if (!is.null(dim(xgrid)) && length(dim(xgrid)) >= 2 && ncol(xgrid) > 1) {
        # Multi-bandwidth (N x dh): KEEP THE FULL MATRIX
        # This ensures element-wise multiplication: kd[i,j] * xgrid_norm[i,j]
        x_coords[[d]] <- xgrid
      } else {
        # Single bandwidth or vector
        x_coords[[d]] <- as.vector(xgrid)
      }
    } else {
      # For multi-dimensional grids
      # xgrid is (N1 x N2 x ... x dx) or (N1 x N2 x ... x dx x dh)
      # Extract slice for dimension d

      if (has_dh_dim) {
        # xgrid: (N1, N2, ..., dx, dh)
        #
        # For multi-bandwidth case, we need to keep ALL bandwidths!
        # Each bandwidth has DIFFERENT normalized coordinates: xgrid_norm[i,j,d,h] = grid[i,j,d] / h[h,d]
        # We need: xgrid[:, :, ..., d, :] -> (N1, N2, ..., dh) (ALL bandwidths)
        #
        # Previous bug: extracted only first bandwidth and broadcast it, causing wrong design matrix
        #
        n_spatial_dims <- length(grid_dims) - 2
        dh_dim <- grid_dims[length(grid_dims)]
        indices <- rep(list(quote(expr = )), length(grid_dims))
        indices[[n_spatial_dims + 1]] <- d  # Select d-th coordinate
        # NOTE: Keep ALL bandwidths (no index for last dimension)

        x_slice <- do.call(`[`, c(list(xgrid), indices, list(drop = FALSE)))

        # Result is (N1, N2, ..., 1, dh), reshape to (N1, N2, ..., dh) (in-place)
        dim(x_slice) <- c(grid_dims[1:n_spatial_dims], dh_dim)
      } else {
        # xgrid: (N1, N2, ..., dx)
        # Extract [:, :, ..., d]
        indices <- rep(list(quote(expr = )), length(grid_dims))
        indices[[length(grid_dims)]] <- d

        x_slice <- do.call(`[`, c(list(xgrid), indices, list(drop = FALSE)))

        # Remove the singleton dimension (the dx dimension) (in-place)
        # x_slice is (N1, N2, ..., 1), need to make it (N1, N2, ...)
        dim(x_slice) <- grid_dims[-length(grid_dims)]
      }

      x_coords[[d]] <- x_slice
    }
  }

  terms <- list()

  if (order == 1) {
    # Order 1: {1, x1, x2, ..., xd}
    # Constant term - handle both vector (1D) and array (multi-D) cases
    x_dims <- dim(x_coords[[1]])
    if (is.null(x_dims)) {
      # 1D case: x_coords[[1]] is a vector
      ones <- rep(1, length(x_coords[[1]]))
    } else {
      # Multi-D case: x_coords[[1]] is an array
      ones <- array(1, dim = x_dims)
    }
    terms[[1]] <- ones

    # Linear terms
    for (d in 1:dx) {
      terms[[length(terms) + 1]] <- x_coords[[d]]
    }

    if (dx == 1 && length(x_coords[[1]]) >= 5) {
    }

  } else if (order == 2) {
    # Order 2: {1, x1, x2, ..., xd, x1^2, x1*x2, ..., xd^2}
    # Constant term - handle both vector (1D) and array (multi-D) cases
    x_dims <- dim(x_coords[[1]])
    if (is.null(x_dims)) {
      # 1D case: x_coords[[1]] is a vector
      ones <- rep(1, length(x_coords[[1]]))
    } else {
      # Multi-D case: x_coords[[1]] is an array
      ones <- array(1, dim = x_dims)
    }
    terms[[1]] <- ones

    # Linear terms
    for (d in 1:dx) {
      terms[[length(terms) + 1]] <- x_coords[[d]]
    }

    # Quadratic terms (lower triangular)
    for (i in 1:dx) {
      for (j in i:dx) {
        terms[[length(terms) + 1]] <- x_coords[[i]] * x_coords[[j]]
      }
    }
  }

  return(terms)
}


# ============================================================================
# TOP-LEVEL FUNCTIONS FOR JIT EFFICIENCY
# ============================================================================
# Move mfun computation outside get_lwp_estimator
# R's JIT compiler has trouble with closures, causing 16+ second delay on
# the second call to mfun (when computing DOF). By making the heavy computation
# a top-level function, the JIT compiler can handle it efficiently.
#
# The pattern:
#   - `.mfun_internal(S, T, order, dx, iscomplex, regularization)` - top-level, JIT-friendly
#   - `mfun <- function(S, T, reg) .mfun_internal(S, T, order, dx, iscomplex, reg)` - thin wrapper
#
# BROADCASTING HELPER (Top-level function for JIT efficiency)
# ============================================================================
# MOVED OUTSIDE get_lwp_estimator to reduce closure complexity and JIT overhead.
# R's JIT compiler had trouble with deeply nested closures, causing 12+ second
# compilation delay on the second call to mfun. By making this a top-level
# function, the JIT compiler can handle it more efficiently.
#
# MATLAB/Python use automatic broadcasting: when T has more dimensions than S,
# operations like s * t automatically expand s to match t's shape.
# R requires explicit array() calls to add dimensions.
#
# Example:
#   S elements: shape (N, dh) - no dy dimension (kernel-only)
#   T elements: shape (N, dh, dy) - has dy dimension (kernel * response)
#   Need to broadcast S: (N, dh) -> (N, dh, dy) for element-wise operations
#'
#' @keywords internal
.prepare_broadcasting <- function(s_list, t_list) {
  # Check if T has more dimensions than S
  if (length(t_list) == 0 || length(s_list) == 0) {
    return(list(s = s_list, t = t_list))
  }

  # Get T's dimensions
  t_first <- t_list[[1]]
  t_dims <- dim(t_first)
  if (is.null(t_dims)) {
    # T is a vector, no broadcasting needed
    return(list(s = s_list, t = t_list))
  }
  t_ndim <- length(t_dims)

  # Get S's dimensions (handle vector case)
  s_first <- s_list[[1]]
  s_dims <- dim(s_first)
  if (is.null(s_dims)) {
    # S is a vector - treat as 1D
    s_ndim <- 1
  } else {
    s_ndim <- length(s_dims)
  }

  # If T has an extra dimension (dy > 1), broadcast S to match
  if (t_ndim > s_ndim) {
    num_dims_to_add <- t_ndim - s_ndim
    s_broadcast <- list()

    for (i in seq_along(s_list)) {
      si <- s_list[[i]]
      si_dims <- dim(si)
      si_was_vector <- is.null(si_dims)

      # Handle vector case: convert to matrix first
      if (si_was_vector) {
        si <- matrix(si, ncol = 1)
        si_dims <- dim(si)
      }

      # Get target extra dimensions from T
      # Use s_ndim (computed from first element) not length(si_dims) (after conversion)
      if (s_ndim < t_ndim) {
        t_extra_dims <- t_dims[(s_ndim + 1):t_ndim]
      } else {
        # Should not reach here, but safeguard
        next
      }

      # Handle different numbers of dimensions to add
      # Use array() with recycling - R recycles the vector automatically
      if (num_dims_to_add == 1) {
        # Add one dimension: (N, dh) -> (N, dh, dy)
        dy <- t_extra_dims[1]
        si_broadcast <- array(si, dim = c(si_dims, dy))

      } else if (num_dims_to_add == 2) {
        # Add two dimensions: (N,) -> (N, dh, dy)
        dh <- t_extra_dims[1]
        dy <- t_extra_dims[2]
        si_broadcast <- array(si[, 1], dim = c(si_dims[1], dh, dy))

      } else {
        stop(sprintf("prepare_broadcasting: Adding %d dimensions not implemented (max 2)", num_dims_to_add))
      }

      s_broadcast[[i]] <- si_broadcast
    }

    return(list(s = s_broadcast, t = t_list))
  }

  # No broadcasting needed
  return(list(s = s_list, t = t_list))
}


#' Complex-aware sign function
#'
#' R's base sign() doesn't support complex numbers. This function provides
#' a complex-aware alternative: sign(z) = z / |z| for z != 0.
#'
#' @param x Numeric or complex vector/array
#' @return Sign of x (same shape as input)
#' @keywords internal
complex_sign <- function(x) {
  # For real numbers, use base sign()
  if (!is.complex(x)) {
    return(sign(x))
  }
  # For complex: sign(z) = z / |z|, with sign(0) = 0
  abs_x <- abs(x)
  result <- x / abs_x
  result[abs_x < .Machine$double.eps] <- 0
  return(result)
}

# ============================================================================
# JIT-EFFICIENT INTERNAL FUNCTION FOR mfun
# ============================================================================
# This top-level function avoids R JIT recompilation overhead on closures.
# The mfun closure in get_lwp_estimator() is now a thin wrapper calling this.
# Moved from closure to avoid 16+ second JIT delays on DOF computation.
# ============================================================================
.mfun_internal <- function(S, T, order, dx, regularization = 0) {
  # Detect if S needs broadcasting to match T's extra dimensions.
  # When T has more dimensions (e.g. dy probe vectors in DoF), the old code
  # copied every S element to match T's shape -- ~1.6 GB for typical DoF calls.
  # Instead, we keep S at its original shape and use R's column-major vector
  # recycling: as.vector(2D) * as.vector(3D) recycles the shorter vector along
  # the trailing dimension, which is equivalent to NumPy-style broadcasting.
  needs_broadcast <- FALSE
  t_dims <- dim(T[[1]])
  s_dims <- dim(S[[1]])
  t_ndim <- if (is.null(t_dims)) 1L else length(t_dims)
  s_ndim <- if (is.null(s_dims)) 1L else length(s_dims)
  if (t_ndim > s_ndim) {
    needs_broadcast <- TRUE
    # bmul: multiply 2D S-product by 3D T array via vector recycling
    bmul <- function(s_prod, t_arr) {
      array(as.vector(s_prod) * as.vector(t_arr), dim = dim(t_arr))
    }
    # bdiv: divide 3D numerator by 2D denominator via vector recycling
    bdiv <- function(num_arr, den_arr) {
      array(as.vector(num_arr) / as.vector(den_arr), dim = dim(num_arr))
    }
  } else {
    # Same dimensions: use .prepare_broadcasting for any remaining edge cases
    broadcast_result <- .prepare_broadcasting(S, T)
    S <- broadcast_result$s
    T <- broadcast_result$t
  }

  # ORDER 0: m = T / S (Nadaraya-Watson)
  if (order == 0) {
    S_arr <- S[[1]]
    T_arr <- T[[1]]
    S_threshold <- pmax(S_arr, abs(S_arr) * regularization)
    if (needs_broadcast) return(bdiv(T_arr, S_threshold))
    return(T_arr / S_threshold)
  }

  # ORDER 1, 1D: Analytical formula
  if (order == 1 && dx == 1) {
    s1 <- S[[1]]; s2 <- S[[2]]; s3 <- S[[3]]
    t1 <- T[[1]]; t2 <- T[[2]]
    det_S <- s1 * s3 - s2^2
    det_S <- pmax(abs(det_S), .Machine$double.eps * 1e6) * complex_sign(det_S)
    if (needs_broadcast) return(bdiv(bmul(s3, t1) - bmul(s2, t2), det_S))
    return((s3 * t1 - s2 * t2) / det_S)
  }

  # ORDER 2, 1D: Analytical formula
  if (order == 2 && dx == 1) {
    s1 <- S[[1]]; s2 <- S[[2]]; s3 <- S[[3]]; s4 <- S[[4]]; s5 <- S[[5]]; s6 <- S[[6]]
    t1 <- T[[1]]; t2 <- T[[2]]; t3 <- T[[3]]
    if (needs_broadcast) {
      numerator <- (bmul(s5^2, t1) - bmul(s2 * s5, t3) + bmul(s2 * s6, t2) +
                    bmul(s3 * s4, t3) - bmul(s3 * s5, t2) - bmul(s4 * s6, t1))
    } else {
      numerator <- (s5^2 * t1 - s2 * s5 * t3 + s2 * s6 * t2 + s3 * s4 * t3 - s3 * s5 * t2 - s4 * s6 * t1)
    }
    denominator <- (s1 * s5^2 + s3^2 * s4 + s2^2 * s6 - s2 * s3 * s5 * 2.0 - s1 * s4 * s6)
    denominator <- pmax(abs(denominator), .Machine$double.eps * 1e6) * complex_sign(denominator)
    if (needs_broadcast) return(bdiv(numerator, denominator))
    return(numerator / denominator)
  }

  # ORDER 1, 2D: Same formula as ORDER 2, 1D (3-parameter systems)
  if (order == 1 && dx == 2) {
    s1 <- S[[1]]; s2 <- S[[2]]; s3 <- S[[3]]; s4 <- S[[4]]; s5 <- S[[5]]; s6 <- S[[6]]
    t1 <- T[[1]]; t2 <- T[[2]]; t3 <- T[[3]]
    if (needs_broadcast) {
      numerator <- (bmul(s5^2, t1) - bmul(s2 * s5, t3) + bmul(s2 * s6, t2) +
                    bmul(s3 * s4, t3) - bmul(s3 * s5, t2) - bmul(s4 * s6, t1))
    } else {
      numerator <- (s5^2 * t1 - s2 * s5 * t3 + s2 * s6 * t2 + s3 * s4 * t3 - s3 * s5 * t2 - s4 * s6 * t1)
    }
    denominator <- (s1 * s5^2 + s3^2 * s4 + s2^2 * s6 - s2 * s3 * s5 * 2.0 - s1 * s4 * s6)
    denominator <- pmax(abs(denominator), .Machine$double.eps * 1e6) * complex_sign(denominator)
    if (needs_broadcast) return(bdiv(numerator, denominator))
    return(numerator / denominator)
  }

  # ORDER 1, 3D: 10-parameter system
  if (order == 1 && dx == 3) {
    s1 <- S[[1]]; s2 <- S[[2]]; s3 <- S[[3]]; s4 <- S[[4]]; s5 <- S[[5]]
    s6 <- S[[6]]; s7 <- S[[7]]; s8 <- S[[8]]; s9 <- S[[9]]; s10 <- S[[10]]
    t1 <- T[[1]]; t2 <- T[[2]]; t3 <- T[[3]]; t4 <- T[[4]]
    if (needs_broadcast) {
      numerator <- (-bmul(s2 * s9^2, t2) - bmul(s3 * s7^2, t3) - bmul(s4 * s6^2, t4) +
                    bmul(s5 * s9^2, t1) + bmul(s7^2 * s8, t1) + bmul(s6^2 * s10, t1) +
                    bmul(s3 * s6 * s7, t4) + bmul(s4 * s6 * s7, t3) +
                    bmul(s2 * s6 * s9, t4) - bmul(s2 * s6 * s10, t3) -
                    bmul(s2 * s7 * s8, t4) + bmul(s2 * s7 * s9, t3) -
                    bmul(s3 * s5 * s9, t4) + bmul(s3 * s5 * s10, t3) -
                    bmul(s3 * s6 * s10, t2) + bmul(s3 * s7 * s9, t2) +
                    bmul(s4 * s5 * s8, t4) - bmul(s4 * s5 * s9, t3) +
                    bmul(s4 * s6 * s9, t2) - bmul(s4 * s7 * s8, t2) +
                    bmul(s2 * s8 * s10, t2) - bmul(s6 * s7 * s9, t1) * 2.0 -
                    bmul(s5 * s8 * s10, t1))
    } else {
      numerator <- (-s2 * s9^2 * t2 - s3 * s7^2 * t3 - s4 * s6^2 * t4 + s5 * s9^2 * t1 +
                    s7^2 * s8 * t1 + s6^2 * s10 * t1 + s3 * s6 * s7 * t4 + s4 * s6 * s7 * t3 +
                    s2 * s6 * s9 * t4 - s2 * s6 * s10 * t3 - s2 * s7 * s8 * t4 + s2 * s7 * s9 * t3 -
                    s3 * s5 * s9 * t4 + s3 * s5 * s10 * t3 - s3 * s6 * s10 * t2 + s3 * s7 * s9 * t2 +
                    s4 * s5 * s8 * t4 - s4 * s5 * s9 * t3 + s4 * s6 * s9 * t2 - s4 * s7 * s8 * t2 +
                    s2 * s8 * s10 * t2 - s6 * s7 * s9 * t1 * 2.0 - s5 * s8 * s10 * t1)
    }
    denominator <- (-s3^2 * s7^2 - s4^2 * s6^2 - s2^2 * s9^2 + s1 * s5 * s9^2 + s1 * s7^2 * s8 +
                    s1 * s6^2 * s10 + s4^2 * s5 * s8 + s3^2 * s5 * s10 + s2^2 * s8 * s10 +
                    s3 * s4 * s6 * s7 * 2.0 - s2 * s3 * s6 * s10 * 2.0 + s2 * s3 * s7 * s9 * 2.0 +
                    s2 * s4 * s6 * s9 * 2.0 - s2 * s4 * s7 * s8 * 2.0 - s3 * s4 * s5 * s9 * 2.0 -
                    s1 * s6 * s7 * s9 * 2.0 - s1 * s5 * s8 * s10)
    denominator <- pmax(abs(denominator), .Machine$double.eps * 1e6) * complex_sign(denominator)
    if (needs_broadcast) return(bdiv(numerator, denominator))
    return(numerator / denominator)
  }

  # ORDER 2, 2D: Delegate to legacy - formula is too complex to duplicate
  # Standalone function to avoid closure overhead
  stop(sprintf(".mfun_internal: Unsupported dx=%d, order=%d (ORDER 2,2D not migrated)", dx, order))
}

#' Get local weighted polynomial estimator functions
#'
#' Generates functions for computing design matrix (S), weighted response (T),
#' and regression coefficients (m) for local polynomial regression.
#'
#' @param dx Number of predictor dimensions
#' @param order Polynomial order (0, 1, or 2)
#' @param iscomplex Whether data is complex-valued (default: FALSE)
#' @return List with Sxfun, Txfun, mfun, ns, nt
#'
#' @details
#' Unlike MATLAB which uses symbolic math, this uses numerical computation.
#' The functions operate on grid-based kernel values.
#'
#' @keywords internal
get_lwp_estimator <- function(dx, order, iscomplex = FALSE) {
  if (dx > 2 && order >= 2) {
    stop(sprintf(
      "Polynomial order %d for %dD data is not supported. High-dimensional (dx > 2) high-order (order >= 2) regression requires large computational power. Please use order <= 1 for dx > 2.",
      order, dx
    ))
  }

  # Determine number of polynomial terms
  if (order == 0) {
    nt <- 1
  } else if (order == 1) {
    nt <- 1 + dx
  } else if (order == 2) {
    nt <- 1 + dx + dx * (dx + 1) / 2
  } else {
    stop(sprintf("Order %d not supported", order))
  }

  # Number of unique elements in lower triangular S
  ns <- nt * (nt + 1) / 2

  # Function to compute design matrix S from kernel values
  # Mirrors MATLAB's Sxfun: @(k,x){k; k.*x1; k.*x1.^2; ...}
  # S is symmetric, so we only store lower triangular elements
  #
  # CRITICAL: MATLAB uses COLUMN-MAJOR LOWER TRIANGULAR ordering!
  # For a 6x6 matrix (nt=6), elements are stored as:
  #   S{1}=S(1,1), S{2}=S(2,1), S{3}=S(2,2), S{4}=S(3,1), S{5}=S(3,2), S{6}=S(3,3), ...
  # This is: for j in 1:nt, for i in j:nt, store S(i,j)
  # NOT: for i in 1:nt, for j in i:nt (which is upper triangular row-major)
  Sxfun <- function(kd, xgrid) {
    # kd: kernel values (base kernel K(x/h))
    # xgrid: normalized grid coordinates

    if (order == 0) {
      # Order 0: S = K
      return(list(kd))
    } else {
      # Order >= 1: Compute polynomial terms from grid
      # For 1D multi-bandwidth: X_terms are [N, dh] (full matrix, same shape as kd)
      # For multi-D: X_terms are [N1, N2, ...] (spatial dims only)
      X_terms <- get_polynomial_terms_from_grid(xgrid, order, dx)

      # CRITICAL: R needs explicit broadcasting unlike MATLAB
      # kd is [N1, N2, dh], X_terms[[i]] is [N1, N2]
      # MATLAB does implicit broadcasting: [N1, N2] * [N1, N2, dh] -> [N1, N2, dh]
      # R needs explicit: replicate X_terms to match kd's trailing dimension(s)
      kd_dims <- dim(kd)
      x_dims <- dim(X_terms[[1]])

      # Check if broadcasting is needed (kd has more dimensions than X_terms)
      need_broadcast <- !is.null(kd_dims) && length(kd_dims) > length(x_dims)

      if (need_broadcast) {
        # Get the trailing dimensions to broadcast over (e.g., dh)
        extra_dims <- kd_dims[(length(x_dims) + 1):length(kd_dims)]
        target_dims <- c(x_dims, extra_dims)

        # Broadcast each X_term to match kd's shape
        X_terms_broadcast <- lapply(X_terms, function(xt) {
          # Replicate xt across the extra dimensions
          array(xt, dim = target_dims)
        })
      } else {
        X_terms_broadcast <- X_terms
      }

      # Compute S_ij = K * X_i * X_j (lower triangular elements)
      # MATLAB column-major lower triangular: iterate column j, then row i from j to nt
      # For complex data: S_ij = Conj(X_i) * K * X_j (conjugate transpose)
      # For real data: S_ij = X_i * K * X_j (regular transpose)
      S_list <- list()
      for (j in seq_len(nt)) {
        for (i in seq(j, nt)) {
          if (iscomplex) {
            # Complex data: use conjugate (MATLAB: X'*W*X)
            S_list[[length(S_list) + 1]] <- Conj(X_terms_broadcast[[i]]) * kd * X_terms_broadcast[[j]]
          } else {
            # Real data: no conjugate (MATLAB: X.'*W*X)
            S_list[[length(S_list) + 1]] <- kd * X_terms_broadcast[[i]] * X_terms_broadcast[[j]]
          }
        }
      }

      return(S_list)
    }
  }

  # Function to compute weighted response T
  # Mirrors MATLAB's Txfun: @(k,x){k; k.*x1; k.*x2; ...}
  Txfun <- function(kd, xgrid) {
    # kd: kernel values (base kernel K(x/h))
    # xgrid: normalized grid coordinates

    if (order == 0) {
      # Order 0: T = K
      return(list(kd))
    } else {
      # Order >= 1: Compute polynomial terms from grid
      # For 1D multi-bandwidth: X_terms are [N, dh] (full matrix, same shape as kd)
      # For multi-D: X_terms are [N1, N2, ...] (spatial dims only)
      X_terms <- get_polynomial_terms_from_grid(xgrid, order, dx)

      # CRITICAL: R needs explicit broadcasting unlike MATLAB
      # kd is [N1, N2, dh], X_terms[[i]] is [N1, N2]
      # MATLAB does implicit broadcasting: [N1, N2] * [N1, N2, dh] -> [N1, N2, dh]
      # R needs explicit: replicate X_terms to match kd's trailing dimension(s)
      kd_dims <- dim(kd)
      x_dims <- dim(X_terms[[1]])

      # Check if broadcasting is needed (kd has more dimensions than X_terms)
      need_broadcast <- !is.null(kd_dims) && length(kd_dims) > length(x_dims)

      if (need_broadcast) {
        # Get the trailing dimensions to broadcast over (e.g., dh)
        extra_dims <- kd_dims[(length(x_dims) + 1):length(kd_dims)]
        target_dims <- c(x_dims, extra_dims)

        # Broadcast each X_term to match kd's shape
        X_terms_broadcast <- lapply(X_terms, function(xt) {
          # Replicate xt across the extra dimensions
          array(xt, dim = target_dims)
        })
      } else {
        X_terms_broadcast <- X_terms
      }

      # Compute T_i = K * X_i
      # For complex data: T_i = Conj(X_i) * K (conjugate transpose)
      # For real data: T_i = X_i * K (regular transpose)
      T_list <- list()
      for (i in seq_len(nt)) {
        if (iscomplex) {
          # Complex data: use conjugate (MATLAB: X'*W)
          T_list[[length(T_list) + 1]] <- Conj(X_terms_broadcast[[i]]) * kd
        } else {
          # Real data: no conjugate (MATLAB: X.'*W)
          T_list[[length(T_list) + 1]] <- kd * X_terms_broadcast[[i]]
        }
      }
      return(T_list)
    }
  }

  # ============================================================================
  # VECTORIZED mfun - NO LOOPS!
  # ============================================================================
  # Uses pre-computed analytical formulas instead of nested loops
  # Processes ALL bandwidths and samples simultaneously
  #
  # NOTE: Uses .prepare_broadcasting (top-level function) to reduce closure
  # complexity and avoid JIT compilation overhead on second call.

  mfun <- function(S, T, regularization = 0) {
    # ========================================================================
    # JIT FIX: Delegate to top-level .mfun_internal() for most cases
    # ========================================================================
    # This avoids R's JIT recompilation overhead on second call (16+ seconds!)
    # Root cause: R's JIT compiler recompiles large closure bodies repeatedly.
    # Solution: Move heavy computation to top-level function .mfun_internal()
    # ORDER 2,2D is handled inline below (formula too complex to duplicate)
    if (!(dx == 2 && order == 2)) {
      return(.mfun_internal(S, T, order, dx, regularization))
    }


    # ========================================================================
    # ORDER 2, 2D: Delegate to top-level function (JIT optimization)
    # ========================================================================
    # The massive symbolic Cramer's rule formula has been extracted to
    # .mfun_order2_2d() in mfun_order2_2d.R to avoid JIT compilation overhead.
    # This reduces mfun closure size from 73KB to ~1KB.
    return(.mfun_order2_2d(S, T, regularization))
  }

  list(
    Sxfun = Sxfun,
    Txfun = Txfun,
    mfun = mfun,
    ns = ns,
    nt = nt
  )
}

