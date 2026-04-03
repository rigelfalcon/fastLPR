# Copyright (c) 2024-2025 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later

#' Multi-linear Interpolation Functions (EXACT MATLAB Match)
#'
#' These functions implement true multi-linear interpolation that EXACTLY matches
#' MATLAB's griddedInterpolant with 'linear' interpolation and 'linear' extrapolation.
#'
#' Key differences from previous R implementations:
#'   - 1D: Uses TRUE linear extrapolation (not constant extrapolation from endpoints)
#'   - 2D: Allows extrapolation via negative/excess weights (not clamping)
#'   - 3D: Allows extrapolation via negative/excess weights (not clamping)
#'
#' Goal: Maximum difference from MATLAB < 1e-10
#'
#' @name multilinear_interp
NULL

#' 1D Linear Interpolation with Linear Extrapolation
#'
#' Exactly matches MATLAB's griddedInterpolant(x, y, 'linear', 'linear')
#'
#' @param grid 1D grid vector (sorted, length n)
#' @param values Values at grid points (length n)
#' @param query_points Query points (vector)
#' @return Interpolated values at query points
#' @keywords internal
interp1_linear <- function(grid, values, query_points) {
  n <- length(grid)
  n_query <- length(query_points)
  result <- numeric(n_query)

  for (i in seq_len(n_query)) {
    x <- query_points[i]

    # Find bracketing interval
    idx <- findInterval(x, grid)

    if (idx == 0) {
      # Extrapolate below grid using first two points
      idx <- 1
    } else if (idx == n) {
      # Extrapolate above grid using last two points
      idx <- n - 1
    }

    # Get grid bounds
    x1 <- grid[idx]
    x2 <- grid[idx + 1]
    y1 <- values[idx]
    y2 <- values[idx + 1]

    # Linear interpolation/extrapolation
    t <- (x - x1) / (x2 - x1)
    result[i] <- (1 - t) * y1 + t * y2
  }

  return(result)
}

#' Vectorized 1D Linear Interpolation with Linear Extrapolation
#'
#' Exactly matches MATLAB's griddedInterpolant(x, y, 'linear', 'linear')
#' Vectorized for performance.
#'
#' @param grid 1D grid vector (sorted, length n)
#' @param values Values at grid points (length n)
#' @param query_points Query points (vector)
#' @return Interpolated values at query points
#' @keywords internal
interp1_linear_vec <- function(grid, values, query_points) {
  n <- length(grid)

  # Find bracketing intervals
  idx <- findInterval(query_points, grid)

  # Handle boundary cases for extrapolation
  # idx = 0 means x < grid[1], use first segment
  # idx = n means x >= grid[n], use last segment
  idx[idx == 0] <- 1
  idx[idx == n] <- n - 1

  # Get grid bounds
  x1 <- grid[idx]
  x2 <- grid[idx + 1]
  y1 <- values[idx]
  y2 <- values[idx + 1]

  # Linear interpolation/extrapolation
  # t can be < 0 or > 1 for extrapolation
  t <- (query_points - x1) / (x2 - x1)
  result <- (1 - t) * y1 + t * y2

  return(result)
}

#' 2D Bilinear Interpolation with Linear Extrapolation
#'
#' Exactly matches MATLAB's griddedInterpolant({x1,x2}, Z, 'linear', 'linear')
#' Vectorized for performance.
#'
#' @param grid_x Grid vector for dimension 1 (sorted)
#' @param grid_y Grid vector for dimension 2 (sorted)
#' @param values Values at grid points (nx x ny matrix)
#' @param query_points Query points (n_query x 2 matrix)
#' @return Interpolated values at query points (vector of length n_query)
#' @keywords internal
interp2_bilinear <- function(grid_x, grid_y, values, query_points) {
  nx <- length(grid_x)
  ny <- length(grid_y)
  n_query <- nrow(query_points)

  # Extract query coordinates
  xi <- query_points[, 1]
  yi <- query_points[, 2]

  # Find bracketing intervals (NO clamping of coordinates)
  ix <- findInterval(xi, grid_x)
  iy <- findInterval(yi, grid_y)

  # Handle boundary cases for extrapolation
  # idx = 0 means below grid, use first segment
  # idx = n means at/above last point, use last segment
  ix[ix == 0] <- 1
  ix[ix == nx] <- nx - 1
  iy[iy == 0] <- 1
  iy[iy == ny] <- ny - 1

  # Get grid bounds
  x1 <- grid_x[ix]
  x2 <- grid_x[ix + 1]
  y1 <- grid_y[iy]
  y2 <- grid_y[iy + 1]

  # Compute interpolation weights (can be < 0 or > 1 for extrapolation)
  tx <- (xi - x1) / (x2 - x1)
  ty <- (yi - y1) / (y2 - y1)

  # Get corner values using linear indexing
  # R stores matrices in column-major order: linear index = i + (j-1)*nrow
  idx_11 <- ix + (iy - 1) * nx
  idx_21 <- (ix + 1) + (iy - 1) * nx
  idx_12 <- ix + iy * nx
  idx_22 <- (ix + 1) + iy * nx

  z11 <- values[idx_11]
  z21 <- values[idx_21]
  z12 <- values[idx_12]
  z22 <- values[idx_22]

  # Bilinear interpolation formula
  result <- (1 - tx) * (1 - ty) * z11 +
            tx * (1 - ty) * z21 +
            (1 - tx) * ty * z12 +
            tx * ty * z22

  return(result)
}

#' 3D Trilinear Interpolation with Linear Extrapolation
#'
#' Exactly matches MATLAB's griddedInterpolant({x1,x2,x3}, V, 'linear', 'linear')
#' Vectorized for performance.
#'
#' @param grid_x Grid vector for dimension 1 (sorted)
#' @param grid_y Grid vector for dimension 2 (sorted)
#' @param grid_z Grid vector for dimension 3 (sorted)
#' @param values Values at grid points (nx x ny x nz array)
#' @param query_points Query points (n_query x 3 matrix)
#' @return Interpolated values at query points (vector of length n_query)
#' @keywords internal
interp3_trilinear <- function(grid_x, grid_y, grid_z, values, query_points) {
  nx <- length(grid_x)
  ny <- length(grid_y)
  nz <- length(grid_z)
  n_query <- nrow(query_points)

  # Extract query coordinates
  xi <- query_points[, 1]
  yi <- query_points[, 2]
  zi <- query_points[, 3]

  # Find bracketing intervals (NO clamping of coordinates)
  ix <- findInterval(xi, grid_x)
  iy <- findInterval(yi, grid_y)
  iz <- findInterval(zi, grid_z)

  # Handle boundary cases for extrapolation
  ix[ix == 0] <- 1
  ix[ix == nx] <- nx - 1
  iy[iy == 0] <- 1
  iy[iy == ny] <- ny - 1
  iz[iz == 0] <- 1
  iz[iz == nz] <- nz - 1

  # Get grid bounds
  x1 <- grid_x[ix]
  x2 <- grid_x[ix + 1]
  y1 <- grid_y[iy]
  y2 <- grid_y[iy + 1]
  z1 <- grid_z[iz]
  z2 <- grid_z[iz + 1]

  # Compute interpolation weights (can be < 0 or > 1 for extrapolation)
  tx <- (xi - x1) / (x2 - x1)
  ty <- (yi - y1) / (y2 - y1)
  tz <- (zi - z1) / (z2 - z1)

  # Get corner values using linear indexing
  # R stores arrays in column-major order
  # linear index = i + (j-1)*nx + (k-1)*nx*ny
  stride_y <- nx
  stride_z <- nx * ny

  idx_111 <- ix +       (iy - 1) * stride_y + (iz - 1) * stride_z
  idx_211 <- (ix + 1) + (iy - 1) * stride_y + (iz - 1) * stride_z
  idx_121 <- ix +       iy * stride_y +       (iz - 1) * stride_z
  idx_221 <- (ix + 1) + iy * stride_y +       (iz - 1) * stride_z
  idx_112 <- ix +       (iy - 1) * stride_y + iz * stride_z
  idx_212 <- (ix + 1) + (iy - 1) * stride_y + iz * stride_z
  idx_122 <- ix +       iy * stride_y +       iz * stride_z
  idx_222 <- (ix + 1) + iy * stride_y +       iz * stride_z

  z111 <- values[idx_111]
  z211 <- values[idx_211]
  z121 <- values[idx_121]
  z221 <- values[idx_221]
  z112 <- values[idx_112]
  z212 <- values[idx_212]
  z122 <- values[idx_122]
  z222 <- values[idx_222]

  # Trilinear interpolation formula
  result <- (1 - tx) * (1 - ty) * (1 - tz) * z111 +
            tx * (1 - ty) * (1 - tz) * z211 +
            (1 - tx) * ty * (1 - tz) * z121 +
            tx * ty * (1 - tz) * z221 +
            (1 - tx) * (1 - ty) * tz * z112 +
            tx * (1 - ty) * tz * z212 +
            (1 - tx) * ty * tz * z122 +
            tx * ty * tz * z222

  return(result)
}

#' General N-D Multilinear Interpolation
#'
#' Matches MATLAB's griddedInterpolant for any dimension.
#' Uses tensor product of 1D linear interpolations.
#'
#' @param grid_list List of grid vectors (one per dimension)
#' @param values N-dimensional array of values
#' @param query_points Query points matrix (n_query x ndim)
#' @return Interpolated values at query points
#' @keywords internal
interp_multilinear <- function(grid_list, values, query_points) {
  ndim <- length(grid_list)

  if (ndim == 1) {
    return(interp1_linear_vec(grid_list[[1]], as.vector(values), query_points[, 1]))
  } else if (ndim == 2) {
    return(interp2_bilinear(grid_list[[1]], grid_list[[2]], values, query_points))
  } else if (ndim == 3) {
    return(interp3_trilinear(grid_list[[1]], grid_list[[2]], grid_list[[3]], values, query_points))
  } else {
    # For ndim > 3, use recursive tensor product approach
    return(interp_nd_multilinear(grid_list, values, query_points))
  }
}

#' Vectorized N-D Multilinear Interpolation (O(N) complexity)
#'
#' Fully vectorized implementation for arbitrary dimensions.
#' Uses tensor product approach with linear indexing - NO for-loops over query points.
#'
#' Algorithm:
#'   1. Find bracketing indices and weights for ALL query points in each dimension
#'   2. Compute linear indices for ALL 2^ndim corners of ALL query points
#'   3. Sum weighted corner values (tensor product)
#'
#' Complexity: O(N * 2^ndim) where N is number of query points
#' For ndim <= 10, this is effectively O(N) since 2^10 = 1024 is constant
#'
#' @param grid_list List of grid vectors (one per dimension)
#' @param values N-dimensional array
#' @param query_points Query points matrix (n_query x ndim)
#' @return Interpolated values (vector of length n_query)
#' @keywords internal
interp_nd_multilinear <- function(grid_list, values, query_points) {
  ndim <- length(grid_list)
  n_query <- nrow(query_points)
  dims <- sapply(grid_list, length)

  # Step 1: Find bracketing indices and weights for ALL query points
  # idx_lo[i, d] = lower bracket index for query i, dimension d
  # weights[i, d] = interpolation weight for query i, dimension d
  idx_lo <- matrix(0L, nrow = n_query, ncol = ndim)
  idx_hi <- matrix(0L, nrow = n_query, ncol = ndim)
  weights <- matrix(0, nrow = n_query, ncol = ndim)

  for (d in seq_len(ndim)) {
    grid <- grid_list[[d]]
    n <- length(grid)
    x <- query_points[, d]

    # Find bracketing intervals for ALL query points at once
    idx <- findInterval(x, grid)

    # Handle boundaries for extrapolation
    idx[idx == 0] <- 1
    idx[idx == n] <- n - 1

    idx_lo[, d] <- idx
    idx_hi[, d] <- idx + 1

    # Compute weights (can be < 0 or > 1 for extrapolation)
    weights[, d] <- (x - grid[idx]) / (grid[idx + 1] - grid[idx])
  }

  # Step 2: Compute strides for linear indexing
  # R uses column-major order: linear_idx = i1 + (i2-1)*n1 + (i3-1)*n1*n2 + ...
  strides <- c(1, cumprod(dims[-ndim]))  # [1, n1, n1*n2, ...]

  # Step 3: Sum over all 2^ndim corners using tensor product
  result <- numeric(n_query)
  n_corners <- 2^ndim

  for (corner in 0:(n_corners - 1)) {
    # Compute corner weight and indices for ALL query points at once
    corner_weight <- rep(1, n_query)
    linear_idx <- rep(1L, n_query)  # Start with 1 (R is 1-indexed)

    for (d in seq_len(ndim)) {
      # Check if this corner uses high index in dimension d
      use_hi <- bitwAnd(corner, 2^(d - 1)) > 0

      if (use_hi) {
        corner_weight <- corner_weight * weights[, d]
        # Add (idx_hi - 1) * stride for this dimension
        linear_idx <- linear_idx + (idx_hi[, d] - 1L) * strides[d]
      } else {
        corner_weight <- corner_weight * (1 - weights[, d])
        # Add (idx_lo - 1) * stride for this dimension
        linear_idx <- linear_idx + (idx_lo[, d] - 1L) * strides[d]
      }
    }

    # Add weighted corner values to result
    result <- result + corner_weight * values[linear_idx]
  }

  return(result)
}

#' Legacy: Loop-based N-D Interpolation (DEPRECATED)
#'
#' Kept for reference and testing. Use interp_nd_multilinear instead.
#' This version is O(N²) and should not be used for large N.
#'
#' @keywords internal
interp_nd_multilinear_loop <- function(grid_list, values, query_points) {
  ndim <- length(grid_list)
  n_query <- nrow(query_points)
  result <- numeric(n_query)

  for (i in seq_len(n_query)) {
    result[i] <- interp_point_multilinear(grid_list, values, query_points[i, ])
  }

  return(result)
}

#' Single Point Multilinear Interpolation
#'
#' @param grid_list List of grid vectors
#' @param values N-dimensional array
#' @param point Single query point (vector of length ndim)
#' @return Interpolated value
#' @keywords internal
interp_point_multilinear <- function(grid_list, values, point) {
  ndim <- length(grid_list)
  dims <- sapply(grid_list, length)

  # Find bracketing indices and weights for each dimension
  idx_lo <- integer(ndim)
  idx_hi <- integer(ndim)
  weights <- numeric(ndim)

  for (d in seq_len(ndim)) {
    grid <- grid_list[[d]]
    n <- length(grid)
    x <- point[d]

    idx <- findInterval(x, grid)

    # Handle boundaries for extrapolation
    if (idx == 0) idx <- 1
    if (idx == n) idx <- n - 1

    idx_lo[d] <- idx
    idx_hi[d] <- idx + 1
    weights[d] <- (x - grid[idx]) / (grid[idx + 1] - grid[idx])
  }

  # Tensor product: sum over all 2^ndim corners
  result <- 0
  for (corner in 0:(2^ndim - 1)) {
    w <- 1
    idx <- integer(ndim)
    for (d in seq_len(ndim)) {
      if (bitwAnd(corner, 2^(d - 1)) > 0) {
        idx[d] <- idx_hi[d]
        w <- w * weights[d]
      } else {
        idx[d] <- idx_lo[d]
        w <- w * (1 - weights[d])
      }
    }
    # Get value at this corner using dynamic indexing
    val <- do.call(`[`, c(list(values), as.list(idx)))
    result <- result + w * val
  }

  return(result)
}
