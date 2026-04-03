# Copyright (c) 2024-2025 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later

#' Local polynomial regression with adaptive regularization
#'
#' Computes the regression estimate using precomputed density (design matrix).
#' For higher-order polynomials (order >= 1), applies adaptive regularization
#' based on the maximum diagonal element to prevent instability in sparse regions.
#'
#' Mirrors: fastLPR/utility/core/fastLPR_reg.m
#'
#' @param regs Regression structure (must have regs$s precomputed)
#' @param y_ft Response data in Fourier domain
#'
#' @return mq Regression estimate on grid
#' @keywords internal
fastlpr_reg <- function(regs, y_ft) {
  if (!("s" %in% names(regs))) {
    stop("fastlpr_reg: need precomputed density (regs$s)")
  }

  if (regs$opt$order == 0) {
    # Order 0: Nadaraya-Watson estimator
    # mq = sum(K * y) / sum(K)

    # Compute T = conv(K, y) using NUFFT convolution (matches MATLAB)
    t <- fastlpr_conv(regs, regs$kdf, y_ft, flag_transformed = FALSE)

    # Ensure s and t have compatible dimensions
    # s dims: (spatial_dims, dh, 1)  - singleton for single "response" (density)
    # t dims: (spatial_dims, dh, dy) - multiple responses for DOF computation
    # Need to broadcast s to match t for element-wise division

    s <- regs$s
    # Use abs() for complex safety:
    # - s represents kernel sums (always real positive in theory)
    # - R may store as complex type due to computation pathway
    # - abs() handles both real and complex correctly
    # - For real positive s: abs(s) == s
    # - MATLAB's max() on complex returns element with largest magnitude
    s_threshold <- max(abs(s)) * 1e-6
    s_safe <- pmax(abs(s), s_threshold)  # Real positive denominator

    # Manual broadcasting for division: t / s
    # Both arrays have same spatial dimensions and dh dimension
    # s has singleton last dim (dy=1), t has dy dimension
    # R's division should broadcast automatically, but let's be explicit

    t_dims <- dim(t)
    s_dims <- dim(s)

    if (length(s_dims) == length(t_dims) && s_dims[length(s_dims)] == 1 && t_dims[length(t_dims)] > 1) {
      # Case: s is (..., dh, 1), t is (..., dh, dy)
      # Use sweep to divide along the last dimension
      dy <- t_dims[length(t_dims)]

      # Reshape for sweep: flatten all but last dimension
      total_size <- prod(s_dims[-length(s_dims)])
      s_flat <- matrix(s_safe, nrow = total_size, ncol = 1)
      t_flat <- matrix(t, nrow = total_size, ncol = dy)

      # Divide each column of t by s
      mq_flat <- sweep(t_flat, 1, s_flat[, 1], `/`)

      # Reshape back to original dimensions
      mq <- array(mq_flat, dim = t_dims)

    } else {
      # Simpler case: direct division works
      mq <- t / s_safe
    }

  } else {
    # Order >= 1: Local polynomial regression
    # Solve: S * beta = T, where S is the design matrix

    # Compute convolutions T = sum(K * X^j * y)
    t <- list()
    for (i in regs$lwp$nt:1) {
      # For each polynomial term, compute T_i = conv(K_i, y)
      # kdf[[i]] contains the kernel for the i-th polynomial term
      t_i <- fastlpr_conv(regs, regs$kdf[[i]], y_ft, flag_transformed = FALSE)

      # t_i has shape: (L1, L2, ..., dh, dy) for multi-D
      # mfun expects: (L, dh, dy) where L is spatial grid product
      # Need to flatten spatial dimensions for mfun

      t_i_dims <- dim(t_i)
      if (length(t_i_dims) == 3 && t_i_dims[3] == 1) {
        # Special case: single response, single bandwidth -> remove singleton
        t[[i]] <- t_i[, , 1]
      } else if (regs$dx > 1) {
        # Multi-dimensional: reshape (L1, L2, ..., dh, dy) -> (L, dh, dy)
        # Spatial dims: t_i_dims[1:dx]
        # Other dims: t_i_dims[(dx+1):end] = [dh, dy]
        L_total <- prod(t_i_dims[1:regs$dx])
        other_dims <- t_i_dims[(regs$dx + 1):length(t_i_dims)]
        t[[i]] <- array(t_i, dim = c(L_total, other_dims))
      } else {
        # 1D case: keep as is
        t[[i]] <- t_i
      }
    }

    # Apply adaptive regularization to design matrix S
    s_reg <- apply_adaptive_regularization_r(regs)

    # Remove trailing singleton dimensions from S elements (for 1D case)
    # This handles the case where S has shape (N, dh, 1) due to DOF computation
    # but T has shape (N, dh) because the singleton was removed
    if (regs$dx == 1) {
      s_squeezed <- list()
      for (i in seq_along(s_reg)) {
        s_i <- s_reg[[i]]
        s_i_dims <- dim(s_i)

        # Check if has trailing singleton: (N, dh, 1) -> (N, dh)
        if (length(s_i_dims) == 3 && s_i_dims[3] == 1) {
          s_squeezed[[i]] <- s_i[, , 1]  # Remove singleton third dimension
        } else {
          s_squeezed[[i]] <- s_i
        }
      }
      s_reg <- s_squeezed
    }

    # Flatten S elements for multi-dimensional grids (same as T flattening above)
    if (regs$dx > 1) {
      s_flat <- list()
      for (i in seq_along(s_reg)) {
        s_i <- s_reg[[i]]
        s_i_dims <- dim(s_i)
        
        # Flatten spatial dimensions: (L1, L2, ..., dh, ...) -> (L, dh, ...)
        L_total <- prod(s_i_dims[1:regs$dx])
        if (length(s_i_dims) > regs$dx) {
          # Has extra dimensions after spatial
          other_dims <- s_i_dims[(regs$dx + 1):length(s_i_dims)]
          # Remove trailing singletons (e.g., (dh, 1) -> (dh,))
          while (length(other_dims) > 1 && other_dims[length(other_dims)] == 1) {
            other_dims <- other_dims[-length(other_dims)]
          }
          s_flat[[i]] <- array(s_i, dim = c(L_total, other_dims))
        } else {
          # No extra dimensions
          s_flat[[i]] <- array(s_i, dim = c(L_total))
        }
      }
      s_reg <- s_flat
    }

    # Solve for regression coefficients using mfun
    mq_flat <- regs$lwp$mfun(s_reg, t)


    # Reshape output back to multi-dimensional grid
    # mfun returns various shapes depending on input:
    # - Multi-sample: (L, dh, dy)
    # - Single response, multiple bandwidths: (L, dh)
    # - Single response, single bandwidth: (L,) vector
    # Need to reshape to: (L1, L2, ..., dh, dy) or (L1, L2, ..., dh) or (L1, L2, ...)

    if (regs$dx > 1) {
      mq_dims <- dim(mq_flat)
      spatial_dims <- if (length(regs$N) > 1) regs$N else rep(regs$N, regs$dx)

      if (is.null(mq_dims)) {
        # mq_flat is a vector (no dimensions) - need to determine intended shape
        # Check if it's (n_grid,) or should be (n_grid, dh)
        L_total <- prod(spatial_dims)
        dh <- nrow(regs$h)

        if (length(mq_flat) == L_total) {
          # Single bandwidth: (L,) -> (L1, L2, ...)
          mq <- array(mq_flat, dim = spatial_dims)
        } else if (length(mq_flat) == L_total * dh) {
          # Multiple bandwidths: (L*dh,) -> (L1, L2, ..., dh)
          mq <- array(mq_flat, dim = c(spatial_dims, dh))
        } else {
          warning(sprintf("fastlpr_reg: Cannot reshape vector of length %d to expected shape",
                         length(mq_flat)))
          mq <- mq_flat
        }
      } else if (length(mq_dims) == 1) {
        # 1D array - same as vector case
        L_total <- prod(spatial_dims)
        dh <- nrow(regs$h)

        if (mq_dims[1] == L_total) {
          mq <- array(mq_flat, dim = spatial_dims)
        } else if (mq_dims[1] == L_total * dh) {
          mq <- array(mq_flat, dim = c(spatial_dims, dh))
        } else {
          warning(sprintf("fastlpr_reg: Cannot reshape 1D array of length %d",
                         mq_dims[1]))
          mq <- mq_flat
        }
      } else if (length(mq_dims) == 2) {
        # Shape: (L, dh) -> (L1, L2, ..., dh)
        mq <- array(mq_flat, dim = c(spatial_dims, mq_dims[2]))
      } else if (length(mq_dims) == 3) {
        # Shape: (L, dh, dy) -> (L1, L2, ..., dh, dy)
        mq <- array(mq_flat, dim = c(spatial_dims, mq_dims[2], mq_dims[3]))
      } else {
        # Unexpected shape - keep as is
        warning(sprintf("fastlpr_reg: Unexpected mq_flat shape (%s), keeping as is",
                       paste(mq_dims, collapse = ", ")))
        mq <- mq_flat
      }
    } else {
      # 1D: no reshaping needed
      mq <- mq_flat
    }
  }
  
  return(mq)
}

#' Apply adaptive regularization to design matrix
#'
#' Adds regularization only where needed based on maximum diagonal element.
#' This ensures stability without suppressing the fit in high-signal areas.
#'
#' @param regs Regression structure
#' @return s_reg Regularized design matrix
#' @keywords internal
apply_adaptive_regularization_r <- function(regs) {
  # CRITICAL: MATLAB uses order+1 for regularization, NOT lwp$nt
  # This applies regularization to the first (order+1) diagonal elements only
  # For 2D order 2: lwp$nt=6 (polynomial terms), but MATLAB uses nt=3 (order+1)
  # This is intentional - only regularize lower-order term diagonals
  nt <- regs$opt$order + 1  # Match MATLAB: nt = order + 1

  # Find the maximum diagonal element across all spatial points
  max_diag <- max(abs(regs$s[[1]]))  # Start with S11
  for (k in 2:nt) {
    diag_idx <- k * (k + 1) / 2  # Diagonal index in lower triangular storage
    max_diag <- max(max_diag, max(abs(regs$s[[diag_idx]])))
  }

  # Add a small, fixed regularization relative to the max diagonal
  # alpha = 1e-6: regularize at 0.0001% of maximum signal
  alpha <- 1e-6
  lambda_fixed <- alpha * max_diag

  # Apply regularization to diagonal elements: S_reg = S + lambda*I
  # OPTIMIZATION: Deep copy to avoid R's copy-on-modify overhead
  # When s_reg shares references with regs$s, subsequent mfun calls
  # trigger expensive copy operations. By copying upfront, we avoid this.
  s_reg <- lapply(regs$s, function(x) x + 0)  # Force copy via arithmetic
  for (k in 1:nt) {
    diag_idx <- k * (k + 1) / 2
    s_reg[[diag_idx]] <- s_reg[[diag_idx]] + lambda_fixed
  }

  return(s_reg)
}

