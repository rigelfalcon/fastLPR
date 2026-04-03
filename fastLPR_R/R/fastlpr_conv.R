# Copyright (c) 2024-2025 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later

#' Fast convolution using NUFFT for scattered data
#'
#' Computes convolution of kernel with data using NUFFT (Non-Uniform FFT).
#' This is the core operation for local polynomial regression, enabling
#' O(N + M log M) complexity instead of naive O(N*M).
#'
#' Mirrors: fastLPR/utility/core/fastLPR_conv.m
#'
#' @param regs Regression structure from fastlpr_create
#' @param kdf Kernel density function in Fourier domain (optional for direct mode)
#' @param y Data to convolve
#' @param flag_transformed Whether y is already Fourier-transformed (optional)
#' @param add_eps Whether to add eps for numerical stability (default: FALSE)
#'
#' @return Convolution result on evaluation grid
#' @keywords internal
#'
#' @details
#' This is a simplified implementation that uses direct computation for now.
#' Full NUFFT-based implementation will be added later for performance.
#'
fastlpr_conv <- function(regs, kdf = NULL, y = NULL, flag_transformed = NULL, add_eps = FALSE) {
  # Handle two calling modes:
  # 1. Old mode: fastlpr_conv(regs) - computes regs$t from regs$y
  # 2. New mode: fastlpr_conv(regs, kdf, y) - computes convolution of kdf with y

  if (is.null(kdf) && is.null(y)) {
    # Old mode: compute T = conv(K, y) and store in regs$t
    return(fastlpr_conv_old_mode(regs))
  }

  # Get dimensions
  sz <- dim(y)
  if (is.null(sz)) {
    dy <- 1
    Ny <- length(y)
    y <- matrix(y, ncol = 1)
  } else {
    dy <- sz[length(sz)]
    Ny <- sz[1]
  }

  # Determine if y is already Fourier-transformed
  if (is.null(flag_transformed)) {
    # FIX 2025-12-18 v2: Robust detection of whether y is already transformed
    #
    # Key insight: Raw data y ALWAYS has shape (Tx, dy) where:
    #   - Tx = number of samples (can be ANY value)
    #   - dy = number of response variables
    #   - ndim(y) = 2
    #
    # Transformed data y has shape matching the padded grid:
    #   - 1D: (L,) or (L, dh) or (L, dh, dy) -> ndim >= 1, first dim = L
    #   - 2D: (L1, L2) or (L1, L2, dh) or (L1, L2, dh, dy) -> ndim >= 2, first 2 dims = L
    #   - 3D: (L1, L2, L3, ...) -> ndim >= 3, first 3 dims = L
    #
    # The critical distinction: transformed data has ndim >= dx+1 (or exactly dx for simple case)
    # AND its first dx dimensions match L exactly.
    #
    # Raw data (Tx, dy) is ALWAYS 2D, regardless of dx.
    # So for dx >= 2, if y is 2D, it's definitely raw data (not transformed).
    # For dx == 1, we need to check: if y is 2D with shape (Tx, dy) where Tx could equal L,
    # we check if Tx == regs$Tx (the actual sample count) to disambiguate.

    y_dims <- dim(y)
    n_y_dims <- length(y_dims)

    if (regs$dx == 1) {
      # 1D case: L is a scalar
      L_val <- if (length(regs$L) > 1) regs$L[1] else regs$L

      if (n_y_dims == 2 && y_dims[1] == regs$Tx) {
        # y has shape (Tx, dy) - this is RAW data, even if Tx == L
        flag_transformed <- FALSE
      } else if (y_dims[1] == L_val) {
        # First dim matches L and it's NOT (Tx, dy) shape -> transformed
        flag_transformed <- TRUE
        if (n_y_dims > 1) {
          dy <- y_dims[n_y_dims]
        }
      } else {
        # First dim doesn't match L -> raw data
        flag_transformed <- FALSE
      }
    } else {
      # Multi-D case (dx >= 2)
      # Raw y always has shape (Tx, dy) = 2 dims
      # Transformed y has >= dx dims with first dx dims matching L

      if (n_y_dims == 2) {
        # 2D array with multi-D regression -> this is raw data (Tx, dy)
        flag_transformed <- FALSE
      } else if (n_y_dims >= regs$dx && all(y_dims[1:regs$dx] == regs$L)) {
        # First dx dims match L -> transformed data
        flag_transformed <- TRUE
        if (n_y_dims > regs$dx) {
          dy <- y_dims[n_y_dims]
        }
      } else {
        # Shape doesn't match L -> raw data
        flag_transformed <- FALSE
      }
    }
  }

  # Convolution in Fourier domain
  # By convolution theorem: conv(f, g) = ifft(fft(f) .* fft(g))
  # Here: kdf = fft(kernel), y_ft = fft(data)

  if (!flag_transformed) {
    # Transform y to Fourier domain using NUFFT
    y_ft_raw <- fastlpr_nufft(regs, y)

    # Handle broadcasting for multiple bandwidths and response variables
    # kdf shape: (L1, L2, ..., dh) where dh is number of bandwidths
    # y_ft shape: (L1, L2, ..., dy) where dy is number of response variables
    # Result shape: (L1, L2, ..., dh, dy)

    # Extract spatial grid size L (for convenience in broadcasting code)
    L <- if (length(regs$L) > 1) regs$L[1] else regs$L

    # Determine number of bandwidths (dh) from kdf
    kdf_dims <- dim(kdf)
    if (length(kdf_dims) == regs$dx) {
      dh <- 1
    } else {
      dh <- kdf_dims[length(kdf_dims)]
    }

    # Determine number of responses (dy) from y_ft
    y_ft_dims <- dim(y_ft_raw)
    if (length(y_ft_dims) == regs$dx) {
      dy <- 1
    } else {
      dy <- y_ft_dims[length(y_ft_dims)]
    }

    target_dims <- c(kdf_dims[1:regs$dx], dh, dy)

    # OPTIMIZATION v3.1: Use dimension-agnostic Rcpp convolution pipeline
    # Handles ANY dimension (1D, 2D, 3D, etc.) like MATLAB's code:
    #   for ix = regs.dx:-1:1; m = ifft(m, [], ix); end
    # FIX 2025-12-14: Use get0() instead of exists() because exists() doesn't
    # reliably find functions in the package namespace when called from within
    rcpp_conv_nd_fn <- tryCatch({
      fn <- get0("rcpp_conv_nd_full", envir = globalenv(), inherits = FALSE)
      if (is.null(fn)) fn <- get0("rcpp_conv_nd_full", envir = asNamespace("fastlpr"), inherits = FALSE)
      fn
    }, error = function(e) NULL)
    rcpp_conv_nd_available <- is.function(rcpp_conv_nd_fn)
    
    use_rcpp_full <- FALSE
    m <- NULL
    
    if (rcpp_conv_nd_available) {
      # FAST N-D: Full Rcpp pipeline (broadcast + IFFT + extract) for ANY dimension
      m <- tryCatch({
        # Build qout matrix (2 x dx) with 1-based R indices
        if (regs$dx == 1 && is.null(dim(regs$qout))) {
          # 1D: qout is vector [start, end] -> convert to 2x1 matrix
          qout_mat <- matrix(as.integer(regs$qout), nrow = 2, ncol = 1)
        } else {
          # Multi-D: qout is already 2xdx matrix
          qout_mat <- matrix(as.integer(regs$qout), nrow = 2, ncol = regs$dx)
        }

        result <- rcpp_conv_nd_fn(
          as.complex(kdf), as.complex(y_ft_raw),
          as.integer(regs$L), as.integer(dh), as.integer(dy),
          qout_mat, regs$y_isreal
        )
        use_rcpp_full <- TRUE
        
        # Convert to real if y_isreal (Rcpp returns ComplexVector always)
        if (regs$y_isreal) result <- Re(result)
        
        # Reshape to (N1, N2, ..., dh, dy)
        array(result, dim = c(regs$N, dh, dy))
      }, error = function(e) {
        # Fall through to standard path
        NULL
      })
    }
    
    # If Rcpp full pipeline succeeded, skip the rest of convolution processing
    if (use_rcpp_full && !is.null(m)) {
      return(m)
    }
    
    # FALLBACK: Standard path with separate broadcast + IFFT + extract

    # Use complex type since NUFFT output is complex
    kdf_broadcast <- array(0+0i, dim = target_dims)
    y_ft_broadcast <- array(0+0i, dim = target_dims)

    # OPTIMIZATION v2.1: Use Rcpp broadcast multiply with OpenMP if available
    # FIX 2025-12-14: Use get0() instead of exists() because exists() doesn't
    # reliably find functions in the package namespace when called from within
    rcpp_broadcast_1d_fn <- tryCatch({
      fn <- get0("rcpp_broadcast_multiply", envir = globalenv(), inherits = FALSE)
      if (is.null(fn)) fn <- get0("rcpp_broadcast_multiply", envir = asNamespace("fastlpr"), inherits = FALSE)
      fn
    }, error = function(e) NULL)
    rcpp_broadcast_1d <- is.function(rcpp_broadcast_1d_fn)

    rcpp_broadcast_nd_fn <- tryCatch({
      fn <- get0("rcpp_broadcast_multiply_nd", envir = globalenv(), inherits = FALSE)
      if (is.null(fn)) fn <- get0("rcpp_broadcast_multiply_nd", envir = asNamespace("fastlpr"), inherits = FALSE)
      fn
    }, error = function(e) NULL)
    rcpp_broadcast_nd <- is.function(rcpp_broadcast_nd_fn)

    if (regs$dx == 1 && rcpp_broadcast_1d) {
      m_ft <- tryCatch({
        rcpp_broadcast_1d_fn(as.complex(kdf), as.complex(y_ft_raw), L, dh, dy)
      }, error = function(e) NULL)
    } else if (regs$dx >= 2 && rcpp_broadcast_nd) {
      m_ft <- tryCatch({
        L_spatial <- prod(kdf_dims[1:regs$dx])
        kdf_flat <- matrix(as.complex(kdf), nrow = L_spatial, ncol = dh)
        y_ft_flat <- matrix(as.complex(y_ft_raw), nrow = L_spatial, ncol = dy)
        result <- rcpp_broadcast_nd_fn(
          as.complex(kdf_flat), as.complex(y_ft_flat),
          as.integer(kdf_dims[1:regs$dx]), as.integer(dh), as.integer(dy)
        )
        array(result, dim = target_dims)
      }, error = function(e) NULL)
    } else {
      m_ft <- NULL
    }
    
    if (is.null(m_ft)) {
      # FALLBACK: Pure R implementation - check if allowed
      require_rcpp_or_allow_fallback("fastlpr_conv broadcast multiply")
      if (regs$dx == 1) {
        kdf_broadcast <- array(rep(kdf, dy), dim = c(L, dh, dy))
        y_ft_broadcast <- aperm(array(rep(y_ft_raw, dh), dim = c(L, dy, dh)), c(1, 3, 2))
      } else {
        L_spatial <- prod(kdf_dims[1:regs$dx])
        kdf_flat <- matrix(kdf, nrow = L_spatial, ncol = dh)
        y_ft_flat <- matrix(y_ft_raw, nrow = L_spatial, ncol = dy)
        kdf_broadcast_flat <- array(rep(kdf_flat, dy), dim = c(L_spatial, dh, dy))
        y_ft_broadcast_flat <- aperm(array(rep(y_ft_flat, dh), dim = c(L_spatial, dy, dh)), c(1, 3, 2))
        kdf_broadcast <- array(kdf_broadcast_flat, dim = target_dims)
        y_ft_broadcast <- array(y_ft_broadcast_flat, dim = target_dims)
      }
      m_ft <- kdf_broadcast * y_ft_broadcast
    }

    # Reshape for inverse FFT
    spatial_dims <- target_dims[1:regs$dx]
    m <- array(m_ft, dim = c(spatial_dims, dh * dy))
  } else {
    # y is already in Fourier domain
    if (!regs$opt$y_corr_bandwidth) {
      # FIX 2025-12-18: Handle 2D y (L, dy) properly for ALL dimensions
      # When flag_transformed=TRUE (Ny == L), y has shape (L, dy) not (L, dh, dy)
      # This happens when regs$Tx == regs$L (sample size equals padded grid size)
      y_dims <- dim(y)
      n_y_dims <- length(y_dims)

      if (regs$dx == 1) {
        # 1D case
        if (n_y_dims == 2) {
          # y is (L, dy), kdf is (L, dh)
          L <- y_dims[1]
          dy_local <- y_dims[2]
          kdf_dims <- dim(kdf)
          dh_local <- if (is.null(kdf_dims) || length(kdf_dims) < 2) 1 else kdf_dims[2]

          # Broadcast: kdf (L, dh) * y (L, dy) -> (L, dh, dy)
          if (dh_local == 1 && dy_local == 1) {
            m_ft <- kdf * y
          } else {
            kdf_3d <- array(kdf, dim = c(L, dh_local, 1))
            y_3d <- array(y, dim = c(L, 1, dy_local))
            m_ft <- kdf_3d * y_3d
          }
        } else if (n_y_dims == 3) {
          # y is already 3D (L, dh, dy) - original code path
          m_ft <- kdf * aperm(y, c(1, 3, 2))
        } else {
          # y is a vector
          m_ft <- kdf * y
        }
      } else {
        # Multi-D case (dx >= 2)
        # Expected y shape: (L1, L2, ..., Ldx, dh, dy) = (dx + 2) dimensions
        # But when flag_transformed=TRUE due to Ny==L, y is (L1, L2, ..., Ldx, dy) = (dx + 1) dimensions
        expected_dims <- regs$dx + 2  # spatial + dh + dy

        if (n_y_dims == regs$dx + 1) {
          # y is (L1, ..., Ldx, dy) - missing dh dimension
          # kdf is (L1, ..., Ldx, dh)
          # Need to broadcast: kdf * y -> (L1, ..., Ldx, dh, dy)
          L_spatial <- prod(y_dims[1:regs$dx])
          dy_local <- y_dims[regs$dx + 1]
          kdf_dims <- dim(kdf)
          dh_local <- if (length(kdf_dims) == regs$dx) 1 else kdf_dims[regs$dx + 1]

          if (dh_local == 1 && dy_local == 1) {
            # Simple case: just element-wise multiply
            m_ft <- kdf * y
          } else {
            # Reshape to broadcast
            # kdf: (L_spatial, dh) -> (L_spatial, dh, 1)
            # y:   (L_spatial, dy) -> (L_spatial, 1, dy)
            kdf_flat <- array(kdf, dim = c(L_spatial, dh_local, 1))
            y_flat <- array(y, dim = c(L_spatial, 1, dy_local))
            m_ft <- kdf_flat * y_flat  # (L_spatial, dh, dy)
            # Reshape back to spatial dimensions (may avoid a copy when unshared)
            dim(m_ft) <- c(y_dims[1:regs$dx], dh_local, dy_local)
          }
        } else if (n_y_dims == expected_dims) {
          # y is already (L1, ..., Ldx, dh, dy) - original code path
          perm <- c(1:regs$dx, regs$dx + 2, regs$dx + 1)
          m_ft <- kdf * aperm(y, perm)
        } else {
          stop(sprintf("fastlpr_conv: unexpected y dimensions %d (expected %d or %d)",
                       n_y_dims, regs$dx + 1, expected_dims))
        }
      }
      if (regs$dx == 1) {
        m <- array(m_ft, dim = c(regs$L, regs$dh * dy))
      } else {
        m <- array(m_ft, dim = c(regs$L, regs$dh * dy))
      }
    } else {
      # Special case: different bandwidth for each response variable
      m_ft <- kdf * array(y, dim = c(regs$L, regs$dh, dy / regs$dh))
      m <- array(m_ft, dim = c(regs$L, dy))
    }
  }

  # Inverse FFT to spatial domain
  # OPTIMIZATION: Use apply_fft_spatial which uses R's native fft() on slices
  # This avoids expensive aperm calls (2.9x faster for 3D KDE!)
  # m has shape (L1, L2, ..., dh*dy) where last dim is combined bandwidth*response
  m <- apply_fft_spatial(m, regs$dx, inverse = TRUE)

  # Reshape result (may avoid a copy when unshared)
  if (!flag_transformed || !regs$opt$y_corr_bandwidth) {
    dim(m) <- c(regs$L, regs$dh, dy)
  } else {
    dim(m) <- c(regs$L, regs$dh, dy / regs$dh)
  }

  # For real-valued data, take real part (imaginary part is numerical noise)
  if (regs$y_isreal) {
    m <- Re(m)
  }

  # Extract evaluation grid points (remove padding)
  # m has shape (L1, L2, ..., dh, dy)
  # Extract N1:N2 range from each padded dimension L
  # NOTE (2025-11-28): Use qout (not qin) for OUTPUT extraction
  # MATLAB uses: idx_mq=get_patch_index(regs.qout,sz); m=m(idx_mq);
  # qin is for INPUT grid points, qout is for OUTPUT extraction after convolution
  indices <- list()
  for (d in 1:regs$dx) {
    # NOTE (2025-11-28): Handle 1D case where qout is a vector [start, end]
    # vs multi-D where qout is a matrix [2, dx]
    if (regs$dx == 1 && is.null(dim(regs$qout))) {
      # 1D: qout is c(start, end)
      indices[[d]] <- regs$qout[1]:regs$qout[2]
    } else {
      # Multi-D: qout is matrix with qout[1, d] = start, qout[2, d] = end
      indices[[d]] <- regs$qout[1, d]:regs$qout[2, d]
    }
  }
  # Keep all dh and dy dimensions
  indices[[regs$dx + 1]] <- quote(expr = )  # All dh
  indices[[regs$dx + 2]] <- quote(expr = )  # All dy

  m <- do.call(`[`, c(list(m), indices, list(drop = FALSE)))

  return(m)
}

#' Old mode: fastlpr_conv_old_mode
#' @keywords internal
fastlpr_conv_old_mode <- function(regs) {
  regs$t <- fastlpr_conv(regs, regs$kdf, regs$y, flag_transformed = FALSE)
  return(regs)
}

# NOTE: apply_fft_axis is defined in nufft.R - use that version (optimized)
