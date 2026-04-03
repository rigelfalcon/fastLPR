# Copyright (c) 2024-2025 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later

#' Apply multi-dimensional FFT to all spatial dimensions at once
#'
#' MATLAB-style vectorized FFT: applies FFT axis-by-axis across ALL bandwidths
#' at once using mvfft(), instead of looping over each bandwidth.
#'
#' Key insight from MATLAB's fastLPR_conv.m:
#'   for ix = regs.dx:-1:1
#'       m = ifft(m, [], ix);
#'   end
#' MATLAB's ifft works on all columns (bandwidths) simultaneously.
#'
#' @section IMPORTANT - DO NOT "OPTIMIZE" WITH RCPP FFT:
#' Benchmark results (2025-12-28, N=1M complex):
#'
#' | Backend            | Time    | Notes                              |
#' |--------------------|---------|-------------------------------------|
#' | R base fft()       | 30 ms   | Uses FFTPACK, no copy overhead      |
#' | Rcpp/Armadillo+FFTW3 | 38 ms | R<->C++ data copy kills performance |
#' | Python pyfftw      | 28 ms   | Reference (32 threads)              |
#'
#' The R<->C++ data copy overhead (~8ms) negates any FFTW3 advantage.
#' Rcpp FFT functions (rcpp_fft2d_batch, rcpp_fft3d_batch) exist in
#' fastlpr_rcpp.cpp but are SLOWER than base R for standalone FFT calls.
#'
#' They ARE still useful internally within C++ (e.g., rcpp_nufft_type1,
#' rcpp_conv_nd_full) where there's no R<->C++ boundary crossing.
#'
#' TL;DR: DO NOT add Rcpp FFT fallback here. Base R fft() is optimal.
#'
#' @param arr Input array (N1, N2, ..., Nd, dh) where last dim is bandwidth
#' @param dx Number of spatial dimensions to FFT (1, 2, or 3)
#' @param inverse Whether to apply inverse FFT
#'
#' @return Transformed array with same dimensions
#'
#' @keywords internal
apply_fft_spatial <- function(arr, dx, inverse = FALSE) {
  dims <- dim(arr)
  ndim <- length(dims)

  # If no bandwidth dimension (ndim == dx), just apply fft directly
  if (ndim == dx) {
    if (inverse) {
      return(fft(arr, inverse = TRUE) / prod(dims))
    } else {
      return(fft(arr))
    }
  }

  # Extract bandwidth dimension (last dimension)
  dh <- dims[ndim]
  spatial_dims <- dims[1:dx]
  prod_spatial <- prod(spatial_dims)

  # -------------------------------------------------------------------
  # 2D spatial + extra dims (3D+ array): apply fft() on each 2D slice

  # NOTE: We use base R fft() here, NOT Rcpp. See @section above for why.
  # -------------------------------------------------------------------
  if (dx == 2 && ndim >= 3) {
    n_extra <- prod(dims[(dx+1):ndim])
    dim(arr) <- c(spatial_dims, n_extra)
    result <- arr
    if (inverse) {
      for (ih in seq_len(n_extra)) {
        result[,,ih] <- fft(arr[,,ih], inverse = TRUE) / prod_spatial
      }
    } else {
      for (ih in seq_len(n_extra)) {
        result[,,ih] <- fft(arr[,,ih])
      }
    }
    dim(result) <- dims
    return(result)
  }

  # -------------------------------------------------------------------
  # 3D spatial + bandwidth (4D array): apply fft() on each 3D slice
  # NOTE: We use base R fft() here, NOT Rcpp. See @section above for why.
  # -------------------------------------------------------------------
  if (dx == 3 && ndim == 4) {
    result <- arr
    if (inverse) {
      for (ih in 1:dh) {
        result[,,,ih] <- fft(arr[,,,ih], inverse = TRUE) / prod_spatial
      }
    } else {
      for (ih in 1:dh) {
        result[,,,ih] <- fft(arr[,,,ih])
      }
    }
    return(result)
  }

  # For 1D spatial or other cases: axis-by-axis using apply_fft_axis
  # This vectorizes over all bandwidths using mvfft() internally
  result <- arr
  for (ix in dx:1) {
    result <- apply_fft_axis(result, ix, inverse)
  }

  return(result)
}

#' Apply FFT along a specific axis (OPTIMIZED VERSION)
#'
#' This function MUST be defined early (in design_matrix.R) because it's used by
#' fastlpr_conv.R and fastlpr_kdf.R which are sourced before nufft.R.
#'
#' @param arr Input array
#' @param axis Axis along which to apply FFT
#' @param inverse Whether to apply inverse FFT
#'
#' @return Transformed array
#'
#' @keywords internal
apply_fft_axis <- function(arr, axis, inverse = FALSE) {
  dims <- dim(arr)
  ndim <- length(dims)
  if (is.null(dims)) {
    # 1D vector case - direct FFT
    if (inverse) {
      return(fft(arr, inverse = TRUE) / length(arr))
    } else {
      return(fft(arr))
    }
  }

  n <- dims[axis]
  if (n <= 1) return(arr)

  # OPTIMIZATION: Special cases to avoid expensive aperm
  if (ndim == 2) {
    # 2D array: use t() instead of aperm (much faster!)
    if (axis == 1) {
      if (inverse) {
        return(mvfft(arr, inverse = TRUE) / n)
      } else {
        return(mvfft(arr, inverse = FALSE))
      }
    } else {
      mat <- t(arr)
      if (inverse) {
        mat <- mvfft(mat, inverse = TRUE) / n
      } else {
        mat <- mvfft(mat, inverse = FALSE)
      }
      return(t(mat))
    }
  }

  # OPTIMIZATION: 3D array special cases
  if (ndim == 3) {
    if (axis == 1) {
      other_dim <- dims[2] * dims[3]
      mat <- matrix(arr, nrow = n, ncol = other_dim)
      if (inverse) {
        mat <- mvfft(mat, inverse = TRUE) / n
      } else {
        mat <- mvfft(mat, inverse = FALSE)
      }
      dim(mat) <- dims  # In-place reshape (no allocation)
      return(mat)

    } else if (axis == 2) {
      d1 <- dims[1]
      d3 <- dims[3]
      arr_perm <- aperm(arr, c(2, 1, 3))
      mat <- matrix(arr_perm, nrow = n, ncol = d1 * d3)
      if (inverse) {
        mat <- mvfft(mat, inverse = TRUE) / n
      } else {
        mat <- mvfft(mat, inverse = FALSE)
      }
      dim(mat) <- c(n, d1, d3)  # In-place reshape
      return(aperm(mat, c(2, 1, 3)))

    } else {
      d1d2 <- dims[1] * dims[2]
      mat <- matrix(arr, nrow = d1d2, ncol = n, byrow = FALSE)
      mat_t <- t(mat)
      if (inverse) {
        mat_t <- mvfft(mat_t, inverse = TRUE) / n
      } else {
        mat_t <- mvfft(mat_t, inverse = FALSE)
      }
      result <- t(mat_t)
      dim(result) <- dims  # In-place reshape
      return(result)
    }
  }

  # OPTIMIZATION: 4D array special cases (critical for 3D KDE with bandwidths!)
  # arr is (d1, d2, d3, d4) where d4 is typically dh (bandwidths)
  if (ndim == 4) {
    if (axis == 1) {
      # FFT along dim 1: reshape (n, d2, d3, d4) -> (n, d2*d3*d4)
      other_dim <- dims[2] * dims[3] * dims[4]
      mat <- matrix(arr, nrow = n, ncol = other_dim)
      if (inverse) {
        mat <- mvfft(mat, inverse = TRUE) / n
      } else {
        mat <- mvfft(mat, inverse = FALSE)
      }
      dim(mat) <- dims
      return(mat)

    } else if (axis == 2) {
      # FFT along dim 2: arr is (d1, n, d3, d4)
      # aperm approach is faster than loop in R (benchmarked)
      d1 <- dims[1]
      d3 <- dims[3]
      d4 <- dims[4]
      arr_perm <- aperm(arr, c(2, 1, 3, 4))
      mat <- matrix(arr_perm, nrow = n, ncol = d1 * d3 * d4)
      if (inverse) {
        mat <- mvfft(mat, inverse = TRUE) / n
      } else {
        mat <- mvfft(mat, inverse = FALSE)
      }
      dim(mat) <- c(n, d1, d3, d4)
      return(aperm(mat, c(2, 1, 3, 4)))

    } else if (axis == 3) {
      # FFT along dim 3: arr is (d1, d2, n, d4)
      # aperm approach is faster than loop in R (benchmarked)
      d1 <- dims[1]
      d2 <- dims[2]
      d4 <- dims[4]
      arr_perm <- aperm(arr, c(3, 1, 2, 4))
      mat <- matrix(arr_perm, nrow = n, ncol = d1 * d2 * d4)
      if (inverse) {
        mat <- mvfft(mat, inverse = TRUE) / n
      } else {
        mat <- mvfft(mat, inverse = FALSE)
      }
      dim(mat) <- c(n, d1, d2, d4)
      return(aperm(mat, c(2, 3, 1, 4)))

    } else {
      # axis == 4: FFT along dim 4 (bandwidth dimension)
      # arr is (d1, d2, d3, n)
      d123 <- dims[1] * dims[2] * dims[3]
      mat <- matrix(arr, nrow = d123, ncol = n, byrow = FALSE)
      mat_t <- t(mat)
      if (inverse) {
        mat_t <- mvfft(mat_t, inverse = TRUE) / n
      } else {
        mat_t <- mvfft(mat_t, inverse = FALSE)
      }
      result <- t(mat_t)
      dim(result) <- dims
      return(result)
    }
  }

  # GENERAL CASE: axis == 1 fast path
  if (axis == 1) {
    other_dim <- prod(dims[-1])
    mat <- matrix(arr, nrow = n, ncol = other_dim)
    if (inverse) {
      mat <- mvfft(mat, inverse = TRUE) / n
    } else {
      mat <- mvfft(mat, inverse = FALSE)
    }
    dim(mat) <- dims  # In-place reshape
    return(mat)
  }

  # FALLBACK: General case with permutation
  perm <- c(axis, setdiff(seq_along(dims), axis))
  arr_perm <- aperm(arr, perm)
  other_dim <- prod(dims[-axis])
  mat <- matrix(arr_perm, nrow = n, ncol = other_dim)
  if (inverse) {
    mat <- mvfft(mat, inverse = TRUE) / n
  } else {
    mat <- mvfft(mat, inverse = FALSE)
  }
  dim(mat) <- c(n, dims[-axis])  # In-place reshape
  aperm(mat, order(perm))
}

#' Compute design matrix for local polynomial regression
#'
#' Computes the design matrix S = X'*W*X for local polynomial regression.
#' For order 0 (Nadaraya-Watson), S is a scalar (sum of kernel weights).
#' For order >= 1, S is a matrix with elements computed via convolution.
#' This is a direct translation of MATLAB's fastLPR_s.m
#'
#' @param regs Regression structure from fastlpr_create containing:
#'          .Tx       - Number of data samples
#'          .dtype    - Data type (e.g., 'double', 'single')
#'          .opt.order - Polynomial order (0, 1, or 2)
#'          .kdf      - Kernel density function in Fourier domain
#'          .lwp.ns   - Number of design matrix elements (for order >= 1)
#'
#' @return regs Updated regression structure with:
#'          .s - Design matrix elements
#'               Order 0: scalar (sum of weights)
#'               Order >= 1: cell array of ns elements
#'
#' @noRd
fastlpr_s <- function(regs) {
  # Create vector of ones for convolution (represents constant function)
  Is <- matrix(1, nrow = regs$Tx, ncol = 1)

  if (regs$opt$order == 0) {
    # Order 0: Nadaraya-Watson estimator
    # S = sum of kernel weights at each evaluation point
    # Add eps for numerical stability (prevents division by zero)
    s_raw <- fastlpr_conv(regs, regs$kdf, Is, FALSE)
    regs$s <- s_raw + .Machine$double.eps
  } else if (regs$opt$order == 1) {
    # Order 1: Local linear regression
    # Compute each element of the design matrix S via convolution
    # S has ns unique elements (lower triangular due to symmetry)
    regs$s <- list()
    for (i in 1:regs$lwp$ns) {
      regs$s[[i]] <- fastlpr_conv(regs, regs$kdf[[i]], Is, FALSE)
    }
  } else if (regs$opt$order == 2) {
    # Order 2: Local quadratic regression
    # Compute each element of the design matrix S via convolution
    # S has ns unique elements (lower triangular due to symmetry)
    regs$s <- list()
    for (i in 1:regs$lwp$ns) {
      regs$s[[i]] <- fastlpr_conv(regs, regs$kdf[[i]], Is, FALSE)
    }
  } else {
    stop("Unknown polynomial order: ", regs$opt$order)
  }

  return(regs)
}