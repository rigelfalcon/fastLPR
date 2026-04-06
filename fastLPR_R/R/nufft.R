# Copyright (c) 2024-2025 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later

#' Non-Uniform Fast Fourier Transform for scattered data
#'
#' Computes the Fourier transform of data at non-uniform (scattered) sample points.
#' This is a key component enabling O(N + M log M) complexity for kernel regression.
#' This is a direct translation of MATLAB's fastLPR_nufft.m
#'
#' @param regs Regression structure containing:
#'   \itemize{
#'     \item knot: Normalized sample positions in [-0.5, 0.5]
#'     \item hf: Frequency grid spacing
#'     \item opt$accuracy: NUFFT accuracy (number of digits)
#'     \item opt$nufft_deconv: Whether to apply deconvolution correction
#'     \item dx: Number of dimensions
#'   }
#' @param y Data values at sample points (Tx x dy matrix)
#'
#' @return y_ft Fourier transform on uniform grid (L x L x ... x dy array)
#'
#' @noRd
fastlpr_nufft <- function(regs, y) {
  # Apply Type-1 NUFFT: non-uniform points -> uniform grid
  y_ft <- nufftn_type1(regs$knot, y, matrix(), regs$hf, -1,
                       regs$opt$accuracy, regs$opt$nufft_deconv)

  # Shift zero frequency to center (standard FFT convention)
  # Apply ifftshift along each dimension
  for (i in 1:regs$dx) {
    y_ft <- ifftshift_array(y_ft, i)
  }

  return(y_ft)
}

#' Multi-dimensional ifftshift
#'
#' Equivalent to MATLAB's ifftshift applied along a specific dimension.
#' ifftshift undoes fftshift: it moves the first ceil(n/2) elements to the end.
#'
#' MATLAB formula for ifftshift(x):
#'   shift = floor(n/2)
#'   x = x([(shift+1):n, 1:shift])
#'
#' @param x Input array
#' @param dim Dimension along which to apply ifftshift
#'
#' @return Shifted array
#'
#' @noRd
ifftshift_array <- function(x, dim = NULL) {
  if (is.null(dim)) {
    # Apply ifftshift to all dimensions
    dims <- dim(x)
    for (i in 1:length(dims)) {
      x <- ifftshift_array(x, dim = i)
    }
    return(x)
  } else {
    # Apply ifftshift along specific dimension
    dims <- dim(x)
    if (is.null(dims) || length(dims) < dim || dims[dim] <= 1) {
      return(x)
    }

    n <- dims[dim]
    shift <- floor(n / 2)

    # Create index vectors for all dimensions
    # All dimensions get : except the target dimension
    # ifftshift formula: [(shift+1):n, 1:shift] where shift = floor(n/2)
    # This moves the first ceil(n/2) elements to the end
    indices <- lapply(seq_along(dims), function(i) {
      if (i == dim) {
        c((shift + 1):n, 1:shift)
      } else {
        1:dims[i]
      }
    })

    # Use do.call with [ to subset the array
    result <- do.call(`[`, c(list(x), indices, list(drop = FALSE)))

    return(result)
  }
}

#' Multi-dimensional fftshift
#'
#' Equivalent to MATLAB's fftshift applied along a specific dimension
#'
#' @param x Input array
#' @param dim Dimension along which to apply fftshift
#'
#' @return Shifted array
#'
#' @noRd
fftshift_array <- function(x, dim = NULL) {
  if (is.null(dim)) {
    # Apply fftshift to all dimensions
    dims <- dim(x)
    for (i in 1:length(dims)) {
      x <- fftshift_array(x, dim = i)
    }
    return(x)
  } else {
    # Apply fftshift along specific dimension
    dims <- dim(x)
    if (is.null(dims) || length(dims) < dim || dims[dim] <= 1) {
      return(x)
    }

    n <- dims[dim]
    shift <- ceiling(n / 2)

    # Create index vectors for all dimensions
    # All dimensions get : except the target dimension
    indices <- lapply(seq_along(dims), function(i) {
      if (i == dim) {
        c((shift + 1):n, 1:shift)
      } else {
        1:dims[i]
      }
    })
    
    # Use do.call with [ to subset the array
    result <- do.call(`[`, c(list(x), indices, list(drop = FALSE)))

    return(result)
  }
}

#' Type-1 Non-Uniform Fast Fourier Transform
#'
#' Transforms from non-uniform points to uniform grid.
#' This is a direct translation of MATLAB's nufftn_type1.m
#'
#' @param x Non-uniform sample locations (M x dx matrix)
#' @param y Function values at sample points (M x dy matrix)
#' @param Fs Sampling frequency (default: auto)
#' @param df Frequency spacing (default: auto)
#' @param iflag Sign for exponential (default: -1)
#' @param acc Accuracy in digits (default: 6)
#' @param isdeconv Whether to apply deconvolution (default: TRUE)
#' @param nufft_loop Whether to use loop optimization (default: TRUE)
#'
#' @return Yq Uniform grid Fourier transform values
#'
#' @noRd
nufftn_type1 <- function(x, y, Fs = matrix(), df = numeric(), iflag = -1,
                         acc = 6, isdeconv = TRUE, nufft_loop = TRUE) {
  M <- nrow(x)
  dx <- ncol(x)
  dy <- ncol(y)
  
  # OPTIMIZATION v3.2: Use Rcpp NUFFT for ALL N (not just M >= 1000)
  # Always try Rcpp first for maximum performance
  # Look up the Rcpp binding - first try global environment (for sourced files/testthat),
  # then try package namespace (for installed package)
  rcpp_nufft_fn <- tryCatch({
    # First check global environment (for sourced scripts and testthat)
    fn <- get0("rcpp_nufft_type1", envir = globalenv(), inherits = FALSE)
    if (is.null(fn)) {
      # Then check package namespace (for installed package)
      fn <- get0("rcpp_nufft_type1", envir = asNamespace("fastlpr"), inherits = FALSE)
    }
    fn
  }, error = function(e) NULL)

  rcpp_nufft_complex_fn <- tryCatch({
    # First check global environment (for sourced scripts and testthat)
    fn <- get0("rcpp_nufft_type1_complex", envir = globalenv(), inherits = FALSE)
    if (is.null(fn)) {
      # Then check package namespace (for installed package)
      fn <- get0("rcpp_nufft_type1_complex", envir = asNamespace("fastlpr"), inherits = FALSE)
    }
    fn
  }, error = function(e) NULL)

  rcpp_nufft_available <- is.function(rcpp_nufft_fn)
  rcpp_nufft_complex_available <- is.function(rcpp_nufft_complex_fn)

  # Look up Rcpp binning functions (for accuracy=0 mode)
  rcpp_binning_fn <- tryCatch({
    fn <- get0("rcpp_nufft_binning", envir = globalenv(), inherits = FALSE)
    if (is.null(fn)) {
      fn <- get0("rcpp_nufft_binning", envir = asNamespace("fastlpr"), inherits = FALSE)
    }
    fn
  }, error = function(e) NULL)

  rcpp_binning_complex_fn <- tryCatch({
    fn <- get0("rcpp_nufft_binning_complex", envir = globalenv(), inherits = FALSE)
    if (is.null(fn)) {
      fn <- get0("rcpp_nufft_binning_complex", envir = asNamespace("fastlpr"), inherits = FALSE)
    }
    fn
  }, error = function(e) NULL)

  rcpp_binning_available <- is.function(rcpp_binning_fn)
  rcpp_binning_complex_available <- is.function(rcpp_binning_complex_fn)

  # RE-ENABLE Rcpp NUFFT after fixing all algorithmic bugs:
  # - Fixed tau formula: now uses ratio*(ratio-0.5) in denominator
  # - Fixed hx formula: now uses 2*pi/Mr instead of 1/Mr
  # - Fixed coordinate system: expects pre-scaled knots in [0, 2*pi]
  # - Fixed normalization: uses computed ratio, not hardcoded 2.0
  y_is_complex <- is.complex(y)
  use_rcpp_real <- rcpp_nufft_available && !y_is_complex && acc > 0
  use_rcpp_complex <- rcpp_nufft_complex_available && y_is_complex && acc > 0
  # Enable Rcpp binning for accuracy=0 mode
  use_rcpp_binning <- rcpp_binning_available && !y_is_complex && acc == 0
  use_rcpp_binning_complex <- rcpp_binning_complex_available && y_is_complex && acc == 0

  # Early exit for Rcpp binning mode (accuracy=0)
  if (use_rcpp_binning || use_rcpp_binning_complex) {
    # Compute N and Fs for scale_knots
    if (missing(df) || length(df) == 0) {
      N <- ceiling(M^(1/dx))
      N <- rep(N, dx)
    } else {
      N <- round(1/df)
    }
    if (missing(Fs) || length(Fs) == 0 || all(is.na(Fs))) {
      Fs <- N
    }

    # Compute grid parameters for binning
    params <- compute_grid_params(N, acc)
    Mr <- params$Mr
    ratio <- params$ratio

    # Scale knots to [0, 2*pi]
    xmod <- scale_knots(x, N, Fs)

    # Call Rcpp binning
    Ftau <- tryCatch({
      if (use_rcpp_binning_complex) {
        rcpp_binning_complex_fn(as.matrix(xmod), as.matrix(y + 0i), as.integer(Mr))
      } else {
        rcpp_binning_fn(as.matrix(xmod), as.matrix(y), as.integer(Mr))
      }
    }, error = function(e) NULL)

    if (!is.null(Ftau)) {
      # Rcpp binning succeeded - apply FFT and extract
      iflag_val <- if (missing(iflag) || length(iflag) == 0) -1 else iflag
      if (iflag_val < 0) {
        for (ix in dx:1) {
          Ftau_temp <- apply_fft_axis(Ftau, axis = ix, inverse = FALSE)
          Ftau <- fftshift_array(Ftau_temp, dim = ix)
        }
        Ftau <- Ftau / (M * (ratio^dx))
      } else {
        for (ix in dx:1) {
          Ftau <- ifftshift_array(apply_fft_axis(Ftau, axis = ix, inverse = TRUE), dim = ix)
        }
      }

      # Extract output region
      q <- (Mr - N) / 2
      q_start <- ceiling(q) + 1
      q_end <- q_start + N - 1

      if (dx == 1) {
        Yq <- Ftau[q_start[1]:q_end[1], , drop = FALSE]
      } else if (dx == 2) {
        Yq <- Ftau[q_start[1]:q_end[1], q_start[2]:q_end[2], , drop = FALSE]
      } else if (dx == 3) {
        Yq <- Ftau[q_start[1]:q_end[1], q_start[2]:q_end[2], q_start[3]:q_end[3], , drop = FALSE]
      }

      return(Yq)
    }
    # Fall through to pure R if Rcpp failed
  }

  if (use_rcpp_real || use_rcpp_complex) {
    # Compute N and Fs for scale_knots
    if (missing(df) || length(df) == 0) {
      N <- ceiling(M^(1/dx))
      N <- rep(N, dx)
    } else {
      N <- round(1/df)
    }
    if (missing(Fs) || length(Fs) == 0 || all(is.na(Fs))) {
      Fs <- N
    }

    # Scale knots BEFORE passing to C++ (matching Pure R algorithm)
    x_scaled <- scale_knots(x, N, Fs)

      # Try Rcpp NUFFT with pre-scaled knots
      result <- tryCatch({
        if (use_rcpp_complex) {
          # Use complex-aware Rcpp function
          rcpp_nufft_complex_fn(as.matrix(x_scaled), as.matrix(y + 0i), as.integer(N), as.integer(acc))
        } else if (use_rcpp_real) {
          rcpp_nufft_fn(as.matrix(x_scaled), as.matrix(y), as.integer(N), as.integer(acc))
        } else {
          NULL
        }
      }, error = function(e) {
        # Rcpp NUFFT failed - fall back to pure R silently
        NULL
      })

      if (!is.null(result)) {
        # Rcpp NUFFT succeeded - apply deconvolution if needed
        # (Rcpp version doesn't include deconvolution)
        if (isdeconv) {
          # Compute deconvolution kernel using CORRECTED params matching Pure R/MATLAB
          # CRITICAL: Must use same tau formula as Rcpp: tau = pi*Msp / (N^2 * ratio * (ratio-0.5))
          Msp <- ceiling(acc)
          # Determine ratio based on accuracy (same as compute_grid_params and Rcpp)
          if (acc <= 6) {
            ratio <- 1
          } else if (acc <= 11) {
            ratio <- 2
          } else {
            ratio <- 3
          }
          tau <- (pi * Msp) / (N^2 * ratio * (ratio - 0.5))

          # Get frequencies
          freqs_result <- nufftfreqs(N)
          freqs <- freqs_result$grid

          if (dx == 1) {
            freqs_vec <- as.vector(freqs)
            Kn <- sqrt(tau / pi) * exp(-tau * freqs_vec^2)
          } else {
            tau_vec <- as.vector(tau)
            weighted_freqs_sq <- sweep(freqs^2, 2, tau_vec, `*`)
            Kn <- sqrt(prod(tau) / (pi^dx)) * exp(-rowSums(weighted_freqs_sq))
          }
          Kninv <- 1 / Kn

          # Apply deconvolution
          dy <- ncol(y)
          if (dx == 1) {
            result <- sweep(result, 1, Kninv, `*`)
          } else {
            if (dy == 1) {
              # Rcpp result has dim (N[1], N[2], dy) even when dy=1
              # So Kninv_array must include the dy dimension to be conformable
              Kninv_array <- array(Kninv, dim = c(N, dy))
              result <- result * Kninv_array
            } else {
              result_flat <- matrix(result, nrow = prod(N), ncol = dy)
              result_flat <- result_flat * Kninv
              result <- array(result_flat, dim = c(N, dy))
            }
          }
        }
        return(result)
      }
      # Fall through to pure R (for verification or if Rcpp fails)
  }

  # Pure R NUFFT - check if fallback is allowed (skip for binning mode)
  if (acc > 0) {
    require_rcpp_or_allow_fallback("nufft Pure R implementation")
  }

  # Default parameters
  if (missing(df) || length(df) == 0) {
    N <- ceiling(M^(1/dx))
    N <- rep(N, dx)
    df <- 1/N
  } else {
    N <- round(1/df)
  }

  # Check if Fs is empty or contains only NAs
  if (missing(Fs) || length(Fs) == 0 || all(is.na(Fs))) {
    Fs <- N
  }

  if (length(iflag) == 0) {
    iflag <- -1
  }

  if (length(acc) == 0) {
    acc <- 6
  }

  if (length(isdeconv) == 0) {
    isdeconv <- TRUE
  }

  if (length(nufft_loop) == 0) {
    nufft_loop <- TRUE
  }

  # Compute grid parameters
  params <- compute_grid_params(N, acc)
  Msp <- params$Msp
  Mr <- params$Mr
  tau <- params$tau
  ratio <- params$ratio

  # Binning mode: accuracy=0 means no spreading, direct accumarray
  if (acc == 0) {
    hx <- 2 * pi / Mr
    xmod <- scale_knots(x, N, Fs)

    # Use Rcpp binning for performance (5x faster than R loops)
    if (use_rcpp_binning || use_rcpp_binning_complex) {
      Ftau <- tryCatch({
        if (use_rcpp_binning_complex) {
          rcpp_binning_complex_fn(as.matrix(xmod), as.matrix(y + 0i), as.integer(Mr))
        } else {
          rcpp_binning_fn(as.matrix(xmod), as.matrix(y), as.integer(Mr))
        }
      }, error = function(e) {
        # Fall back to pure R on error
        NULL
      })

      if (!is.null(Ftau)) {
        # Rcpp binning succeeded - apply FFT and extract
        if (iflag < 0) {
          for (ix in dx:1) {
            Ftau_temp <- apply_fft_axis(Ftau, axis = ix, inverse = FALSE)
            Ftau <- fftshift_array(Ftau_temp, dim = ix)
          }
          Ftau <- Ftau / (M * (ratio^dx))
        } else {
          for (ix in dx:1) {
            Ftau <- ifftshift_array(apply_fft_axis(Ftau, axis = ix, inverse = TRUE), dim = ix)
          }
        }

        # Extract output region
        q <- (Mr - N) / 2
        q_start <- ceiling(q) + 1
        q_end <- q_start + N - 1

        if (dx == 1) {
          Yq <- Ftau[q_start[1]:q_end[1], , drop = FALSE]
        } else if (dx == 2) {
          Yq <- Ftau[q_start[1]:q_end[1], q_start[2]:q_end[2], , drop = FALSE]
        } else if (dx == 3) {
          Yq <- Ftau[q_start[1]:q_end[1], q_start[2]:q_end[2], q_start[3]:q_end[3], , drop = FALSE]
        }

        return(Yq)
      }
      # Fall through to pure R if Rcpp failed
    }

    # Pure R binning (fallback)
    m_orig <- round(xmod / hx)

    # Compute grid indices (mod for periodic boundary)
    xindx <- (m_orig %% Mr) + 1  # R is 1-indexed

    # Direct binning (equivalent to MATLAB accumarray)
    sz <- Mr
    Ftau <- array(0 + 0i, dim = c(sz, dy))

    # Compute linear indices for accumulation
    if (dx == 1) {
      linear_idx <- xindx
    } else {
      strides <- cumprod(c(1, sz[-dx]))
      linear_idx <- rowSums((xindx - 1) * matrix(strides, nrow = M, ncol = dx, byrow = TRUE)) + 1
    }

    # Accumulate (R equivalent of accumarray)
    for (iy in 1:dy) {
      for (i in 1:M) {
        if (dx == 1) {
          Ftau[linear_idx[i], iy] <- Ftau[linear_idx[i], iy] + y[i, iy]
        } else {
          Ftau[linear_idx[i] + (iy - 1) * prod(sz)] <- Ftau[linear_idx[i] + (iy - 1) * prod(sz)] + y[i, iy]
        }
      }
    }

    # FFT (use apply_fft_axis like spreading mode)
    if (iflag < 0) {
      for (ix in dx:1) {
        Ftau_temp <- apply_fft_axis(Ftau, axis = ix, inverse = FALSE)
        Ftau <- fftshift_array(Ftau_temp, dim = ix)
      }
      Ftau <- Ftau / (M * (ratio^dx))
    } else {
      for (ix in dx:1) {
        Ftau <- ifftshift_array(apply_fft_axis(Ftau, axis = ix, inverse = TRUE), dim = ix)
      }
    }

    # Extract output region (no deconvolution needed for binning)
    q <- (Mr - N) / 2
    q_start <- ceiling(q) + 1
    q_end <- q_start + N - 1

    if (dx == 1) {
      Yq <- Ftau[q_start[1]:q_end[1], , drop = FALSE]
    } else if (dx == 2) {
      Yq <- Ftau[q_start[1]:q_end[1], q_start[2]:q_end[2], , drop = FALSE]
    } else if (dx == 3) {
      Yq <- Ftau[q_start[1]:q_end[1], q_start[2]:q_end[2], q_start[3]:q_end[3], , drop = FALSE]
    }

    return(Yq)
  }

  # Construct the convolved grid
  hx <- 2 * pi / Mr
  xmod <- scale_knots(x, N, Fs)

  # Compute spreading indices
  m <- round(xmod / hx)
  Msp2 <- Msp * 2 + 1

  # Prepare for broadcasting
  # MATLAB: m=permute(m,[1,3,2]) makes m (M, 1, dx) for dx>1, or (M, 1) for dx=1
  if (dx == 1) {
    m_perm <- matrix(m, ncol = 1)  # (M, 1)
    hx_perm <- hx
    xmod_perm <- matrix(xmod, ncol = 1)  # (M, 1)
    mpmm <- m_perm
  } else {
    m_perm <- aperm(array(m, dim = c(M, dx, 1)), c(1, 3, 2))  # (M, 1, dx)
    hx_perm <- array(hx, dim = c(1, 1, dx))
    xmod_perm <- aperm(array(xmod, dim = c(M, dx, 1)), c(1, 3, 2))
    mpmm <- m_perm
  }

  # Compute spreading neighborhoods for each dimension
  # MATLAB: for i=1:dx; m=mpmm; mm=zeros(1,Msp2(i),dx); mm(1,:,i)=...; mpmm=reshape(m+mm,[],1,dx); end
  for (i in 1:dx) {
    m_current <- mpmm

    if (dx == 1) {
      # For 1D: mm is (1, Msp2[i])
      mm_vec <- round(seq(-Msp[i], Msp[i], length.out = Msp2[i]))
      # m_current is (M_current, 1), mm is (1, Msp2[i])
      # Broadcasting: m_current + mm gives (M_current, Msp2[i])
      # In R, use outer() for broadcasting: outer(m_current[,1], mm_vec, "+")
      m_plus_mm <- outer(m_current[, 1], mm_vec, "+")
      # Reshape to (M_current * Msp2[i], 1)
      mpmm <- matrix(as.vector(m_plus_mm), ncol = 1)
    } else {
      # For multi-D: mm is (1, Msp2[i], dx)
      mm <- array(0, dim = c(1, Msp2[i], dx))
      mm[1, , i] <- round(seq(-Msp[i], Msp[i], length.out = Msp2[i]))

      # m_current is (M_current, 1, dx), mm is (1, Msp2[i], dx)
      # Need to broadcast: result is (M_current, Msp2[i], dx)
      dims_m <- dim(m_current)
      M_current <- dims_m[1]

      # Broadcasting for 3D arrays - outer is efficient for this
      m_plus_mm <- array(0, dim = c(M_current, Msp2[i], dx))
      for (j in 1:dx) {
        m_plus_mm[, , j] <- outer(m_current[, 1, j], mm[1, , j], "+")
      }

      # Reshape to (M_current * Msp2[i], 1, dx)
      total_size <- M_current * Msp2[i]
      mpmm <- array(m_plus_mm, dim = c(total_size, 1, dx))
    }
  }

  # Reshape for processing
  if (dx == 1) {
    # For 1D: reshape to (M, prod(Msp2))
    mpmm_reshaped <- matrix(mpmm, nrow = M, ncol = prod(Msp2), byrow = FALSE)
  } else {
    # For multi-D: mpmm is (M*prod(Msp2), 1, dx), need to reshape to (M, prod(Msp2), dx)
    # Convert to vector first, then reshape
    mpmm_vec <- as.vector(mpmm)
    mpmm_reshaped <- array(mpmm_vec, dim = c(M, prod(Msp2), dx))
  }
  
  # Test indexing
  if (dx == 1) {
  } else {
  }

  # Compute heat kernel weights
  pMsp2 <- prod(Msp2)  # Cache this value

  if (dx == 1) {
    # For 1D: xmod_perm is (M, 1), mpmm_reshaped is (M, prod(Msp2))
    # OPTIMIZED: Use rep() instead of outer() - same effect, less overhead
    diff <- rep(xmod_perm[, 1], times = pMsp2) - hx * as.vector(mpmm_reshaped)
    diff <- matrix(diff, nrow = M, ncol = pMsp2)
    # For 1D, heat_kernel expects (M, prod(Msp2), 1) array
    diff_3d <- array(diff, dim = c(M, pMsp2, 1))
    weight <- heat_kernel(diff_3d, tau)
    # y_perm for 1D: just y as (M, dy)
    if (is.null(dim(y))) {
      y_perm <- matrix(y, ncol = 1)
    } else {
      y_perm <- y
    }
    # OPTIMIZED: Vectorized ysp computation without loop
    # For dy == 1 (common case), just multiply directly
    if (dy == 1) {
      ysp_mat <- rep(y_perm[, 1], times = pMsp2)
      ysp_mat <- matrix(ysp_mat, nrow = M, ncol = pMsp2) * weight
      ysp <- array(ysp_mat, dim = c(M, pMsp2, 1))
    } else {
      # For dy > 1: reshape y_perm and weight for vectorized multiply
      # y_perm: (M, dy), weight: (M, pMsp2) -> ysp: (M, pMsp2, dy)
      # FIX: Use 0+0i for complex y to preserve imaginary parts
      init_val <- if (is.complex(y_perm)) 0+0i else 0
      ysp <- array(init_val, dim = c(M, pMsp2, dy))
      for (k in 1:dy) {
        ysp_mat <- rep(y_perm[, k], times = pMsp2)
        ysp[, , k] <- matrix(ysp_mat, nrow = M, ncol = pMsp2) * weight
      }
    }
  } else {
    # For multi-D: xmod_perm is (M, 1, dx), mpmm_reshaped is (M, prod(Msp2), dx)
    # OPTIMIZED: Vectorized expansion using aperm - no loop!
    # xmod_perm[, 1, ] is (M, dx), replicate each column pMsp2 times
    xmod_2d <- matrix(xmod_perm, nrow = M, ncol = dx)  # (M, dx)

    # OPTIMIZATION: Use aperm to replicate without loop
    # Create (M, dx, pMsp2) then permute to (M, pMsp2, dx)
    xmod_expanded <- aperm(array(xmod_2d, dim = c(M, dx, pMsp2)), c(1, 3, 2))

    # Compute difference: xmod_expanded - hx * mpmm_reshaped
    # hx_perm is (1, 1, dx), will broadcast properly
    diff <- xmod_expanded - as.vector(hx) * mpmm_reshaped

    weight <- heat_kernel(diff, tau)

    # OPTIMIZED: Direct y extraction without intermediate arrays
    if (is.null(dim(y))) {
      y_vec <- as.vector(y)
    } else {
      y_vec <- y[, 1]
    }

    # OPTIMIZED: Vectorized ysp computation using aperm - no loop!
    # Result should be (M, pMsp2, dy)
    if (dy == 1) {
      # OPTIMIZATION: dim<- may avoid a copy for unshared, non-ALTREP objects
      ysp_mat <- matrix(rep(y_vec, times = pMsp2), nrow = M, ncol = pMsp2) * weight
      dim(ysp_mat) <- c(M, pMsp2, 1)
      ysp <- ysp_mat
    } else {
      # For dy > 1: Use aperm trick to avoid loop
      # y is (M, dy), replicate each column pMsp2 times, multiply by weight
      y_mat <- if (is.null(dim(y))) matrix(y, ncol = 1) else y
      # Create (M, dy, pMsp2) then permute to (M, pMsp2, dy)
      y_replicated <- aperm(array(y_mat, dim = c(M, dy, pMsp2)), c(1, 3, 2))
      # weight is (M, pMsp2), need to broadcast to (M, pMsp2, dy)
      weight_3d <- array(weight, dim = c(M, pMsp2, dy))
      ysp <- y_replicated * weight_3d
    }
  }

  # Reshape ysp - dim<- may avoid a copy for unshared, non-ALTREP objects
  dim(ysp) <- c(M * pMsp2, dy)
  ysp_reshaped <- ysp

  # Compute indices
  if (nufft_loop) {
    xindx <- (mpmm_reshaped %% Mr) + 1
    # dim<- may avoid a copy for unshared, non-ALTREP objects
    dim(xindx) <- c(M * pMsp2, dx)
    xindx_reshaped <- xindx
  } else {
    xindx <- (mpmm_reshaped %% Mr) + 1
    if (dy > 1) {
      xindx <- cbind(xindx, matrix(1, nrow = nrow(xindx), ncol = 1))
    }
    xindx_rep <- do.call(rbind, replicate(dy, xindx, simplify = FALSE))
    if (dy > 1) {
      xindx_rep[, dx + 1] <- rep(1:dy, each = M * prod(Msp2))
    }
    xindx_reshaped <- xindx_rep
  }

  # Accumulate using sparse operations
  mu <- bit_convert(0, matrix(0, 1, 1))$x1

  # FIX: Use 0+0i for complex y to preserve imaginary parts
  ftau_init <- if (is.complex(y)) 0+0i else 0

  # Create Ftau with proper dimensions based on dx
  if (dx == 1) {
    Ftau <- array(ftau_init, dim = c(Mr, dy))
  } else {
    # For multi-D, Mr is a vector, so c(Mr, dy) creates proper multi-D array
    # e.g., for 2D: Mr = c(32, 32), dy = 1 → dim = c(32, 32, 1)
    Ftau <- array(ftau_init, dim = c(Mr, dy))
  }
  


  if (nufft_loop) {
    # For 2D, sz needs to be c(Mr[1], Mr[2]), not just Mr
    if (dx == 1) {
      sz <- c(Mr, 1)
    } else {
      # For multi-D, Mr is already a vector
      sz <- Mr
    }
    

    for (iy in 1:dy) {
      # Use sparse matrix multiplication for accumulation
      indices <- xindx_reshaped[, 1:dx, drop = FALSE]
      
      # Handle ysp_reshaped indexing properly
      if (dy == 1) {
        # For single response, ysp_reshaped might be a vector
        values <- as.vector(ysp_reshaped)
      } else {
        values <- ysp_reshaped[, iy]
      }

      
      # Convert multi-dimensional indices to linear indices
      linear_indices <- arrayInd_to_linear(indices, sz)
      
      result <- sparse_accumulator(linear_indices, values, sz)
      
      # Assign result to Ftau using generalized N-D indexing
      if (dx == 1) {
        Ftau[, iy] <- result
      } else {
        # For multi-D, Ftau is (Mr[1], Mr[2], ..., dy) and result is (Mr[1], Mr[2], ...)
        # Use generalized assignment for any dx
        idx_list <- c(
          lapply(1:dx, function(d) seq_len(dim(Ftau)[d])),
          list(iy)
        )
        Ftau <- do.call(`[<-`, c(list(Ftau), idx_list, list(value = result)))
      }
    }
  } else {
    if (dx == 1 || dy > 1) {
      sz <- c(Mr, dy)
    } else {
      sz <- Mr
    }

    linear_indices <- arrayInd_to_linear(xindx_reshaped, sz)
    Ftau_values <- sparse_accumulator(linear_indices, ysp_reshaped, sz)
    Ftau <- array(Ftau_values, dim = sz)
  }

  # Apply FFT
  
  if (iflag < 0) {
    for (ix in dx:1) {
      Ftau_temp <- apply_fft_axis(Ftau, axis = ix, inverse = FALSE)
      Ftau <- fftshift_array(Ftau_temp, dim = ix)
    }
    Ftau <- Ftau / (M * (ratio^dx))
  } else {
    for (ix in dx:1) {
      Ftau <- ifftshift_array(apply_fft_axis(Ftau, axis = ix, inverse = TRUE), dim = ix)
    }
  }

  # Extract output patch
  q <- (Mr - N) / 2
  q_bounds <- cbind(ceiling(q) + 1, ceiling(q) + N)

  sz_full <- c(Mr, dy)
  patch_indices <- get_patch_index(q_bounds, sz_full)

  Ftau_extracted <- Ftau[patch_indices]
  Yq <- array(Ftau_extracted, dim = c(N, dy))

  # Deconvolution
  if (isdeconv) {
    # MATLAB: nufftfreqs(N) returns frequencies WITHOUT df scaling
    # Then the formula uses 2*pi*freqs in heat_kernel
    freqs_result <- nufftfreqs(N)  # Don't pass df
    freqs <- freqs_result$grid
    if (dx == 1) {
      # For 1D: freqs is a matrix with 1 column, extract as vector
      freqs_vec <- as.vector(freqs)
      # MATLAB formula: Kn = sqrt(tau/pi) * exp(-tau * freqs^2)
      # where freqs is from nufftfreqs(N) (integer frequencies)
      Kn <- sqrt(tau / pi) * exp(-tau * freqs_vec^2)
    } else {
      # For multi-D: freqs is a matrix (nfreq x dx)
      # Need to compute sum(tau[d] * freqs[,d]^2) for each frequency point
      # Use sweep to multiply each column by corresponding tau, then rowSums
      tau_vec <- as.vector(tau)
      weighted_freqs_sq <- sweep(freqs^2, 2, tau_vec, `*`)
      Kn <- sqrt(prod(tau) / (pi^dx)) * exp(-rowSums(weighted_freqs_sq))
    }
    Kninv <- 1 / Kn

    # Apply deconvolution - OPTIMIZED using array recycling
    if (dx == 1) {
      # For 1D, sweep works correctly
      Yq <- sweep(Yq, 1, Kninv, `*`)
    } else {
      # For multi-D, Kninv is a vector of length prod(N)
      # but Yq has shape (N[1], N[2], ..., N[dx], dy)
      # Yq has dim (N[1], N[2], dy) even for dy=1
      # So Kninv_array must have matching dimension
      if (dy == 1) {
        # Single response - Kninv_array must include dy dimension
        Kninv_array <- array(Kninv, dim = c(N, dy))
        Yq <- Yq * Kninv_array
      } else {
        # Multiple responses - reshape for vectorized multiplication
        # Flatten spatial dims, multiply, then reshape back
        Yq_flat <- matrix(Yq, nrow = prod(N), ncol = dy)
        Yq_flat <- Yq_flat * Kninv  # Kninv recycles across columns
        Yq <- array(Yq_flat, dim = c(N, dy))
      }
    }
  }

  return(Yq)
}

#' Heat kernel function
#'
#' Gaussian kernel for NUFFT spreading
#'
#' @param x Input values
#' @param h Kernel width parameter
#'
#' @return Kernel weights
#'
#' @noRd
heat_kernel <- function(x, h) {
  # MATLAB: h=permute(h(:),[3:-1:1]); x=exp(-sum((x.^2)./(4*h),3));
  # h is permuted to (1, 1, length(h))
  # x.^2 ./ (4*h) broadcasts to (M, prod(Msp2), dx)
  # sum(..., 3) sums along 3rd dimension to get (M, prod(Msp2))
  #
  # OPTIMIZATION: Replace slow apply() with vectorized rowSums()

  h_vec <- as.vector(h)
  dx <- length(h_vec)
  x_dims <- dim(x)

  if (dx == 1) {
    # For 1D: x is (M, prod(Msp2), 1), h is scalar
    # Since 3rd dim is 1, just drop it
    result <- exp(-(x[, , 1]^2) / (4 * h_vec[1]))
  } else {
    # For multi-D: FULLY VECTORIZED - no array creation, no loops
    # Reshape x from (M, N, dx) to (M*N, dx) directly
    MN <- x_dims[1] * x_dims[2]
    mat_2d <- matrix(x, nrow = MN, ncol = dx)
    # Square and divide by 4*h using sweep (broadcasts h across columns)
    mat_scaled <- sweep(mat_2d^2, 2, 4 * h_vec, "/")
    # Sum across columns and exp
    result <- matrix(exp(-rowSums(mat_scaled)), nrow = x_dims[1], ncol = x_dims[2])
  }

  result
}

#' Multi-dimensional array indices to linear
#'
#' Convert multi-dimensional array indices to linear indices
#'
#' @param indices Matrix of multi-dimensional indices
#' @param dims Array dimensions
#'
#' @return Linear indices
#'
#' @noRd
arrayInd_to_linear <- function(indices, dims) {
  # NOTE: Validation removed for performance. Re-enable for debugging:
  # if (any(is.na(indices))) stop("NAs in indices")

  if (ncol(indices) == 1) {
    return(indices[, 1])
  }

  # OPTIMIZED: Use matrix multiplication for linear index computation
  # This avoids the loop and is vectorized
  multipliers <- c(1, cumprod(dims[-length(dims)]))
  # indices is (n, dx), multipliers is (dx,)
  # linear = indices %*% multipliers, but indices are 1-based
  # so we need: sum over i of (indices[,i] - (i>1)) * multipliers[i]
  # = indices[,1] + (indices[,2]-1)*dims[1] + (indices[,3]-1)*dims[1]*dims[2] + ...

  # Direct formula: linear = 1 + sum((indices - 1) * multipliers)
  # But we have 1-based indices, so: indices[,1] + (indices[,2]-1)*dims[1] + ...
  # Equivalent to: sum(indices * multipliers) - sum(multipliers[2:end])

  offset_adjust <- sum(multipliers[-1])  # sum of multipliers for dims 2 and beyond
  linear_indices <- as.integer(indices %*% multipliers) - offset_adjust

  return(linear_indices)
}


#' Sparse accumulator function
#'
#' Efficient accumulation using sparse matrices
#'
#' @param indices Linear indices
#' @param values Values to accumulate
#' @param dims Output dimensions
#'
#' @return Accumulated values
#'
#' @noRd
sparse_accumulator <- function(indices, values, dims) {
  # OPTIMIZED: Use rowsum instead of tapply for 10-20× speedup
  # tapply() calls factor() internally which is extremely slow (O(N log N) with huge constant)
  # rowsum() just sorts integers - much faster!
  # COMPLEX SUPPORT: Split real/imaginary parts for complex-valued data

  # Convert multi-dimensional indices to linear indices if needed
  if (is.matrix(indices)) {
    # Multi-dimensional indices: convert to linear
    linear_idx <- sub2ind(dims, indices)
  } else {
    linear_idx <- indices
  }

  # DEFENSIVE FIX: Remove NA entries before rowsum
  # This prevents "missing values for 'group'" warning and all-zero output
  valid_mask <- !is.na(linear_idx) & !is.na(values)
  if (sum(valid_mask) < length(linear_idx)) {
    n_removed <- sum(!valid_mask)
    n_na_values <- sum(is.na(values))
    n_na_indices <- sum(is.na(linear_idx))
    warning(sprintf("sparse_accumulator: Removed %d NA entries (%d values, %d indices). Check NUFFT computation.",
                    n_removed, n_na_values, n_na_indices))
  }

  # Filter to valid entries only
  linear_idx <- linear_idx[valid_mask]
  values <- values[valid_mask]

  # Early return if no valid data
  if (length(linear_idx) == 0) {
    warning("sparse_accumulator: No valid data after NA filtering. Returning zero array.")
    if (length(values) > 0 && is.complex(values[1])) {
      return(array(0 + 0i, dim = dims))
    } else {
      return(array(0, dim = dims))
    }
  }


  # Check if values are complex
  if (is.complex(values)) {
    # Split complex into real and imaginary parts
    # Process separately since rowsum() doesn't handle complex
    values_real <- Re(values)
    values_imag <- Im(values)

    # Accumulate real part
    accumulated_real <- rowsum(values_real, linear_idx, reorder = FALSE)
    idx_unique <- as.integer(rownames(accumulated_real))

    # Accumulate imaginary part
    accumulated_imag <- rowsum(values_imag, linear_idx, reorder = FALSE)

    # Combine back to complex
    result <- array(0 + 0i, dims)  # Complex array
    result[idx_unique] <- complex(real = accumulated_real[, 1],
                                  imaginary = accumulated_imag[, 1])
  } else {
    # Real-valued: use fast rowsum path
    result <- array(0, dims)

    # OPTIMIZATION: Use rowsum instead of tapply - 10-20× faster!
    # rowsum does NOT create factor levels (no string processing, no level sorting)
    # Just sorts integer indices and sums - exactly what we need
    accumulated <- rowsum(values, linear_idx, reorder = FALSE)
    idx_unique <- as.integer(rownames(accumulated))
    result[idx_unique] <- accumulated[, 1]
  }

  return(result)
}

# NOTE: apply_fft_axis is defined in design_matrix.R (optimized version)
# It must be defined there because it's sourced before fastlpr_conv.R and fastlpr_kdf.R
