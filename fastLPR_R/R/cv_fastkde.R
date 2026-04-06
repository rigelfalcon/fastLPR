# Copyright (c) 2024-2025 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later

#' Fast Kernel Density Estimation with automatic bandwidth selection
#'
#' This function performs nonparametric density estimation using kernel
#' smoothing with NUFFT (Non-Uniform Fast Fourier Transform) acceleration.
#' It automatically selects the optimal bandwidth via Likelihood Cross-
#' Validation (LCV) when multiple bandwidth candidates are provided.
#'
#' @param x N x d matrix of data points (N samples, d dimensions).
#'   Can be real or complex-valued. Each row is one observation.
#'   For complex-valued data, use x = real + 1i*imag.
#' @param h Bandwidth parameter(s) (optional, default: automatic selection).
#'   Scalar: same bandwidth for all dimensions.
#'   1 x d vector: different bandwidth per dimension.
#'   k x d matrix: grid of k bandwidth combinations for LCV selection.
#'   Use get_hlist() to generate bandwidth candidates.
#' @param opt Options list (optional) with fields:
#'   \itemize{
#'     \item kernel_type: Kernel function (default: 'gaussian')
#'       'gaussian' or 'epanechnikov'
#'     \item N: Grid resolution for evaluation (default: auto)
#'     \item xrange: Evaluation range (default: data range)
#'     \item verbose: Display progress (default: false)
#'   }
#'
#' @return Density estimation results list containing:
#'   \itemize{
#'     \item h: Selected bandwidth (scalar for 1D, vector for multi-D)
#'     \item fhat: Density estimate at evaluation grid points for SELECTED bandwidth only.
#'       (N_grid x 1 for 1D, N_grid x N_grid for 2D, N_grid x N_grid x N_grid for 3D).
#'       Note: Only contains result for selected bandwidth, not all bandwidths.
#'     \item fpp: Interpolator object for evaluation at any point.
#'       Use: f_pred = kde$fpp(x_new)
#'     \item lcv: LCV results list (if multiple bandwidths provided)
#'       \itemize{
#'         \item h1se: Selected bandwidth (1-SE rule)
#'         \item hmin: Bandwidth with minimum LCV
#'         \item lcv_m: LCV values for all bandwidths
#'         \item id1se: Index of selected bandwidth
#'       }
#'     \item xlist: Grid vectors (list)
#'     \item opt: Options used for estimation
#'     \item xraw: Original data points
#'   }
#'
#' @examples
#' # Example 1: 1D density estimation with automatic bandwidth selection
#' x <- c(rnorm(100), rnorm(100) + 3)  # Bimodal distribution
#' hlist <- get_hlist(20, c(0.01, 1), "logspace")
#' kde <- cv_fastkde(x, hlist)
#'
#' @export
cv_fastkde <- function(x, h = NULL, opt = NULL) {
  # Input validation
  if (!is.matrix(x)) {
    stop("Input x must be a matrix.")
  }
  if (!is.numeric(x) && !is.complex(x)) {
    stop("Input x must be numeric or complex.")
  }

  n <- nrow(x)
  dx <- ncol(x)

  if (n < 2) {
    stop(sprintf("Input x must have at least 2 observations. Got %d observations.", n))
  }

  # Dimension limits - same as Python implementation
  if (dx > 10) {
    stop(sprintf(paste0(
      "dx=%d exceeds maximum supported dimension (10). ",
      "High-dimensional regression suffers from the curse of dimensionality."
    ), dx))
  } else if (dx > 6) {
    warning(sprintf(paste0(
      "dx=%d is high-dimensional and may cause memory issues. ",
      "Consider dimensionality reduction techniques."
    ), dx))
  } else if (dx > 3) {
    message(sprintf("dx=%d: Using generalized N-D KDE (order=0 only).", dx))
  }

  # Set defaults for optional inputs
  if (is.null(opt)) {
    opt <- list()
  }
  if (is.null(h)) {
    h <- matrix()
  }

  # Validate opt fields if provided
  if (!is.null(opt$kernel_type)) {
    valid_kernels <- c("gaussian", "epanechnikov")
    if (!tolower(opt$kernel_type) %in% valid_kernels) {
      stop("opt$kernel_type must be 'gaussian' or 'epanechnikov'.\nGot: ", opt$kernel_type)
    }
  }

  # Set KDE-specific options
  # KDE is equivalent to order=0 regression with y=1
  opt$order <- 0  # Nadaraya-Watson
  opt$calc_dof <- FALSE  # Don't need DOF for KDE
  opt$y_type_out <- "mean"  # Standard output

  # Set default kernel type
  opt <- set_defaults(opt, "kernel_type", "gaussian")
  opt <- set_defaults(opt, "verbose", FALSE)

  # Grid size N: use adaptive default from nufft module
  # N = ceiling(n^(1/dx)) where n = sample size

  # Create dummy response (all ones) for KDE
  # KDE: f(x) = (1/n) * sum_i K_h(x - x_i)
  # This is equivalent to order=0 regression with y=1, then divide by n
  y_dummy <- matrix(1, n, 1)

  # Initialization
  regs <- fastlpr_create(x, y_dummy, h, opt)

  # Compute design matrix
  # Calculate kernel-weighted design matrix S using NUFFT
  # For order=0, this computes sum_i K_h(x - x_i) at each grid point
  regs <- fastlpr_s(regs)

  # Compute density estimates
  # KDE formula: f(x) = (1/n) * sum_i K_h(x - x_i)
  # regs$s contains sum_i K_h(x - x_i) for each bandwidth
  #
  # However, the NUFFT-based convolution loses some mass due to the finite
  # spreading width (Msp). To correct for this, we normalize so that the
  # integral of the density equals 1.
  #
  # Theoretical: integral[f(x)dx] = sum_x f(x)*dx = 1
  # So we normalize: fhat = regs$s / (sum(regs$s)*dx)
  #
  # This data-driven normalization corrects for NUFFT approximation error
  # without needing to increase Msp (which would increase computation).

  # Compute grid spacing (product of spacing in each dimension)
  dx_grid <- prod(sapply(regs$xlist, function(vec) vec[2] - vec[1]))

  # Create colon indexing for spatial dimensions
  ndcolon <- replicate(dx, ":", simplify = FALSE)

  # Normalize each bandwidth separately
  # Note: Some bandwidths may have been removed (ihbad), so we only normalize good ones
  fhat_all <- array(0, dim = dim(regs$s))
  for (ih in 1:regs$dh) {
    # Skip bad bandwidths (they will remain zero)
    if (regs$ihbad[ih]) {
      next
    }

    # Extract current bandwidth slice for the specific bandwidth
    # regs$s has structure: (grid dimensions..., dh, dy)
    # where dy=1 for KDE
    # 1D: s_current = regs$s[, ih, 1]  (3D array: Nx x dh x dy)
    # 2D: s_current = regs$s[, , ih, 1]  (4D array: Nx x Ny x dh x dy)
    # The dy dimension is always last and equals 1 for KDE

    # Use generalized N-D slice extraction (works for dx = 1, 2, 3, ... 10)
    # regs$s has structure: (grid dimensions..., dh, dy)
    s_current <- extract_slice_nd(regs$s, dx, ih, 1)

    # Take real part to remove numerical noise from FFT
    # (imaginary part should be negligible for real-valued data)
    s_current <- Re(s_current)

    # Normalize so integral equals 1
    sum_s_dx <- sum(s_current) * dx_grid

    # Normalize current bandwidth
    s_normalized <- s_current / sum_s_dx

    # Assign normalized values back using generalized N-D assignment
    fhat_all <- assign_slice_nd(fhat_all, dx, ih, 1, s_normalized)
  }

  # Create interpolator for each bandwidth
  regs$fpp_yhat <- fastlpr_gridinterp(regs, fhat_all, regs$xlist,
                                      regs$opt$y_grid_method, regs$opt$y_grid_opt)
  regs$yhat <- fhat_all

  # Bandwidth selection via LCV
  # If multiple bandwidths provided, select optimal one using LCV
  if (regs$dh > 1) {
    lcv_result <- fastkde_lcv(regs, x)
  } else {
    lcv_result <- NULL
  }

  # Save fields before compacting
  # fastlpr_compact removes some fields, so save them first
  xraw_save <- regs$xraw
  opt_save <- regs$opt
  h_save <- regs$h
  dh_save <- regs$dh
  dx_save <- regs$dx
  xlist_save <- regs$xlist
  N_save <- regs$N

  # Compact results
  # Remove intermediate variables to save memory
  regs <- fastlpr_compact(regs)

  # Prepare output structure
  kde <- list()

  if (!is.null(lcv_result)) {
    # Multiple bandwidths: use selected bandwidth
    kde$lcv <- lcv_result
    kde$h <- lcv_result$h1se  # Selected bandwidth (1-SE rule)

    # Extract only the selected bandwidth's density from fpp
    # regs$fpp_yhat contains all bandwidths, we need to extract the 1-SE one
    # Similar to fastlpr_gcv.m line 142
    id1se <- lcv_result$id1se

    # Create NEW interpolator with extracted values instead of
    # modifying kde$fpp$Values in place. The old approach broke the closure
    # in kde$fpp$evaluate() due to R's copy-on-write semantics - the closure
    # would still reference the old interpolator with ALL bandwidths.
    if (dh_save > 1) {
      # Extract only the selected bandwidth using dimension-aware indexing
      # For 1D: Values is (N x dh), extract column id1se
      # For 2D: Values is (N1 x N2 x dh), extract slice [:, :, id1se]
      # For 3D: Values is (N1 x N2 x N3 x dh), extract slice [:, :, :, id1se]

      if (dx_save == 1) {
        # 1D case: Values is (N x dh)
        values_1se <- regs$fpp_yhat$Values[, id1se, drop = FALSE]
      } else if (dx_save == 2) {
        # 2D case: Values is (N1 x N2 x dh)
        values_1se <- regs$fpp_yhat$Values[, , id1se, drop = FALSE]
      } else if (dx_save == 3) {
        # 3D case: Values is (N1 x N2 x N3 x dh)
        values_1se <- regs$fpp_yhat$Values[, , , id1se, drop = FALSE]
      } else {
        # General N-D case (dx > 3)
        # Build index list programmatically
        idx_list <- c(
          lapply(1:dx_save, function(d) seq_len(N_save[d])),
          list(id1se)
        )
        values_1se <- do.call(`[`, c(list(regs$fpp_yhat$Values), idx_list, list(drop = FALSE)))
      }

      # Create NEW interpolator object with extracted values
      # This ensures the closure captures the correct values
      fake_regs <- list(
        dx = dx_save,
        N = N_save,
        xlist = xlist_save
      )
      kde$fpp <- fastlpr_gridinterp(
        fake_regs, values_1se, xlist_save,
        regs$fpp_yhat$Method,
        list(Method = regs$fpp_yhat$Method,
             ExtrapolationMethod = regs$fpp_yhat$ExtrapolationMethod)
      )
    } else {
      kde$fpp <- regs$fpp_yhat
    }

    # Also extract the selected bandwidth's yhat
    if (dh_save > 1) {
      # Extract specific bandwidth slice accounting for dy dimension
      if (dx_save == 1) {
        kde$fhat <- regs$yhat[, id1se, 1]
      } else if (dx_save == 2) {
        kde$fhat <- regs$yhat[, , id1se, 1]
      } else if (dx_save == 3) {
        kde$fhat <- regs$yhat[, , , id1se, 1]
      }
    } else {
      # Single bandwidth case: extract dy=1 slice with proper dimensions
      if (dx_save == 1) {
        kde$fhat <- regs$yhat[, 1, 1]  # (N, dh=1, dy=1) -> (N)
      } else if (dx_save == 2) {
        kde$fhat <- regs$yhat[, , 1, 1]  # (N1, N2, dh=1, dy=1) -> (N1, N2)
      } else if (dx_save == 3) {
        kde$fhat <- regs$yhat[, , , 1, 1]  # (N1, N2, N3, dh=1, dy=1) -> (N1, N2, N3)
      }
    }
  } else {
    # Single bandwidth
    kde$h <- h_save
    kde$fpp <- regs$fpp_yhat
    kde$fhat <- regs$yhat
  }

  # Get xlist from fpp$GridVectors (this is the correct grid)
  kde$xlist <- kde$fpp$GridVectors

  kde$opt <- opt_save
  kde$xraw <- xraw_save

  # Add class for S3 methods
  class(kde) <- "fastkde_result"

  return(kde)
}

#' Helper function for LCV computation
#' Helper function for LCV computation
#'
#' Computes Likelihood Cross-Validation scores for bandwidth selection.
#' Uses the efficient leave-one-out formula to avoid O(n^2) computation.
#'
#' OPTIMIZED: Uses vectorized interpolation to evaluate ALL bandwidths at once
#' instead of looping. This is critical for 3D KDE with 1000+ bandwidths.
#'
#' Algorithm:
#'   1. For each bandwidth, compute leave-one-out density at data points
#'   2. Use efficient formula: f_{-i}(x_i) = (n/(n-1)) * [f(x_i) - (1/n)*K(0)]
#'   3. Compute LCV = sum(log(f_{-i}(x_i)))
#'   4. Select bandwidth with maximum LCV
#'   5. Apply 1-SE rule for robustness
#'
#' @param regs Regression structure from fastlpr_create
#' @param x Original data points (N x dx matrix)
#' @return LCV result structure with fields:
#'   - lcv_m: LCV scores for all bandwidths (dh x 1)
#'   - lcv_sd: Standard errors (dh x 1)
#'   - idmax: Index of bandwidth with maximum LCV
#'   - id1se: Index of bandwidth selected by 1-SE rule
#'   - h1se: Selected bandwidth (1 x dx)
#'   - hmax: Bandwidth with maximum LCV (1 x dx)
#'   - hlist: All bandwidths (dh x dx)
#' @noRd
fastkde_lcv <- function(regs, x) {
  n <- nrow(x)
  dx <- regs$dx
  dh <- regs$dh

  # OPTIMIZED: Evaluate ALL bandwidths at data points at once
  # regs$yhat structure:
  #   1D: (Ng x dh x dy) where dy=1 for KDE
  #   2D: (N1 x N2 x dh x dy)
  #   3D: (N1 x N2 x N3 x dh x dy)
  # Extract the dy=1 slice to get (Ng... x dh)

  # Extract dy=1 slice for any dx
  # yhat shape: (N1 x ... x Ndx x dh x dy) where dy=1 for KDE
  # Extract to get: (N1 x ... x Ndx x dh)
  yhat_dims <- dim(regs$yhat)
  # Build index list: all indices for spatial dims, all for dh, and 1 for dy
  idx_list <- c(
    lapply(1:dx, function(d) seq_len(yhat_dims[d])),  # All spatial indices
    list(seq_len(dh)),                                 # All bandwidth indices
    list(1)                                            # dy = 1 slice
  )
  fhat_all <- do.call(`[`, c(list(regs$yhat), idx_list, list(drop = FALSE)))
  # Reshape to remove trailing dy=1 dimension
  dim(fhat_all) <- c(yhat_dims[1:dx], dh)

  # Create a single interpolator for ALL bandwidths at once
  # This avoids creating dh separate interpolators
  # FIX: Use 'linear' interpolation to match MATLAB's fastkde_lcv exactly
  # MATLAB uses: griddedInterpolant(regs.xlist, fhat_current, 'linear', 'linear')
  interp <- fastlpr_gridinterp(regs, fhat_all, regs$xlist, 
                                "griddedInterpolant",
                                list(Method = "linear", ExtrapolationMethod = "linear"))

  # Evaluate at data points - returns (n x dh) matrix
  f_at_data_all <- interp$evaluate(x)

  # Ensure it's a matrix (n x dh)
  if (!is.matrix(f_at_data_all)) {
    f_at_data_all <- matrix(f_at_data_all, nrow = n, ncol = dh)
  }

  # Compute K_h(0) for each bandwidth - vectorized
  # NOTE: MATLAB uses 1/sqrt(2*pi) for ALL dimensions
  # This is because the bandwidth normalization (dividing by prod(h)) handles the multi-D scaling
  # The actual multi-dimensional normalization factor cancels out in the LCV ratio
  if (regs$opt$kernel_type == "gaussian") {
    K0 <- 1 / sqrt(2 * pi)  # Match MATLAB exactly
  } else {
    K0 <- 0.75  # Epanechnikov
  }

  # K0_normalized is a vector of length dh (one per bandwidth)
  # h is (dh x dx), we need prod of each row
  K0_normalized <- K0 / apply(regs$h, 1, prod)

  # Compute leave-one-out density for ALL bandwidths at once
  # f_loo[i, ih] = (n/(n-1)) * (f_at_data_all[i, ih] - K0_normalized[ih]/n)
  # Use sweep to subtract K0_normalized/n from each column
  f_loo_all <- (n / (n - 1)) * sweep(f_at_data_all, 2, K0_normalized / n, "-")

  # Avoid log(0)
  f_loo_all <- pmax(f_loo_all, 1e-10)

  # Compute LCV scores for all bandwidths at once - vectorized
  log_f_loo_all <- log(f_loo_all)
  lcv_scores <- colSums(log_f_loo_all)
  lcv_std <- apply(log_f_loo_all, 2, sd) * sqrt(n)

  # Find maximum LCV bandwidth
  idmax <- which.max(lcv_scores)

  # Apply 1-SE rule to select smoothest bandwidth within 1 SE of maximum
  semax <- lcv_scores[idmax] - 1 * lcv_std[idmax]

  # Find all bandwidths with LCV >= threshold
  # Among these, select the one with largest bandwidth (sum of h)
  hsum <- rowSums(regs$h)
  valid_indices <- which(lcv_scores >= semax)

  # Print 1-SE selection details (when debug mode enabled)
  if (getOption("fastlpr.debug", FALSE)) {
    cat("\n=== 1-SE RULE DEBUG ===\n")
    cat(sprintf("Max LCV index: %d\n", idmax))
    cat(sprintf("Max LCV value: %.6f\n", lcv_scores[idmax]))
    cat(sprintf("SE at max: %.6f\n", lcv_std[idmax]))
    cat(sprintf("1-SE threshold: %.6f\n", semax))
    cat(sprintf("Number of candidates: %d\n", length(valid_indices)))
    if (length(valid_indices) > 0) {
      cat("Candidate indices:\n")
      for (idx in valid_indices) {
        if (dx == 1) {
          cat(sprintf("  idx=%d, h=%.4f, lcv=%.6f\n",
                      idx, regs$h[idx, 1], lcv_scores[idx]))
        } else if (dx == 2) {
          cat(sprintf("  idx=%d, h=[%.4f,%.4f], sum=%.4f, lcv=%.6f\n",
                      idx, regs$h[idx, 1], regs$h[idx, 2],
                      hsum[idx], lcv_scores[idx]))
        } else {
          cat(sprintf("  idx=%d, h=[%.4f,%.4f,%.4f], sum=%.4f, lcv=%.6f\n",
                      idx, regs$h[idx, 1], regs$h[idx, 2], regs$h[idx, 3],
                      hsum[idx], lcv_scores[idx]))
        }
      }
    }
  }

  if (length(valid_indices) > 0) {
    # Among valid bandwidths, select largest (sum of h)
    # Use which.max to match MATLAB behavior exactly
    hsum_candidates <- hsum[valid_indices]
    max_idx <- which.max(hsum_candidates)
    id1se <- valid_indices[max_idx]

    if (getOption("fastlpr.debug", FALSE)) {
      cat(sprintf("Selected index: %d (max_idx in candidates: %d)\n", id1se, max_idx))
    }
  } else {
    id1se <- idmax
    if (getOption("fastlpr.debug", FALSE)) {
      cat(sprintf("No candidates found, using idmax: %d\n", idmax))
    }
  }

  if (getOption("fastlpr.debug", FALSE)) {
    cat("=== END 1-SE DEBUG ===\n\n")
  }

  # Package results
  lcv_result <- list(
    lcv_m = lcv_scores,
    lcv_sd = lcv_std,
    idmax = idmax,
    id1se = id1se,
    h1se = regs$h[id1se, , drop = FALSE],
    hmax = regs$h[idmax, , drop = FALSE],
    hlist = regs$h
  )

  return(lcv_result)
}

#' S3 print method for fastkde_result
#' @noRd
print.fastkde_result <- function(x, ...) {
  cat("Fast Kernel Density Estimation Results\n")
  cat("====================================\n")
  cat(sprintf("Number of samples: %d\n", nrow(x$xraw)))
  cat(sprintf("Dimensions: %d\n", ncol(x$xraw)))
  cat(sprintf("Kernel type: %s\n", x$opt$kernel_type))
  cat(sprintf("Selected bandwidth: %s\n", paste(x$h, collapse = ", ")))
  cat("\nUse summary() for more details or plot() for visualization.\n")
}