# Copyright (c) 2024-2025 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later

#' Compute fitted values and GCV
#'
#' Mirrors: fastLPR/utility/core/fastLPR_y.m
#'
#' Unified implementation for arbitrary dx.
#' - 1D case: Specialized with approxfun for performance
#' - dx >= 2: Single unified code path using dynamic N-D array operations
#'
#' @keywords internal
fastlpr_y <- function(regs) {

# OPTIMIZATION HELPER: Fast row SD (10-100x faster than row_sd(x))
row_sd <- function(x) {
  if (is.matrix(x)) {
    n <- ncol(x)
    if (n <= 1) return(rep(0, nrow(x)))
    means <- rowMeans(x)
    sq_diff <- (x - means)^2
    variances <- rowSums(sq_diff) / (n - 1)
    return(sqrt(variances))
  } else if (is.array(x) && length(dim(x)) == 2) {
    return(row_sd(as.matrix(x)))
  } else {
    # Fallback to apply for higher-dimensional arrays
    return(row_sd(x))
  }
}


  # ==========================================================================
  # VARIANCE ESTIMATION: Apply log transform for numerical stability
  # When y_type_out = 'variance', we estimate log(variance) instead of variance
  # This is more stable numerically and handles non-negativity constraints
  # ==========================================================================
  y_type_out <- regs$opt$y_type_out
  y_raw_for_variance <- NULL

  if (!is.null(y_type_out) && y_type_out == "variance") {
    # Save original y for normalization later
    y_raw_for_variance <- regs$y
    # MATLAB: y = log(regs.y + 1/regs.Ty)
    # Add small constant 1/Ty to avoid log(0)
    Ty <- nrow(regs$y)
    y_shifted <- regs$y + 1.0 / Ty

    # Handle negative values: MATLAB returns complex log, R returns NaN
    # For variance estimation, the typical input is squared residuals (non-negative)
    # If any values are negative, this indicates incorrect input or unusual use case
    # We match MATLAB by using complex log
    if (any(y_shifted < 0)) {
      # Use complex logarithm for negative values (matches MATLAB behavior)
      # log(negative) = log(|negative|) + i*pi
      regs$y <- complex(real = log(abs(y_shifted)),
                        imaginary = ifelse(y_shifted < 0, pi, 0))
    } else {
      regs$y <- log(y_shifted)
    }
  }

  # Compute design matrix S
  regs <- fastlpr_s(regs)

  # Transform y to Fourier domain (for NUFFT-based convolution)
  # For now, use direct computation, so just pass y as-is
  y_ft <- regs$y

  # Compute regression estimate on grid
  mq <- fastlpr_reg(regs, y_ft)

  # ==========================================================================
  # DIMENSION-SPECIFIC PROCESSING
  # ==========================================================================

  if (regs$dx == 1) {
    # ========================================================================
    # 1D CASE: Specialized implementation with approxfun for performance
    # ========================================================================

  # Extract data in RAW space (for interpolation)
  x_raw <- as.numeric(regs$xraw[, 1])
  # FIX: Do not use as.numeric() for y - it discards imaginary parts!
  y <- regs$y
  if (is.matrix(y)) {
    y <- y[, 1]
  }
  grid_raw <- as.numeric(regs$xlist[[1]])

  # Extract data in SCALED space (for DOF computation)
  x_scaled <- as.numeric(regs$x[, 1])

  # Bandwidth is in SCALED space
  h_vals <- as.numeric(regs$h[, 1])

  Tx <- length(x_raw)
  Ng <- length(grid_raw)
  dh <- length(h_vals)

  # FIX: Use complex matrix if mq is complex (for complex-valued y data)
  if (is.complex(mq)) {
    yhat_grid <- matrix(complex(1), nrow = Ng, ncol = dh)
    yhat_data <- matrix(complex(1), nrow = Tx, ncol = dh)
  } else {
    yhat_grid <- matrix(0, nrow = Ng, ncol = dh)
    yhat_data <- matrix(0, nrow = Tx, ncol = dh)
  }
  diagS_mat <- matrix(0, nrow = Tx, ncol = dh)
  gcv <- numeric(dh)
  mse <- numeric(dh)
  rss <- numeric(dh)
  df_vals <- numeric(dh)
  pdof_vals <- numeric(dh)
  penalty_vals <- numeric(dh)


  # Step 1: Interpolate grid predictions to data points for all bandwidths
  # OPTIMIZATION: Use interp_batch_fast for batch interpolation (unified N-D, O(N))

  # Extract grid values for all bandwidths: mq is (Ng, dh)
  # Always ensure yhat_grid is a matrix (Ng x dh)
  if (is.null(dim(mq))) {
    yhat_grid <- matrix(mq, ncol = 1)
  } else if (length(dim(mq)) == 2) {
    # Ensure matrix even for single bandwidth (drop=FALSE equivalent)
    yhat_grid <- matrix(mq, nrow = nrow(mq), ncol = ncol(mq))
  } else {
    # Multi-response: extract first response
    yhat_grid <- matrix(mq[, , 1], ncol = dim(mq)[2])
  }

  # Batch interpolate all bandwidths at once
  # grid_raw is the grid, x_raw are query points
  # yhat_grid is (Ng x dh), output is (Tx x dh)
  yhat_data <- interp_batch_fast(
    x_query = matrix(x_raw, ncol = 1),
    grid = list(grid_raw),
    values = yhat_grid,
    dx = 1
  )

  # Ensure yhat_data is always a matrix (Tx x dh)
  # interp_batch_fast may return a vector when dh=1
  if (is.null(dim(yhat_data))) {
    yhat_data <- matrix(yhat_data, ncol = dh)
  }

  # Compute residuals and MSE for all bandwidths (vectorized)
  residual <- sweep(yhat_data, 1, y, "-")  # (Tx x dh)
  rss <- colSums(abs(residual)^2)  # Vector of length dh - use abs() for complex support
  mse <- colMeans(abs(residual)^2)  # Vector of length dh - use abs() for complex support

  # Step 2: Compute DOF (only if requested)
  # When calc_dof=FALSE, skip DOF computation entirely (use default penalty=1.0)
  # When calc_dof=TRUE, use Hutchinson trace estimator for O(N) complexity
  calc_dof <- if (is.null(regs$opt$calc_dof)) TRUE else regs$opt$calc_dof

  if (!calc_dof) {
    # Skip DOF - use default penalty
    df_vals <- rep(1, dh)
    pdof_vals <- rep(1, dh)
    penalty_vals <- rep(1, dh)
    penalty_sd <- rep(0, dh)
    diagS_mat <- matrix(NA, nrow = Tx, ncol = dh)
  } else {
    # Use Hutchinson's randomized trace estimator: O(N + M log M)
    dof_result <- fastlpr_dof_hutchinson(regs, num_samples = 10)
    df_vals <- dof_result$df_m
    pdof_vals <- dof_result$pdof_m
    penalty_vals <- dof_result$pdof_inv_m
    penalty_sd <- dof_result$pdof_inv_sd
    diagS_mat <- matrix(NA, nrow = Tx, ncol = dh)
  }

  # Compute GCV using MSE and DOF penalty
  gcv <- mse * penalty_vals
  gcv_sd <- mse * penalty_sd  # GCV standard deviation for 1-SE rule

  # Find minimum GCV bandwidth
  idmin <- which.min(gcv)

  # ==========================================================================
  # Apply 1-SE rule (select largest bandwidth within dstd SE of minimum)
  # The 1-SE rule prefers larger (smoother) bandwidths when GCV is similar
  # ==========================================================================
  dstd <- regs$opt$dstd
  if (is.null(dstd)) dstd <- 0  # Default: no 1-SE rule

  if (dstd > 0) {
    # Compute threshold: minimum GCV + dstd * SE
    semin <- gcv[idmin] + dstd * gcv_sd[idmin]

    # Find all bandwidths with GCV <= threshold
    candidates <- which(gcv <= semin)

    if (length(candidates) == 0) {
      # No candidates found, use minimum GCV bandwidth
      id1se <- idmin
    } else {
      # Among candidates, select the one with largest bandwidth
      # For 1D, h_vals is a vector, so just pick the largest h
      h_candidates <- h_vals[candidates]
      max_h_idx <- which.max(h_candidates)
      id1se <- candidates[max_h_idx]

      # If multiple bandwidths have same max h, pick one with smallest GCV
      same_h_candidates <- candidates[h_candidates == h_candidates[max_h_idx]]
      if (length(same_h_candidates) > 1) {
        id1se <- same_h_candidates[which.min(gcv[same_h_candidates])]
      }
    }
  } else {
    # No 1-SE rule: use minimum GCV bandwidth
    id1se <- idmin
  }

  # Use interpolated data values at the selected bandwidth
  # Handle single bandwidth case properly (dh=1)
  # Use drop=FALSE or direct indexing to avoid dimension dropping
  if (dh == 1) {
    regs$yhat <- yhat_data[, 1]
    regs$yhat_data <- yhat_data[, 1]
    yhat_grid_selected <- yhat_grid[, 1]
  } else {
    regs$yhat <- yhat_data[, id1se]
    regs$yhat_data <- yhat_data[, id1se]
    yhat_grid_selected <- yhat_grid[, id1se]
  }
  regs$dof <- df_vals[id1se]
  regs$diagS <- diagS_mat
  regs$df_m <- matrix(df_vals, ncol = 1)
  regs$pdof_m <- matrix(pdof_vals, ncol = 1)
  regs$pdof_inv_m <- matrix(penalty_vals, ncol = 1)
  regs$pdof_inv_sd <- matrix(penalty_sd, ncol = 1)

  # Create interpolator from GRID values (not data values)
  # MATLAB: fpp = griddedInterpolant(xlist, mq)
  # Use complex_approxfun to handle complex y without warnings
  regs$fpp_yhat <- complex_approxfun(
    x = grid_raw,
    y = yhat_grid_selected,  # Use 1-SE bandwidth grid values
    method = "linear",
    rule = 2
  )

  regs$gcv_yhat <- list(
    gcv_m = matrix(gcv, ncol = 1),
    gcv_sd = matrix(gcv_sd, ncol = 1),  # GCV standard deviation
    hmin = matrix(h_vals[idmin], nrow = 1),  # Minimum GCV bandwidth
    h1se = matrix(h_vals[id1se], nrow = 1),  # 1-SE rule bandwidth
    h = matrix(h_vals[id1se], nrow = 1),     # Alias for h1se (API compatibility)
    idmin = idmin,
    id1se = id1se,
    mse = matrix(mse, ncol = 1),
    rss = matrix(rss, ncol = 1),
    yhat_1se = if (dh == 1) yhat_data[, 1] else yhat_data[, id1se],
    pdof_m = if (exists("pdof_vals")) matrix(pdof_vals, ncol = 1) else NULL,
    df_m = if (exists("df_vals")) matrix(df_vals, ncol = 1) else NULL
  )

  } else {
    # ========================================================================
    # MULTI-D CASE (dx >= 2): Unified implementation for 2D, 3D, 4D, etc.
    # Uses dynamic N-D array operations for dimension-agnostic processing
    # ========================================================================

    dx <- regs$dx
    Tx <- nrow(regs$xraw)
    dh <- nrow(regs$h)
    dy <- ncol(regs$y)
    num_dof_sample <- regs$opt$num_dof_sample
    calc_dof <- regs$opt$calc_dof

    # ========================================================================
    # STEP 1: Compute DOF using Hutchinson's trace estimator
    # ========================================================================
    if (calc_dof) {
      # Generate random vectors: p ~ N(0, I)
      # Support external random vectors for exact reproducibility
      if (!is.null(regs$opt$dof_random_vectors)) {
        p <- regs$opt$dof_random_vectors
        # Validate dimensions
        if (nrow(p) != Tx || ncol(p) != num_dof_sample) {
          stop(sprintf(
            "dof_random_vectors must be (%d x %d), got (%d x %d)",
            Tx, num_dof_sample, nrow(p), ncol(p)
          ))
        }
      } else {
        # Generate random probe vectors for DOF estimation
        # NOTE: R's rnorm will NOT match MATLAB's randn (different algorithms)
        # For cross-language tests, use opt$dof_random_vectors from MATLAB
        set.seed(42)  # Fixed seed for R-internal reproducibility
        p <- matrix(rnorm(Tx * num_dof_sample), nrow = Tx, ncol = num_dof_sample)
      }

      # Call fastlpr_reg with MULTI-COLUMN p
      # Output shape: (N[1], N[2], ..., N[dx], dh, num_dof_sample)
      mq_p <- fastlpr_reg(regs, p)

      # Verify output shape dynamically
      expected_dims <- c(regs$N, dh, num_dof_sample)
      actual_dims <- dim(mq_p)
      if (length(actual_dims) != length(expected_dims) || !all(actual_dims == expected_dims)) {
        stop(sprintf(
          "fastlpr_reg output shape mismatch: expected (%s), got (%s)",
          paste(expected_dims, collapse=", "),
          paste(actual_dims, collapse=", ")
        ))
      }

      # Reshape mq_p for interpolation: (N[1], ..., N[dx], dh*num_dof_sample)
      n_grid <- prod(regs$N)
      mq_p_flat <- array(mq_p, dim = c(regs$N, dh * num_dof_sample))

      # Batch interpolation using fastlpr_gridinterp (supports any dx)
      interp <- fastlpr_gridinterp(regs, mq_p_flat)
      phat_flat_all <- interp$evaluate(regs$xraw)  # (Tx, dh*num_dof_sample)
      phat_flat <- matrix(phat_flat_all, nrow = Tx, ncol = dh * num_dof_sample)

      # Reshape phat_flat: (Tx, dh*num_dof_sample) -> (Tx, dh, num_dof_sample)
      phat <- array(phat_flat, dim = c(Tx, dh, num_dof_sample))

      # ======================================================================
      # Compute tr(H) using Hutchinson's method (vectorized)
      # ======================================================================

      # p shape: (Tx, num_dof_sample)
      # phat shape: (Tx, dh, num_dof_sample)

      # Compute p dot p: sum(p^2) for each sample
      p_dot_p <- colSums(p * p)  # Shape: (num_dof_sample,)

      # VECTORIZED broadcasting: replicate p to (Tx, dh, num_dof_sample)
      p_broadcast <- aperm(array(p, dim = c(Tx, num_dof_sample, dh)), c(1, 3, 2))

      # Element-wise multiplication and sum over Tx
      product <- p_broadcast * phat  # Shape: (Tx, dh, num_dof_sample)
      product_2d <- matrix(product, nrow = Tx, ncol = dh * num_dof_sample)
      sums <- Re(colSums(product_2d))  # Re() for complex data DOF
      p_dot_phat <- matrix(sums, nrow = dh, ncol = num_dof_sample)  # Shape: (dh, num_dof_sample)

      # Estimate tr(H) for each bandwidth and sample
      p_dot_p_broadcast <- matrix(rep(p_dot_p, each = dh), nrow = dh, ncol = num_dof_sample)
      trace_H <- Tx * (p_dot_phat / p_dot_p_broadcast)  # Shape: (dh, num_dof_sample)

      # Handle NaN values
      trace_H[is.nan(trace_H)] <- Tx

      # Clamp trace_H to valid range [0, Tx] (reshape may avoid a copy when unshared)
      trace_H_dims <- dim(trace_H)
      trace_H <- pmax(0, pmin(Tx, trace_H))
      dim(trace_H) <- trace_H_dims

      # Compute mean and stderr across samples for each bandwidth
      dof_mean <- rowMeans(trace_H)  # Shape: (dh,)
      if (num_dof_sample > 1) {
        dof_stderr <- row_sd(trace_H) / sqrt(num_dof_sample)  # Shape: (dh,)
      } else {
        dof_stderr <- rep(0, dh)
      }

      # Compute penalty mean and SD
      pdof_samples <- 1 - trace_H / Tx  # Shape: (dh, num_dof_sample)
      pdof_samples_dims <- dim(pdof_samples)
      # Two-step clamping: first to [0,1], then enforce minimum 0.01
      pdof_samples_clamped <- pmax(0, pmin(1, pdof_samples))  # First: clamp to [0,1]
      pdof_samples_clamped <- pmax(0.01, pdof_samples_clamped)  # Then: clamp min to 0.01
      dim(pdof_samples_clamped) <- pdof_samples_dims  # reshape may avoid a copy when unshared
      penalty_samples <- 1 / pdof_samples_clamped^2  # Shape: (dh, num_dof_sample)

      penalty_mean <- rowMeans(penalty_samples)  # Shape: (dh,)
      if (num_dof_sample > 1) {
        penalty_sd <- row_sd(penalty_samples)  # Shape: (dh,)
      } else {
        penalty_sd <- rep(0, dh)
      }

      # Store results
      df_m <- matrix(dof_mean, ncol = 1)
      df_sd <- matrix(dof_stderr, ncol = 1)
      pdof_inv_m <- matrix(penalty_mean, ncol = 1)
      pdof_inv_sd <- matrix(penalty_sd, ncol = 1)
      pdof_m <- matrix(rowMeans(pdof_samples_clamped), ncol = 1)
      pdof_sd <- matrix(if (num_dof_sample > 1) row_sd(pdof_samples_clamped) else rep(0, dh), ncol = 1)

    } else {
      # calc_dof = FALSE: Skip DOF computation
      pdof_inv_m <- NULL
      pdof_inv_sd <- NULL
      df_m <- NULL
      df_sd <- NULL
      pdof_m <- NULL
      pdof_sd <- NULL
    }

    # ========================================================================
    # STEP 2: Compute fitted values on grid for actual data
    # ========================================================================
    mq <- fastlpr_reg(regs, regs$y)

    # Interpolate from grid to data points for all bandwidths
    N <- regs$N
    # FIX: Use complex matrix if mq is complex (for complex-valued y data)
    if (is.complex(mq)) {
      yhat_data <- matrix(complex(1), nrow = Tx, ncol = dh)
    } else {
      yhat_data <- matrix(0, nrow = Tx, ncol = dh)
    }
    mq_dims <- dim(mq)

    # Handle singleton dimension for single response
    # mq may be (N[1], ..., N[dx], dh, 1) for single response
    if (length(mq_dims) == dx + 2 && mq_dims[dx + 2] == 1) {
      dim(mq) <- mq_dims[1:(dx+1)]  # Remove singleton (reshape may avoid a copy when unshared)
      mq_dims <- dim(mq)
    }

    for (ih in seq_len(dh)) {
      # Extract values for this bandwidth using dynamic slicing
      if (length(mq_dims) == dx + 1) {
        # Single response: mq is (N[1], ..., N[dx], dh)
        values <- extract_slice_nd(mq, dx, ih = ih, iy = NULL)
      } else if (length(mq_dims) == dx + 2) {
        # Multiple responses: mq is (N[1], ..., N[dx], dh, dy)
        values <- extract_slice_nd(mq, dx, ih = ih, iy = 1)
      } else {
        stop(sprintf("Unexpected mq dimensions for dx=%d: %s", dx, paste(mq_dims, collapse=" x ")))
      }

      # Interpolate to data points
      interp <- fastlpr_gridinterp(regs, values)
      yhat_data[, ih] <- interp$evaluate(regs$xraw)
    }

    # ========================================================================
    # STEP 3: Compute GCV scores and apply 1-SE rule
    # ========================================================================
    if (calc_dof && !is.null(pdof_inv_m)) {
      # VECTORIZED GCV: Process all bandwidths at once
      y_vec <- as.vector(regs$y)  # Convert to vector for broadcasting
      # Subtract y from each column of yhat_data
      residual <- sweep(yhat_data, 1, y_vec, "-")  # (Tx, dh) matrix
      rss <- colSums(abs(residual)^2)  # Vector of length dh
      mse <- colMeans(abs(residual)^2)  # Vector of length dh

      # Mark bandwidths with essentially zero yhat as NaN (match MATLAB behavior)
      yhat_var <- apply(yhat_data, 2, var)  # Variance for each bandwidth
      invalid_bandwidth <- yhat_var < .Machine$double.eps * 1e6  # Essentially constant
      mse[invalid_bandwidth] <- NaN  # Mark as invalid (will be skipped by which.min)

      gcv_scores <- mse * pdof_inv_m[, 1]  # Vector of length dh

      # Find minimum GCV bandwidth
      idmin <- which.min(gcv_scores)

      # Apply 1-SE rule (select largest bandwidth within 1 SE of minimum)
      dstd <- regs$opt$dstd
      semin <- gcv_scores[idmin] + dstd * (mse[idmin] * pdof_inv_sd[idmin, 1])

      # Find bandwidths with GCV <= threshold
      ihgood <- which(gcv_scores <= semin)

      # Among these, select largest bandwidth (sum of h)
      if (length(ihgood) > 0) {
        hsum <- rowSums(regs$h[ihgood, , drop = FALSE])
        id1se <- ihgood[which.max(hsum)]
      } else {
        id1se <- idmin
      }

      # Extract selected bandwidth
      h1se <- regs$h[id1se, ]
      hmin <- regs$h[idmin, ]

      # Store results
      regs$yhat <- yhat_data[, id1se]
      regs$yhat_data <- yhat_data
      regs$dof <- df_m[id1se, 1]
      regs$df_m <- df_m
      regs$df_sd <- df_sd
      regs$pdof_m <- pdof_m
      regs$pdof_sd <- pdof_sd
      regs$pdof_inv_m <- pdof_inv_m
      regs$pdof_inv_sd <- pdof_inv_sd

      # Create griddedInterpolant for 1-SE fit
      if (length(mq_dims) == dx + 1) {
        # Single response: mq is (N[1], ..., N[dx], dh)
        values_1se <- extract_slice_nd(mq, dx, ih = id1se, iy = NULL)
        values_1se_flat <- as.vector(values_1se)  # For fgrid storage
      } else if (length(mq_dims) == dx + 2) {
        # Multi-response: mq is (N[1], ..., N[dx], dh, dy)
        # Use drop=FALSE to preserve dimensions
        dy_actual <- mq_dims[dx + 2]

        # Extract all responses at selected bandwidth
        values_1se_list <- lapply(1:dy_actual, function(iy) {
          extract_slice_nd(mq, dx, ih = id1se, iy = iy)
        })

        # Flatten spatial dimensions for fgrid storage: (prod(N), dy)
        n_grid <- prod(N)
        values_1se_flat <- matrix(0, nrow = n_grid, ncol = dy_actual)
        for (i in 1:dy_actual) {
          values_1se_flat[, i] <- as.vector(values_1se_list[[i]])
        }

        # For interpolator, use first response
        values_1se <- values_1se_list[[1]]
      } else {
        stop("Unexpected mq dimensions")
      }

      fpp_yhat <- fastlpr_gridinterp(regs, values_1se)
      regs$fpp_yhat <- fpp_yhat

      # Store GCV results
      regs$gcv_yhat <- list(
        gcv_m = matrix(gcv_scores, ncol = 1),
        gcv_sd = matrix(mse * pdof_inv_sd[, 1], ncol = 1),
        hmin = matrix(hmin, nrow = dx),
        h1se = matrix(h1se, nrow = dx),
        h = matrix(h1se, nrow = dx),  # Alias for h1se (API compatibility)
        idmin = idmin,
        id1se = id1se,
        mse = matrix(mse, ncol = 1),
        rss = matrix(rss, ncol = 1),
        yhat_1se = yhat_data[, id1se],
        pdof_m = pdof_m,
        df_m = df_m,
        fgrid = values_1se_flat,  # Fitted values on evaluation grid (flattened)
        xlist = regs$grid  # Add grid points for plotting
      )

      # Display selected bandwidth
      cat(sprintf("[fastlpr]: [%dd] regression with order %d\n", dx, regs$opt$order))
      cat(sprintf(" - selected bandwidth - [%dth] h\n", id1se))
      cat(sprintf(" - normalized scale h = [%s]\n", paste(round(h1se, 4), collapse = ", ")))
      cat(sprintf(" - original scale h = [%s]\n", paste(round(h1se * regs$x_std, 4), collapse = ", ")))

    } else {
      # No GCV computed: return FIRST bandwidth's fitted values
      regs$yhat <- yhat_data[, 1]  # Select first bandwidth
      regs$yhat_data <- yhat_data  # Keep all bandwidths for reference

      # Create fpp_yhat for single bandwidth case
      # This was missing, causing benchmark failures for N <= 65536
      if (length(mq_dims) == dx + 1) {
        # Single response: mq is (N[1], ..., N[dx], dh)
        values_single <- extract_slice_nd(mq, dx, ih = 1, iy = NULL)
      } else if (length(mq_dims) == dx + 2) {
        # Multi-response: mq is (N[1], ..., N[dx], dh, dy)
        values_single <- extract_slice_nd(mq, dx, ih = 1, iy = 1)
      } else {
        values_single <- mq  # Fallback
      }
      regs$fpp_yhat <- fastlpr_gridinterp(regs, values_single)

      regs$gcv_yhat <- list(
        h1se = matrix(regs$h[1, ], nrow = dx),  # First bandwidth
        h = matrix(regs$h[1, ], nrow = dx),     # Alias for h1se (API compatibility)
        hmin = matrix(regs$h[1, ], nrow = dx),
        id1se = 1,
        idmin = 1
      )
      regs$df_m <- df_m
      regs$pdof_m <- pdof_m
    }
  }

  # ==========================================================================
  # VARIANCE ESTIMATION: Apply inverse transform
  # Transform back from log scale and apply normalization factor d
  # ==========================================================================
  if (!is.null(y_raw_for_variance)) {
    # Ty is the number of DATA POINTS
    Ty <- nrow(y_raw_for_variance)

    # Compute normalization factor d
    # MATLAB: d = 1./((1/regs.Ty).*sum(yraw.*exp(-regs.yhat)))
    d_normalization <- 1.0 / ((1.0 / Ty) * sum(y_raw_for_variance * exp(-regs$yhat)))

    # Transform yhat back from log scale
    # MATLAB: regs.yhat = exp(regs.yhat)./d
    regs$yhat <- exp(regs$yhat) / d_normalization

    # Also transform yhat_data if it exists (for all bandwidths)
    if (!is.null(regs$yhat_data)) {
      # For yhat_data, need to compute d for each bandwidth
      # But simplified: use the selected bandwidth's d
      regs$yhat_data <- exp(regs$yhat_data) / d_normalization
    }

    # Store normalization factor (for prediction at new points)
    regs$d <- d_normalization

    # Restore original y (before log transform) for residual computation
    regs$y <- y_raw_for_variance
  }

  regs
}
