# Copyright (c) 2024-2025 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later

#' Mirrors: fastLPR/utility/core/ fastLPR_create.m
#'
#' Initializes the regression structure used by the fastLPR pipeline.
#' Mirrors the MATLAB helper line-by-line, relying on the exact utility
#' functions defined in fastlpr_exact_utils.R and existing helpers such as
#' `set_defaults` and `multispace`.
#'
#' @keywords internal
fastlpr_create <- function(x, y, h = NULL, opt = NULL) {
  if (is.null(opt)) opt <- list()
  if (is.null(h)) h <- matrix(numeric(0), nrow = 0)

  # Process complex-valued predictors (split columns with imaginary part)
  x_is_complex <- fastlpr_is_complex(x, 1)
  if (is.matrix(x)) {
    real_part <- Re(x)
  } else {
    real_part <- matrix(Re(x), ncol = 1)
  }
  if (any(x_is_complex)) {
    imag_cols <- Im(x)[, x_is_complex, drop = FALSE]
    x <- cbind(real_part, imag_cols)
  } else {
    x <- real_part
  }

  Tx <- nrow(x)
  dx <- ncol(x)

  if (length(dim(y)) == 2L) {
    Ty <- nrow(y)
    dy <- ncol(y)
    opt$y_corr_bandwidth <- FALSE
  } else if (length(dim(y)) == 3L) {
    dims_y <- dim(y)
    Ty <- dims_y[1]
    dhy <- dims_y[2]
    dy <- dims_y[3]
    opt$y_corr_bandwidth <- TRUE
  } else {
    stop("fastlpr_create: y must be a matrix or 3D array.")
  }

  if (Tx != Ty) {
    stop(
      sprintf(
        "fastlpr_create: x (%d rows) and y (%d rows) must match.",
        Tx,
        Ty
      )
    )
  }

  # Default options (mirroring MATLAB set_defaults calls)
  opt <- set_defaults(opt, "y_type_out", "mean")
  opt <- set_defaults(opt, "kernel_type", "gaussian")
  opt <- set_defaults(opt, "order", 0)
  opt <- set_defaults(opt, "Nratio", 1)
  default_N <- ceiling(opt$Nratio * (Tx^(1 / dx)))
  opt <- set_defaults(opt, "N", default_N)
  N <- opt$N
  if (length(N) == 1L) {
    N <- rep(N, dx)
  }
  opt$N <- N
  opt$Nratio <- N[1] / (Tx^(1 / dx))

  # FFT padding: use power-of-2 length for speed and accuracy
  # Must match MATLAB default (flag_power2=true) for cross-language consistency
  # Note: For 3D, power-of-2 causes L³ to explode (e.g., N=50 -> L=128 -> 15x more memory)
  # but is required for correct accuracy. Use flag_power2=FALSE manually if memory-constrained.
  opt <- set_defaults(opt, "flag_power2", TRUE)
  opt <- set_defaults(opt, "accuracy", max(6 - ceiling(log10(Tx)), 4))
  opt <- set_defaults(opt, "nufft_deconv", opt$accuracy > 6)
  opt <- set_defaults(opt, "dstd", 0)
  opt <- set_defaults(opt, "calc_dof", TRUE)
  opt <- set_defaults(opt, "num_dof_sample", 10)

  if (!is.list(opt$y_grid_opt)) {
    opt$y_grid_opt <- list(
      # Changed from "spline" to "linear" to match:
      # 1. MATLAB's griddedInterpolant('linear') used in cv_fastKDE
      # 2. The hardcoded "linear" in fastkde_lcv() for LCV computation
      # This ensures consistency between LCV values and fpp.evaluate()
      Method = "linear",
      ExtrapolationMethod = "linear"
    )
  }

  opt <- set_defaults(opt, "compact", TRUE)
  opt <- set_defaults(opt, "y_corr_bandwidth", FALSE)
  opt <- set_defaults(opt, "y_gcv_metric", NULL)
  opt <- set_defaults(opt, "y_grid_method", "griddedInterpolant")
  opt <- set_defaults(opt, "isgpu", FALSE)

  dtype <- typeof(y)

  converted <- fastlpr_bit_convert(opt$accuracy, x, y)
  x <- converted[[1]]
  y <- converted[[2]]
  dtype <- typeof(y)

  ndcolon <- replicate(dx, quote(expr = ), simplify = FALSE)

  if (isTRUE(opt$flag_power2)) {
    L <- 2^ceiling(log2(2 * N - 1))
  } else {
    L <- N + 2
  }
  if (length(L) == 1L) {
    L <- rep(L, dx)
  }

  # Ensure x is a matrix before saving to xraw
  if (!is.matrix(x)) {
    x <- as.matrix(x)
  }

  xraw <- x
  zs <- zscore(x)
  x <- zs$z

  # Ensure x is a matrix (zscore should return matrix, but safeguard)
  if (!is.matrix(x)) {
    x <- as.matrix(x)
  }

  x_min <- apply(x, 2, min)
  x_max <- apply(x, 2, max)
  x_scale <- x_max - x_min
  x_delta <- x_scale / N

  x <- sweep(
    x,
    2,
    (x_max + x_min) / 2,
    FUN = "-"
  )
  x_min <- apply(x, 2, min)
  x_max <- apply(x, 2, max)
  x_mean <- colMeans(x)

  qin <- matrix(0, nrow = 2, ncol = dx)
  pad_left <- floor((L - N) / 2)
  qin[1, ] <- pad_left + 1
  qin[2, ] <- pad_left + N

  # MATLAB passes z-scored x to fastLPR_kdf (see fastLPR_create.m:228)
  # Grid is created from z-scored data for kernel computation
  kdf_result <- fastlpr_kdf(x, h, N, opt)
  kdf <- kdf_result$kdf
  ihbad <- kdf_result$ihbad
  dh <- kdf_result$dh
  dh_input <- kdf_result$dh_input
  h <- kdf_result$h
  lwp <- kdf_result$lwp

  if (opt$y_corr_bandwidth && exists("dhy") && dh != dhy && dhy != 1) {
    stop(
      sprintf(
        "fastlpr_create: bandwidth count (%d) must match responses (%d)",
        dh,
        dhy
      )
    )
  }

  if (dx == 1) {
    idx_knot <- seq.int(qin[1], qin[2])
    idx_range <- seq(-0.5, 0.5 - 1 / L, length.out = L)
    hf <- 1 / L

    if (N != L) {
      knot_scale <- idx_range[idx_knot[length(idx_knot)]] - idx_range[idx_knot[1]]
      knot <- (x - x_min) / x_scale * knot_scale + idx_range[idx_knot[1]]
    } else {
      knot_scale <- idx_range[length(idx_range)] - idx_range[1]
      knot <- (x - x_min) / x_scale * knot_scale - 0.5
    }
  } else {
    idx_knot <- vector("list", dx)
    idx_range <- vector("list", dx)
    hf <- numeric(dx)
    knot <- matrix(0, nrow = Tx, ncol = dx)
    knot_scale <- numeric(dx)

    for (d in seq_len(dx)) {
      idx_knot[[d]] <- seq.int(qin[1, d], qin[2, d])
      idx_range[[d]] <- seq(-0.5, 0.5 - 1 / L[d], length.out = L[d])
      hf[d] <- 1 / L[d]

      if (N[d] != L[d]) {
        knot_scale[d] <- idx_range[[d]][idx_knot[[d]][length(idx_knot[[d]])]] -
          idx_range[[d]][idx_knot[[d]][1]]
        knot[, d] <- (x[, d] - x_min[d]) / x_scale[d] * knot_scale[d] +
          idx_range[[d]][idx_knot[[d]][1]]
      } else {
        knot_scale[d] <- idx_range[[d]][length(idx_range[[d]])] - idx_range[[d]][1]
        knot[, d] <- (x[, d] - x_min[d]) / x_scale[d] * knot_scale[d] - 0.5
      }
    }
  }

  grid_params <- fastlpr_compute_grid_params(Ty, opt$accuracy)
  Msp <- grid_params$Msp

  if (dx == 1) {
    qout_val <- (L - N) / 2
    qout <- c(ceiling(qout_val) + 1, ceiling(qout_val) + N)
  } else {
    # NOTE (2025-11-28): Use rbind (not cbind) to match MATLAB format
    # MATLAB qout format: row 1 = start indices, row 2 = end indices
    # Each column corresponds to one dimension
    qout <- rbind(
      ceiling((L - N) / 2) + 1,  # Row 1: start indices for all dims
      ceiling((L - N) / 2) + N   # Row 2: end indices for all dims
    )
  }

  xlist <- multispace(
    apply(xraw, 2, min),
    apply(xraw, 2, max),
    N
  )

  if (!is.list(xlist)) {
    xlist <- lapply(seq_len(dx), function(d) xlist[d, ])
  }
  xlist <- lapply(xlist, as.numeric)

  regs <- list(
    xraw = xraw,
    x = x,
    y = y,
    Tx = Tx,
    Ty = Ty,
    dx = dx,
    dy = dy,
    dtype = dtype,
    Msp = Msp,
    L = L,
    N = N,
    ndcolon = ndcolon,
    x_min = x_min,
    x_max = x_max,
    x_scale = x_scale,
    x_mean = x_mean,
    x_std = zs$sigma,
    x_delta = x_delta,
    h = h,
    kdf = kdf,
    ihbad = ihbad,
    dh = dh,
    dh_input = dh_input,
    qin = qin,
    qout = qout,
    knot_scale = if (dx == 1) knot_scale else knot_scale,
    knot = knot,
    hf = hf,
    xlist = xlist,
    opt = opt,
    y_isreal = !fastlpr_is_complex(y, "all"),
    lwp = lwp
  )

  regs
}