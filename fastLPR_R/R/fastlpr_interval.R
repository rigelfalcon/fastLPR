# Copyright (c) 2024-2025 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later

#' Compute pointwise intervals for regression estimates
#'
#' Port from fastLPR/utility/fastLPR_interval.m
#'
#' Two interval types are supported:
#' \itemize{
#'   \item \code{confidence}: Approximate asymptotic pointwise confidence interval
#'     for the conditional mean m(x): CI = m_hat +/- z * se(m_hat),
#'     where se^2 = sigma^2 * nu / (|H| * s_0).
#'   \item \code{prediction}: Pointwise prediction interval for a new observation
#'     y|x: PI = m_hat +/- z * sqrt(sigma^2 + se^2).
#' }
#'
#' @param mu Mean estimate from cv_fastlpr (must contain fpp_s0 and s0 fields)
#' @param sigma Variance estimate from cv_fastlpr OR scalar/vector
#' @param alpha Significance level (default: 0.05)
#' @param type Interval type: "confidence" (default) or "prediction"
#'
#' @return ci List with $Values (array, last dim = [upper, lower]) and $GridVectors
#'
#' @note Intervals are pointwise, not simultaneous. The CI is an asymptotic
#'   approximation valid at interior points. No boundary or bias correction.
#'
#' @export
fastlpr_interval <- function(mu, sigma, alpha = 0.05, type = "confidence") {
  # Input validation
  if (!is.list(mu) || !("fpp_yhat" %in% names(mu))) {
    stop("mu must be a list with field fpp_yhat (output from cv_fastlpr)")
  }
  if (!(is.list(sigma) || is.numeric(sigma))) {
    stop("sigma must be a list (from cv_fastlpr) or numeric")
  }
  if (is.list(sigma) && !("fpp_yhat" %in% names(sigma))) {
    stop("If sigma is a list, it must have field fpp_yhat")
  }
  if (!is.numeric(alpha) || length(alpha) != 1 || alpha <= 0 || alpha >= 1) {
    stop("alpha must be a scalar in (0, 1)")
  }
  type <- tolower(type)
  if (!type %in% c("confidence", "prediction")) {
    stop("type must be 'confidence' or 'prediction'")
  }
  if (is.null(mu$s0)) {
    stop("mu$s0 is missing. Re-run cv_fastlpr with a recent version.")
  }

  # nu constants for Gaussian kernel (precomputed)
  nu_table <- list(
    "1_0" = 0.282094791774, "1_1" = 0.282094791774, "1_2" = 0.476034961118,
    "2_0" = 0.079577471546, "2_1" = 0.079577471546, "2_2" = 0.198943678865,
    "3_0" = 0.022448390266, "3_1" = 0.022448390266, "3_2" = 0.077166341538
  )

  # z-score
  z <- qnorm(1 - alpha / 2)

  # Detect interpolant type
  is_structured_interp <- is.list(mu$fpp_yhat) &&
                         "GridVectors" %in% names(mu$fpp_yhat) &&
                         "Values" %in% names(mu$fpp_yhat)

  # Get grid vectors and values
  if (is_structured_interp) {
    grid_vectors <- mu$fpp_yhat$GridVectors
    dx <- length(grid_vectors)
    mu_values <- mu$fpp_yhat$Values
    if (is.list(sigma)) {
      sigma_values <- sigma$fpp_yhat$Values
    } else {
      sigma_values <- sigma
    }
  } else {
    # Simple approxfun (1D)
    if (!("xlist" %in% names(mu))) {
      stop("For simple interpolator, mu must have xlist field")
    }
    grid_vectors <- mu$xlist
    dx <- length(grid_vectors)
    grid_x <- grid_vectors[[1]]
    mu_values <- mu$fpp_yhat(grid_x)
    # For sigma: evaluate on the SAME grid as mu.
    # sigma$fpp_yhat may have pre-transform (log-scale) values for variance
    # mode, so we interpolate the transformed yhat onto mu's grid.
    if (is.list(sigma)) {
      if (length(sigma$yhat) == length(grid_x)) {
        # Same length as grid -- use directly
        sigma_values <- sigma$yhat
      } else {
        # yhat is at sample points (N != grid size) -- interpolate onto grid
        # Use xraw (original sample x) as interpolation source
        if (!is.null(sigma$xraw)) {
          x_src <- sort(sigma$xraw[, 1])
          y_src <- sigma$yhat[order(sigma$xraw[, 1])]
        } else {
          # Fallback: assume yhat is ordered by xlist
          x_src <- seq(min(grid_x), max(grid_x), length.out = length(sigma$yhat))
          y_src <- sigma$yhat
        }
        sigma_interp <- approxfun(x_src, y_src, rule = 2)
        sigma_values <- sigma_interp(grid_x)
      }
    } else {
      sigma_values <- sigma
    }
  }

  # Extract parameters for se computation
  ell <- mu$opt$order
  h <- mu$gcv_yhat$h1se
  if (is.null(h)) {
    if (!is.null(mu$h)) {
      h <- mu$h
    } else {
      stop("Cannot determine bandwidth. mu must have gcv_yhat$h1se or h field.")
    }
  }
  prod_h <- prod(h)

  nu_key <- paste0(dx, "_", ell)
  nu <- nu_table[[nu_key]]
  if (is.null(nu)) {
    stop(sprintf("nu not available for d=%d, order=%d", dx, ell))
  }

  # Get s_0 values on grid
  if (is_structured_interp) {
    # Multi-D: s0 is an array matching grid shape
    s0_values <- mu$s0
  } else {
    # 1D: evaluate s0 interpolant on grid
    if (is.function(mu$fpp_s0)) {
      s0_values <- mu$fpp_s0(grid_x)
    } else {
      s0_values <- mu$s0
    }
  }
  s0_values <- pmax(s0_values, 1e-10)

  # Grid compatibility check
  if (length(mu_values) != length(s0_values)) {
    stop(sprintf("Shape mismatch: mu has %d elements but s0 has %d. Ensure same grid.",
                 length(mu_values), length(s0_values)))
  }
  if (length(sigma_values) != length(s0_values)) {
    stop(sprintf("Shape mismatch: sigma has %d elements but s0 has %d. Ensure same grid.",
                 length(sigma_values), length(s0_values)))
  }

  # Compute se^2 = sigma^2 * nu / (|H| * s_0)
  sigma_pos <- pmax(sigma_values, 0)
  se_sq <- sigma_pos * nu / (prod_h * s0_values)

  # Compute bounds
  if (type == "confidence") {
    half_width <- z * sqrt(se_sq)
  } else {
    half_width <- z * sqrt(sigma_pos + se_sq)
  }

  upper <- mu_values + half_width
  lower <- mu_values - half_width

  # Squeeze trailing singleton dimensions
  mu_dims <- dim(mu_values)
  if (!is.null(mu_dims) && length(mu_dims) > 1 && mu_dims[length(mu_dims)] == 1) {
    mu_dims <- mu_dims[-length(mu_dims)]
    dim(mu_values) <- mu_dims
    dim(upper) <- mu_dims
    dim(lower) <- mu_dims
  }

  # Stack along new dimension
  if (is.null(mu_dims) || length(mu_dims) <= 1) {
    ci_values <- cbind(upper, lower)
  } else {
    ci_values <- array(0, dim = c(mu_dims, 2))
    if (length(mu_dims) == 2) {
      ci_values[, , 1] <- upper
      ci_values[, , 2] <- lower
    } else if (length(mu_dims) == 3) {
      ci_values[, , , 1] <- upper
      ci_values[, , , 2] <- lower
    }
  }

  ci <- list(
    Values = ci_values,
    GridVectors = grid_vectors
  )

  return(ci)
}
