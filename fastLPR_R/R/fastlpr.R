# Copyright (c) 2024-2025 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later

#' Fast Local Polynomial Regression (fastLPR) - Main Implementation
#' This file combines all essential fastLPR functions into a single cohesive module
#'
#' Author: Ying Wang, Min Li (Original MATLAB)
#'         R Translation: Claude Assistant
#' Create Time: 2025
#' License: GNU General Public License v3.0

# Source required dependencies
if (!requireNamespace("Matrix", quietly = TRUE)) {
  stop("Package 'Matrix' is required but not installed")
}
if (!requireNamespace("akima", quietly = TRUE)) {
  stop("Package 'akima' is required but not installed")
}
if (!requireNamespace("interp", quietly = TRUE)) {
  stop("Package 'interp' is required but not installed")
}

#' Fast Local Polynomial Regression with NUFFT acceleration
#'
#' @param x Input covariate(s) - matrix of size n x d (d = dimension)
#' @param y Response variable - vector of size n x 1
#' @param x0 Points for estimation - matrix of size n0 x d
#' @param order Polynomial order (0 = Nadaraya-Watson, 1 = local linear)
#' @param h Bandwidth (vector of length d for anisotropic bandwidths)
#' @param kernel Kernel function ('gaussian', 'epanechnikov', 'tricube')
#' @param method Method for estimation ('NUFFT', 'direct')
#'
#' @return List containing:
#'         - mu: Estimated function values at x0
#'         - dof: Degrees of freedom
#'         - cv_scores: Cross-validation scores (if applicable)
#'         - bandwidth: Final bandwidth used
#'         - execution_time: Computation time
#'
#' @examples
#' \dontrun{
#' # 1D example
#' n <- 200
#' x <- matrix(runif(n), ncol = 1)
#' y <- sin(2 * pi * x) + rnorm(n, sd = 0.1)
#' result <- fastlpr(x, y, order = 1, h = 0.1)
#' plot(x, y); lines(sort(x), result$mu[order(x)], col = 'red', lwd = 2)
#' }
#'
#' @keywords internal
fastlpr <- function(x, y, x0 = NULL, order = 1, h = NULL,
                    kernel = "gaussian", method = "NUFFT") {

  # Start timing
  start_time <- proc.time()

  # Input validation and preprocessing
  x <- as.matrix(x)
  y <- as.numeric(y)
  n <- nrow(x)
  d <- ncol(x)

  # If x0 not provided, use original x points
  if (is.null(x0)) {
    x0 <- x
    n0 <- n
  } else {
    x0 <- as.matrix(x0)
    n0 <- nrow(x0)
  }

  # Validate dimensions
  if (length(y) != n) {
    stop("Length of y must equal number of rows in x")
  }
  if (ncol(x0) != d) {
    stop("x and x0 must have same number of columns")
  }

  # Default bandwidth selection if not provided
  if (is.null(h)) {
    if (d == 1) {
      h <- bw.nrd0(x)  # Silverman's rule for 1D
    } else {
      h <- apply(x, 2, function(col) bw.nrd0(col))
    }
  }

  # Convert h to vector if scalar
  if (length(h) == 1) {
    h <- rep(h, d)
  }

  # Create regression control structure
  regs <- list(
    x = x,
    y = y,
    x0 = x0,
    order = order,
    h = h,
    kernel = kernel,
    method = method,
    n = n,
    d = d,
    n0 = n0,
    dx = d,  # dimension index
    dh = length(h),  # number of bandwidths
    Tx = n0,  # target points
    x_delta = diff(apply(x, 2, range)) / (n - 1),
    dof = NULL,
    cv_scores = NULL
  )

  # Data standardization for NUFFT
  if (method == "NUFFT") {
    x_min <- apply(x, 2, min)
    x_max <- apply(x, 2, max)
    # Use sweep() for memory-efficient broadcasting (avoids full-size matrix replication)
    x_range <- x_max - x_min
    x_std <- sweep(sweep(x, 2, x_min, "-"), 2, x_range, "/")
    x0_std <- sweep(sweep(x0, 2, x_min, "-"), 2, x_range, "/")

    regs$x_std <- x_std
    regs$x0_std <- x0_std
    regs$x_min <- x_min
    regs$x_max <- x_max
  }

  # Main estimation
  if (method == "NUFFT") {
    regs <- fastlpr_compute_s(regs)
    regs <- fastlpr_compute_dof(regs)
    result <- fastlpr_compute_mu(regs)
  } else {
    # Direct computation method
    result <- fastlpr_direct_computation(regs)
  }

  # End timing
  execution_time <- (proc.time() - start_time)[3]

  # Prepare output
  output <- list(
    mu = result,
    x0 = x0,
    dof = regs$dof,
    bandwidth = h,
    kernel = kernel,
    method = method,
    order = order,
    execution_time = execution_time,
    n = n,
    d = d
  )

  class(output) <- "fastlpr"
  return(output)
}

#' Enhanced helper function to compute design matrix with NUFFT
#' @noRd
fastlpr_compute_s <- function(regs) {
  # Compute kernel density function (KDF) in Fourier domain
  regs <- fastlpr_compute_kdf(regs)

  # Compute NUFFT of response data
  y_ft <- fastlpr_nufft(regs, regs$y)

  # Apply kernel density function in Fourier domain
  if (regs$dx == 1) {
    # 1D case: element-wise multiplication
    regs$s <- y_ft * regs$kdf
  } else {
    # Multi-dimensional case: use broadcasting
    regs$s <- y_ft
    for (h_idx in 1:regs$dh) {
      if (regs$dx == 2) {
        regs$s[, , h_idx] <- y_ft[, , h_idx] * regs$kdf[, , h_idx]
      } else if (regs$dx == 3) {
        regs$s[, , , h_idx] <- y_ft[, , , h_idx] * regs$kdf[, , , h_idx]
      }
    }
  }

  return(regs)
}

#' Enhanced helper function to estimate degrees of freedom
#' @noRd
fastlpr_compute_dof <- function(regs) {
  if (regs$opt$order == 0) {
    # Nadaraya-Watson: trace of smoother matrix
    dof <- sum(regs$kdf) * prod(regs$x_delta)
  } else {
    # Local polynomial: more complex DOF estimation
    # Simplified approximation for local polynomial
    dof <- regs$Tx * (regs$order + 1) * 0.6  # Approximate effective DOF
  }

  regs$dof <- dof
  return(regs)
}

#' Enhanced helper function to compute fitted values
#' @noRd
fastlpr_compute_mu <- function(regs) {
  # Inverse NUFFT to get fitted values
  mu_hat <- fastlpr_inufft(regs, regs$s)

  return(mu_hat)
}

#' Direct computation method (fallback for small datasets)
#' @noRd
fastlpr_direct_computation <- function(regs) {
  n0 <- regs$Tx
  d <- regs$dx
  h <- regs$h
  order <- regs$order
  x <- regs$x
  y <- regs$y
  x0 <- regs$x0

  mu_hat <- numeric(n0)

  for (i in 1:n0) {
    # Compute weights
    distances_sq <- rowSums((t(t(x) - x0[i, ]) / h)^2)

    if (regs$kernel == "gaussian") {
      weights <- exp(-0.5 * distances_sq)
    } else if (regs$kernel == "epanechnikov") {
      weights <- pmax(0, 1 - 0.5 * distances_sq)
    } else if (regs$kernel == "tricube") {
      weights <- pmax(0, (1 - distances_sq^(3/2))^3)
    } else {
      weights <- exp(-0.5 * distances_sq)  # default to gaussian
    }

    if (order == 0) {
      # Nadaraya-Watson estimator
      mu_hat[i] <- sum(weights * y) / sum(weights)
    } else if (order == 1) {
      # Local linear estimator
      X_mat <- cbind(1, t(t(x) - x0[i, ]) / h)
      weights_mat <- diag(weights)

      # Weighted least squares
      if (sum(weights) > 1e-10) {
        beta_hat <- solve(t(X_mat) %*% weights_mat %*% X_mat +
                         diag(1e-6, ncol = ncol(X_mat))) %*%
                   t(X_mat) %*% weights_mat %*% y
        mu_hat[i] <- beta_hat[1]
      } else {
        mu_hat[i] <- mean(y)
      }
    }
  }

  return(mu_hat)
}

#' Kernel Density Estimation using fastLPR methodology
#'
#' @param x Data points for density estimation
#' @param h Bandwidth
#' @param x_grid Points where density should be estimated
#' @param kernel Kernel function
#'
#' @return List with density values and metadata
#'
#' @keywords internal
fastkde <- function(x, h = NULL, x_grid = NULL, kernel = "gaussian") {
  x <- as.numeric(x)
  n <- length(x)

  # Default bandwidth if not provided
  if (is.null(h)) {
    h <- bw.nrd0(x)
  }

  # Default grid if not provided
  if (is.null(x_grid)) {
    x_grid <- seq(min(x) - 3*h, max(x) + 3*h, length.out = 512)
  }

  # Use fastlpr with order = 0 for KDE
  result <- fastlpr(
    x = matrix(x, ncol = 1),
    y = rep(1/n, n),  # Unit weights
    x0 = matrix(x_grid, ncol = 1),
    order = 0,  # Nadaraya-Watson
    h = h,
    kernel = kernel,
    method = "NUFFT"
  )

  # Normalize to proper density
  density <- result$mu / h

  # Ensure non-negative and integrate to 1
  density <- pmax(0, density)
  density <- density / (sum(density) * diff(x_grid)[1])

  output <- list(
    x_grid = x_grid,
    density = density,
    bandwidth = h,
    kernel = kernel,
    data = x,
    n = n,
    execution_time = result$execution_time
  )

  class(output) <- "fastkde"
  return(output)
}

#' Cross-validation for bandwidth selection
#'
#' @param x Input covariates
#' @param y Response variable
#' @param order Polynomial order
#' @param h_grid Vector of bandwidths to test
#' @param cv_method CV method ('GCV', 'LCV', 'CV')
#' @param kernel Kernel function
#'
#' @return Optimal bandwidth and CV scores
#'
#' @keywords internal
fastlpr_cv <- function(x, y, order = 1, h_grid = NULL,
                      cv_method = "GCV", kernel = "gaussian") {
  x <- as.matrix(x)
  y <- as.numeric(y)
  n <- nrow(x)

  # Default bandwidth grid if not provided
  if (is.null(h_grid)) {
    h_range <- bw.nrd0(x) * c(0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0)
    h_grid <- h_range[h_range > 0]
  }

  cv_scores <- numeric(length(h_grid))

  for (i in seq_along(h_grid)) {
    h_current <- h_grid[i]

    # Fit fastLPR model
    fit <- fastlpr(
      x = x,
      y = y,
      x0 = x,  # Predict at training points for CV
      order = order,
      h = h_current,
      kernel = kernel
    )

    # Compute CV score
    if (cv_method == "GCV") {
      # Generalized Cross-Validation
      if (fit$dof > 0 && fit$dof < n) {
        residuals <- y - fit$mu
        cv_scores[i] <- mean(residuals^2) / (1 - fit$dof/n)^2
      } else {
        cv_scores[i] <- Inf
      }
    } else if (cv_method == "LCV") {
      # Likelihood Cross-Validation (assuming Gaussian)
      if (order == 0) {
        kde_fit <- fastkde(x, h = h_current)
        log_lik <- sum(log(fastkde_eval(kde_fit, x) + 1e-10))
        cv_scores[i] <- -log_lik  # Negative log-likelihood (minimize)
      } else {
        cv_scores[i] <- Inf
      }
    } else {
      # Leave-One-Out CV (approximate)
      cv_scores[i] <- mean((y - fit$mu)^2) * n / (n - fit$dof)
    }
  }

  # Find optimal bandwidth
  best_idx <- which.min(cv_scores)
  optimal_h <- h_grid[best_idx]

  output <- list(
    optimal_h = optimal_h,
    h_grid = h_grid,
    cv_scores = cv_scores,
    cv_method = cv_method,
    best_idx = best_idx
  )

  return(output)
}

#' Plot method for fastlpr objects
#'
#' @param x fastlpr object
#' @param add Whether to add to existing plot
#' @param col Color for fitted line
#' @param lwd Line width
#' @param ... Additional plot parameters
#'
#' @export
plot.fastlpr <- function(x, add = FALSE, col = "blue", lwd = 2, ...) {
  if (x$d > 1) {
    stop("Plotting only available for 1D regression")
  }

  if (!add) {
    plot(x$x, x$y, ...)
  }

  lines(sort(x$x0), x$mu[order(x$x0)], col = col, lwd = lwd)
}

#' Plot method for fastkde objects
#'
#' @param x fastkde object
#' @param main Plot title
#' @param col Color for density line
#' @param lwd Line width
#' @param ... Additional plot parameters
#'
#' @export
plot.fastkde <- function(x, main = "Kernel Density Estimate",
                        col = "blue", lwd = 2, ...) {
  plot(x$x_grid, x$density, type = "l", main = main,
       xlab = "x", ylab = "Density", col = col, lwd = lwd, ...)
}

# Source NUFFT helper functions
plot.fastkde <- function(x, main = "Kernel Density Estimate",
                        col = "blue", lwd = 2, ...) {
  plot(x$x_grid, x$density, type = "l", main = main,
       xlab = "x", ylab = "Density", col = col, lwd = lwd, ...)
}

# Note: NUFFT helper functions should be sourced separately if needed
# Note: NUFFT helper functions should be sourced separately
