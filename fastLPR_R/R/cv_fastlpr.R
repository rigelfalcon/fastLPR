# Copyright (c) 2024-2025 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later

#' Fast Local Polynomial Regression with automatic bandwidth selection
#'
#' This is the main function for fastLPR toolbox. It performs nonparametric
#' regression using kernel-weighted local polynomial methods with NUFFT
#' (Non-Uniform Fast Fourier Transform) acceleration. The function automatically
#' selects the optimal bandwidth via Generalized Cross-Validation (GCV) when
#' multiple bandwidth candidates are provided.
#'
#' @param x N x d matrix of predictors (N samples, d dimensions).
#'   Can be real or complex-valued. Each row is one observation.
#'   For complex-valued predictors, use x = real + 1i*imag.
#' @param y N x 1 vector of responses.
#'   Can be real or complex-valued. Must have same number of rows as x.
#' @param h Bandwidth parameter(s) (optional, default: automatic selection).
#'   Scalar: same bandwidth for all dimensions.
#'   1 x d vector: different bandwidth per dimension.
#'   k x d matrix: grid of k bandwidth combinations for GCV selection.
#'   Use get_hlist() to generate bandwidth candidates.
#' @param opt Options list (optional) with fields:
#'   \itemize{
#'     \item order: Polynomial order (default: 0)
#'       0 = Nadaraya-Watson (local constant)
#'       1 = Local linear regression
#'       2 = Local quadratic regression
#'     \item kernel_type: Kernel function (default: 'gaussian')
#'       'gaussian' or 'epanechnikov'
#'     \item dstd: Number of DOF samples for variance estimation (default: 0)
#'       Set to 5-20 for heteroscedastic variance estimation
#'     \item y_type_out: Output type (default: 'mean')
#'       'mean' = estimate conditional mean E[Y|X]
#'       'variance' = estimate conditional variance Var[Y|X]
#'     \item N: Grid resolution for evaluation (default: auto)
#'     \item xrange: Evaluation range (default: data range)
#'   }
#'
#' @return Regression results list containing:
#'   \itemize{
#'     \item yhat: Fitted values at evaluation grid points (N_grid x 1)
#'     \item fpp_yhat: Interpolator object for prediction at any point.
#'       Use: y_pred = regs$fpp_yhat(x_new)
#'     \item gcv_yhat: GCV results list (if multiple bandwidths provided)
#'       \itemize{
#'         \item h1se: Selected bandwidth (1-SE rule)
#'         \item hmin: Bandwidth with minimum GCV
#'         \item gcv: GCV values for all bandwidths
#'       }
#'     \item xq: Evaluation grid points (list for multi-dimensional)
#'     \item xlist: Grid vectors (list)
#'     \item opt: Options used for regression
#'     \item xraw: Original predictor data
#'     \item yraw: Original response data
#'   }
#'
#' @examples
#' # Example 1: 1D regression with automatic bandwidth selection
#' x <- matrix(runif(500) * 20, ncol = 1)
#' y <- sin(x) + 0.2 * rnorm(500)
#' hlist <- get_hlist(20, c(0.01, 1), "logspace")
#' opt <- list(order = 1)  # Local linear
#' regs <- cv_fastlpr(x, y, hlist, opt)
#'
#' @export
cv_fastlpr <- function(x, y, h = NULL, opt = NULL) {
  if (!is.matrix(x)) {
    x <- as.matrix(x)
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }

  # Use MATLAB-aligned implementation
  regs <- fastlpr_create(x, y, h, opt)
  regs <- fastlpr_y(regs)

  # Extract s_0 (zero-th kernel moment) for CI/PI construction
  # s_0 = sum_i K_h(x - x_i) = n * f_hat(x)
  tryCatch({
    if (regs$opt$order == 0) {
      s0_all <- regs$s
    } else {
      s0_all <- regs$s[[1]]
    }
    # Drop trailing singleton dimensions (like ns=1), then select bandwidth
    s0_all <- drop(s0_all)
    s0_ndim <- length(dim(s0_all))  # 0 if vector, or ndims of array
    has_bw_dim <- !is.null(dim(s0_all)) && s0_ndim > regs$dx

    if (has_bw_dim && !is.null(regs$gcv_yhat) && !is.null(regs$gcv_yhat$id1se)) {
      ih <- regs$gcv_yhat$id1se
      if (regs$dx == 1) {
        s0_selected <- Re(s0_all[, ih])
      } else {
        idx <- rep(list(quote(expr = )), regs$dx)
        s0_selected <- Re(do.call(`[`, c(list(s0_all), idx, ih)))
      }
    } else {
      # No bandwidth dimension (dh=1 was dropped) or no GCV
      s0_selected <- Re(if (is.null(dim(s0_all))) s0_all else s0_all)
    }
    regs$s0 <- s0_selected
    if (regs$dx == 1) {
      grid_raw <- regs$xlist[[1]]
      regs$fpp_s0 <- approxfun(x = grid_raw, y = s0_selected, method = "linear", rule = 2)
    } else {
      regs$fpp_s0 <- fastlpr_gridinterp(regs, s0_selected)
    }
  }, error = function(e) {
    warning(sprintf("s_0 extraction failed: %s. CI/PI intervals unavailable.", conditionMessage(e)))
    regs$s0 <<- NULL
    regs$fpp_s0 <<- NULL
  })

  regs <- fastlpr_compact(regs)

  class(regs) <- "fastlpr_result"

  return(regs)
}

# Note: The core implementation functions are defined in separate files:
# - fastlpr_create.R - Mirrors fastLPR_create.m
# - fastlpr_kdf.R - Mirrors fastLPR_kdf.m
# - fastlpr_conv.R - Mirrors fastLPR_conv.m
# - fastlpr_s.R - Mirrors fastLPR_s.m
# - fastlpr_dof.R - Mirrors fastLPR_dof.m
# - fastlpr_y.R - Mirrors fastLPR_y.m
# - fastlpr_compact.R - Mirrors fastLPR_compact.m



#' S3 print method for fastlpr_result
#' @noRd
print.fastlpr_result <- function(x, ...) {
  cat("Fast Local Polynomial Regression Results\n")
  cat("====================================\n")
  cat(sprintf("Number of samples: %d\n", x$Tx))
  cat(sprintf("Dimensions: %d\n", x$dx))
  cat(sprintf("Polynomial order: %d\n", x$opt$order))
  cat(sprintf("Kernel type: %s\n", x$opt$kernel_type))
  cat("\nUse summary() for more details or plot() for visualization.\n")
}

#' S3 summary method for fastlpr_result
#' @noRd
summary.fastlpr_result <- function(object, ...) {
  cat("Summary of Fast Local Polynomial Regression Results\n")
  cat("===============================================\n")
  cat(sprintf("Data: %d samples, %d dimensions\n", object$Tx, object$dx))
  cat(sprintf("Response: %d variables\n", object$dy))
  cat(sprintf("Polynomial order: %d\n", object$opt$order))
  cat(sprintf("Kernel type: %s\n", object$opt$kernel_type))

  if (!is.null(object$gcv_yhat)) {
    cat(sprintf("Selected bandwidth: %s\n", paste(object$gcv_yhat$h1se, collapse = ", ")))
    cat(sprintf("Minimum GCV bandwidth: %s\n", paste(object$gcv_yhat$hmin, collapse = ", ")))
  }

  cat("\nDegrees of freedom: ", object$dof, "\n")
}
