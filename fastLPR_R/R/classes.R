# Copyright (c) 2024-2025 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later

# =============================================================================
# S3 CLASS DOCUMENTATION FOR FASTLPR PACKAGE
# =============================================================================
# This file documents the S3 classes used by the fastLPR package.
# Classes are created in cv_fastlpr.R and cv_fastkde.R.
# =============================================================================

#' fastlpr_result: Local Polynomial Regression Output
#'
#' @description
#' S3 class representing the output of \code{\link{cv_fastlpr}}.
#' Contains fitted values, bandwidth selection results, and interpolation functions.
#'
#' @section Structure Fields:
#' \describe{
#'   \item{\code{yhat}}{Numeric vector of fitted values at data points}
#'   \item{\code{fpp_yhat}}{Interpolation function for prediction at new points.
#'                          Use \code{fpp_yhat(x_new)} to predict.}
#'   \item{\code{gcv_yhat}}{List with GCV-based bandwidth selection results:
#'     \itemize{
#'       \item \code{h1se}: Selected bandwidth (using 1-SE rule)
#'       \item \code{hmin}: Bandwidth with minimum GCV
#'       \item \code{idmin}: Index of minimum GCV bandwidth
#'       \item \code{id1se}: Index of 1-SE rule bandwidth
#'       \item \code{gcv_m}: GCV scores for all bandwidths
#'       \item \code{gcv_sd}: Standard errors of GCV scores
#'       \item \code{pdof_m}: Effective degrees of freedom
#'     }
#'   }
#'   \item{\code{opt}}{List of options used for computation:
#'     \itemize{
#'       \item \code{order}: Polynomial order (0, 1, or 2)
#'       \item \code{kernel_type}: Kernel function ("gaussian")
#'       \item \code{N}: Grid resolution
#'       \item \code{accuracy}: NUFFT accuracy
#'     }
#'   }
#'   \item{\code{Tx}}{Number of samples}
#'   \item{\code{dx}}{Number of dimensions}
#'   \item{\code{x}}{Original predictor matrix (N x dx)}
#'   \item{\code{y}}{Original response vector}
#'   \item{\code{grid}}{List of grid vectors for each dimension}
#' }
#'
#' @section S3 Methods:
#' \describe{
#'   \item{\code{print(x)}}{Print summary of results}
#'   \item{\code{summary(x)}}{Detailed summary with GCV statistics}
#'   \item{\code{plot(x)}}{Visualize regression fit}
#'   \item{\code{predict(x, newdata)}}{Predict at new locations}
#' }
#'
#' @seealso \code{\link{cv_fastlpr}}, \code{\link{fastlpr_predict}}
#'
#' @examples
#' \dontrun{
#' # Fit regression
#' x <- matrix(runif(500), ncol=1)
#' y <- sin(2*pi*x) + 0.1*rnorm(500)
#' hlist <- get_hlist(20, c(0.05, 0.5))
#' regs <- cv_fastlpr(x, y, hlist, list(order=1))
#'
#' # Access results
#' regs$gcv_yhat$h1se      # Selected bandwidth
#' regs$yhat               # Fitted values
#' regs$fpp_yhat(x_new)    # Predict at new points
#'
#' # S3 methods
#' print(regs)             # Summary
#' plot(regs)              # Visualization
#' predict(regs, x_new)    # Prediction
#' }
#'
#' @name fastlpr_result
#' @aliases fastlpr_result-class
NULL


#' fastkde_result: Kernel Density Estimation Output
#'
#' @description
#' S3 class representing the output of \code{\link{cv_fastkde}}.
#' Contains density estimates, bandwidth selection results, and interpolation functions.
#'
#' @section Structure Fields:
#' \describe{
#'   \item{\code{fhat}}{Numeric vector of density estimates at data points}
#'   \item{\code{fpp}}{Interpolation function for density evaluation at new points.
#'                     Use \code{fpp(x_new)} to evaluate density.}
#'   \item{\code{lcv}}{List with LCV-based bandwidth selection results:
#'     \itemize{
#'       \item \code{h1se}: Selected bandwidth (using 1-SE rule)
#'       \item \code{hmax}: Bandwidth with maximum LCV (log-likelihood)
#'       \item \code{idmax}: Index of maximum LCV bandwidth
#'       \item \code{id1se}: Index of 1-SE rule bandwidth
#'       \item \code{lcv_m}: LCV scores for all bandwidths
#'       \item \code{lcv_sd}: Standard errors of LCV scores
#'     }
#'   }
#'   \item{\code{opt}}{List of options used for computation:
#'     \itemize{
#'       \item \code{kernel_type}: Kernel function ("gaussian")
#'       \item \code{N}: Grid resolution
#'       \item \code{accuracy}: NUFFT accuracy
#'     }
#'   }
#'   \item{\code{Tx}}{Number of samples}
#'   \item{\code{dx}}{Number of dimensions}
#'   \item{\code{x}}{Original data matrix (N x dx)}
#'   \item{\code{h}}{Bandwidth matrix (Nh x dx)}
#'   \item{\code{grid}}{List of grid vectors for each dimension}
#' }
#'
#' @section S3 Methods:
#' \describe{
#'   \item{\code{print(x)}}{Print summary of results}
#'   \item{\code{plot(x)}}{Visualize density estimate}
#' }
#'
#' @seealso \code{\link{cv_fastkde}}
#'
#' @examples
#' \dontrun{
#' # Fit KDE
#' x <- matrix(rnorm(500), ncol=1)
#' hlist <- get_hlist(20, c(0.1, 1.0))
#' kde <- cv_fastkde(x, hlist)
#'
#' # Access results
#' kde$lcv$h1se            # Selected bandwidth
#' kde$fhat                # Density estimates
#' kde$fpp(x_new)          # Evaluate at new points
#'
#' # S3 methods
#' print(kde)              # Summary
#' plot(kde)               # Visualization
#' }
#'
#' @name fastkde_result
#' @aliases fastkde_result-class
NULL


#' Check if object is a fastlpr_result
#'
#' @param x Object to test
#' @return Logical indicating if x is a fastlpr_result
#' @export
#' @examples
#' \dontrun{
#' regs <- cv_fastlpr(x, y, hlist)
#' is_fastlpr_result(regs)  # TRUE
#' }
is_fastlpr_result <- function(x) {
  inherits(x, "fastlpr_result")
}


#' Check if object is a fastkde_result
#'
#' @param x Object to test
#' @return Logical indicating if x is a fastkde_result
#' @export
#' @examples
#' \dontrun{
#' kde <- cv_fastkde(x, hlist)
#' is_fastkde_result(kde)  # TRUE
#' }
is_fastkde_result <- function(x) {
  inherits(x, "fastkde_result")
}
