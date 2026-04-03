# Copyright (c) 2024-2025 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later

#' Plot fastLPR regression results
#'
#' Creates publication-quality plots of regression results.
#' Similar to MATLAB's fastLPR_plot function.
#'
#' @param fpp_yhat Interpolator object from cv_fastlpr
#' @param x_range Optional range for plotting (default: data range)
#' @param n_points Number of points for plotting (default: 1000)
#' @param ... Additional graphical parameters
#'
#' @return Invisibly returns the plot
#'
#' @examples
#' x <- matrix(runif(500) * 20, ncol = 1)
#' y <- sin(x) + 0.2 * rnorm(500)
#' hlist <- get_hlist(20, c(0.01, 1), "logspace")
#' opt <- list(order = 1)
#' regs <- cv_fastlpr(x, y, hlist, opt)
#' fastlpr_plot(regs$fpp_yhat)
#'
#' @export
fastlpr_plot <- function(fpp_yhat, x_range = NULL, n_points = 1000, ...) {
  # Handle both structured interpolator and function cases
  if (is.function(fpp_yhat)) {
    # fpp_yhat is a function (1D approxfun case)
    # Get environment to extract grid info
    env <- environment(fpp_yhat)
    if (exists("x", env) && exists("y", env)) {
      xlist <- list(get("x", env))
      values <- get("y", env)
    } else {
      stop("Cannot extract grid from function interpolator")
    }
  } else if (is.list(fpp_yhat) && "GridVectors" %in% names(fpp_yhat)) {
    # Structured interpolator
    xlist <- fpp_yhat$GridVectors
    values <- fpp_yhat$Values
  } else {
    stop("fpp_yhat must be a function or list with GridVectors and Values")
  }

  # Determine dimensionality
  dims <- length(xlist)

  if (dims == 1) {
    # 1D plot
    x_plot <- xlist[[1]]
    y_plot <- as.vector(values)

    plot(x_plot, y_plot, type = "l", lwd = 2, col = "blue",
         xlab = "X", ylab = "Y", main = "Fast Local Polynomial Regression", ...)
  } else if (dims == 2) {
    # 2D plot
    x1 <- xlist[[1]]
    x2 <- xlist[[2]]
    z <- values

    # Create contour plot
    filled.contour(x1, x2, z, color.palette = terrain.colors,
                 xlab = "X1", ylab = "X2",
                 main = "Fast Local Polynomial Regression (2D)", ...)
  } else {
    stop("Plotting only supported for 1D and 2D data.")
  }

  invisible(NULL)
}

#' Plot fastKDE density estimation results
#'
#' Creates publication-quality plots of density estimation results.
#' Similar to MATLAB's fastKDE_plot function.
#'
#' @param fpp Interpolator object from cv_fastkde
#' @param x_range Optional range for plotting (default: data range)
#' @param n_points Number of points for plotting (default: 1000)
#' @param ... Additional graphical parameters
#'
#' @return Invisibly returns the plot
#'
#' @examples
#' x <- c(rnorm(100), rnorm(100) + 3)
#' hlist <- get_hlist(20, c(0.01, 1), "logspace")
#' kde <- cv_fastkde(x, hlist)
#' fastkde_plot(kde$fpp)
#'
#' @export
fastkde_plot <- function(fpp, x_range = NULL, n_points = 1000, ...) {
  # Extract grid and values from interpolator
  xlist <- fpp$GridVectors
  values <- fpp$Values

  # Determine dimensionality
  dims <- length(xlist)

  if (dims == 1) {
    # 1D density plot
    x_plot <- xlist[[1]]
    y_plot <- as.vector(values)

    plot(x_plot, y_plot, type = "l", lwd = 2, col = "red",
         xlab = "X", ylab = "Density", main = "Kernel Density Estimation", ...)
  } else if (dims == 2) {
    # 2D density plot
    x1 <- xlist[[1]]
    x2 <- xlist[[2]]
    z <- values

    # Create contour plot
    filled.contour(x1, x2, z, color.palette = heat.colors,
                 xlab = "X1", ylab = "X2",
                 main = "Kernel Density Estimation (2D)", ...)
  } else {
    stop("Plotting only supported for 1D and 2D data.")
  }

  invisible(NULL)
}

#' S3 plot method for fastlpr_result
#' @noRd
plot.fastlpr_result <- function(x, ...) {
  if (!is.null(x$fpp_yhat)) {
    fastlpr_plot(x$fpp_yhat, ...)
  } else {
    plot.new()
    text(0.5, 0.5, "No interpolator available for plotting", cex = 1.5)
  }
}

#' S3 plot method for fastkde_result
#' @noRd
plot.fastkde_result <- function(x, ...) {
  if (!is.null(x$fpp)) {
    fastkde_plot(x$fpp, ...)
  } else {
    plot.new()
    text(0.5, 0.5, "No interpolator available for plotting", cex = 1.5)
  }
}

#' S3 predict method for fastlpr_result
#' @noRd
predict.fastlpr_result <- function(object, newdata, ...) {
  if (is.null(object$fpp_yhat)) {
    stop("No interpolator available for prediction.")
  }

  # This is a simplified implementation
  # In practice, we need to evaluate the interpolator at newdata points
  return(fastlpr_predict(object, newdata))
}