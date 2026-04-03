# Copyright (c) 2024-2025 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later

#' Plot interval bands for regression estimates
#'
#' Port from fastLPR/utility/fastLPR_plot_interval.m and fastLPR_py/src/fastlpr/confidence.py
#'
#' This function visualizes intervals computed by fastlpr_interval.
#' For 1D data, it creates a filled area between upper and lower bounds.
#' For 2D data, it creates transparent surfaces for the bounds.
#'
#' @param ci Confidence interval structure from fastlpr_interval
#'           List with $Values (array with last dim = 2) and $GridVectors
#'
#' @param col Color for interval band (default: 'green')
#'
#' @param alpha Transparency level (default: 0.2)
#'
#' @param add Logical, if TRUE add to existing plot, if FALSE create new plot (default: TRUE)
#'
#' @param ... Additional plotting arguments
#'
#' @return Invisibly returns NULL
#'
#' @examples
#' \dontrun{
#' # 1D intervals
#' ci <- fastlpr_interval(regs_mu, regs_sigma, 0.05)
#' plot(x, y, pch = 20)
#' fastlpr_plot(regs_mu$fpp_yhat, add = TRUE, col = 'blue', lwd = 2)
#' fastlpr_plot_interval(ci, col = 'green', alpha = 0.2, add = TRUE)
#' }
#'
#' @export
fastlpr_plot_interval <- function(ci, col = "green", alpha = 0.2, add = TRUE, ...) {
  # Validate input
  if (!is.list(ci) || !("Values" %in% names(ci)) || !("GridVectors" %in% names(ci))) {
    stop("ci must be a list with Values and GridVectors fields (output from fastlpr_interval)")
  }

  # Get dimensions
  dims <- dim(ci$Values)
  dx <- length(dims) - 1  # Exclude last dimension (which is 2 for [upper, lower])

  # Extract bounds
  if (dx == 1) {
    # 1D case: ci$Values is (N, 2)
    upper <- ci$Values[, 1]
    lower <- ci$Values[, 2]
    x_grid <- ci$GridVectors[[1]]

    # Sort by x_grid to ensure proper polygon behavior
    sort_idx <- order(x_grid)
    x_sorted <- x_grid[sort_idx]
    upper_sorted <- upper[sort_idx]
    lower_sorted <- lower[sort_idx]

    # Create filled polygon
    polygon_x <- c(x_sorted, rev(x_sorted))
    polygon_y <- c(upper_sorted, rev(lower_sorted))

    # Convert alpha to RGB with transparency
    col_rgb <- col2rgb(col) / 255
    col_trans <- rgb(col_rgb[1], col_rgb[2], col_rgb[3], alpha = alpha)

    # Plot
    if (!add) {
      plot(x_sorted, upper_sorted, type = "n",
           ylim = range(c(upper_sorted, lower_sorted), na.rm = TRUE),
           xlab = "", ylab = "", ...)
    }

    polygon(polygon_x, polygon_y, col = col_trans, border = NA)

  } else if (dx == 2) {
    # 2D case: ci$Values is (N1, N2, 2)
    upper <- ci$Values[, , 1]
    lower <- ci$Values[, , 2]

    x1_grid <- ci$GridVectors[[1]]
    x2_grid <- ci$GridVectors[[2]]

    # Convert alpha to RGB with transparency
    col_rgb <- col2rgb(col) / 255
    col_trans <- rgb(col_rgb[1], col_rgb[2], col_rgb[3], alpha = alpha)

    # Plot upper surface
    if (!requireNamespace("rgl", quietly = TRUE)) {
      warning("rgl package not available, using persp() instead")
      # Use base graphics persp() - less interactive but always available
      persp(x1_grid, x2_grid, upper, col = col_trans, border = NA, ...)
      persp(x1_grid, x2_grid, lower, col = col_trans, border = NA, add = TRUE, ...)
    } else {
      # Use rgl for interactive 3D - better but requires rgl
      if (!add) {
        rgl::open3d()
      }
      rgl::surface3d(x1_grid, x2_grid, upper, color = col, alpha = alpha, ...)
      rgl::surface3d(x1_grid, x2_grid, lower, color = col, alpha = alpha, ...)
    }

  } else if (dx == 3) {
    # 3D case: Not supported for visualization
    warning("3D interval visualization not supported. Use slices.")
    return(invisible(NULL))

  } else {
    stop(sprintf("Unsupported dimensionality: %d. Only 1D and 2D supported.", dx))
  }

  invisible(NULL)
}
