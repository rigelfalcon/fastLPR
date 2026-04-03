# Copyright (c) 2024-2025 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later

#' Visualize bandwidth selection results from cv_fastkde
#'
#' Port from fastLPR/utility/fastkde_plot_bandwidth.m
#'
#' Creates a visualization of LCV (Likelihood Cross-Validation) scores
#' vs candidate bandwidths. For 1D: a log-scale line plot. For 2D: a heatmap.
#' The selected bandwidth is highlighted with a red marker.
#'
#' @param kde KDE result from cv_fastkde. Must contain \code{$lcv} with fields
#'   \code{hlist}, \code{lcv_m}, \code{id1se}, and \code{$h} (selected bandwidth).
#' @param main Plot title (default: "Bandwidth Selection (LCV)")
#' @param xlab X-axis label (default: auto)
#' @param ylab Y-axis label (default: auto)
#' @param ... Additional arguments passed to plot functions
#'
#' @examples
#' \dontrun{
#' kde <- cv_fastkde(x, hlist)
#' fastkde_plot_bandwidth(kde)
#' }
#'
#' @export
fastkde_plot_bandwidth <- function(kde, main = "Bandwidth Selection (LCV)",
                                   xlab = NULL, ylab = NULL, ...) {
  if (!is.list(kde) || is.null(kde$lcv)) {
    stop("kde must be a result from cv_fastkde with $lcv field")
  }

  lcv <- kde$lcv
  h_selected <- kde$h
  dx <- length(h_selected)

  if (dx == 1) {
    # 1D: semilog line plot
    hlist_vec <- lcv$hlist[, 1]
    lcv_scores <- lcv$lcv_m

    if (is.null(xlab)) xlab <- "Bandwidth h"
    if (is.null(ylab)) ylab <- "Log-Likelihood CV Score"

    plot(hlist_vec, lcv_scores, type = "o", log = "x",
         pch = 16, cex = 0.8, lwd = 1.5,
         xlab = xlab, ylab = ylab, main = main, ...)
    grid()

    # Mark selected bandwidth
    id1se <- lcv$id1se
    points(h_selected, lcv_scores[id1se], pch = 8, col = "red",
           cex = 2.5, lwd = 3)
    legend("bottomright",
           legend = c("LCV scores", sprintf("Selected (h=%.3f)", h_selected)),
           pch = c(16, 8), col = c("black", "red"),
           pt.cex = c(0.8, 2), lwd = c(1.5, 3))

  } else if (dx == 2) {
    # 2D: heatmap
    h1_vals <- unique(lcv$hlist[, 1])
    h2_vals <- unique(lcv$hlist[, 2])
    n1 <- length(h1_vals)
    n2 <- length(h2_vals)
    lcv_matrix <- matrix(lcv$lcv_m, nrow = n1, ncol = n2)

    if (is.null(xlab)) xlab <- expression(log[10](h[1]))
    if (is.null(ylab)) ylab <- expression(log[10](h[2]))

    image(log10(h1_vals), log10(h2_vals), lcv_matrix,
          xlab = xlab, ylab = ylab, main = main,
          col = heat.colors(64), ...)

    # Mark selected bandwidth
    points(log10(h_selected[1]), log10(h_selected[2]),
           pch = 8, col = "red", cex = 3, lwd = 3)
    text(log10(h_selected[1]), log10(h_selected[2]),
         labels = sprintf("h=[%.2f, %.2f]", h_selected[1], h_selected[2]),
         col = "red", pos = 1, font = 2, cex = 0.8)

  } else {
    stop(sprintf("fastkde_plot_bandwidth: %dD not supported (only 1D and 2D)", dx))
  }

  invisible(NULL)
}
