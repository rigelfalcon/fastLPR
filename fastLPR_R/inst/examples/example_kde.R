#!/usr/bin/env Rscript
# Code to generate Figure 2: Kernel Density Estimation (1D, 2D, and 3D)
#
# This script reproduces Figure 2 from the fastLPR paper, demonstrating:
#   Row 1 - KDE Examples:
#   - Panel (a): 1D KDE with automatic bandwidth selection
#   - Panel (b): 2D KDE with contour plot
#   - Panel (c):3D KDE volume rendering
#   Row 2 - Bandwidth Selection:
#   - Panel (d): 1D bandwidth selection via LCV
#   - Panel (e): 2D bandwidth selection via LCV heatmap
#   - Panel (f): 3D bandwidth selection via LCV
#
# The figure follows JSS publication standards with:
#   - Fixed random seed for reproducibility
#   - Consistent styling (fonts, colors, sizes)
#   - 300 DPI resolution for publication
#   - Self-contained code (no external dependencies except fastLPR_R)
#
# Copyright (c) 2024 fastLPR Development Team

# Auto-detect working directory
script_dir <- tryCatch({
  dirname(sys.frame(1)$ofile)
}, error = function(e) {
  "."
})
repo_root <- normalizePath(file.path(script_dir, "../.."), mustWork = FALSE)
if (dir.exists(file.path(repo_root, "fastLPR_R"))) {
  setwd(repo_root)
}
# Load fastLPR package using setup.R
source("fastLPR_R/setup.R")

# Helper function for parula colormap (matching MATLAB's default)
parula <- function(n) {
  # MATLAB parula colormap - 256 colors interpolated
  colorRampPalette(c(
    "#352A87", "#363093", "#3637A0", "#353DAD", "#3243BA",
    "#2C4DC6", "#2456CD", "#1E60D5", "#1B6ADC", "#1873E0",
    "#177AE3", "#1882E5", "#1A8AE6", "#1D92E6", "#209AE5",
    "#23A1E3", "#27A8E0", "#2BAEDD", "#30B4D9", "#36B9D4",
    "#3CBDCF", "#43C1C9", "#4AC4C3", "#52C7BC", "#5AC9B5",
    "#63CBAF", "#6CCCA8", "#76CDA1", "#80CE9A", "#8ACF93",
    "#94CF8C", "#9ED085", "#A8D07F", "#B2D079", "#BCD073",
    "#C6D06D", "#D0CF67", "#DACE61", "#E4CC5C", "#EECA56",
    "#F7C650", "#FEC044", "#FEB635", "#FCA928", "#FA9C1D",
    "#F79015", "#F4840D", "#F17807", "#ED6D01", "#E96101",
    "#E45501", "#DE4901", "#D93E01", "#D33501", "#CD2C01"
  ))(n)
}

# MATLAB jet colormap - blue to cyan to yellow to red
jet <- function(n) {
  colorRampPalette(c(
    "#00007F", "#0000FF", "#0080FF", "#00FFFF",
    "#80FF80", "#FFFF00", "#FF8000", "#FF0000", "#7F0000"
  ))(n)
}

cat("\n")
cat(strrep("=", 80), "\n")
cat("Figure 2: Kernel Density Estimation (1D, 2D, and 3D Examples)\n")
cat(strrep("=", 80), "\n")
cat("\n")

################################################################################
# Panel (a) and (c): 1D Bimodal Distribution
################################################################################

cat("Generating Panel (a) and (c): 1D KDE...\n")

# Load Old Faithful geyser data (R built-in 'faithful', n = 272).
# Columns: 1 = eruption duration (min), 2 = waiting time (min).
# Panel (a) uses eruption durations, a classic bimodal real dataset.
faithful_data <- read.table(file.path(script_dir, "faithful.txt"))
x <- as.matrix(faithful_data[, 1])                         # Eruption duration (min)

cat(sprintf("  - Loaded %d Old Faithful eruption-duration samples\n", nrow(x)))

# Create bandwidth list (log-spaced) for the eruption-duration scale (~1.6-5.1 min)
hlist <- get_hlist(20, c(0.05, 2))  # Log-spaced
cat(sprintf("  - Testing %d bandwidths\n", length(hlist)))

# Compute KDE with automatic bandwidth selection via LCV
start_time <- Sys.time()
kde <- cv_fastkde(x, hlist)
elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

cat(sprintf("  - Computation time: %.3f seconds\n", elapsed))
cat(sprintf("  - Selected bandwidth: h = %.4f\n", kde$h))

################################################################################
# Panel (b) and (d): 2D Bimodal Distribution
################################################################################

cat("\nGenerating Panel (b) and (d): 2D KDE...\n")

# Old Faithful 2D: eruption duration vs waiting time (both columns).
x2d <- as.matrix(faithful_data[, c(1, 2)])  # [eruptions (min), waiting (min)]

cat(sprintf("  - Loaded %d Old Faithful 2D samples\n", nrow(x2d)))

# Create 2D bandwidth list (log-spaced grid) matched to each axis scale:
#   eruptions ~ [1.6, 5.1] min, waiting ~ [43, 96] min
hlist2d <- get_hlist(c(20, 20), matrix(c(0.05, 1.5, 1, 20), nrow = 2, byrow = TRUE))
cat(sprintf("  - Testing %d bandwidth combinations\n", nrow(hlist2d)))

# Compute 2D KDE with automatic bandwidth selection
opt2d <- list(
  verbose = FALSE,
  N = c(51, 51),  # Grid size
  xrange = matrix(c(1, 6, 35, 100), nrow = 2, byrow = TRUE)  # Evaluation range (eruptions, waiting)
)
start_time <- Sys.time()
kde2d <- cv_fastkde(x2d, hlist2d, opt2d)
elapsed2d <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

cat(sprintf("  - Computation time: %.3f seconds\n", elapsed2d))
cat(sprintf("  - Selected bandwidth: h = [%.4f, %.4f]\n", kde2d$h[1], kde2d$h[2]))

################################################################################
# Panel (c) and (f): 3D Trimodal Distribution
################################################################################

cat("\nGenerating Panel (c) and (f): 3D KDE...\n")

# Set random seed for reproducibility (matches MATLAB rng(45))
set.seed(45)

# Generate 3D data with three clusters (same structure as MATLAB)
n_cluster <- 800
x3d_1 <- matrix(rnorm(n_cluster * 3, sd = 0.3), ncol = 3) + matrix(rep(c(-1, -1, -1), n_cluster), ncol = 3, byrow = TRUE)  # Cluster 1
x3d_2 <- matrix(rnorm(n_cluster * 3, sd = 0.4), ncol = 3) + matrix(rep(c(1, 1, 0), n_cluster), ncol = 3, byrow = TRUE)      # Cluster 2
x3d_3 <- matrix(rnorm(n_cluster * 3, sd = 0.25), ncol = 3) + matrix(rep(c(0, -1, 1), n_cluster), ncol = 3, byrow = TRUE)   # Cluster 3
x3d <- rbind(x3d_1, x3d_2, x3d_3)

cat(sprintf("  - Generated %d samples from 3D trimodal distribution\n", nrow(x3d)))

# Create 3D bandwidth list (log-spaced) - use 10x10x10 grid for consistency
hlist3d <- get_hlist(c(10, 10, 10), matrix(c(0.2, 0.6, 0.2, 0.6, 0.2, 0.6), nrow = 3, byrow = TRUE))
cat(sprintf("  - Testing %d bandwidth combinations\n", nrow(hlist3d)))

# Compute 3D KDE with automatic bandwidth selection
opt3d <- list(
  verbose = FALSE,
  N = c(31, 31, 31),  # Grid size for evaluation
  xrange = matrix(c(-2.5, 2.5, -2.5, 2.5, -2.5, 2.5), nrow = 3, byrow = TRUE)  # Evaluation range
)
start_time <- Sys.time()
kde3d <- cv_fastkde(x3d, hlist3d, opt3d)
elapsed3d <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

cat(sprintf("  - Computation time: %.3f seconds\n", elapsed3d))
cat(sprintf("  - Selected bandwidth: h = %.4f\n", kde3d$h))

################################################################################
# Create Figure with 2x3 Layout (Row 1: KDE, Row 2: Bandwidth Selection)
################################################################################

cat("\nCreating figure...\n")

# Create figure with publication-quality size (wider for 3 columns)
png("fastLPR_R/fig/reproduced/fig2_fastkde_r.png", width = 4800, height = 2400, res = 300)
par(mfrow = c(2, 3), mar = c(4.5, 4.5, 3, 1), oma = c(0, 0, 2, 0))

################################################################################
# ROW 1: KDE EXAMPLES
################################################################################

################################################################################
# Panel (a): 1D KDE with Multiple Bandwidths (5 curves)
################################################################################

# Create evaluation grid
x_grid <- seq(min(x) - 1, max(x) + 1, length.out = 200)

# Plot histogram (light gray, semi-transparent)
hist(x, breaks = 30, freq = FALSE, col = rgb(0.7, 0.7, 0.7, 0.5),
     border = "black", main = "", xlab = "Eruption duration (min)", ylab = "Density",
     xlim = range(x_grid), cex.lab = 1.3, cex.axis = 1.1)

# Plot 5 different bandwidths to show effect of bandwidth selection
# Similar to MATLAB: test_h_idx = [1, 5, 10, 15, 20]
test_h_idx <- c(1, 5, 10, 15, 20)
colors_5 <- c("blue", "green", "purple", "orange", "cyan")

cat("  - Computing KDE for 5 different bandwidths...
")
for (i in 1:length(test_h_idx)) {
  idx <- test_h_idx[i]
  
  # Compute KDE for this specific bandwidth
  opt_temp <- list()
  opt_temp$verbose <- FALSE
  kde_temp <- cv_fastkde(x, matrix(hlist[idx], ncol=1), opt_temp)
  
  # Extract density values at grid points
  if (!is.null(kde_temp$fpp) && inherits(kde_temp$fpp, "fastlpr_interpolator")) {
    # Get values - handle case where Values may be a matrix
    fpp_values <- kde_temp$fpp$Values
    if (is.matrix(fpp_values)) {
      fpp_values <- fpp_values[, 1]  # Take first column
    }
    kde_vals_temp <- stats::approx(kde_temp$fpp$GridVectors[[1]],
                                    fpp_values,
                                    xout = x_grid, rule = 2)$y

    # Plot this bandwidth's curve
    lines(x_grid, kde_vals_temp, col = colors_5[i], lwd = 2)
    cat(sprintf("    - Plotted h[%d]=%.4f\n", idx, hlist[idx]))
  }
}

# Plot selected bandwidth KDE with thick red line (on top)
if (!is.null(kde$fpp)) {
  if (inherits(kde$fpp, "fastlpr_interpolator")) {
    # Get values - handle case where Values may be a matrix
    fpp_values_opt <- kde$fpp$Values
    if (is.matrix(fpp_values_opt)) {
      fpp_values_opt <- fpp_values_opt[, 1]  # Take first column
    }
    kde_vals_opt <- stats::approx(kde$fpp$GridVectors[[1]],
                                   fpp_values_opt,
                                   xout = x_grid, rule = 2)$y
  } else if (is.function(kde$fpp)) {
    kde_vals_opt <- kde$fpp(x_grid)
  } else {
    kde_vals_opt <- rep(NA, length(x_grid))
  }
  lines(x_grid, kde_vals_opt, col = "red", lwd = 3)
  cat(sprintf("    - Plotted SELECTED h=%.4f (red, thick)
", kde$h))
}

# Add legend for all 6 curves (5 test + 1 selected)
legend_labels <- c(sprintf("h=%.3f", hlist[test_h_idx]),
                   paste0("h=", sprintf("%.3f", kde$h), " (selected)"))
legend_cols <- c(colors_5, "red")
legend_lwd <- c(rep(2, 5), 3)
legend("topright",
       legend = legend_labels,
       col = legend_cols,
       lwd = legend_lwd,
       cex = 0.8, bg = "white")

title("(a) 1D KDE with Bandwidth Comparison", font.main = 2, cex.main = 1.4)
grid()

################################################################################
# Panel (b): 2D KDE (Contour Plot)
################################################################################

# Extract grid and reshape fhat for contour plotting
x1_grid <- kde2d$xlist[[1]]
x2_grid <- kde2d$xlist[[2]]
n1 <- length(x1_grid)
n2 <- length(x2_grid)

# kde2d$fhat should be (n1 x n2) after fix
# Transpose for R's contour convention (x=rows, y=cols)
fhat_2d <- matrix(kde2d$fhat, nrow = n1, ncol = n2)

# Create image plot with contours (compatible with multi-panel layout)
image(x1_grid, x2_grid, fhat_2d,
      col = parula(100),
      xlab = "Eruption duration (min)",
      ylab = "Waiting time (min)",
      main = "(b) 2D KDE Density Contours",
      font.main = 2, cex.main = 1.4, cex.lab = 1.3, cex.axis = 1.1)

# Add contour lines
contour(x1_grid, x2_grid, fhat_2d, add = TRUE, col = "black", lwd = 0.5)

# Overlay data points (small black dots)
points(x2d[,1], x2d[,2], pch = ".", col = "black", cex = 1.5)

grid()

################################################################################
# Panel (c): 3D KDE (Volume Rendering)
################################################################################

# Extract grid dimensions
x1_grid_3d <- kde3d$xlist[[1]]
x2_grid_3d <- kde3d$xlist[[2]]
x3_grid_3d <- kde3d$xlist[[3]]
n1_3d <- length(x1_grid_3d)
n2_3d <- length(x2_grid_3d)
n3_3d <- length(x3_grid_3d)

# Reshape fhat to 3D array
fhat_3d <- array(kde3d$fhat, dim = c(n1_3d, n2_3d, n3_3d))

# Filter to show only significant density regions (top 70% percentile)
density_thresh <- quantile(kde3d$fhat[kde3d$fhat > 0], 0.30)
idx_sig <- which(kde3d$fhat > density_thresh)

# Create grid points for visualization
grid_3d <- expand.grid(x1 = x1_grid_3d, x2 = x2_grid_3d, x3 = x3_grid_3d)
x_sig <- grid_3d[idx_sig, ]
c_data <- kde3d$fhat[idx_sig]

# Normalize for coloring and transparency
alpha_data <- (c_data - min(c_data)) / (max(c_data) - min(c_data) + .Machine$double.eps)
alpha_data <- 0.05 + 0.7 * alpha_data  # Map to [0.05, 0.75]

# Use scatterplot3d for 3D KDE volume rendering (if available)
if (!requireNamespace("scatterplot3d", quietly = TRUE)) {
  cat("WARNING: scatterplot3d not installed. Installing from CRAN...\n")
  install.packages("scatterplot3d", repos = "https://cloud.r-project.org")
}

if (requireNamespace("scatterplot3d", quietly = TRUE)) {
  library(scatterplot3d)

  # Map density values to jet colormap (MATLAB uses jet for this panel)
  jet_colors <- jet(100)
  color_indices <- cut(c_data, breaks = 100)
  point_colors <- jet_colors[color_indices]

  # Create plot with wider right margin for colorbar
  old_par <- par(mar = c(3, 3, 3, 5))  # Increase right margin

  s3d <- scatterplot3d(x_sig$x1, x_sig$x2, x_sig$x3,
                       color = point_colors,
                       pch = 19, cex.symbols = 0.5,
                       main = "(c) 3D KDE Volume Rendering",
                       xlab = expression(x[1]),
                       ylab = expression(x[2]),
                       zlab = expression(x[3]),
                       font.main = 2, cex.main = 1.4,
                       box = TRUE, grid = TRUE,
                       angle = 45)

  # Add simple colorbar using legend
  legend_labels <- format(seq(min(c_data), max(c_data), length.out = 5), digits = 2)
  legend("right", legend = legend_labels, fill = jet(5),
         title = "Density", xpd = TRUE, inset = c(-0.15, 0),
         cex = 0.7, bty = "n")

  par(old_par)  # Restore original par settings
} else {
  # Fallback: 2D slice at z=0
  cat("WARNING: scatterplot3d package required for 3D visualization\n")
  plot(0, 0, type = "n", axes = FALSE, xlab = "", ylab = "")
  text(0, 0, "3D KDE visualization requires scatterplot3d package\nInstall with: install.packages('scatterplot3d')",
       cex = 1.0, font = 2)
  title("(c) 3D KDE Volume Rendering", font.main = 2, cex.main = 1.4)
}

################################################################################
# ROW 2: BANDWIDTH SELECTION
################################################################################

################################################################################
# Panel (d): 1D Bandwidth Selection (LCV Scores)
################################################################################

if (!is.null(kde$lcv)) {
  # Plot LCV scores for all bandwidths
  plot(hlist, kde$lcv$lcv_m, type = "b", pch = 19, col = "blue", lwd = 2,
       xlab = expression(log[10](h)), ylab = "LCV Score",
       main = "(d) 1D Bandwidth Selection via LCV",
       font.main = 2, cex.main = 1.4, cex.lab = 1.3, cex.axis = 1.1,
       log = "x")  # Log scale for x-axis
  grid()

  # Mark optimal bandwidth
  abline(v = kde$h, col = "red", lwd = 2, lty = 2)
  points(kde$h, kde$lcv$lcv_m[kde$lcv$id1se],
         col = "red", pch = 19, cex = 1.5)

  # Add text annotation
  text(kde$h, kde$lcv$lcv_m[kde$lcv$id1se],
       labels = sprintf("h*=%.3f", kde$h),
       pos = 4, col = "red", cex = 1.1, font = 2)
} else {
  plot(0, 0, type = "n", axes = FALSE, xlab = "", ylab = "")
  text(0, 0, "Single bandwidth (no selection)", cex = 1.4)
  title("(d) 1D Bandwidth Selection", font.main = 2, cex.main = 1.4)
}

################################################################################
# Panel (e): 2D Bandwidth Selection (LCV Heatmap)
################################################################################

if (!is.null(kde2d$lcv)) {
  # Reshape LCV scores into 2D grid
  h1_unique <- unique(kde2d$lcv$hlist[,1])
  h2_unique <- unique(kde2d$lcv$hlist[,2])
  lcv_grid <- matrix(kde2d$lcv$lcv_m, nrow = length(h1_unique), ncol = length(h2_unique))

  # Plot heatmap with parula colormap (transpose for R's image convention)
  image(log10(h1_unique), log10(h2_unique), lcv_grid,
        col = parula(100),
        xlab = expression(log[10](h[1])),
        ylab = expression(log[10](h[2])),
        main = "(e) 2D Bandwidth Selection via LCV",
        font.main = 2, cex.main = 1.4, cex.lab = 1.3, cex.axis = 1.1)

  # Add contour lines
  contour(log10(h1_unique), log10(h2_unique), lcv_grid, add = TRUE, col = "black", lwd = 0.5)

  # Mark selected bandwidth with red star (1-SE rule)
  points(log10(kde2d$h[1]), log10(kde2d$h[2]),
         pch = 8, col = "red", cex = 2, lwd = 2)

  # Add text annotation
  text(log10(kde2d$h[1]), log10(kde2d$h[2]) + 0.1,
       labels = "1-SE",
       col = "white", cex = 1.1, font = 2)
} else {
  plot(0, 0, type = "n", axes = FALSE, xlab = "", ylab = "")
  text(0, 0, "Single bandwidth (no selection)", cex = 1.4)
  title("(e) 2D Bandwidth Selection", font.main = 2, cex.main = 1.4)
}

################################################################################
# Panel (f): 3D Bandwidth Selection (LCV Volume)
################################################################################

if (!is.null(kde3d$lcv)) {
  # Get LCV scores and bandwidth list
  lcv_scores <- kde3d$lcv$lcv_m
  h_list <- kde3d$lcv$hlist

  # For 3D anisotropic bandwidth, reshape LCV scores to 3D grid
  n_h <- round(nrow(h_list)^(1/3))  # Should be 10
  LCV_3d <- array(lcv_scores, dim = c(n_h, n_h, n_h))

  # Get unique bandwidth values for each dimension
  h1_values <- unique(h_list[,1])
  h2_values <- unique(h_list[,2])
  h3_values <- unique(h_list[,3])

  # Convert to log10 scale for consistency with panels (d) and (e)
  log10_h1 <- log10(h1_values)
  log10_h2 <- log10(h2_values)
  log10_h3 <- log10(h3_values)

  # Create 3D grid with log10 bandwidth values
  grid_3d <- expand.grid(h1 = log10_h1, h2 = log10_h2, h3 = log10_h3)

  # Normalize LCV for coloring and transparency
  lcv_norm <- (lcv_scores - min(lcv_scores)) / (max(lcv_scores) - min(lcv_scores) + .Machine$double.eps)
  alpha_data <- 0.05 + 0.6 * lcv_norm  # Higher LCV = more visible

  # Create size variation based on LCV scores
  marker_size <- 10 + 30 * lcv_norm

  # Get selected bandwidth (needed by both visualization paths)
  h_selected <- kde3d$lcv$h1se

  # Plot 3D scatter (with fallback if scatterplot3d not available)
  if (requireNamespace("scatterplot3d", quietly = TRUE)) {
    library(scatterplot3d)
    s3d <- scatterplot3d(grid_3d$h1, grid_3d$h2, grid_3d$h3,
                         color = parula(100)[cut(lcv_scores, 100)],
                         pch = 19, cex.symbols = marker_size/10,
                         main = "(f) 3D Bandwidth Selection via LCV",
                         xlab = expression(log[10](h[1])),
                         ylab = expression(log[10](h[2])),
                         zlab = expression(log[10](h[3])),
                         font.main = 2, cex.main = 1.4,
                         box = TRUE, grid = TRUE,
                         angle = 45)  # Matches MATLAB view(45, 30)

    # Mark selected bandwidth with large red star
    s3d$points3d(log10(h_selected[1]), log10(h_selected[2]), log10(h_selected[3]),
                 col = "red", pch = 8, cex = 2.5)
  } else {
    # Fallback: 2D slice at median h3
    h3_median_idx <- which.min(abs(log10_h3 - median(log10_h3)))
    lcv_slice <- LCV_3d[, , h3_median_idx]

    image(log10_h1, log10_h2, lcv_slice,
          col = parula(100),
          xlab = expression(log[10](h[1])),
          ylab = expression(log[10](h[2])),
          main = sprintf("(f) 3D Bandwidth Selection (h3 fixed at %.2f)", 10^log10_h3[h3_median_idx]),
          font.main = 2, cex.main = 1.2, cex.lab = 1.3, cex.axis = 1.1)

    contour(log10_h1, log10_h2, lcv_slice, add = TRUE, col = "black", lwd = 0.5)

    # Mark selected bandwidth
    points(log10(h_selected[1]), log10(h_selected[2]),
           pch = 8, col = "red", cex = 2, lwd = 2)
  }
} else {
  plot(0, 0, type = "n", axes = FALSE, xlab = "", ylab = "")
  text(0, 0, "Single bandwidth (no selection needed)", cex = 1.4)
  title("(f) 3D Bandwidth Selection", font.main = 2, cex.main = 1.4)
}

# Overall title
mtext("Figure 2: Kernel Density Estimation (1D, 2D, and 3D)",
      outer = TRUE, cex = 1.6, font = 2, line = 0.5)

dev.off()

cat("\nFigure saved to: fastLPR_R/fig/reproduced/fig2_fastkde_r.png\n")
cat("\nExample completed successfully!\n")
cat(sprintf("Total time: 1D=%.2fs, 2D=%.2fs, 3D=%.2fs\n", elapsed, elapsed2d, elapsed3d))
