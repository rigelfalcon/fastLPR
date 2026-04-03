#!/usr/bin/env Rscript
# Figure 4: Complex-Valued Regression (log(z) function)
#
# This script reproduces Figure 4 from the fastLPR paper, demonstrating:
#   - Panel (a): Real part of log(z)
#   - Panel (b): Imaginary part of log(z)
#   - Panel (c): Magnitude |log(z)|
#   - Panel (d): Angle of log(z)
#
# The figure follows JSS publication standards with:
#   - Fixed random seed for reproducibility
#   - Consistent styling (fonts, colors, sizes)
#   - 300 DPI resolution for publication
#   - Self-contained code (no external dependencies except fastLPR)
#
# Copyright (c) 2024 fastLPR Development Team

cat("\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat("Figure 4: Complex-Valued Regression (log(z) function)\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat("\n")

# Source all required functions
# Auto-detect working directory
script_dir <- tryCatch({ dirname(sys.frame(1)$ofile) }, error = function(e) ".")
repo_root <- normalizePath(file.path(script_dir, "../.."), mustWork = FALSE)
if (dir.exists(file.path(repo_root, "fastLPR_R"))) setwd(repo_root)
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

# Helper function to map z-values to color matrix for persp()
# This creates solid surface rendering with continuous colormap
get_color_matrix <- function(z_matrix, n_colors = 100) {
  # Normalize z values to [0, 1]
  z_range <- range(z_matrix, na.rm = TRUE)
  z_norm <- (z_matrix - z_range[1]) / (z_range[2] - z_range[1])

  # Get parula colors
  parula_pal <- parula(n_colors)

  # Map normalized z to color indices
  z_indices <- pmin(pmax(floor(z_norm * (n_colors - 1)) + 1, 1), n_colors)

  # Create color matrix (needs to be (nrow-1) x (ncol-1) for persp)
  nr <- nrow(z_matrix)
  nc <- ncol(z_matrix)

  # Average adjacent cells to get facet colors
  color_matrix <- matrix(NA, nrow = nr - 1, ncol = nc - 1)
  for (i in 1:(nr - 1)) {
    for (j in 1:(nc - 1)) {
      # Average the 4 corners of each facet
      avg_idx <- round(mean(c(z_indices[i, j], z_indices[i+1, j],
                              z_indices[i, j+1], z_indices[i+1, j+1])))
      color_matrix[i, j] <- parula_pal[avg_idx]
    }
  }

  return(color_matrix)
}


################################################################################
# Generate Complex-Valued Data
################################################################################

cat("Generating complex-valued data...\n")

# Set random seed for reproducibility
set.seed(42)

# Grid resolution
n <- 100

# Create uniform grid in complex plane
# Real part: [0.1, 2], Imaginary part: [-2, 2]
x1 <- seq(0.1, 2, length.out = n)
x2 <- seq(-2, 2, length.out = n)

# Create meshgrid
X1 <- outer(x1, rep(1, n))
X2 <- outer(rep(1, n), x2)

# Convert to complex numbers: z = x1 + i*x2
# For fastLPR, we need to pass x as 2D array [real, imag]
x_complex <- as.vector(X1) + 1i * as.vector(X2)

# x is a 2-column matrix: [Re(z), Im(z)]
x <- cbind(Re(x_complex), Im(x_complex))

# Compute complex logarithm: log(z)
y_complex <- log(x_complex)

# Add complex-valued noise
yr_complex <- y_complex +
  0.1 * sd(Re(y_complex)) * rnorm(length(y_complex)) +
  0.1 * 1i * sd(Im(y_complex)) * rnorm(length(y_complex))

# Split complex response into [real, imag] columns for fastLPR
yr <- cbind(Re(yr_complex), Im(yr_complex))

cat(sprintf("  - Generated %d samples\n", nrow(x)))
cat(sprintf("  - Real(z) range: [%.2f, %.2f]\n", min(Re(x_complex)), max(Re(x_complex))))
cat(sprintf("  - Imag(z) range: [%.2f, %.2f]\n", min(Im(x_complex)), max(Im(x_complex))))

################################################################################
# Perform Complex-Valued Regression
################################################################################

cat("\nPerforming complex-valued regression...\n")

# Use fixed bandwidth (from Python's selected bandwidth for simplicity)
# Python selected: h = [0.2458, 0.2458]
h_fixed <- c(0.25, 0.25)

opt <- list(
  order = 1,      # Local linear regression
  dstd = 0,       # No variance estimation for simplicity
  verbose = FALSE
)

start_time <- Sys.time()
regs <- cv_fastlpr(x, yr, h_fixed, opt)
elapsed_regression <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

cat(sprintf("  - Regression computation time: %.3f seconds\n", elapsed_regression))
cat(sprintf("  - Using fixed bandwidth: h = [%.4f, %.4f]\n", h_fixed[1], h_fixed[2]))

# Get fitted surface from regs (for surface plots)
cat("\nPreparing fitted surface for visualization...\n")
start_predict <- Sys.time()

# regs$gcv_yhat$fgrid is the fitted values on grid
# Need to reshape and convert back to complex
yhat_grid_flat <- regs$gcv_yhat$fgrid

# Grid shape is (n, n) for 2D
grid_shape <- c(n, n)

# yhat_grid_flat is a 2-column matrix [real, imag]
# Reshape each column separately
yhat_real_grid <- matrix(yhat_grid_flat[, 1], nrow = n, ncol = n)
yhat_imag_grid <- matrix(yhat_grid_flat[, 2], nrow = n, ncol = n)

# Combine into complex-valued grid
yhat_complex_grid <- yhat_real_grid + 1i * yhat_imag_grid

elapsed_predict <- as.numeric(difftime(Sys.time(), start_predict, units = "secs"))
cat(sprintf("  - Surface preparation time: %.3f seconds\n", elapsed_predict))
cat(sprintf("  - Grid shape: %d x %d\n", grid_shape[1], grid_shape[2]))

################################################################################
# Create Figure with 2x2 Layout
################################################################################

cat("\nCreating figure...\n")
start_plot <- Sys.time()

# Create output directory if it doesn't exist
fig_dir <- "fig"
if (!dir.exists(fig_dir)) {
  dir.create(fig_dir, recursive = TRUE)
}

# Open PNG device with publication-quality resolution
png(file.path(fig_dir, "fig4_complex_r.png"),
    width = 14, height = 10, units = "in", res = 300)

# Set up 2x2 layout
par(mfrow = c(2, 2), mar = c(4, 4, 3, 2))

# Color for scatter plots (dark red from jet colormap)
cl <- rgb(0.8, 0.2, 0.2, 0.3)

# View angle for all subplots (theta=-60, phi=30 in persp)
theta_view <- -60
phi_view <- 30

################################################################################
# Panel (a): Real Part of log(z)
################################################################################

# Plot fitted surface with persp
# Get color matrix for solid surface rendering
col_matrix_real <- get_color_matrix(yhat_real_grid)

# Plot fitted surface with persp and continuous colormap
persp_result <- persp(x1, x2, yhat_real_grid,
                      theta = theta_view, phi = phi_view,
                      col = col_matrix_real,
                      shade = 0.4,
                      border = NA,  # Remove grid lines for solid rendering
                      xlab = "Real(z)", ylab = "Imag(z)", zlab = "Real(log(z))",
                      main = "(a) Real(log(z))",
                      zlim = c(-4, 4),
                      ticktype = "detailed")

# Add noisy data as points
# Convert 3D coordinates to projection
points3d <- trans3d(Re(x_complex), Im(x_complex), Re(yr_complex), persp_result)
points(points3d, pch = 20, col = cl, cex = 0.5)

################################################################################
# Panel (b): Imaginary Part of log(z)
################################################################################

# Plot fitted surface with persp
# Get color matrix for solid surface rendering
col_matrix_imag <- get_color_matrix(yhat_imag_grid)

# Plot fitted surface with persp and continuous colormap
persp_result <- persp(x1, x2, yhat_imag_grid,
                      theta = theta_view, phi = phi_view,
                      col = col_matrix_imag,
                      shade = 0.4,
                      border = NA,  # Remove grid lines for solid rendering
                      xlab = "Real(z)", ylab = "Imag(z)", zlab = "Imag(log(z))",
                      main = "(b) Imag(log(z))",
                      zlim = c(-4, 4),
                      ticktype = "detailed")

# Add noisy data as points
points3d <- trans3d(Re(x_complex), Im(x_complex), Im(yr_complex), persp_result)
points(points3d, pch = 20, col = cl, cex = 0.5)

################################################################################
# Panel (c): Magnitude |log(z)|
################################################################################

# Compute magnitude
yhat_abs_grid <- Mod(yhat_complex_grid)
yr_abs <- Mod(yr_complex)

# Plot fitted surface with persp
# Get color matrix for solid surface rendering
col_matrix_abs <- get_color_matrix(yhat_abs_grid)

# Plot fitted surface with persp and continuous colormap
persp_result <- persp(x1, x2, yhat_abs_grid,
                      theta = theta_view, phi = phi_view,
                      col = col_matrix_abs,
                      shade = 0.4,
                      border = NA,  # Remove grid lines for solid rendering
                      xlab = "Real(z)", ylab = "Imag(z)", zlab = "|log(z)|",
                      main = "(c) |log(z)|",
                      zlim = c(0, 4),
                      ticktype = "detailed")

# Add noisy data as points
points3d <- trans3d(Re(x_complex), Im(x_complex), yr_abs, persp_result)
points(points3d, pch = 20, col = cl, cex = 0.5)

################################################################################
# Panel (d): Angle of log(z)
################################################################################

# Compute angle
yhat_angle_grid <- Arg(yhat_complex_grid)
yr_angle <- Arg(yr_complex)

# Plot fitted surface with persp
# Get color matrix for solid surface rendering
col_matrix_angle <- get_color_matrix(yhat_angle_grid)

# Plot fitted surface with persp and continuous colormap
persp_result <- persp(x1, x2, yhat_angle_grid,
                      theta = theta_view, phi = phi_view,
                      col = col_matrix_angle,
                      shade = 0.4,
                      border = NA,  # Remove grid lines for solid rendering
                      xlab = "Real(z)", ylab = "Imag(z)", zlab = "angle(log(z))",
                      main = "(d) angle(log(z))",
                      zlim = c(-pi, pi),
                      ticktype = "detailed")

# Add noisy data as points
points3d <- trans3d(Re(x_complex), Im(x_complex), yr_angle, persp_result)
points(points3d, pch = 20, col = cl, cex = 0.5)

# Close device
dev.off()

elapsed_plot <- as.numeric(difftime(Sys.time(), start_plot, units = "secs"))
cat(sprintf("  - Plotting time: %.3f seconds\n", elapsed_plot))

################################################################################
# Summary
################################################################################

cat("\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat("Figure 4 generation complete!\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat(sprintf("Figure saved to: %s\n", file.path(fig_dir, "fig4_complex_r.png")))
cat("\nExample completed successfully!\n")
