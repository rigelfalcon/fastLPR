################################################################################
# Figure 5: Heteroscedastic Regression (1D and 2D)
#
# This script reproduces Figure 5 from the fastLPR paper, demonstrating:
#   - Panel (a): 1D mean estimation with intervals (CI and PI)
#   - Panel (b): 1D variance estimation (log-scale)
#   - Panel (c): 2D mean estimation (contour plot)
#   - Panel (d): 2D variance estimation (contour plot)
#
# The figure follows JSS publication standards with:
#   - Fixed random seed for reproducibility
#   - Consistent styling (fonts, colors, sizes)
#   - Self-contained code (no external dependencies except fastLPR)
#
# NOTE: This R implementation uses contour plots for 2D visualization
#       instead of 3D surfaces due to R plotting limitations.
#
# Copyright (c) 2024 fastLPR Development Team
################################################################################

# Navigate to project root
# Auto-detect working directory
script_dir <- tryCatch({ dirname(sys.frame(1)$ofile) }, error = function(e) ".")
repo_root <- normalizePath(file.path(script_dir, "../.."), mustWork = FALSE)
if (dir.exists(file.path(repo_root, "fastLPR_R"))) setwd(repo_root)

# Load fastLPR package using setup.R (sources all R files)
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

# Load required libraries
suppressPackageStartupMessages({
  library(R.matlab)
  library(Matrix)
})

# Setup
cat("\n")
cat(strrep("=", 80), "\n", sep = "")
cat("Figure 5: Heteroscedastic Regression (1D and 2D)\n")
cat(strrep("=", 80), "\n", sep = "")
cat("\n")

################################################################################
# Panel (a) and (b): 1D Heteroscedastic Regression
################################################################################

cat("Generating 1D heteroscedastic data...\n")

# Set random seed for reproducibility
set.seed(42)

# Generate 1D data
n1d <- 10000
x1d <- matrix(2 * 2 * (runif(n1d) - 0.5), ncol = 1)  # Uniform in [-2, 2]

# Define ground truth functions
fun_mu_1d <- function(x) x^3  # Cubic mean function
fun_sigma_1d <- function(x) 1 + 4 * exp(-(x^2))  # Variance depends on x

# Generate data
y1d <- as.vector(fun_mu_1d(x1d))
s1d <- as.vector(fun_sigma_1d(x1d)) * 0.5 * var(y1d)
yr1d <- y1d + rnorm(length(y1d)) * sqrt(s1d)

cat(sprintf("  - Generated %d 1D samples\n", n1d))

# Estimate conditional mean
cat("  - Estimating 1D mean...\n")
opt1d <- list(order = 1, dstd = 0, verbose = FALSE, num_dof_sample = 3)  # REDUCED from 10 to 3
hlist1d <- get_hlist(20, c(0.01, 1.0))  # Match MATLAB: 20 bandwidths
start_time <- Sys.time()
regs_mu_1d <- cv_fastlpr(x1d, yr1d, hlist1d, opt1d)
time_mu_1d <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
yp1d <- regs_mu_1d$yhat
cat(sprintf("    Computation time: %.3f seconds\n", time_mu_1d))
cat(sprintf("    Selected bandwidth: %.4f\n", regs_mu_1d$h_opt))

# Estimate conditional variance
cat("  - Estimating 1D variance...\n")
sr1d <- (yr1d - yp1d)^2
opt1d$y_type_out <- "variance"
opt1d$dstd <- 1
start_time <- Sys.time()
regs_sigma_1d <- cv_fastlpr(x1d, sr1d, hlist1d, opt1d)
time_sigma_1d <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
cat(sprintf("    Computation time: %.3f seconds\n", time_sigma_1d))
cat(sprintf("    Selected bandwidth: %.4f\n", regs_sigma_1d$h_opt))

# Compute intervals (CI and PI)
cat("  - Computing intervals (CI and PI)...\n")
ci1d <- fastlpr_interval(regs_mu_1d, regs_sigma_1d, 0.05, type = "confidence")
pi1d <- fastlpr_interval(regs_mu_1d, regs_sigma_1d, 0.05, type = "prediction")
cat(sprintf("    CI dimensions: %d x %d\n", nrow(ci1d$Values), ncol(ci1d$Values)))

################################################################################
# Panel (c) and (d): 2D Heteroscedastic Regression
################################################################################

cat("\nGenerating 2D heteroscedastic data...\n")

# Set random seed (Python uses 44 for 2D)
set.seed(44)

# Generate 2D data
n2d <- 1200
x2d <- matrix(2 * 2 * (runif(n2d * 2) - 0.5), ncol = 2)  # Uniform in [-2, 2] x [-2, 2]

# Define ground truth functions
fun_mu_2d <- function(x1, x2) x1^3 + x2^3
fun_sigma_2d <- function(x1, x2) 1 + 4 * exp(-(x1^2 + x2^2))

# Generate data
y2d <- as.vector(fun_mu_2d(x2d[, 1], x2d[, 2]))
s2d <- as.vector(fun_sigma_2d(x2d[, 1], x2d[, 2])) * 1 * sd(y2d)
yr2d <- y2d + sqrt(s2d) * rnorm(length(y2d))

cat(sprintf("  - Generated %d 2D samples\n", n2d))

# Estimate conditional mean
cat("  - Estimating 2D mean...\n")
opt2d <- list(order = 1, dstd = 0, verbose = FALSE, num_dof_sample = 3)  # REDUCED from 10 to 3
hlist2d <- get_hlist(c(10, 10), list(c(0.01, 1.0), c(0.01, 1.0)))  # Match MATLAB: 10x10=100 bandwidths
start_time <- Sys.time()
regs_mu_2d <- cv_fastlpr(x2d, yr2d, hlist2d, opt2d)
time_mu_2d <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
yp2d <- regs_mu_2d$yhat
cat(sprintf("    Computation time: %.3f seconds\n", time_mu_2d))
cat(sprintf("    Selected bandwidth: [%.4f, %.4f]\n", regs_mu_2d$h_opt[1], regs_mu_2d$h_opt[2]))

# Estimate conditional variance
cat("  - Estimating 2D variance...\n")
sr2d <- (yr2d - yp2d)^2
opt2d$y_type_out <- "variance"
opt2d$dstd <- 1  # Use 1-SE rule for robust bandwidth selection
start_time <- Sys.time()
regs_sigma_2d <- cv_fastlpr(x2d, sr2d, hlist2d, opt2d)
time_sigma_2d <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
cat(sprintf("    Computation time: %.3f seconds\n", time_sigma_2d))
cat(sprintf("    Selected bandwidth: [%.4f, %.4f]\n", regs_sigma_2d$h_opt[1], regs_sigma_2d$h_opt[2]))

# Compute intervals (CI and PI)
cat("  - Computing intervals (CI and PI)...\n")
ci2d <- fastlpr_interval(regs_mu_2d, regs_sigma_2d, 0.05, type = "confidence")
pi2d <- fastlpr_interval(regs_mu_2d, regs_sigma_2d, 0.05, type = "prediction")
cat(sprintf("    CI dimensions: %d x %d\n", dim(ci2d$Values)[1], dim(ci2d$Values)[2]))

################################################################################
# Create Figure with 2x2 Layout
################################################################################

cat("\nCreating figure...\n")

# Create output directory if it doesn't exist
fig_dir <- "fastLPR_R/fig/reproduced"
if (!dir.exists(fig_dir)) {
  dir.create(fig_dir, recursive = TRUE)
}

# Open PNG device for publication-quality output (300 DPI)
png_path <- file.path(fig_dir, "fig5_heteroscedasticity_r.png")
png(png_path, width = 14, height = 10, units = "in", res = 300)

# Set up 2x2 layout
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1), oma = c(0, 0, 0, 0))

################################################################################
# Panel a): 1D Mean with Confidence Intervals
################################################################################

# Extract grid and values from GCV result
x_grid <- regs_mu_1d$gcv_yhat$xlist[[1]]
y_mean <- regs_mu_1d$gcv_yhat$fgrid[, 1]  # Extract mean values

# Extract intervals at grid points
ci_upper <- ci1d$Values[, 1]
ci_lower <- ci1d$Values[, 2]
pi_upper <- pi1d$Values[, 1]
pi_lower <- pi1d$Values[, 2]

# Plot data (DARK blue for y+ε, green circles for y)
plot(x1d, yr1d, pch = 20, col = rgb(0, 0, 0.8), cex = 0.4,
     xlim = c(-2.2, 2.2), ylim = c(-30, 20),
     xlab = "", ylab = "", main = "(a) 1D Mean with CI and PI",
     cex.main = 1.1, font.main = 1)
points(x1d, y1d, pch = 1, col = rgb(0, 0.7, 0), cex = 0.4, lwd = 0.8)
# Plot PI (blue, wider) then CI (green, narrower)
polygon(c(x_grid, rev(x_grid)), c(pi_lower, rev(pi_upper)),
        col = rgb(0.5, 0.7, 1, 0.15), border = NA)
polygon(c(x_grid, rev(x_grid)), c(ci_lower, rev(ci_upper)),
        col = rgb(0, 1, 0, 0.2), border = NA)

# Plot true mean (red line) - AESTHETIC FIX APPLIED
x_eval <- seq(-2.2, 2.2, length.out = 200)
y_true <- fun_mu_1d(x_eval)
lines(x_eval, y_true, lwd = 2.5, col = "red")

# Plot estimated mean (yellow line)
lines(x_grid, y_mean, lwd = 2.5, col = rgb(1, 0.8, 0))

# Plot variance distribution (magenta, shifted down to floor)
# Match MATLAB values exactly: y_offset = -28, pdf_scale = 10
sigma_dist <- fun_sigma_1d(x_eval) * 0.5 * var(y1d)
y_offset <- -28  # Match MATLAB exactly
pdf_scale <- 10  # Match MATLAB exactly
# Bell opens upward (high variance = peak) - more intuitive visualization
lines(x_eval, y_offset + pdf_scale * sigma_dist / max(sigma_dist),
      col = "magenta", lwd = 2.5)

# Add legend (updated with true mean ��)
legend("bottomright",
       legend = c(expression(y + epsilon), expression(y), "PI", "CI", expression(hat(mu)), expression(mu), expression(sigma^2)),
       col = c(rgb(0, 0, 0.8), rgb(0, 0.7, 0), rgb(0.5, 0.7, 1, 0.15), rgb(0, 1, 0, 0.2), rgb(1, 0.8, 0), "red", "magenta"),
       pch = c(20, 1, 15, 15, NA, NA, NA), lty = c(NA, NA, NA, NA, 1, 1, 1),
       lwd = c(NA, NA, NA, NA, 2.5, 2.5, 2), cex = 0.9)
grid(col = rgb(0, 0, 0, 0.3))

################################################################################
# Panel b): 1D Variance Estimation (Log-scale)
################################################################################

# Compute log-transformed values
log_sigma_true <- 2 * log(sqrt(s1d))
# Add minimum threshold to avoid log(0) causing -Inf values
residuals_abs <- abs(yr1d - yp1d)
residuals_abs_clipped <- pmax(residuals_abs, 1e-10)
log_residuals <- 2 * log(residuals_abs_clipped)

# Match MATLAB plotting order and style:
# Plot MAGENTA SQUARES for true log-variance FIRST (background)
# Use small filled squares (pch=15) matching MATLAB 's' marker with MarkerSize=3
plot(x1d, log_sigma_true, pch = 15, col = "magenta", cex = 0.3,
     xlim = c(-2.2, 2.2), ylim = c(-20, 10),
     xlab = "", ylab = "", main = "(b) 1D Variance Estimation (Log-scale)",
     cex.main = 1.1, font.main = 1)
# GREEN DOTS for residuals SECOND (foreground, dense layer)
# Match MATLAB '.' marker with MarkerSize=4 - NO transparency for visibility
points(x1d, log_residuals, pch = 20, col = rgb(0, 0.7, 0), cex = 0.4)

# Plot estimated log-variance (grey line)
x_eval <- seq(-2.2, 2.2, length.out = 200)
sp1d_eval <- regs_sigma_1d$fpp_yhat(matrix(x_eval, ncol = 1))
sp1d_eval_clipped <- pmax(sp1d_eval, 1e-10)
log_sigma_est <- 2 * log(sqrt(sp1d_eval_clipped))
lines(x_eval, log_sigma_est, col = rgb(0.5, 0.5, 0.5), lwd = 2.5)

# Add legend (order matches MATLAB: magenta squares, green dots, grey line)
legend("bottomleft",
       legend = c(expression(2*log(sigma)), expression(2*log("|"*y*"-"*hat(y)*"|")),
                  expression(2*log(sqrt(hat(Sigma))))),
       col = c("magenta", rgb(0, 0.7, 0), rgb(0.5, 0.5, 0.5)),
       pch = c(0, 20, NA), lty = c(NA, NA, 1), lwd = c(0.8, NA, 2.5), cex = 0.9)
grid(col = rgb(0, 0, 0, 0.3))

################################################################################
# Panel c): 2D Mean Estimation (3D Surface with Data Points)
################################################################################

# Extract grid
grid_1 <- regs_mu_2d$xlist[[1]]
grid_2 <- regs_mu_2d$xlist[[2]]
n_grid_1 <- length(grid_1)
n_grid_2 <- length(grid_2)

# Evaluate mean on grid
grid_points <- expand.grid(x1 = grid_1, x2 = grid_2)
y_mean_2d <- regs_mu_2d$fpp_yhat$evaluate(as.matrix(grid_points))
y_mean_2d_mat <- matrix(y_mean_2d, nrow = n_grid_1, ncol = n_grid_2)

# TASK 1: Evaluate CI on grid (ci2d structure from line ~206)
# ci2d$Values has shape (n_grid_1, n_grid_2, 2) where [:,:,1]=upper, [:,:,2]=lower
ci_upper_mat <- ci2d$Values[, , 1]
ci_lower_mat <- ci2d$Values[, , 2]

# Get color matrix for height-based encoding
col_matrix_mean <- get_color_matrix(y_mean_2d_mat)

# Determine zlim to include all surfaces
zlim_range <- range(c(ci_lower_mat, ci_upper_mat, y_mean_2d_mat, yr2d, y2d), na.rm = TRUE)

# Plot mean surface first (solid colormap surface)
persp_result_c <- persp(grid_1, grid_2, y_mean_2d_mat,
                        theta = -37, phi = 20,
                        col = col_matrix_mean,
                        shade = 0.4,
                        border = NA,
                        xlab = "x1", ylab = "x2", zlab = "Mean",
                        main = "(c) 2D Mean with CI and PI",
                        cex.main = 1.1, font.main = 1,
                        ticktype = "detailed",
                        zlim = zlim_range)

# Add CI surfaces as wireframe overlays for transparency effect
# Lower CI - green wireframe (thin lines)
par(new = TRUE)
persp(grid_1, grid_2, ci_lower_mat,
      theta = -37, phi = 20,
      col = NA,  # No fill
      border = rgb(0, 0.7, 0, 0.4),  # Green wireframe with transparency
      lwd = 0.5,
      xlab = "", ylab = "", zlab = "",
      main = "",
      axes = FALSE,
      zlim = zlim_range)

# Upper CI - green wireframe (thin lines)
par(new = TRUE)
persp(grid_1, grid_2, ci_upper_mat,
      theta = -37, phi = 20,
      col = NA,  # No fill
      border = rgb(0, 0.7, 0, 0.4),  # Green wireframe with transparency
      lwd = 0.5,
      xlab = "", ylab = "", zlab = "",
      main = "",
      axes = FALSE,
      zlim = zlim_range)

# Project data points onto surface (blue for y+ε, semi-transparent)
points3d_yr <- trans3d(x2d[, 1], x2d[, 2], yr2d, persp_result_c)
points(points3d_yr, pch = 20, col = rgb(0, 0, 0.8, 0.3), cex = 0.4)

# Project true data points (red circles, larger)
points3d_y <- trans3d(x2d[, 1], x2d[, 2], y2d, persp_result_c)
points(points3d_y, pch = 1, col = rgb(0.8, 0, 0, 0.5), cex = 0.4, lwd = 0.8)

# CRITICAL: Shift variance for visualization (on the floor) - matches MATLAB line 227-228
# This creates the magenta scatter at the bottom showing variance distribution
s2d_low <- s2d - min(s2d) - 30
points3d_s2d <- trans3d(x2d[, 1], x2d[, 2], s2d_low, persp_result_c)
points(points3d_s2d, pch = 0, col = "magenta", cex = 0.4, lwd = 0.5)

################################################################################
# Panel d): 2D Variance Estimation (3D Surface with Data Points - Log-scale)
################################################################################

# Evaluate variance on grid
sp2d_eval <- regs_sigma_2d$fpp_yhat$evaluate(as.matrix(grid_points))
sp2d_eval_clipped <- pmax(sp2d_eval, 1e-10)
log_sigma_est_2d <- 2 * log(sqrt(sp2d_eval_clipped))
log_sigma_est_2d_mat <- matrix(log_sigma_est_2d, nrow = n_grid_1, ncol = n_grid_2)

# Compute true log-variance for scatter points
log_sigma_true_2d <- 2 * log(sqrt(s2d))
residuals_abs_2d <- abs(yr2d - yp2d)
residuals_abs_2d_clipped <- pmax(residuals_abs_2d, 1e-10)
log_residuals_2d <- 2 * log(residuals_abs_2d_clipped)

# FIX Issue 3: Match MATLAB z-axis scale
# Compute sensible zlim based on surface + scatter points
# MATLAB shows roughly [-15, 5] for panel (d)
# Use surface range with reasonable margin for scatter points
zlim_d <- c(min(log_sigma_est_2d_mat) - 2, max(log_sigma_est_2d_mat) + 2)

# Get color matrix for height-based encoding
col_matrix_var <- get_color_matrix(log_sigma_est_2d_mat)

# Plot 3D surface with persp - ADD EXPLICIT ZLIM
persp_result_d <- persp(grid_1, grid_2, log_sigma_est_2d_mat,
                        theta = -37, phi = 20,
                        col = col_matrix_var,
                        shade = 0.4,
                        border = NA,
                        xlab = "x1", ylab = "x2", zlab = "Log-variance",
                        main = "(d) 2D Variance Estimation (Log-scale)",
                        cex.main = 1.1, font.main = 1,
                        ticktype = "detailed",
                        zlim = zlim_d)  # FIX: Explicit z-axis limits

# Project true log-variance points (magenta squares)
points3d_sigma <- trans3d(x2d[, 1], x2d[, 2], log_sigma_true_2d, persp_result_d)
points(points3d_sigma, pch = 0, col = rgb(1, 0, 1, 0.5), cex = 0.4, lwd = 0.8)

# Project residual points (green dots)
points3d_res <- trans3d(x2d[, 1], x2d[, 2], log_residuals_2d, persp_result_d)
points(points3d_res, pch = 20, col = rgb(0, 0.7, 0, 0.3), cex = 0.4)

# Close device
dev.off()

################################################################################
# Summary
################################################################################

cat("\n")
cat(strrep("=", 80), "\n", sep = "")
cat("Figure 5 Generation Complete!\n")
cat(strrep("=", 80), "\n", sep = "")
cat("\n")

cat("Summary:\n")
cat(sprintf("  - 1D data: %d samples\n", n1d))
cat(sprintf("  - 2D data: %d samples\n", n2d))
cat("  - Demonstrates heteroscedastic regression:\n")
cat("    * Panel (a): 1D mean with intervals (CI and PI)\n")
cat("    * Panel (b): 1D variance estimation (log-scale)\n")
cat("    * Panel (c): 2D mean estimation (filled contour plot)\n")
cat("    * Panel (d): 2D variance estimation (log-scale filled contour)\n")
cat(sprintf("  - Figure saved to: %s\n", normalizePath(png_path)))
cat("\n")

cat("Key observations:\n")
cat("  - Variance depends on predictor values (heteroscedastic)\n")
cat("  - CI (green) and PI (blue) adapt to local variance\n")
cat("  - Log-scale visualization helps identify variance patterns\n")
cat("  - 3D surfaces show mean and variance estimation in 2D space\n")
cat("\n")
