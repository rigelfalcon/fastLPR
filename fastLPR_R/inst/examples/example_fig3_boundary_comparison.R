#!/usr/bin/env Rscript
# Figure 3: Boundary Comparison (NW vs LL vs LQ Regression)
# UNIFIED VERSION - Uses R native RNG (equivalent to MATLAB rng(0))

# Auto-detect working directory
script_dir <- tryCatch({ dirname(sys.frame(1)$ofile) }, error = function(e) ".")
repo_root <- normalizePath(file.path(script_dir, "../.."), mustWork = FALSE)
if (dir.exists(file.path(repo_root, "fastLPR_R"))) setwd(repo_root)
# Load fastLPR package using setup.R
source("fastLPR_R/setup.R")

cat("\n", strrep("=", 80), "\n")
cat("Figure 3: Boundary Comparison (NW vs LL vs LQ Regression)\n")
cat(strrep("=", 80), "\n\n")

# Set random seed (MATLAB uses rng(0), R uses set.seed(0))
set.seed(0)

# Generate test data (matching MATLAB)
n <- 500
x <- matrix(runif(n, 0, 20), ncol = 1)

# True function: Bessel J1 (matching MATLAB besselj(1,x))
y_true <- besselJ(x, 1)

# Add Gaussian noise (matching MATLAB: 0.4*std(y_true)*randn)
y <- matrix(y_true + 0.4 * sd(y_true) * rnorm(n), ncol = 1)

cat(sprintf("Generated %d samples\n", n))
cat(sprintf("X range: [%.1f, %.1f]\n", min(x), max(x)))

# Create bandwidth list
hlist <- get_hlist(50, c(0.01, 2.0))
cat(sprintf("\nBandwidth list: %d values\n", length(hlist)))

# Common options
opt <- list(N = 500, verbose = FALSE)

# Fit three regression models
cat("\n=== Order 0 (Nadaraya-Watson) ===\n")
opt$order <- 0
regs0 <- cv_fastlpr(x, y, hlist, opt)
cat(sprintf("Selected bandwidth: %.4f\n", regs0$gcv_yhat$h1se))

cat("\n=== Order 1 (Local Linear) ===\n")
opt$order <- 1
regs1 <- cv_fastlpr(x, y, hlist, opt)
cat(sprintf("Selected bandwidth: %.4f\n", regs1$gcv_yhat$h1se))

cat("\n=== Order 2 (Local Quadratic) ===\n")
opt$order <- 2
regs2 <- cv_fastlpr(x, y, hlist, opt)
cat(sprintf("Selected bandwidth: %.4f\n", regs2$gcv_yhat$h1se))

# Create evaluation grid
x_grid <- seq(0, 20, length.out = 500)
y_grid_0 <- regs0$fpp_yhat(matrix(x_grid, ncol = 1))
y_grid_1 <- regs1$fpp_yhat(matrix(x_grid, ncol = 1))
y_grid_2 <- regs2$fpp_yhat(matrix(x_grid, ncol = 1))

# Create figure
dir.create("fastLPR_R/fig/reproduced", showWarnings = FALSE, recursive = TRUE)
png("fastLPR_R/fig/reproduced/fig3_boundary_comparison_r.png",
    width = 1200, height = 700, res = 150)

par(mar = c(4.5, 4.5, 3, 1))

# Plot all data and 3 regression curves
plot(x, y, pch = 20, cex = 0.4, col = "black",
     xlim = c(0, 20), ylim = c(-0.6, 1.0),
     xlab = "x", ylab = "y",
     main = "Boundary Effects: Local Polynomial Orders 0, 1, 2",
     cex.main = 1.3, cex.lab = 1.2, font.main = 1)

lines(x_grid, y_grid_0, col = rgb(0, 0.7, 0), lwd = 3, lty = 1)  # NW: solid green
lines(x_grid, y_grid_1, col = rgb(0.8, 0, 0), lwd = 3, lty = 2)  # LL: dashed red  
lines(x_grid, y_grid_2, col = rgb(0, 0, 0.8), lwd = 3, lty = 4)  # LQ: dash-dot blue

legend("topleft",
       legend = c("Noisy data", "NW (order 0)", "LL (order 1)", "LQ (order 2)"),
       col = c(rgb(0, 0, 0, 0.3), rgb(0, 0.7, 0), rgb(0.8, 0, 0), rgb(0, 0, 0.8)),
       pch = c(20, NA, NA, NA),
       lty = c(NA, 1, 2, 4),
       lwd = c(NA, 3, 3, 3),
       cex = 1.1, bg = "white", box.lwd = 1)

grid(col = rgb(0, 0, 0, 0.2))
dev.off()

cat("\nFigure saved to: fastLPR_R/fig/reproduced/fig3_boundary_comparison_r.png\n")
cat("\nDone!\n")
