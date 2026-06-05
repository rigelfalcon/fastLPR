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

# Load real data: MASS::mcycle motorcycle crash test
# Column 1 = time (ms) after impact, column 2 = head acceleration (g).
# A classic boundary-bias benchmark dataset.
mcycle_path <- file.path(script_dir, "mcycle.txt")
if (!file.exists(mcycle_path)) {
  mcycle_path <- "fastLPR_R/inst/examples/mcycle.txt"
}
mcycle <- read.table(mcycle_path)
x <- matrix(mcycle[, 1], ncol = 1)
y <- matrix(mcycle[, 2], ncol = 1)
n <- nrow(x)

cat(sprintf("Loaded %d samples (MASS::mcycle)\n", n))
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

# Create evaluation grid spanning the observed time range
x_grid <- seq(min(x), max(x), length.out = 500)
y_grid_0 <- regs0$fpp_yhat(matrix(x_grid, ncol = 1))
y_grid_1 <- regs1$fpp_yhat(matrix(x_grid, ncol = 1))
y_grid_2 <- regs2$fpp_yhat(matrix(x_grid, ncol = 1))

# Create figure
dir.create("fastLPR_R/fig/reproduced", showWarnings = FALSE, recursive = TRUE)
png("fastLPR_R/fig/reproduced/fig3_boundary_comparison_r.png",
    width = 1200, height = 700, res = 150)

par(mar = c(4.5, 4.5, 3, 1))

# Plot all data and 3 regression curves
plot(x, y, pch = 20, cex = 0.7, col = "black",
     xlim = c(min(x), max(x)), ylim = c(-150, 100),
     xlab = "Time (ms)", ylab = "Acceleration (g)",
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
