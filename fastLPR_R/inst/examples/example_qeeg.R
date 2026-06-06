#!/usr/bin/env Rscript
# Code to generate the qEEG figure (Figure: fig_qeeg) for the fastLPR paper.
#
# qEEG cross-spectral normative modeling (Manuscript Section 4).
#   - Data: data_qeeg_cross_only.csv (N = 66505, complex-valued response)
#   - Native complex-valued local polynomial regression (order = 1)
#   - GCV-based bandwidth selection with the 1-SE rule, effective DoF tracking
#   - Prediction and pointwise confidence bands on a dense grid
#
# Five-panel figure:
#   (a) Raw data scatter on (age, frequency), colored by |y|
#   (b) GCV bandwidth selection surface over the (h1, h2) grid, 1-SE marker
#   (c) Fitted real-part surface Re(m_hat)
#   (d) Fitted imaginary-part surface Im(m_hat)
#   (e) 95% confidence band at the f = 10 Hz slice (real top, imaginary bottom)
#
# Self-contained (no external dependencies except fastLPR_R).

# Auto-detect working directory (works both when source()d and via Rscript).
# The script lives in fastLPR_R/inst/examples; walk up to find the repo root
# (the directory that contains fastLPR_R/setup.R) and set it as the cwd.
script_dir <- tryCatch({
  dirname(sys.frame(1)$ofile)
}, error = function(e) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
  if (length(file_arg) == 1 && nzchar(file_arg)) dirname(normalizePath(file_arg)) else "."
})
find_repo_root <- function(start) {
  d <- normalizePath(start, mustWork = FALSE)
  for (i in 1:6) {
    if (file.exists(file.path(d, "fastLPR_R", "setup.R"))) return(d)
    parent <- dirname(d)
    if (parent == d) break
    d <- parent
  }
  getwd()
}
repo_root <- find_repo_root(script_dir)
setwd(repo_root)
# Load fastLPR package using setup.R
source("fastLPR_R/setup.R")

# Helper function for parula colormap (matching MATLAB's default)
parula <- function(n) {
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

cat("\n", strrep("=", 80), "\n", sep = "")
cat("qEEG Cross-Spectral Normative Modeling\n")
cat(strrep("=", 80), "\n\n", sep = "")

################################################################################
# Load and explore the data
################################################################################

cat("Loading data...\n")
qeeg <- read.csv("fastLPR_R/data/data_qeeg_cross_only.csv")
# R's read.csv parses the complex column into the native complex type.
y <- qeeg$riemlogm10_1
x <- cbind(qeeg$age, qeeg$freq)
cat(sprintf("  - Observations: %d\n", nrow(x)))
cat(sprintf("  - Real part range: [%.3f, %.3f]\n", range(Re(y))[1], range(Re(y))[2]))
cat(sprintf("  - Imaginary part range: [%.3f, %.3f]\n", range(Im(y))[1], range(Im(y))[2]))

################################################################################
# Bandwidth selection and model fitting
################################################################################

cat("\nFitting complex-valued LPR (order = 1, GCV bandwidth selection)...\n")
hlist <- get_hlist(c(9, 9), rbind(c(1e-3, 2), c(0.05, 2)))
opt <- list(order = 1, calc_dof = TRUE, dstd = 1, seed = 42, verbose = FALSE)

t0 <- proc.time()
result <- cv_fastlpr(x, y, hlist, opt)
elapsed <- (proc.time() - t0)[3]

h1se <- as.numeric(result$gcv_yhat$h1se)
hmin <- as.numeric(result$gcv_yhat$hmin)
cat(sprintf("  - Selected bandwidth (1-SE): [%.4f, %.4f]\n", h1se[1], h1se[2]))
cat(sprintf("  - Selected bandwidth (min):  [%.4f, %.4f]\n", hmin[1], hmin[2]))
if (!is.null(result$dof)) cat(sprintf("  - Effective DoF: %.1f\n", result$dof))
cat(sprintf("  - Computation time: %.1f seconds\n", elapsed))

################################################################################
# Prediction and confidence bands on a dense grid
################################################################################

cat("\nPredicting on 100 x 100 evaluation grid...\n")
n_grid <- 100
age_grid  <- seq(min(x[, 1]), max(x[, 1]), length.out = n_grid)
freq_grid <- seq(min(x[, 2]), max(x[, 2]), length.out = n_grid)
x_eval <- as.matrix(expand.grid(age_grid, freq_grid))
pred <- fastlpr_predict(result, x_eval)
pred_mat <- matrix(pred, nrow = n_grid, ncol = n_grid)
re_mat <- matrix(Re(pred), nrow = n_grid, ncol = n_grid)
im_mat <- matrix(Im(pred), nrow = n_grid, ncol = n_grid)

# Pointwise standard error via the local-polynomial expression used for the
# confidence bands: se^2 = sigma^2 * nu / (|H| * s_0), evaluated at each point.
# (See Manuscript Section 4; fastlpr_interval() wraps this same formula.)
resid <- y - fastlpr_predict(result, x)
sig2  <- mean(abs(resid)^2)
nu    <- 0.079577471546          # Gaussian kernel, d = 2, order = 1
prod_h <- prod(h1se)
s0_eval <- pmax(Re(result$fpp_s0$evaluate(x_eval)), 1e-10)
se_eval <- sqrt(sig2 * nu / (prod_h * s0_eval))
zval <- qnorm(0.975)

################################################################################
# Build the 5-panel figure
################################################################################

cat("\nCreating figure...\n")
dir.create("fastLPR_R/fig/reproduced", showWarnings = FALSE, recursive = TRUE)
png("fastLPR_R/fig/reproduced/fig_qeeg.png", width = 4500, height = 3000, res = 300)
# 2 x 3 layout: panels (a)-(d) fill the left/middle columns; the right column
# stacks the two CI sub-plots that together form panel (e).
layout(matrix(c(1, 2, 5,
                3, 4, 6), nrow = 2, byrow = TRUE))
par(mar = c(4.5, 4.5, 3, 2), oma = c(0, 0, 2, 0))

## Panel (a): raw scatter colored by |y|
absy <- Mod(y)
sub <- sample(nrow(x), min(20000, nrow(x)))  # subsample for plotting clarity
col_idx <- pmax(1, pmin(100, ceiling((absy[sub] - min(absy)) /
                        (max(absy) - min(absy) + 1e-12) * 99) + 1))
pal <- parula(100)
plot(x[sub, 1], x[sub, 2], col = pal[col_idx], pch = 19, cex = 0.25,
     xlab = expression(log[10](age)), ylab = "Frequency (Hz)",
     main = "(a) Raw data, colored by |y|", font.main = 2, cex.main = 1.3,
     cex.lab = 1.2)

## Panel (b): GCV bandwidth selection surface
h1_u <- sort(unique(hlist[, 1]))
h2_u <- sort(unique(hlist[, 2]))
gcv_mat <- matrix(NA, length(h1_u), length(h2_u))
for (k in seq_len(nrow(hlist))) {
  i1 <- match(hlist[k, 1], h1_u)
  i2 <- match(hlist[k, 2], h2_u)
  gcv_mat[i1, i2] <- result$gcv_yhat$gcv_m[k]
}
image(log10(h1_u), log10(h2_u), gcv_mat, col = parula(100),
      xlab = expression(log[10](h[1])), ylab = expression(log[10](h[2])),
      main = "(b) GCV bandwidth surface", font.main = 2, cex.main = 1.3,
      cex.lab = 1.2)
contour(log10(h1_u), log10(h2_u), gcv_mat, add = TRUE, col = "black", lwd = 0.4)
points(log10(hmin[1]), log10(hmin[2]), pch = 19, col = "blue", cex = 1.6)
points(log10(h1se[1]), log10(h1se[2]), pch = 8, col = "red", cex = 2, lwd = 2)
legend("topright", legend = c("GCV min", "1-SE"), pch = c(19, 8),
       col = c("blue", "red"), bg = "white", cex = 0.9)

## Panel (c): fitted real-part surface
image(age_grid, freq_grid, re_mat, col = parula(100),
      xlab = expression(log[10](age)), ylab = "Frequency (Hz)",
      main = expression(bold(paste("(c) Fitted real part  ", Re(hat(m))))),
      cex.main = 1.3, cex.lab = 1.2)
contour(age_grid, freq_grid, re_mat, add = TRUE, col = "black", lwd = 0.4)

## Panel (d): fitted imaginary-part surface
image(age_grid, freq_grid, im_mat, col = parula(100),
      xlab = expression(log[10](age)), ylab = "Frequency (Hz)",
      main = expression(bold(paste("(d) Fitted imag part  ", Im(hat(m))))),
      cex.main = 1.3, cex.lab = 1.2)
contour(age_grid, freq_grid, im_mat, add = TRUE, col = "black", lwd = 0.4)

## Panel (e): 95% CI band at f = 10 Hz slice (real top, imag bottom)
f_target <- 10
jf <- which.min(abs(freq_grid - f_target))
# slice indices for this frequency column in the flattened expand.grid order
slice_idx <- ((jf - 1) * n_grid + 1):(jf * n_grid)
ag <- age_grid
re_slice <- Re(pred[slice_idx]); im_slice <- Im(pred[slice_idx])
se_slice <- se_eval[slice_idx]
# top-right cell: real part with CI band
plot(ag, re_slice, type = "l", lwd = 2, col = "black",
     ylim = range(c(re_slice - zval * se_slice, re_slice + zval * se_slice)),
     xlab = expression(log[10](age)), ylab = "Re(m)",
     main = "(e) 95% CI at f = 10 Hz (real)",
     font.main = 2, cex.main = 1.2, cex.lab = 1.1)
polygon(c(ag, rev(ag)),
        c(re_slice + zval * se_slice, rev(re_slice - zval * se_slice)),
        col = rgb(0.2, 0.2, 0.8, 0.25), border = NA)
lines(ag, re_slice, lwd = 2)
# bottom-right cell: imaginary part with CI band
plot(ag, im_slice, type = "l", lwd = 2, col = "black",
     ylim = range(c(im_slice - zval * se_slice, im_slice + zval * se_slice)),
     xlab = expression(log[10](age)), ylab = "Im(m)",
     main = "95% CI at f = 10 Hz (imag)",
     font.main = 2, cex.main = 1.2, cex.lab = 1.1)
polygon(c(ag, rev(ag)),
        c(im_slice + zval * se_slice, rev(im_slice - zval * se_slice)),
        col = rgb(0.8, 0.2, 0.2, 0.25), border = NA)
lines(ag, im_slice, lwd = 2)

mtext("qEEG Cross-Spectral Normative Modeling (fastLPR)",
      outer = TRUE, cex = 1.5, font = 2, line = 0.2)
dev.off()

cat("\nFigure saved to: fastLPR_R/fig/reproduced/fig_qeeg.png\n")
cat("Example completed successfully!\n")
