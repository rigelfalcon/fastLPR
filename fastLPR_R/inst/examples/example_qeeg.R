#!/usr/bin/env Rscript
# Figure 6: Real-World Applications - CORRECTED (Base R Graphics)
#
# CRITICAL FIXES:
#   - Panel (a): 3D surface using persp() showing MODEL OUTPUT
#   - Panel (b): 3D scatter using scatterplot3d showing MRI data

# Auto-detect working directory
script_dir <- tryCatch({ dirname(sys.frame(1)$ofile) }, error = function(e) ".")
repo_root <- normalizePath(file.path(script_dir, "../.."), mustWork = FALSE)
if (dir.exists(file.path(repo_root, "fastLPR_R"))) setwd(repo_root)

# Load fastLPR package using setup.R (sources all R files)
source("fastLPR_R/setup.R")

suppressPackageStartupMessages({
  library(R.matlab)
  library(Matrix)
})

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

cat("\n", strrep("=", 80), "\n")
cat("Figure 6: qEEG and MRI Applications\n")
cat(strrep("=", 80), "\n\n")

### Panel (a): qEEG ###

cat("Panel (a): qEEG Regression...\n")

data_table <- read.csv("fastLPR_R/data/data_qeeg.csv")
cat(sprintf("  - %d samples\n", nrow(data_table)))

X_qeeg <- as.matrix(data_table[, c("age", "freq")])
# FIXED: Use as.complex() then Re() to properly handle negative real parts
y_qeeg <- sapply(as.character(data_table$log10_10), function(s) Re(as.complex(s)))

opt_qeeg <- list(order=0, dstd=10, verbose=FALSE, num_dof_sample=3)
hlist_qeeg <- get_hlist(c(5,5), list(c(0.1,1.0), c(0.1,2.0)))

t1 <- Sys.time()
regs_qeeg <- cv_fastlpr(X_qeeg, y_qeeg, hlist_qeeg, opt_qeeg)
cat(sprintf("  - Time: %.2f sec\n", difftime(Sys.time(), t1, units="secs")))

# Create grid for 3D surface
n_grid <- 40
age_seq <- seq(min(X_qeeg[,1]), max(X_qeeg[,1]), length.out=n_grid)
freq_seq <- seq(min(X_qeeg[,2]), max(X_qeeg[,2]), length.out=n_grid)
grid_pts <- expand.grid(age=age_seq, freq=freq_seq)

# Evaluate predictions on grid using fpp_yhat interpolant
cat("  - Evaluating on grid...\n")
y_pred_vec <- regs_qeeg$fpp_yhat$evaluate(as.matrix(grid_pts))
y_pred_mat <- matrix(y_pred_vec, nrow=n_grid, ncol=n_grid)

### Panel (b): MRI ###

cat("\nPanel (b): MRI Regression...\n")

mri_data <- readMat("fastLPR_R/data/subjectimage_T1.mat")
Cube <- mri_data$Cube
cat(sprintf("  - Dims: [%d,%d,%d]\n", dim(Cube)[1], dim(Cube)[2], dim(Cube)[3]))

indices <- which(Cube > 0, arr.ind=TRUE)
set.seed(42)
idx <- sample(nrow(indices), min(50000, nrow(indices)))  # Reduced for speed
X_mri <- indices[idx,]
y_mri <- as.numeric(Cube[Cube>0][idx])

opt_mri <- list(order=0, dstd=1, verbose=FALSE, N=c(64,64,64))
hlist_mri <- matrix(c(0.03,0.03,0.03), nrow=1)

t2 <- Sys.time()
regs_mri <- cv_fastlpr(X_mri, y_mri, hlist_mri, opt_mri)
cat(sprintf("  - Time: %.2f sec\n", difftime(Sys.time(), t2, units="secs")))

# Create 3D prediction grid (matching MATLAB's 50x50x50 grid)
cat("  - Creating prediction grid...\n")
grid_res <- 50
x_range <- c(min(X_mri[,1]), max(X_mri[,1]))
y_range <- c(min(X_mri[,2]), max(X_mri[,2]))
z_range <- c(min(X_mri[,3]), max(X_mri[,3]))

x_grid_seq <- seq(x_range[1], x_range[2], length.out=grid_res)
y_grid_seq <- seq(y_range[1], y_range[2], length.out=grid_res)
z_grid_seq <- seq(z_range[1], z_range[2], length.out=grid_res)

# Create meshgrid (like MATLAB's meshgrid)
grid_3d <- expand.grid(x=x_grid_seq, y=y_grid_seq, z=z_grid_seq)
x_grid <- as.matrix(grid_3d)

cat(sprintf("  - Grid size: %d points (%d x %d x %d)\n",
            nrow(x_grid), grid_res, grid_res, grid_res))

# Get predictions on the grid using the interpolant
cat("  - Evaluating predictions on grid...\n")
y_pred_mri <- regs_mri$fpp_yhat$evaluate(x_grid)

# Use grid points and predictions for visualization
X_vis <- x_grid
y_vis <- y_pred_mri
cat(sprintf("  - Total grid points for visualization: %d\n", length(y_vis)))

### Create Figure ###

cat("\nCreating figure...\n")
dir.create("fastLPR_R/fig/reproduced", showWarnings=FALSE, recursive=TRUE)

png("fastLPR_R/fig/reproduced/fig6_applications_r.png",
    width=14, height=6, units="in", res=300)
par(mfrow=c(1,2), mar=c(3,3,3,2))

# Panel (a): 3D Surface with Height-Based Color Encoding
# NOTE: R's persp() cannot overlay a separate contour plot at z_bottom like MATLAB's surf()
# Instead, we use persp() with colored facets (MATLAB equivalent approach)
# The MATLAB version shows surface + heatmap at bottom, but persp() doesn't support multiple surfaces

col_matrix_qeeg <- get_color_matrix(y_pred_mat)

persp_result_a <- persp(age_seq, freq_seq, y_pred_mat,
                           theta=-37.5, phi=30,
                           col=col_matrix_qeeg,
                           border=NA, shade=0.4,
                           xlab="Age (log10)", ylab="Freq (Hz)", zlab="Log-spec",
                           main="(a) 2D qEEG Regression",
                           cex.main=1.3, font.main=2,
                           ticktype="detailed")

# Add filled contour heatmap projected at z_bottom (matching MATLAB's surf() at bottom)
# MATLAB uses a second surf() call to create a colored heatmap projection
# R: Create a grid of colored polygons at z_bottom to simulate filled heatmap
z_bottom <- min(y_pred_mat) - 0.5

# Get parula colors for the z values
n_colors <- 100
parula_pal <- parula(n_colors)
z_range <- range(y_pred_mat)
z_norm <- (y_pred_mat - z_range[1]) / (z_range[2] - z_range[1])
# Ensure color_indices is a matrix, not array
color_indices <- matrix(pmax(1, pmin(n_colors, ceiling(z_norm * (n_colors - 1)) + 1)),
                        nrow = nrow(y_pred_mat), ncol = ncol(y_pred_mat))

# Create filled rectangular patches at z_bottom (like MATLAB's surf heatmap)
# For each grid cell, draw a colored polygon at the bottom
for (i in 1:(n_grid - 1)) {
  for (j in 1:(n_grid - 1)) {
    # Get 4 corners of grid cell
    x_corners <- c(age_seq[i], age_seq[i+1], age_seq[i+1], age_seq[i])
    y_corners <- c(freq_seq[j], freq_seq[j], freq_seq[j+1], freq_seq[j+1])
    z_corners <- rep(z_bottom, 4)

    # Transform to 3D perspective coordinates
    coords_3d <- trans3d(x_corners, y_corners, z_corners, persp_result_a)

    # Use average color of 4 corners for this patch (matching get_color_matrix logic)
    # Extract color indices for the 4 corners as scalars
    idx1 <- color_indices[i, j]
    idx2 <- color_indices[i+1, j]
    idx3 <- color_indices[i, j+1]
    idx4 <- color_indices[i+1, j+1]
    avg_color_idx <- round(mean(c(idx1, idx2, idx3, idx4)))
    cell_color <- parula_pal[avg_color_idx]

    # Draw filled polygon at z_bottom with semi-transparency (matching MATLAB's FaceAlpha=0.8)
    polygon(coords_3d$x, coords_3d$y, col = adjustcolor(cell_color, alpha.f = 0.8), border = NA)
  }
}

# Panel (b): 3D MRI Brain Volume with scatterplot3d
if (!requireNamespace("scatterplot3d", quietly = TRUE)) {
  cat("WARNING: scatterplot3d not installed. Installing from CRAN...\n")
  install.packages("scatterplot3d", repos = "https://cloud.r-project.org")
}

if (requireNamespace("scatterplot3d", quietly = TRUE)) {
  library(scatterplot3d)

  # Match MATLAB approach exactly:
  # 1. Normalize intensity to [0,1]
  # 2. Set values below 0.1 to 0 (threshold)
  # 3. Use sigmoid for alpha, linear for size

  # Normalize to [0,1] (like MATLAB lines 205-206)
  y_norm <- (y_vis - min(y_vis)) / (max(y_vis) - min(y_vis) + .Machine$double.eps)

  # Apply threshold: set low values to 0 (like MATLAB line 211)
  threshold <- 0.1
  y_norm[y_norm < threshold] <- 0

  # Keep only non-zero values for plotting
  keep_idx <- which(y_norm > 0)
  X_vis_filtered <- X_vis[keep_idx, ]
  y_norm_filtered <- y_norm[keep_idx]
  y_vis_filtered <- y_vis[keep_idx]

  cat(sprintf("  - Applied threshold %.2f (MATLAB approach)\n", threshold))
  cat(sprintf("  - Kept %d/%d points (%.1f%%)\n",
              length(keep_idx), length(y_vis), 100*length(keep_idx)/length(y_vis)))

  # Compute alpha using sigmoid (like MATLAB line 214)
  # MATLAB: alpha_data = 1./(1+exp(-y_norm))
  alpha_vals <- 1 / (1 + exp(-y_norm_filtered))

  # Compute marker size (like MATLAB line 215)
  # MATLAB: marker_size = 5 + 10 * y_norm
  # Scale down for R: use 0.3-1.3 range
  marker_size <- 0.3 + 1.0 * y_norm_filtered

  # Create color mapping with jet colormap (like MATLAB line 232)
  jet_colors <- colorRampPalette(c("darkblue", "blue", "cyan", "green", "yellow", "orange", "red"))(100)
  color_indices <- pmax(1, pmin(100, ceiling(y_norm_filtered * 99) + 1))
  colors <- jet_colors[color_indices]

  # Add alpha transparency
  colors_with_alpha <- sapply(1:length(colors), function(i) {
    rgb_vals <- col2rgb(colors[i]) / 255
    rgb(rgb_vals[1], rgb_vals[2], rgb_vals[3], alpha = alpha_vals[i])
  })

  # Plot 3D scatter
  s3d <- scatterplot3d(X_vis_filtered[,1], X_vis_filtered[,2], X_vis_filtered[,3],
                       color = colors_with_alpha,
                       pch = 19,
                       cex.symbols = marker_size,
                       main = "(b) 3D MRI T1 Regression",
                       xlab = "x",
                       ylab = "y",
                       zlab = "z",
                       font.main = 2, cex.main = 1.3,
                       box = TRUE, grid = TRUE,
                       angle = 124,
                       type = "p")
} else {
  # Fallback: 2D projection with warning
  plot(X_vis[,1], X_vis[,2], pch=20, cex=0.5,
       main="(b) 3D MRI T1 Regression\n(3D visualization requires scatterplot3d)",
       xlab="x", ylab="y", cex.main=1.0)
  text(mean(range(X_vis[,1])), mean(range(X_vis[,2])),
       "Install scatterplot3d for 3D view:\ninstall.packages('scatterplot3d')",
       cex=0.8, col="red")
}

dev.off()

cat("\n", strrep("=", 80), "\n")
cat("Figure 6 Complete!\n")
cat("Panel (a): 3D Surface with contour projection at z_bottom (qEEG)\n")
cat("Panel (b): 3D MRI brain volume with sigmoid transparency (MATLAB approach)\n")
cat("  - Threshold: 0.1 (normalized), keeps ~90% of points\n")
cat("  - Alpha: sigmoid function 1/(1+exp(-y_norm))\n")
cat("  - Colormap: jet (matching MATLAB)\n")
cat("  - Figure saved to: fastLPR_R/fig/reproduced/fig6_applications_r.png\n")
cat(strrep("=", 80), "\n\n")
