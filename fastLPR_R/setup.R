# fastLPR R Package Setup Script
# ============================================================================
# USAGE: source("setup.R") from fastLPR_R directory
#
# LOADING PRIORITY:
#   1. Installed package (library(fastlpr)) - PREFERRED, includes Rcpp
#   2. Source files (R/*.R) - Fallback, pure R only
#
# For Rcpp acceleration, install the package first:
#   R CMD INSTALL .
# ============================================================================

cat("=== fastLPR R Package Setup ===\n\n")

# Disable JIT to prevent performance issues (15+ second overhead)
old_jit <- compiler::enableJIT(0)

# ============================================================================
# METHOD 1: Try installed package (PREFERRED)
# ============================================================================
if (requireNamespace("fastlpr", quietly = TRUE)) {
  library(fastlpr)
  cat("Loaded: INSTALLED PACKAGE (fastlpr)\n")
  cat("Path:", system.file(package = "fastlpr"), "\n")

  # Check Rcpp status
  if (exists("rcpp_available") && rcpp_available()) {
    info <- tryCatch(get_rcpp_info(), error = function(e) NULL)
    if (!is.null(info)) {
      cat(sprintf("Rcpp: ENABLED (%d OpenMP threads)\n", info$num_threads))
    } else {
      cat("Rcpp: ENABLED\n")
    }
  } else {
    cat("Rcpp: DISABLED (pure R fallback)\n")
  }

} else {
  # ==========================================================================
  # METHOD 2: Fallback to sourcing R files (no Rcpp)
  # ==========================================================================
  cat("Package not installed, using SOURCE FILES fallback\n")
  cat("WARNING: Rcpp acceleration NOT available (pure R only)\n")
  cat("To enable Rcpp: R CMD INSTALL .\n\n")

  # Detect script directory
  script_dir <- tryCatch({
    if (exists(".script_dir")) .script_dir
    else if (!is.null(sys.frames()[[1]]$ofile)) dirname(sys.frames()[[1]]$ofile)
    else getwd()
  }, error = function(e) getwd())

  # Required packages
  required_packages <- c("Matrix", "R.matlab", "Rcpp")
  for (pkg in required_packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      cat("Installing", pkg, "...\n")
      install.packages(pkg, repos = "https://cran.r-project.org")
    }
    suppressWarnings(library(pkg, character.only = TRUE))
  }

  # Source all R files
  r_dir <- file.path(script_dir, "R")
  if (dir.exists(r_dir)) {
    r_files <- list.files(r_dir, pattern = "\\.R$", full.names = TRUE)
    for (f in r_files) {
      source(f)
    }
    cat("Sourced", length(r_files), "R files from", r_dir, "\n")
  } else {
    stop("R/ directory not found. Run from fastLPR_R directory.")
  }
}

cat("\n=== Setup Complete ===\n")
cat("Example: regs <- cv_fastlpr(x, y, get_hlist(20, c(0.05, 0.5)))\n")
