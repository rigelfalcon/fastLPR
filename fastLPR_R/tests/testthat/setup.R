# Setup file for testthat
# This file is automatically sourced before running tests
#
# Strategy: Use installed package if available (proper Rcpp support),
#           otherwise fall back to sourcing R files (no Rcpp acceleration)

# Method 1: Try installed package (preferred)
if (requireNamespace("fastlpr", quietly = TRUE)) {
  library(fastlpr)
} else {
  # Method 2: Fall back to sourcing R files
  # Note: Rcpp acceleration unavailable in this mode

  find_pkg_root <- function() {
    path <- getwd()
    if (grepl("tests[/\\\\]testthat$", path)) return(dirname(dirname(path)))
    if (grepl("tests$", path)) return(dirname(path))
    if (grepl("fastLPR_R$", path)) return(path)
    while (dirname(path) != path) {
      if (dir.exists(file.path(path, "R")) &&
          file.exists(file.path(path, "R", "cv_fastlpr.R"))) {
        return(path)
      }
      path <- dirname(path)
    }
    return(dirname(dirname(getwd())))
  }

  pkg_root <- find_pkg_root()
  r_dir <- file.path(pkg_root, "R")

  if (dir.exists(r_dir)) {
    r_files <- list.files(r_dir, pattern = "\\.R$", full.names = TRUE)
    for (f in r_files) {
      source(f)
    }
  }
}
