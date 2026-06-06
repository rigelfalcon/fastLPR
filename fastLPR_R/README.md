# fastlpr: Fast Local Polynomial Regression for R

[![R package](https://img.shields.io/badge/R-package-source--available-blue)](https://github.com/rigelfalcon/fastLPR)
[![License: GPL-3](https://img.shields.io/badge/License-GPL--3-blue.svg)](https://opensource.org/licenses/GPL-3.0)

NUFFT-accelerated local polynomial regression and kernel density estimation for R.
This is the R port of the fastLPR MATLAB/Python toolbox, achieving O(N + M log M)
computational complexity through Non-Uniform Fast Fourier Transform (NUFFT) with Gaussian gridding.

## Features

- **Fast**: 20-60x faster than ks package for KDE, 100-1000x faster than naive implementations
- **Flexible**: Supports 1D/2D/3D data, complex-valued responses, heteroscedastic variance estimation
- **Automatic**: GCV/LCV-based bandwidth selection with 1-SE rule for robustness
- **Complete**: Confidence intervals, DOF estimation, multiple polynomial orders (0/1/2)

## Installation

### From Source (Development)

```r
# Clone repository and navigate to fastLPR_R
setwd("path/to/fastLPR_R")

# Option 1: Source setup script (for development)
source("setup.R")

# Option 2: Install as package (recommended)
# From command line:
# R CMD INSTALL .
```

### Dependencies

**Required:**
- R >= 4.2.0
- Rcpp (>= 1.0.0)
- RcppArmadillo

**Suggested:**
- testthat (>= 3.0.0) - for testing
- R.matlab - for cross-validation against MATLAB references
- akima - for interpolation
- rgl - for 3D visualization

## Quick Start

### Kernel Density Estimation (KDE)

```r
library(fastlpr)

# 1D KDE
set.seed(42)
x <- matrix(rnorm(500), ncol = 1)
hlist <- get_hlist(20, c(0.05, 0.5), "logspace")
opt <- list(order = 0)  # order=0 for KDE
kde <- cv_fastkde(x, hlist, opt)

# Selected bandwidth
print(kde$h)

# 2D KDE
x2d <- matrix(rnorm(2000), ncol = 2)
hlist2d <- get_hlist(c(10, 10), rbind(c(0.1, 1), c(0.1, 1)), "logspace")
kde2d <- cv_fastkde(x2d, hlist2d, list(order = 0))
```

### Local Polynomial Regression

```r
# 1D Local Linear Regression
x <- matrix(runif(500), ncol = 1)
y <- sin(2 * pi * x) + 0.1 * matrix(rnorm(500), ncol = 1)
hlist <- get_hlist(20, c(0.01, 0.5), "logspace")

opt <- list(
  order = 1,       # 0=Nadaraya-Watson, 1=local linear, 2=local quadratic
  calc_dof = TRUE  # Enable DOF estimation for GCV
)
regs <- cv_fastlpr(x, y, hlist, opt)

# Results
print(regs$gcv_yhat$h)  # Selected bandwidth
yhat <- regs$yhat       # Fitted values
```

### Heteroscedastic Variance Estimation

```r
# Mean estimation
opt_mean <- list(order = 1, calc_dof = TRUE)
res_mean <- cv_fastlpr(x, y, hlist, opt_mean)

# Variance estimation using squared residuals
residuals_sq <- (y - res_mean$yhat)^2
opt_var <- list(order = 1, calc_dof = TRUE, y_type_out = "variance")
res_var <- cv_fastlpr(x, residuals_sq, hlist, opt_var)
```

## Package Structure

```
fastLPR_R/
├── R/                          # Source files (26 files)
│   ├── cv_fastlpr.R           # Main regression API
│   ├── cv_fastkde.R           # Main KDE API
│   ├── fastlpr_*.R            # Core algorithm components
│   ├── nufft.R                # NUFFT implementation
│   ├── lwp_estimator.R        # Local polynomial estimator
│   └── utils.R                # Utility functions
├── src/                        # C++ acceleration (Rcpp/RcppArmadillo)
├── inst/
│   ├── examples/              # JSS paper figure reproduction scripts
│   │   ├── example_kde.R
│   │   ├── example_boundary.R
│   │   ├── example_complex.R
│   │   ├── example_hetero.R
│   │   ├── example_qeeg.R
│   │   └── reproduce_all_figures.R
│   └── benchmarks/            # Performance comparison scripts
│       ├── benchmark_tab4_kde_comparison.R
│       ├── benchmark_kde_large_scale.R
│       └── benchmark_lpr_npregfast.R
├── tests/
│   └── testthat/              # Test suite
│       ├── test-cross-validation.R  # 10-test MATLAB cross-validation
│       └── test-*.R                 # Component tests
├── man/                        # Documentation (roxygen2)
├── data/                       # Example datasets
├── dev/                        # Development files (excluded from build)
├── DESCRIPTION                 # Package metadata
├── NAMESPACE                   # Exported functions
└── LICENSE                     # GPL-3
```

## Examples (JSS Paper Figures)

Reproduce all figures from the JSS paper:

```r
# Run all figure scripts
source(system.file("examples", "reproduce_all_figures.R", package = "fastlpr"))

# Or run individual figures
source(system.file("examples", "example_kde.R", package = "fastlpr"))
```

| Figure | Description | Script |
|--------|-------------|--------|
| Fig 2 | KDE (1D/2D/3D) | `example_kde.R` |
| Fig 3 | Boundary comparison (NW/LL/LQ) | `example_boundary.R` |
| Fig 4 | Complex-valued regression | `example_complex.R` |
| Fig 5 | Heteroscedastic variance | `example_hetero.R` |
| Fig 6 | Applications (qEEG/MRI) | `example_qeeg.R` |

## Benchmarks

Performance comparison with existing R packages:

```r
# Run benchmark scripts
source(system.file("benchmarks", "benchmark_tab4_kde_comparison.R", package = "fastlpr"))
```

**KDE: fastlpr vs ks package (N=50,000)**
- fastlpr: 0.34s
- ks: 19.98s
- **Speedup: 59x**

**LPR: fastlpr vs statsmodels (Python, N=5,000)**
- Naive O(N²): 31.7s
- fastlpr O(N log N): 0.03s
- **Speedup: 1000x+**

## Testing

Run the test suite:

```r
# Using testthat
setwd("fastLPR_R")
testthat::test_dir("tests/testthat")

# Or via devtools
devtools::test()
```

**Cross-validation results (10 tests against MATLAB reference):**

| Test | N | dx | order | Type | MATLAB | R Time | Ratio | GCV MaxErr | BW MaxErr | Mean MaxErr | Var MaxErr | Status |
|------|---|----|-------|------|--------|--------|-------|------------|-----------|-------------|------------|--------|
| 1D KDE | 1400 | 1 | 0 | mean | 0.341s | 0.081s | 0.24x | 1.94e+01* | 0.00e+00 | - | - | **PASS** |
| 2D KDE | 2000 | 2 | 0 | mean | 0.125s | 0.062s | 0.49x | 1.84e-01* | 0.00e+00 | - | - | **PASS** |
| 3D KDE | 2000 | 3 | 0 | mean | 0.145s | 0.869s | 6.00x | 3.32e-03* | 0.00e+00 | - | - | **PASS** |
| 1D Order 1 | 500 | 1 | 1 | mean | 1.546s | 0.060s | 0.04x | 5.86e-04 | 0.00e+00 | 7.31e-03 | - | **PASS** |
| 2D Order 1 | 1200 | 2 | 1 | mean | 0.406s | 0.873s | 2.15x | 9.64e-04 | 1.33e-02 | 1.90e-02 | - | **PASS** |
| 1D Order 2 | 500 | 1 | 2 | mean | 0.185s | 0.116s | 0.63x | 1.46e-03 | 0.00e+00 | 7.77e-03 | - | **PASS** |
| 2D Order 2 | 1200 | 2 | 2 | mean | 1.343s | 5.109s | 3.81x | 1.45e-03 | 0.00e+00 | 8.30e-03 | - | **PASS** |
| Complex | 10000 | 1 | 1 | mean | 0.334s | 7.444s | 22.29x | 6.89e-05 | 0.00e+00 | 3.45e-04 | - | **PASS** |
| Hetero 1D | 10000 | 1 | 1 | mean+var | 0.466s | 1.652s | 3.54x | 2.33e-05 | 0.00e+00 | 2.11e-03 | 1.64e-03 | **PASS** |
| Hetero 2D | 1200 | 2 | 1 | mean+var | 1.167s | 3.916s | 3.35x | 2.89e-02 | 0.00e+00 | 4.29e-02 | 3.02e-02 | **PASS** |

\* For KDE tests (order=0), LCV MaxErr shown in GCV column; All metrics must pass

## API Reference

### Main Functions

| Function | Description |
|----------|-------------|
| `cv_fastlpr(x, y, hlist, opt)` | Cross-validated local polynomial regression |
| `cv_fastkde(x, hlist, opt)` | Cross-validated kernel density estimation |
| `get_hlist(n, range, type)` | Generate bandwidth grid |
| `fastlpr_predict(regs, x_new)` | Predict at new points |
| `fastlpr_interval(regs, alpha)` | Confidence intervals |

### Options

```r
opt <- list(
  order = 1,                # Polynomial order: 0, 1, or 2
  kernel_type = "gaussian", # Kernel: "gaussian" or "epanechnikov"
  N = 100,                  # Grid size per dimension
  accuracy = 6,             # NUFFT accuracy (1-12)
  calc_dof = TRUE,          # Compute DOF for GCV
  dstd = 0                  # Std devs for 1-SE rule (0 = minimum GCV)
)
```

## Citation

If you use fastlpr in your research, please cite:

```bibtex
@article{fastlpr2025,
  title = {fastLPR: Fast Local Polynomial Regression with NUFFT Acceleration},
  author = {Wang, Ying and Li, Min},
  year = {2025},
  note = {R package version 1.0.1},
  url = {https://github.com/rigelfalcon/fastLPR}
}
```

## License

GPL-3

## Related Packages

- [fastLPR (MATLAB)](https://github.com/rigelfalcon/fastLPR) - Reference implementation
- [fastlpr (Python)](https://github.com/rigelfalcon/fastLPR) - Python port
