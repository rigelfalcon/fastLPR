
# fastLPR Standard R Environment
=============================

This setup provides a standard R environment for fastLPR that works with:
- Standard R installations (D:\Software\R\R-4.5.1)
- RStudio IDE integration
- No Pixi or conda dependencies
- CRAN package ecosystem

## Files Created

### Demo Scripts
- `demo_basic.R` - Basic regression demo
- `demo_kde.R` - Kernel density estimation
- `demo_cross_validation.R` - Cross-validation with GCV
- `run_all_demos.R` - Run all demos sequentially

### Result Files
- CSV files with numerical results
- Plots generated automatically

## Quick Start

### 1. In R/RStudio:
```r
# Navigate to working directory
setwd("path/to/fastLPR_R")

# Run setup script
source("../verification/setup_fastLPR_pure.R")

# Run basic demo
source("demo_basic.R")
```

### 2. From Command Line:
```bash
cd path/to/fastLPR_R
Rscript run_all_demos.R
```

### 3. RStudio Workflow:
- Open `demo_basic.R` in RStudio
- Edit as needed
- Run from Source menu
- Use RStudio debugging tools

## Requirements

- R 4.0+ installed at standard location
- Standard R packages (Matrix, akima, interp)
- Optional: ggplot2, dplyr, tidyr for enhanced features

## Troubleshooting

### R not found:
```r
# Install R from: https://cran.r-project.org/
# Set path to: D:\Software\R\R-4.5.1\bin
# Verify with: R --version
```

### Package installation issues:
```r
# Try alternative CRAN mirror:
install.packages("Matrix", repos = "https://cloud.r-project.org")
```

### Working directory issues:
```r
# Change to writable directory:
setwd("C:/Users/YourUsername/Documents")
```

## Features

- ✅ RStudio compatible
- ✅ Standard R packages only
- ✅ No specialized tools required
- ✅ Easy to customize
- ✅ Ready for production use
- ✅ Well-documented code

