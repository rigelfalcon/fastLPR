# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-05

### Added

- Fast local polynomial regression via NUFFT with O(N + M log M) complexity
- Kernel density estimation (KDE) for 1D, 2D, and 3D data
- Local polynomial regression with orders 0 (Nadaraya-Watson), 1 (local linear), and 2 (local quadratic)
- Complex-valued response support
- Heteroscedastic variance estimation
- Automatic bandwidth selection via GCV (regression) and LCV (density estimation)
- 1-SE rule for conservative bandwidth selection
- Confidence interval computation
- Numba JIT compilation support (optional)

### Main Functions

- cv_fastlpr() - Cross-validated local polynomial regression
- cv_fastkde() - Cross-validated kernel density estimation  
- get_hlist() - Generate bandwidth grid
- fastlpr_predict() - Prediction at new data points
- fastlpr_interval() - Confidence interval computation
- fastlpr_plot() - Plotting utilities

### Notes

- Python port of the MATLAB fastLPR toolbox
- Verified against MATLAB reference implementation (MSE < 1e-8)
- Performance: within 8x of MATLAB speed for all test cases
