# fastlpr (development)

## Numerical Stability & Cross-Language Consistency

### Fixed

* Regularize all diagonal elements of the design matrix (was only order+1, missing S33 for 2D order 1)
* ihbad threshold: use `< eps*10` instead of `== 0` to detect underflow bandwidths (matches Python)
* Clamp Inf/NaN grid values to 0 before interpolation (prevents NaN propagation from extreme bandwidths)
* Remove artificial `invalid_bandwidth` NaN assignment in GCV — let GCV naturally penalize bad bandwidths

## Performance Optimizations

### Changed

* Rcpp convolution pipeline: fused broadcast multiply + IFFT + extraction into single C++ call (rcpp_conv_nd_full) with OpenMP parallelism
* Single-precision FFTW3 path for accuracy <= 4, reducing FFT time by ~40%
* Eliminated redundant as.complex() copies in Rcpp wrapper guards
* Eliminated 1.6GB broadcast array copy in DoF mfun computation via as.vector() recycling
* Pre-computed NUFFT of y and ones vector, reused across polynomial terms
* Overall: R within 1.5x of MATLAB speed (down from 8x)

# fastlpr 1.0.1

## CRAN Resubmission (2026-04-13)

### Fixed

* CRAN reviewer feedback compliance fixes

# fastlpr 1.0.0

## Initial CRAN Release (2026-04-03)

### Features

* Fast local polynomial regression via NUFFT with O(N + M log M) complexity
* Kernel density estimation (KDE) for 1D, 2D, and 3D data
* Local polynomial regression with orders 0 (Nadaraya-Watson), 1 (local linear), and 2 (local quadratic)
* Complex-valued response support
* Heteroscedastic variance estimation
* Automatic bandwidth selection via GCV (regression) and LCV (density estimation)
* 1-SE rule for conservative bandwidth selection
* Confidence interval computation
* OpenMP parallelization via Rcpp/RcppArmadillo (optional)

### Main Functions

* cv_fastlpr() - Cross-validated local polynomial regression
* cv_fastkde() - Cross-validated kernel density estimation
* get_hlist() - Generate bandwidth grid
* fastlpr_predict() - Prediction at new data points
* fastlpr_interval() - Confidence interval computation

### Notes

* R port of the MATLAB/Python fastLPR toolbox
* Verified against MATLAB reference implementation (MSE < 1e-8)
