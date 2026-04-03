# fastlpr 1.0.0

## Initial CRAN Release

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
* Performance: within 8x of MATLAB speed for all test cases
