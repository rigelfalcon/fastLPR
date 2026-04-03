# CRAN Submission Comments

## Test environments

- Windows 10 x64 (build 19045), R 4.5.1

## R CMD check results

0 errors ✓ | 0 warnings ✓ | 1 note ℹ

**NOTE details:**

```
Undefined global functions or variables:
  dof_interpolate_batch fastkde_eval fastlpr_compute_kdf fastlpr_inufft
  interp_batch_1d interp_batch_2d interp_batch_3d sub2ind
```

These are internal helper functions called within the package scope, not missing imports from external packages. They are defined in the package source files and used for internal computations.

## Downstream dependencies

None (this is a first submission).

## Additional notes

- This is the R port of the fastLPR MATLAB/Python toolbox
- Companion package to JSS paper (under review)
- Uses Rcpp/RcppArmadillo compiled code for performance-critical operations
- Comprehensive test suite (~100 tests) available in tests/testthat/
- All examples wrapped in \donttest{} to avoid CHECK timeout
