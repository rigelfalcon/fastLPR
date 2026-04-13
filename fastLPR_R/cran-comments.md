# CRAN Submission Comments

## Resubmission

This is a resubmission addressing feedback from Konstanze Lauseker:

- Expanded NUFFT acronym in DESCRIPTION
- Added single quotes around software names in DESCRIPTION
- Added method references with DOIs in DESCRIPTION
- Replaced cat() with message()/warning() in R/fastlpr_kdf.R and R/fastlpr_y.R
- Removed hardcoded set.seed(42) from R/fastlpr_dof.R and R/fastlpr_y.R;
  users can now pass seed via the opt parameter

## Test environments

- Windows 11 x64 (build 26200), R 4.5.1

## R CMD check results

0 errors | 0 warnings | 1 note

- NOTE: New submission

## Downstream dependencies

None.
