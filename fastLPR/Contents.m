% fastLPR - Fast Local Polynomial Regression via NUFFT
% Version 1.0.0  21-Nov-2025
%
% Core Functions (Regression)
%   cv_fastlpr              - Fast local polynomial regression with GCV bandwidth selection
%   fastlpr_predict         - Predict at new points using fitted model
%   fastlpr_interval        - Compute pointwise intervals (CI or PI)
%   fastlpr_plot            - Plot regression results
%   fastlpr_plot_interval   - Plot interval bands
%
% Core Functions (Density Estimation)
%   cv_fastkde              - Fast kernel density estimation with LCV bandwidth selection
%   fastkde_plot            - Plot KDE results
%   fastkde_plot_bandwidth  - Visualize bandwidth selection
%
% Utilities
%   get_hlist               - Generate bandwidth candidate list
%   get_ndgrid              - Create multi-dimensional grids
%   save_open_figures       - Save all open figures
%
% Setup
%   fastlpr_setup           - Add toolbox to MATLAB path
%
% Examples (Reproduce JSS Paper Figures)
%   example/example_kde.m              - 1D kernel density estimation
%   example/example_boundary.m  - Boundary effect comparison
%   example/example_complex.m              - Complex-valued regression
%   example/example_hetero.m   - Heteroscedastic regression
%   example/example_qeeg.m         - Real-world qEEG application
%   example/reproduce_all_figures.m             - Generate all JSS figures
%
% Tests
%   tests/run_all.m                             - Run all tests (12 tests, 100% pass)
%   tests/test_fastlpr_vs_naive_nw.m            - Validate against naive NW (O(N²) baseline)
%
% See also: MATLAB Signal Processing Toolbox, Statistics and Machine Learning Toolbox
%
% Copyright (c) 2025 Ying Wang, Min Li, Deirel Paz-Linares, Pedro A. Valdes-Sosa
% Licensed under GPL-3.0
%
% References:
% [1] Wang, Y., Li, M., Paz-Linares, D., & Valdes-Sosa, P. A. (2025).
%     fastLPR: Fast Local Polynomial Regression via NUFFT in MATLAB,
%     Python, and R. Submitted.
