%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% REPRODUCE_ALL_FIGURES - Master script to reproduce all paper figures
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script runs all figure generation scripts for the fastLPR paper
% and saves the results to fastLPR/fig/reproduced/ directory.
%
% Figures generated:
%   - Figure 2: Kernel Density Estimation (1D and 2D)
%   - Figure 3: Boundary Comparison (NW vs LL vs LQ)
%   - Figure 4: Complex-Valued Regression (log(z))
%   - Figure 5: Heteroscedastic Regression (1D and 2D)
%
% Note: Figure 1 is a conceptual diagram and not code-generated.
%
% The script follows JSS publication standards with:
%   - Fixed random seeds for reproducibility
%   - Consistent styling across all figures
%   - 300 DPI resolution for publication
%   - Self-contained code
%
% Copyright (c) 2024 fastLPR Development Team
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
close all;

% Add fastLPR to path (go up one directory and run setup)
current_dir = pwd;
cd('..');
fastlpr_setup;
cd(current_dir);

fprintf('\n');
fprintf('================================================================================\n');
fprintf('REPRODUCE ALL PAPER FIGURES\n');
fprintf('================================================================================\n');
fprintf('This script runs all figure generation scripts for the fastLPR paper.\n');
fprintf('Each figure will be saved to fastLPR/fig/reproduced/ directory.\n');
fprintf('================================================================================\n\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure 2: Kernel Density Estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('================================================================================\n');
fprintf('Figure 2: Kernel Density Estimation (1D and 2D)\n');
fprintf('================================================================================\n\n');

example_kde;
close all;
clear
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure 3: Boundary Comparison
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('================================================================================\n');
fprintf('Figure 3: Boundary Comparison (NW vs LL vs LQ)\n');
fprintf('================================================================================\n\n');

example_boundary;
close all;
clear
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure 4: Complex-Valued Regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('================================================================================\n');
fprintf('Figure 4: Complex-Valued Regression (log(z))\n');
fprintf('================================================================================\n\n');

example_complex;
close all;
clear
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure 5: Heteroscedastic Regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('================================================================================\n');
fprintf('Figure 5: Heteroscedastic Regression (1D and 2D)\n');
fprintf('================================================================================\n\n');

example_hetero;
close all;
clear
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure 6: Real-World Applications
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('================================================================================\n');
fprintf('Figure 6: Real-World Applications (qEEG and MRI)\n');
fprintf('================================================================================\n\n');

example_qeeg;
close all;
clear
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Summary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n');
fprintf('================================================================================\n');
fprintf('ALL FIGURES GENERATED!\n');
fprintf('================================================================================\n\n');

% Output directory
figDir = fullfile(fileparts(mfilename('fullpath')), '..', 'fig', 'reproduced');
fprintf('All figures saved to: %s\n\n', figDir);

fprintf('Files generated:\n');
fprintf('  - fig2_fastkde_matlab.png (300 DPI)\n');
fprintf('  - fig2_fastkde_matlab.fig\n');
fprintf('  - fig3_boundary_comparison_matlab.png (300 DPI)\n');
fprintf('  - fig3_boundary_comparison_matlab.fig\n');
fprintf('  - fig4_complex_matlab.png (300 DPI)\n');
fprintf('  - fig4_complex_matlab.fig\n');
fprintf('  - fig5_heteroscedasticity_matlab.png (300 DPI)\n');
fprintf('  - fig5_heteroscedasticity_matlab.fig\n');
fprintf('  - fig6_applications_matlab.png (300 DPI)\n');
fprintf('  - fig6_applications_matlab.fig\n');
fprintf('\n');

fprintf('Notes:\n');
fprintf('  - All figures use fixed random seeds for reproducibility\n');
fprintf('  - PNG files are saved at 300 DPI for publication quality\n');
fprintf('  - FIG files can be opened in MATLAB for further editing\n');
fprintf('  - Figure 1 is a conceptual diagram and not code-generated\n');
fprintf('\n');

fprintf('================================================================================\n\n');

