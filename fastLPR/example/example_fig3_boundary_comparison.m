%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code to generate Figure 3: Boundary Comparison (NW vs LL vs LQ Regression)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script reproduces Figure 3 from the fastLPR paper, demonstrating:
%   - Comparison of three local polynomial regression methods:
%     * NW (Nadaraya-Watson, order 0 - local constant)
%     * LL (Local Linear, order 1)
%     * LQ (Local Quadratic, order 2)
%   - Comparison of boundary behavior on real data
%   - Motorcycle crash test data (time vs head acceleration)
%
% The figure follows JSS publication standards with:
%   - Real benchmark dataset (MASS::mcycle)
%   - Consistent styling (fonts, colors, sizes)
%   - 300 DPI resolution for publication
%   - Self-contained code (no external dependencies except fastLPR)
%
% Copyright (c) 2024 fastLPR Development Team
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear all; close all;

% Add fastLPR utility functions to path
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'utility'));

fprintf('\n');
fprintf('================================================================================\n');
fprintf('Figure 3: Boundary Comparison (NW vs LL vs LQ Regression)\n');
fprintf('================================================================================\n\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load Real Data (MASS::mcycle - motorcycle crash test)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Loading mcycle data...\n');

% Motorcycle crash test data (MASS::mcycle): a classic boundary-bias
% benchmark. Column 1 = time in milliseconds after impact, column 2 = head
% acceleration in g.
mcycle = load(fullfile(fileparts(mfilename('fullpath')), 'mcycle.txt'));
x = mcycle(:, 1);
y = mcycle(:, 2);
n = numel(y);

fprintf('  - Loaded %d samples\n', n);
fprintf('  - x range: [%.1f, %.1f]\n', min(x), max(x));
fprintf('  - Data: MASS::mcycle (time vs head acceleration)\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fit Three Regression Models
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nFitting regression models...\n');

% Create evaluation grid spanning the observed time range
x_grid = linspace(min(x), max(x), 500)';

% Use automatic bandwidth selection with appropriate range
% Generate bandwidth candidates (focus on larger range for smoother fits)
hlist = get_hlist(50, [0.01, 2.0], @logspace);
fprintf('  - Using automatic bandwidth selection (GCV)\n');
fprintf('  - Bandwidth range: [%.3f, %.3f] in normalized scale\n', min(hlist), max(hlist));

% Options for regression
opt.N = 500;  % Grid size
opt.verbose = false;

% Order 0: Nadaraya-Watson (local constant)
fprintf('  - Fitting NW (order 0) with auto bandwidth...\n');
opt.order = 0;
tic;
reg_nw = cv_fastlpr(x, y, hlist, opt);
t_nw = toc;
y_nw = reg_nw.fpp_yhat(x_grid);
h_nw = reg_nw.gcv_yhat.h1se;

% Order 1: Local Linear
fprintf('  - Fitting LL (order 1) with auto bandwidth...\n');
opt.order = 1;
tic;
reg_ll = cv_fastlpr(x, y, hlist, opt);
t_ll = toc;
y_ll = reg_ll.fpp_yhat(x_grid);
h_ll = reg_ll.gcv_yhat.h1se;

% Order 2: Local Quadratic
fprintf('  - Fitting LQ (order 2) with auto bandwidth...\n');
opt.order = 2;
tic;
reg_lq = cv_fastlpr(x, y, hlist, opt);
t_lq = toc;
y_lq = reg_lq.fpp_yhat(x_grid);
h_lq = reg_lq.gcv_yhat.h1se;

fprintf('  - Computation times: NW=%.3fs, LL=%.3fs, LQ=%.3fs\n', t_nw, t_ll, t_lq);
fprintf('  - Selected bandwidths: NW=%.3f, LL=%.3f, LQ=%.3f\n', h_nw, h_ll, h_lq);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create Main Figure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nCreating figure...\n');

% Create figure with publication-quality size
fig = figure('Position', [100, 100, 1200, 700], 'Color', 'w');

% Set default font to Arial (sans-serif) for JSS style
set(groot, 'defaultAxesFontName', 'Arial');
set(groot, 'defaultTextFontName', 'Arial');

% Main plot
ax_main = axes('Position', [0.08, 0.12, 0.88, 0.82]);
hold on; box on;

% Plot scattered data points (larger black dots) - now visible in legend
plot(x, y, 'k.', 'MarkerSize', 6, 'DisplayName', 'Noisy data');

% Plot regression curves with DISTINCT line styles AND colors for better visibility
% NW: Solid green line
plot(x_grid, y_nw, '-', 'Color', [0, 0.7, 0], 'LineWidth', 3, 'DisplayName', 'NW (order 0)');
% LL: Dashed red line
plot(x_grid, y_ll, '--', 'Color', [0.8, 0, 0], 'LineWidth', 3, 'DisplayName', 'LL (order 1)');
% LQ: Dash-dot blue line
plot(x_grid, y_lq, '-.', 'Color', [0, 0, 0.8], 'LineWidth', 3, 'DisplayName', 'LQ (order 2)');

% Set axis limits to span the mcycle data
xlim([min(x), max(x)]);
ylim([-150, 100]);
set(gca, 'FontSize', 14);

% Add axis labels
xlabel('Time (ms)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Acceleration (g)', 'FontSize', 16, 'FontWeight', 'bold');

% Add legend at top
legend('Location', 'north', 'FontSize', 14, 'Box', 'on', 'Color', 'w', 'Orientation', 'horizontal');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save Figure for Publication
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nSaving figure...\n');

% Create output directory if it doesn't exist
figDir = fullfile(fileparts(mfilename('fullpath')), '..', 'fig', 'reproduced');
if ~exist(figDir, 'dir')
    mkdir(figDir);
end

% Save as PNG (300 DPI for publication)
pngPath = fullfile(figDir, 'fig3_boundary_comparison_matlab.png');
exportgraphics(fig, pngPath, 'Resolution', 300);
fprintf('  - Saved PNG: %s\n', pngPath);

% Save as MATLAB figure for editing
figPath = fullfile(figDir, 'fig3_boundary_comparison_matlab.fig');
savefig(fig, figPath);
fprintf('  - Saved FIG: %s\n', figPath);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Summary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n');
fprintf('================================================================================\n');
fprintf('Figure 3 Generation Complete!\n');
fprintf('================================================================================\n\n');

fprintf('Summary:\n');
fprintf('  - Data: %d samples (MASS::mcycle motorcycle crash test)\n', n);
fprintf('  - Bandwidths (auto-selected): NW=%.3f, LL=%.3f, LQ=%.3f\n', h_nw, h_ll, h_lq);
fprintf('  - Methods compared: NW (order 0), LL (order 1), LQ (order 2)\n');
fprintf('  - Figure saved to: %s\n', figDir);
fprintf('\n');

fprintf('Key observations:\n');
fprintf('  - NW (green): Smoother but higher bias at boundaries\n');
fprintf('  - LL (red): Reduces boundary bias compared to NW\n');
fprintf('  - LQ (blue): Best fit in high curvature regions\n');
fprintf('\n');

