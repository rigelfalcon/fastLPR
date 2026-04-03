%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code to generate Figure 5: Heteroscedastic Regression (1D and 2D)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script reproduces Figure 5 from the fastLPR paper, demonstrating:
%   - Panel (a): 1D mean estimation with confidence and prediction intervals
%   - Panel (b): 1D variance estimation (log-scale)
%   - Panel (c): 2D mean estimation with confidence and prediction intervals
%   - Panel (d): 2D variance estimation (log-scale)
%
% The figure follows JSS publication standards with:
%   - Fixed random seed for reproducibility
%   - Consistent styling (fonts, colors, sizes)
%   - 300 DPI resolution for publication
%   - Self-contained code (no external dependencies except fastLPR)
%
% Copyright (c) 2024 fastLPR Development Team
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear all; close all;

% Add fastLPR utility functions to path
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));
fastlpr_setup();

fprintf('\n');
fprintf('================================================================================\n');
fprintf('Figure 5: Heteroscedastic Regression (1D and 2D)\n');
fprintf('================================================================================\n\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel (a) and (b): 1D Heteroscedastic Regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Generating 1D heteroscedastic data...\n');

% Set random seed for reproducibility
rng(42);

% Generate 1D data
n1d = 10000;
x1d = 2*2*(rand(n1d, 1) - 0.5);  % Uniform in [-2, 2]

% Define ground truth functions
fun_mu_1d = @(x) x.^3;  % Cubic mean function
fun_sigma_1d = @(x) 1 + 4*exp(-(x.^2));  % Variance depends on x

% Generate data
y1d = fun_mu_1d(x1d);
s1d = fun_sigma_1d(x1d) * 0.5 * var(y1d);
yr1d = y1d + randn(size(y1d)) .* sqrt(s1d);

fprintf('  - Generated %d 1D samples\n', n1d);

% Estimate conditional mean
fprintf('  - Estimating 1D mean...\n');
opt1d.order = 1;
opt1d.dstd = 0;
opt1d.verbose = false;
hlist1d = get_hlist([20], [0.01, 1], @logspace);
[regs_mu_1d] = cv_fastlpr(x1d, yr1d, hlist1d, opt1d);
yp1d = regs_mu_1d.yhat;

% Estimate conditional variance
fprintf('  - Estimating 1D variance...\n');
sr1d = (yr1d - yp1d).^2;
opt1d.y_type_out = 'variance';
opt1d.dstd = 1;
[regs_sigma_1d] = cv_fastlpr(x1d, sr1d, hlist1d, opt1d);

% Compute confidence and prediction intervals
ci1d = fastlpr_interval(regs_mu_1d, regs_sigma_1d, 0.05, 'confidence');
pi1d = fastlpr_interval(regs_mu_1d, regs_sigma_1d, 0.05, 'prediction');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel (c) and (d): 2D Heteroscedastic Regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nGenerating 2D heteroscedastic data...\n');

% Set random seed
rng(44);

% Generate 2D data
n2d = 1200;
x2d = 2*2*(rand(n2d, 2) - 0.5);  % Uniform in [-2, 2] x [-2, 2]

% Define ground truth functions
fun_mu_2d = @(x1, x2) x1.^3 + x2.^3;
fun_sigma_2d = @(x1, x2) 1 + 4*exp(-(x1.^2 + x2.^2));

% Generate data
y2d = fun_mu_2d(x2d(:,1), x2d(:,2));
s2d = fun_sigma_2d(x2d(:,1), x2d(:,2)) * 1 * std(y2d);
yr2d = y2d + sqrt(s2d) .* randn(size(y2d));

fprintf('  - Generated %d 2D samples\n', n2d);

% Estimate conditional mean
fprintf('  - Estimating 2D mean...\n');
opt2d.order = 1;
opt2d.dstd = 0;
opt2d.verbose = false;
hlist2d = get_hlist([10, 10], [0.01, 1; 0.01, 1], @logspace);
[regs_mu_2d] = cv_fastlpr(x2d, yr2d, hlist2d, opt2d);
yp2d = regs_mu_2d.yhat;

% Estimate conditional variance
fprintf('  - Estimating 2D variance...\n');
sr2d = (yr2d - yp2d).^2;
opt2d.y_type_out = 'variance';
opt2d.dstd = 1;
[regs_sigma_2d] = cv_fastlpr(x2d, sr2d, hlist2d, opt2d);

% Compute confidence and prediction intervals
ci2d = fastlpr_interval(regs_mu_2d, regs_sigma_2d, 0.05, 'confidence');
pi2d = fastlpr_interval(regs_mu_2d, regs_sigma_2d, 0.05, 'prediction');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create Figure with 2x2 Layout
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nCreating figure...\n');

% Create figure with publication-quality size
fig = figure('Position', [100, 100, 1400, 1000], 'Color', 'w');

% Set default font to Arial (sans-serif) for JSS style
set(groot, 'defaultAxesFontName', 'Arial');
set(groot, 'defaultTextFontName', 'Arial');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel a): 1D Mean with Confidence and Prediction Intervals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Use tighter subplot spacing
subplot('Position', [0.08, 0.55, 0.40, 0.38]);
hold on; box on;

% Plot data (DARK blue for y+ε, use different marker for y to make it beautiful)
plot(x1d, yr1d, '.', 'Color', [0, 0, 0.8], 'MarkerSize', 4, 'DisplayName', '$y + \epsilon$');
plot(x1d, y1d, 'o', 'Color', [0, 0.7, 0], 'MarkerSize', 4, 'LineWidth', 0.8, 'DisplayName', '$y$');

% Plot prediction interval (light blue with transparency)
% Extract PI bounds and plot manually (fastlpr_plot_interval uses fixed green color)
pi1d_grid = pi1d.GridVectors{1};
pi1d_upper = pi1d.Values(:,1);
pi1d_lower = pi1d.Values(:,2);
fill([pi1d_grid; flipud(pi1d_grid)], [pi1d_upper; flipud(pi1d_lower)], ...
    [0.5, 0.7, 1], 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
% Plot interval (green with transparency)
fastlpr_plot_interval([], ci1d);
% Create small visible patches for legend (outside visible range)
x_patch = [-10, -10, -9.9, -9.9];
y_patch = [0, 1, 1, 0];
h_pi_legend = patch(x_patch, y_patch, [0.5, 0.7, 1], 'FaceAlpha', 0.15, 'EdgeAlpha', 0, 'DisplayName', 'PI');
h_ci_legend = patch(x_patch, y_patch, 'g', 'FaceAlpha', 0.2, 'EdgeAlpha', 0, 'DisplayName', 'CI');

% Plot estimated mean (yellow)
fastlpr_plot(regs_mu_1d.fpp_yhat, [], [], [], 'LineWidth', 2.5, 'Color', [1, 0.8, 0], 'DisplayName', '$\hat{m}$');

% Plot variance distribution (magenta, shifted down to floor)
x_grid = linspace(-2, 2, 200)';
sigma_dist = fun_sigma_1d(x_grid) * 0.5 * var(y1d);
y_offset = -28;
pdf_scale = 10;
% Bell opens upward (high variance = peak) - more intuitive visualization
plot(x_grid, y_offset + pdf_scale * sigma_dist / max(sigma_dist), 'm-', 'LineWidth', 2, 'DisplayName', '$\sigma^2$');

% Labels and styling
xlim([-2.2, 2.2]);
ylim([-30, 20]);
set(gca, 'XTick', -2:1:2, 'YTick', -30:10:20, 'FontSize', 12);
% Add title with panel label
title('(a) 1D Mean with CI and PI', 'FontSize', 16, 'FontWeight', 'bold');
% Create legend with LaTeX interpreter (remove extra "data1" entries by using HandleVisibility)
leg = legend('Location', 'southeast', 'FontSize', 11);
set(leg, 'Interpreter', 'latex');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel b): 1D Variance Estimation (Log-scale)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Use tighter subplot spacing
subplot('Position', [0.56, 0.55, 0.40, 0.38]);
hold on; box on;

% Compute log-transformed values
log_sigma_true = 2*log(sqrt(s1d));
log_residuals = 2*log(abs(yr1d - yp1d));

% Plot scatter points (use different markers for better visibility)
plot(x1d, log_sigma_true, 's', 'Color', 'm', 'MarkerSize', 3, 'LineWidth', 0.5, 'DisplayName', '$2\log(\sigma)$');
plot(x1d, log_residuals, '.', 'Color', [0, 0.7, 0], 'MarkerSize', 4, 'DisplayName', '$2\log(|y-\hat{y}|)$');

% Plot estimated log-variance (grey line, not black)
% Create evaluation grid and compute log-transformed variance
x_eval = linspace(-2.2, 2.2, 200)';
sp1d_eval = regs_sigma_1d.fpp_yhat(x_eval);
log_sigma_est = 2*log(sqrt(sp1d_eval));
plot(x_eval, log_sigma_est, '-', 'Color', [0.5, 0.5, 0.5], 'LineWidth', 2.5, 'DisplayName', '$2\log(\sqrt{\hat{\Sigma}})$');

% Labels and styling
xlim([-2.2, 2.2]);
ylim([-20, 10]);
set(gca, 'XTick', -2:1:2, 'YTick', -20:10:10, 'FontSize', 12);
% Add title with panel label
title('(b) 1D Variance Estimation (Log-scale)', 'FontSize', 16, 'FontWeight', 'bold');
% Create legend with LaTeX interpreter
leg = legend('Location', 'southwest', 'FontSize', 11);
set(leg, 'Interpreter', 'latex');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel c): 2D Mean with Confidence and Prediction Intervals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Use tighter subplot spacing
subplot('Position', [0.08, 0.08, 0.40, 0.38]);
hold on; box on;

% Plot data (BLUE dots for y+ε, use different marker for y to make it beautiful)
scatter3(x2d(:,1), x2d(:,2), yr2d, 15, 'b', 'filled', 'MarkerFaceAlpha', 0.3, 'DisplayName', '$y+\epsilon$');

% Plot true data (use circle marker for y - more beautiful!)
scatter3(x2d(:,1), x2d(:,2), y2d, 20, 'r', 'o', 'MarkerFaceAlpha', 0.5, 'LineWidth', 0.8, 'DisplayName', '$y$');

% Plot prediction interval surfaces (light blue)
% Extract PI bounds and plot manually for custom color
pi2d_gv = pi2d.GridVectors;
[X1_pi, X2_pi] = ndgrid(pi2d_gv{1}, pi2d_gv{2});
pi2d_upper = pi2d.Values(:,:,1);
pi2d_lower = pi2d.Values(:,:,2);
surf(X1_pi, X2_pi, pi2d_upper, 'FaceColor', [0.5, 0.7, 1], 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
surf(X1_pi, X2_pi, pi2d_lower, 'FaceColor', [0.5, 0.7, 1], 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
% Plot interval surfaces (green)
fastlpr_plot_interval([], ci2d);
% Create small visible patches for legend (outside visible range)
x_patch = [-10, -10, -9.9];
y_patch = [-10, -9.9, -9.9];
z_patch = [0, 0, 0];
h_pi_legend = patch(x_patch, y_patch, z_patch, [0.5, 0.7, 1], 'FaceAlpha', 0.15, 'EdgeAlpha', 0, 'DisplayName', 'PI');
h_ci_legend = patch(x_patch, y_patch, z_patch, 'g', 'FaceAlpha', 0.4, 'EdgeAlpha', 0, 'DisplayName', 'CI');

% Plot estimated mean surface with black grid lines and semi-transparency
fastlpr_plot(regs_mu_2d.fpp_yhat, [], [], [], 'DisplayName', '$\hat{m}$', 'EdgeColor', 'k', 'FaceAlpha', 0.8);

% Shift variance for visualization (on the floor) - use different marker
s2d_low = s2d - min(s2d) - 30;
scatter3(x2d(:,1), x2d(:,2), s2d_low, 15, 'm', 's', 'LineWidth', 0.5, 'DisplayName', '$\sigma^2$');

% Labels and styling
% Improved viewing angle for better 3D structure visibility
view(-37, 20);

colormap(jet);
colorbar;
set(gca, 'FontSize', 14);
xlim([-2, 2]); ylim([-2, 2]);
set(gca, 'XTick', -2:2:2, 'YTick', -2:2:2);
% Add title with panel label
title('(c) 2D Mean with CI and PI', 'FontSize', 16, 'FontWeight', 'bold');
% Create legend with LaTeX interpreter
leg = legend('Location', 'southwest', 'FontSize', 10);
set(leg, 'Interpreter', 'latex');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel d): 2D Variance Estimation (Log-scale)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Use tighter subplot spacing
subplot('Position', [0.56, 0.08, 0.40, 0.38]);
hold on; box on;

% Compute log-transformed values
log_sigma_true_2d = 2*log(sqrt(s2d));
log_residuals_2d = 2*log(abs(yr2d - yp2d));

% Plot scatter points (use different markers for better visibility)
scatter3(x2d(:,1), x2d(:,2), log_sigma_true_2d, 15, 'm', 's', 'LineWidth', 0.5, 'DisplayName', '$2\log(\sigma)$');
scatter3(x2d(:,1), x2d(:,2), log_residuals_2d, 10, [0, 0.7, 0], '.', 'DisplayName', '$2\log(|y-\hat{y}|)$');

% Plot estimated log-variance surface with BLACK grid lines
% Use fastlpr_plot with transformation function
opt_plot.yfun = @(y) 2*log(sqrt(y));
fastlpr_plot(regs_sigma_2d.fpp_yhat, [], [], opt_plot, 'EdgeColor', 'k', 'FaceAlpha', 0.8, 'DisplayName', '$2\log(\sqrt{\hat{\Sigma}})$');

% Labels and styling
% Improved viewing angle for better 3D structure visibility
view(-37, 20);

% CRITICAL: Panel d) should NOT have colorbar (unlike panel c))
colormap(parula);
% NO colorbar for panel d)
set(gca, 'FontSize', 14);
xlim([-2, 2]); ylim([-2, 2]);
set(gca, 'XTick', -2:2:2, 'YTick', -2:2:2);
% Add title with panel label
title('(d) 2D Variance Estimation (Log-scale)', 'FontSize', 16, 'FontWeight', 'bold');
% Create legend with LaTeX interpreter
leg = legend('Location', 'southwest', 'FontSize', 10);
set(leg, 'Interpreter', 'latex');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save Figure for Publication
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nSaving figure...\n');

% Create output directories if they don't exist
figDir = fullfile(fileparts(mfilename('fullpath')), '..', 'fig', 'reproduced');
docFigDir = fullfile(fileparts(mfilename('fullpath')), '..', 'doc', 'fig');
if ~exist(figDir, 'dir')
    mkdir(figDir);
end
if ~exist(docFigDir, 'dir')
    mkdir(docFigDir);
end

% Save full composite as PNG (300 DPI for publication)
pngPath = fullfile(figDir, 'fig5_heteroscedasticity_matlab.png');
exportgraphics(fig, pngPath, 'Resolution', 300);
fprintf('  - Saved full composite PNG: %s\n', pngPath);

% Save as MATLAB figure for editing
figPath = fullfile(figDir, 'fig5_heteroscedasticity_matlab.fig');
savefig(fig, figPath);
fprintf('  - Saved FIG: %s\n', figPath);

% Save individual panels for manuscript (doc/fig/)
fprintf('  - Saving individual panels to doc/fig/...\n');
% Panel names: (a) 1D Mean, (b) 1D Variance, (c) 2D Mean, (d) 2D Variance
panel_names = {'a_1d_mean', 'b_1d_variance', 'c_2d_mean', 'd_2d_variance'};
% Get all axes handles (exclude colorbar etc)
allaxes = findall(fig, 'type', 'axes');
plotaxes = allaxes(arrayfun(@(ax) ~strcmp(get(ax, 'Tag'), 'Colorbar'), allaxes));
% Sort by position (top-left first): sort by y descending, then x ascending
pos = cell2mat(arrayfun(@(ax) get(ax, 'Position'), plotaxes, 'UniformOutput', false));
[~, sortIdx] = sortrows([-pos(:,2), pos(:,1)]);
plotaxes = plotaxes(sortIdx);

for i = 1:min(4, length(plotaxes))
    panelPath = fullfile(docFigDir, sprintf('fig5_heteroscedastic_%s.png', panel_names{i}));
    exportgraphics(plotaxes(i), panelPath, 'Resolution', 300);
    fprintf('    Panel (%c): %s\n', 'a'+i-1, panelPath);
end

% Also save full composite to doc/fig/ for reference
compositePath = fullfile(docFigDir, 'fig5_heteroscedastic.png');
exportgraphics(fig, compositePath, 'Resolution', 300);
fprintf('  - Saved full composite to doc/fig/: %s\n', compositePath);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Summary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n');
fprintf('================================================================================\n');
fprintf('Figure 5 Generation Complete!\n');
fprintf('================================================================================\n\n');

fprintf('Summary:\n');
fprintf('  - 1D data: %d samples\n', n1d);
fprintf('  - 2D data: %d samples\n', n2d);
fprintf('  - Demonstrates heteroscedastic regression:\n');
fprintf('    * Panel (a): 1D mean with CI and PI\n');
fprintf('    * Panel (b): 1D variance estimation (log-scale)\n');
fprintf('    * Panel (c): 2D mean with CI and PI\n');
fprintf('    * Panel (d): 2D variance estimation (log-scale)\n');
fprintf('  - Figure saved to: %s\n', figDir);
fprintf('\n');

fprintf('Key observations:\n');
fprintf('  - Variance depends on predictor values (heteroscedastic)\n');
fprintf('  - CI (green) and PI (blue) adapt to local variance\n');
fprintf('  - Log-scale visualization helps identify variance patterns\n');
fprintf('\n');

