%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code to generate Figure 4: Complex-Valued Regression (log(z) function)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script reproduces Figure 4 from the fastLPR paper, demonstrating:
%   - Panel (a): Real part of log(z)
%   - Panel (b): Imaginary part of log(z)
%   - Panel (c): Magnitude |log(z)|
%   - Panel (d): Angle of log(z)
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
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'utility'));

fprintf('\n');
fprintf('================================================================================\n');
fprintf('Figure 4: Complex-Valued Regression (log(z) function)\n');
fprintf('================================================================================\n\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate Complex-Valued Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Generating complex-valued data...\n');

% Set random seed for reproducibility
rng(42);

% Grid resolution
n = 100;

% Create uniform grid in complex plane
% Real part: [0.1, 2], Imaginary part: [-2, 2]
x = get_ndgrid({linspace(0.1, 2, n), linspace(-2, 2, n)}, 'list');

% Convert to complex numbers: z = x1 + i*x2
x = x(:,1) + 1i*x(:,2);

% Compute complex logarithm: log(z)
y = log(x);

% Add complex-valued noise
yr = y + 0.1*std(real(y)) .* randn(size(y)) + ...
         0.1*1i*std(imag(y)) .* randn(size(y));

fprintf('  - Generated %d samples\n', length(x));
fprintf('  - Real(z) range: [%.2f, %.2f]\n', min(real(x)), max(real(x)));
fprintf('  - Imag(z) range: [%.2f, %.2f]\n', min(imag(x)), max(imag(x)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform Complex-Valued Regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nPerforming complex-valued regression...\n');

% Bandwidth selection
% For 2D: dh=[5, 5], hrange=[0.1, 1; 0.1, 1]
hlist = get_hlist([5, 5], [0.1, 1; 0.1, 1], @logspace);
fprintf('  - Testing %d bandwidth combinations\n', size(hlist, 1));
opt.order = 1;  % Local linear regression
opt.dstd = 3;   % Use 3 DOF samples
opt.verbose = false;

tic;
[regs] = cv_fastlpr(x, yr, hlist, opt);
elapsed = toc;

fprintf('  - Computation time: %.3f seconds\n', elapsed);
if ~isempty(regs.gcv_yhat)
    fprintf('  - Selected bandwidth: h = [%.4f, %.4f]\n', regs.gcv_yhat.h1se(1), regs.gcv_yhat.h1se(2));
    h_selected = regs.gcv_yhat.h1se;
else
    fprintf('  - Using single bandwidth\n');
    h_selected = hlist;
end

% Predict at original points for visualization
yhat = fastlpr_predict(regs, x);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create Figure with 2x2 Layout
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nCreating figure...\n');

% Create figure with publication-quality size
fig = figure('Position', [100, 100, 1400, 1000], 'Color', 'w');

% Set default font to Arial (sans-serif) for JSS style
set(groot, 'defaultAxesFontName', 'Arial');
set(groot, 'defaultTextFontName', 'Arial');

% Color for scatter plots (dark red from jet colormap)
cl = jet(10);
cl = cl(10, :);

% View angle for all subplots (elevation=30, azimuth=-60)
view_elev = 30;
view_azim = -60;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel (a): Real Part of log(z)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Use compact layout with balanced spacing
subplot('Position', [0.07, 0.56, 0.40, 0.38]);
hold on; box on;

% Plot noisy data as scatter
yfun = @real;
scatter3(real(x), imag(x), yfun(yr), 20, cl, '.', 'DisplayName', 'Noisy data');

% Plot fitted surface
opt_plot.yfun = yfun;
opt_plot.plotfun = @surf;
h_surf = fastlpr_plot(regs.fpp_yhat, [], [], opt_plot, 'EdgeColor', 'none', 'FaceAlpha', 0.85);
set(h_surf, 'DisplayName', 'Fitted surface');

% Labels and styling
xlabel('Real(z)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Imag(z)', 'FontSize', 16, 'FontWeight', 'bold');
zlabel('Real(log(z))', 'FontSize', 16, 'FontWeight', 'bold');
title('(a) Real(log(z))', 'FontSize', 18, 'FontWeight', 'bold');
view(view_azim, view_elev);
colormap(jet);
colorbar;
set(gca, 'FontSize', 14);
zlim([-4, 4]);
set(gca, 'ZTick', -4:4:4);
legend('Location', 'northeast', 'FontSize', 10);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel (b): Imaginary Part of log(z)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Use compact layout with balanced spacing
subplot('Position', [0.55, 0.56, 0.40, 0.38]);
hold on; box on;

% Plot noisy data as scatter
yfun = @imag;
scatter3(real(x), imag(x), yfun(yr), 20, cl, '.', 'DisplayName', 'Noisy data');

% Plot fitted surface
opt_plot.yfun = yfun;
opt_plot.plotfun = @surf;
h_surf = fastlpr_plot(regs.fpp_yhat, [], [], opt_plot, 'EdgeColor', 'none', 'FaceAlpha', 0.85);
set(h_surf, 'DisplayName', 'Fitted surface');

% Labels and styling
xlabel('Real(z)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Imag(z)', 'FontSize', 16, 'FontWeight', 'bold');
zlabel('Imag(log(z))', 'FontSize', 16, 'FontWeight', 'bold');
title('(b) Imag(log(z))', 'FontSize', 18, 'FontWeight', 'bold');
view(view_azim, view_elev);
colormap(jet);
colorbar;
set(gca, 'FontSize', 14);
zlim([-4, 4]);
set(gca, 'ZTick', -4:4:4);
legend('Location', 'northeast', 'FontSize', 10);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel (c): Magnitude |log(z)|
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Use compact layout with balanced spacing
subplot('Position', [0.07, 0.10, 0.40, 0.38]);
hold on; box on;

% Plot noisy data as scatter
yfun = @abs;
scatter3(real(x), imag(x), yfun(yr), 20, cl, '.', 'DisplayName', 'Noisy data');

% Plot fitted surface
opt_plot.yfun = yfun;
opt_plot.plotfun = @surf;
h_surf = fastlpr_plot(regs.fpp_yhat, [], [], opt_plot, 'EdgeColor', 'none', 'FaceAlpha', 0.85);
set(h_surf, 'DisplayName', 'Fitted surface');

% Labels and styling
xlabel('Real(z)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Imag(z)', 'FontSize', 16, 'FontWeight', 'bold');
zlabel('|log(z)|', 'FontSize', 16, 'FontWeight', 'bold');
title('(c) |log(z)|', 'FontSize', 18, 'FontWeight', 'bold');
view(view_azim, view_elev);
colormap(jet);
colorbar;
set(gca, 'FontSize', 14);
zlim([0, 4]);
set(gca, 'ZTick', 0:2:4);
legend('Location', 'northeast', 'FontSize', 10);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel (d): Angle of log(z)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Use compact layout with balanced spacing
subplot('Position', [0.55, 0.10, 0.40, 0.38]);
hold on; box on;

% Plot noisy data as scatter
yfun = @angle;
scatter3(real(x), imag(x), yfun(yr), 20, cl, '.', 'DisplayName', 'Noisy data');

% Plot fitted surface
opt_plot.yfun = yfun;
opt_plot.plotfun = @surf;
h_surf = fastlpr_plot(regs.fpp_yhat, [], [], opt_plot, 'EdgeColor', 'none', 'FaceAlpha', 0.85);
set(h_surf, 'DisplayName', 'Fitted surface');

% Labels and styling
xlabel('Real(z)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Imag(z)', 'FontSize', 16, 'FontWeight', 'bold');
zlabel('angle(log(z))', 'FontSize', 16, 'FontWeight', 'bold');
title('(d) angle(log(z))', 'FontSize', 18, 'FontWeight', 'bold');
view(view_azim, view_elev);
colormap(jet);
colorbar;
set(gca, 'FontSize', 14);
zlim([-pi, pi]);
set(gca, 'ZTick', -pi:pi/2:pi);
set(gca, 'ZTickLabel', {'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});
legend('Location', 'northeast', 'FontSize', 10);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Add Main Title (DISABLED - causes overlap with subplot titles)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Main title removed to avoid overlap with panel titles
% The figure description will be in the caption instead
% sgtitle('Complex-Valued Regression: log(z) Function', ...
%     'FontSize', 18, 'FontWeight', 'bold');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Add Gray Dotted Separator Lines (DISABLED per user request)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Note: Separator lines turned off to avoid visual clutter
% % Vertical separator (between columns)
% annotation('line', [0.5, 0.5], [0.08, 0.95], ...
%     'Color', [0.7, 0.7, 0.7], 'LineStyle', ':', 'LineWidth', 1.5);
%
% % Horizontal separator (between rows)
% annotation('line', [0.05, 0.95], [0.515, 0.515], ...
%     'Color', [0.7, 0.7, 0.7], 'LineStyle', ':', 'LineWidth', 1.5);

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
pngPath = fullfile(figDir, 'fig4_complex_matlab.png');
exportgraphics(fig, pngPath, 'Resolution', 300);
fprintf('  - Saved full composite PNG: %s\n', pngPath);

% Save as MATLAB figure for editing
figPath = fullfile(figDir, 'fig4_complex_matlab.fig');
savefig(fig, figPath);
fprintf('  - Saved FIG: %s\n', figPath);

% Save individual panels for manuscript (doc/fig/)
fprintf('  - Saving individual panels to doc/fig/...\n');
% Panel names: (a) Real, (b) Imag, (c) Magnitude, (d) Angle
panel_names = {'a_real', 'b_imag', 'c_magnitude', 'd_angle'};
% Get all axes handles (exclude colorbar etc)
allaxes = findall(fig, 'type', 'axes');
plotaxes = allaxes(arrayfun(@(ax) ~strcmp(get(ax, 'Tag'), 'Colorbar'), allaxes));
% Sort by position (top-left first): sort by y descending, then x ascending
pos = cell2mat(arrayfun(@(ax) get(ax, 'Position'), plotaxes, 'UniformOutput', false));
[~, sortIdx] = sortrows([-pos(:,2), pos(:,1)]);
plotaxes = plotaxes(sortIdx);

for i = 1:min(4, length(plotaxes))
    panelPath = fullfile(docFigDir, sprintf('fig4_complex_%s.png', panel_names{i}));
    exportgraphics(plotaxes(i), panelPath, 'Resolution', 300);
    fprintf('    Panel (%c): %s\n', 'a'+i-1, panelPath);
end

% Also save full composite to doc/fig/ for reference
compositePath = fullfile(docFigDir, 'fig4_complex.png');
exportgraphics(fig, compositePath, 'Resolution', 300);
fprintf('  - Saved full composite to doc/fig/: %s\n', compositePath);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Summary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n');
fprintf('================================================================================\n');
fprintf('Figure 4 Generation Complete!\n');
fprintf('================================================================================\n\n');

fprintf('Summary:\n');
fprintf('  - Complex data: %d samples\n', length(x));
fprintf('  - Function: log(z) where z = x1 + i*x2\n');
if ~isempty(regs.gcv_yhat)
    fprintf('  - Selected bandwidth: h = [%.4f, %.4f]\n', h_selected(1), h_selected(2));
else
    fprintf('  - Using single bandwidth\n');
end
fprintf('  - Four components visualized:\n');
fprintf('    * Panel (a): Real part\n');
fprintf('    * Panel (b): Imaginary part\n');
fprintf('    * Panel (c): Magnitude\n');
fprintf('    * Panel (d): Angle\n');
fprintf('  - Figure saved to: %s\n', figDir);
fprintf('\n');

fprintf('Key observations:\n');
fprintf('  - Real part shows log(|z|) behavior\n');
fprintf('  - Imaginary part shows arg(z) with branch cut\n');
fprintf('  - Magnitude shows |log(z)| (always non-negative)\n');
fprintf('  - Angle shows phase of log(z)\n');
fprintf('\n');

