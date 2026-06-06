%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code to generate Figure 2: Kernel Density Estimation (1D and 2D Examples)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script reproduces Figure 2 from the fastLPR paper, demonstrating:
%   Row 1 - KDE Examples:
%   - Panel (a): 1D KDE with multiple bandwidths
%   - Panel (b): 2D KDE contour plot
%   - Panel (c): 3D KDE volume rendering
%   Row 2 - Bandwidth Selection:
%   - Panel (d): 1D bandwidth selection via LCV
%   - Panel (e): 2D bandwidth selection heatmap
%   - Panel (f): 3D bandwidth selection volume rendering
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
fprintf('Figure 2: Kernel Density Estimation (1D and 2D Examples)\n');
fprintf('================================================================================\n\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel (a) and (b): 1D Bimodal Distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Generating Panel (a) and (b): 1D KDE...\n');

% Load Old Faithful geyser data (R built-in 'faithful', n = 272).
% Columns: 1 = eruption duration (min), 2 = waiting time (min).
% Panel (a) uses eruption durations, a classic bimodal real dataset.
faithful = load(fullfile(fileparts(mfilename('fullpath')), 'faithful.txt'));
x = faithful(:, 1);                   % Eruption duration (min)

fprintf('  - Loaded %d Old Faithful eruption-duration samples\n', length(x));

% Create bandwidth list (log-spaced) for the eruption-duration scale (~1.6-5.1 min)
hlist = get_hlist(20, [0.05, 2], @logspace);
fprintf('  - Testing %d bandwidths\n', length(hlist));

% Compute KDE with automatic bandwidth selection via LCV
opt.verbose = false;
tic;
kde = cv_fastkde(x, hlist, opt);
elapsed = toc;

fprintf('  - Computation time: %.3f seconds\n', elapsed);
fprintf('  - Selected bandwidth: h = %.4f\n', kde.h);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel (c) and (d): 2D Density Estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nGenerating Panel (c) and (d): 2D KDE...\n');

% Old Faithful 2D: eruption duration vs waiting time (both columns).
x2d = faithful(:, [1, 2]);            % [eruptions (min), waiting (min)]

fprintf('  - Loaded %d Old Faithful 2D samples\n', size(x2d, 1));

% Create 2D bandwidth list (log-spaced grid) matched to each axis scale:
%   eruptions ~ [1.6, 5.1] min, waiting ~ [43, 96] min
hlist2d = get_hlist([20, 20], [0.05, 1.5; 1, 20], @logspace);
fprintf('  - Testing %d bandwidth combinations\n', size(hlist2d, 1));

% Compute 2D KDE with automatic bandwidth selection
opt2d.verbose = false;
opt2d.N = [51, 51];                   % Grid size
opt2d.xrange = [1, 6; 35, 100];       % Evaluation range (eruptions, waiting)
tic;
kde2d = cv_fastkde(x2d, hlist2d, opt2d);
elapsed2d = toc;

fprintf('  - Computation time: %.3f seconds\n', elapsed2d);
fprintf('  - Selected bandwidth: h = [%.4f, %.4f]\n', kde2d.h(1), kde2d.h(2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel (c) and (f): 3D Density Estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nGenerating Panel (c) and (f): 3D KDE...\n');

% Set random seed for reproducibility
rng(45);

% Generate 3D data with three clusters (similar to demo_3d_cv_fastkde.m)
n_cluster = 800;
x3d_1 = randn(n_cluster, 3) * 0.3 + repmat([-1, -1, -1], n_cluster, 1);  % Cluster 1
x3d_2 = randn(n_cluster, 3) * 0.4 + repmat([1, 1, 0], n_cluster, 1);     % Cluster 2
x3d_3 = randn(n_cluster, 3) * 0.25 + repmat([0, -1, 1], n_cluster, 1);   % Cluster 3
x3d = [x3d_1; x3d_2; x3d_3];

fprintf('  - Generated %d samples from 3D trimodal distribution\n', size(x3d, 1));

% Create 3D bandwidth list (log-spaced) - use anisotropic 10x10x10 grid for consistency with 2D
hlist3d = get_hlist([10, 10, 10], [0.2, 0.6; 0.2, 0.6; 0.2, 0.6], @logspace);
fprintf('  - Testing %d bandwidth combinations\n', size(hlist3d, 1));

% Compute 3D KDE with automatic bandwidth selection
opt3d.verbose = false;
opt3d.N = [31, 31, 31];               % Grid size for evaluation
opt3d.xrange = [-2.5, 2.5; -2.5, 2.5; -2.5, 2.5];  % Evaluation range
tic;
kde3d = cv_fastkde(x3d, hlist3d, opt3d);
elapsed3d = toc;

fprintf('  - Computation time: %.3f seconds\n', elapsed3d);
fprintf('  - Selected bandwidth: h = %.4f\n', kde3d.h);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create Figure with 2x3 Layout (Row 1: KDE, Row 2: Bandwidth Selection)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nCreating figure...\n');

% Create figure with publication-quality size (wider for 3 columns)
fig = figure('Position', [100, 100, 1800, 900], 'Color', 'w');

% Set default font to Arial (sans-serif) for JSS style
set(groot, 'defaultAxesFontName', 'Arial');
set(groot, 'defaultTextFontName', 'Arial');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ROW 1: KDE EXAMPLES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel (a): 1D KDE with Multiple Bandwidths
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(2, 3, 1);
hold on; box on; grid on;

% Plot histogram (light gray, semi-transparent)
histogram(x, 30, 'Normalization', 'pdf', ...
    'FaceColor', [0.7, 0.7, 0.7], 'EdgeColor', 'k', ...
    'FaceAlpha', 0.5, 'DisplayName', 'Histogram');

% Plot KDE curves for selected bandwidths
test_h_idx = [1, 5, 10, 15, 20];  % Show 5 different bandwidths
colors = lines(length(test_h_idx));  % Matlab default color palette

for i = 1:length(test_h_idx)
    idx = test_h_idx(i);
    opt_temp.verbose = false;
    kde_temp = cv_fastkde(x, hlist(idx), opt_temp);
    plot(kde_temp.xlist{1}, kde_temp.fhat, 'Color', colors(i,:), ...
         'LineWidth', 2, 'DisplayName', sprintf('h=%.3f', hlist(idx)));
end

% Plot selected bandwidth with thick red line
plot(kde.xlist{1}, kde.fhat, 'r-', 'LineWidth', 3, ...
     'DisplayName', sprintf('h=%.3f (selected)', kde.h));

xlabel('Eruption duration (min)', 'FontSize', 14);
ylabel('Density', 'FontSize', 14);
title('(a) 1D KDE with Bandwidth Comparison', 'FontSize', 16, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 11);
set(gca, 'FontSize', 12);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel (b): 2D KDE (Contour)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(2, 3, 2);
hold on; box on;

% Plot filled contours with jet colormap
contourf(kde2d.xlist{1}, kde2d.xlist{2}, kde2d.fhat', 20, 'LineStyle', 'none');
colormap(jet);
colorbar;

% Overlay data points (small black dots)
plot(x2d(:,1), x2d(:,2), 'k.', 'MarkerSize', 3);

xlabel('Eruption duration (min)', 'FontSize', 14);
ylabel('Waiting time (min)', 'FontSize', 14);
title('(b) 2D KDE Density Contours', 'FontSize', 16, 'FontWeight', 'bold');
set(gca, 'FontSize', 12);
axis tight;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel (c): 3D KDE (Volume Rendering)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(2, 3, 3);
hold on; box on;

% Evaluate KDE on 3D grid
[X3, Y3, Z3] = meshgrid(kde3d.xlist{1}, kde3d.xlist{2}, kde3d.xlist{3});
pdf_3d = reshape(kde3d.fhat, length(kde3d.xlist{1}), length(kde3d.xlist{2}), length(kde3d.xlist{3}));

% Use volume rendering with transparency based on density
% Filter to show only significant density regions
density_thresh = prctile(kde3d.fhat(kde3d.fhat > 0), 30);
idx_sig = kde3d.fhat > density_thresh;

% Create grid points for visualization
x_grid_3d = [X3(:), Y3(:), Z3(:)];
x_sig = x_grid_3d(idx_sig, :);
c_data = kde3d.fhat(idx_sig);

% Normalize for coloring and transparency
alpha_data = (c_data - min(c_data)) / (max(c_data) - min(c_data) + eps);
alpha_data = 0.05 + 0.7 * alpha_data;  % Map to [0.05, 0.75]

% Plot volume as scatter3 with density-based coloring
s = scatter3(x_sig(:,1), x_sig(:,2), x_sig(:,3), 30, c_data, 'filled');
s.AlphaData = alpha_data;
s.MarkerFaceAlpha = 'flat';

xlabel('x_1', 'FontSize', 14);
ylabel('x_2', 'FontSize', 14);
zlabel('x_3', 'FontSize', 14);
title('(c) 3D KDE Volume Rendering', 'FontSize', 16, 'FontWeight', 'bold');
set(gca, 'FontSize', 12);
view(45, 20);
grid on;
colormap(jet);
colorbar;
set(gca, 'Color', [0.95 0.95 0.95]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ROW 2: BANDWIDTH SELECTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel (d): 1D Bandwidth Selection (LCV)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(2, 3, 4);
hold on; box on; grid on;

if ~isempty(kde.lcv)
    % Plot LCV scores
    plot(log10(kde.lcv.hlist), kde.lcv.lcv_m, ...
         'k-o', 'LineWidth', 2, 'MarkerSize', 6, ...
         'MarkerFaceColor', 'w', 'DisplayName', 'LCV Score');

    % Mark selected bandwidth with red star (1-SE rule)
    plot(log10(kde.lcv.h1se), kde.lcv.lcv_m(kde.lcv.id1se), ...
         'r*', 'MarkerSize', 20, 'LineWidth', 2, 'DisplayName', 'Selected (1-SE)');

    xlabel('log_{10}(h)', 'FontSize', 14);
    ylabel('LCV Score', 'FontSize', 14);
    title('(d) 1D Bandwidth Selection via LCV', 'FontSize', 16, 'FontWeight', 'bold');
    legend('Location', 'southwest', 'FontSize', 11);
    set(gca, 'FontSize', 12);
else
    % Single bandwidth case
    text(0.5, 0.5, 'Single bandwidth (no selection needed)', ...
        'HorizontalAlignment', 'center', 'FontSize', 14);
    axis off;
    title('(d) 1D Bandwidth Selection (LCV)', 'FontSize', 16, 'FontWeight', 'bold');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel (e): 2D Bandwidth Selection (LCV Heatmap)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(2, 3, 5);
hold on; box on;

if ~isempty(kde2d.lcv)
    % Reshape LCV scores into 2D grid
    h1_unique = unique(kde2d.lcv.hlist(:,1));
    h2_unique = unique(kde2d.lcv.hlist(:,2));
    lcv_grid = reshape(kde2d.lcv.lcv_m, length(h1_unique), length(h2_unique));

    % Plot heatmap with jet colormap
    imagesc(log10(h1_unique), log10(h2_unique), lcv_grid');
    colormap(jet);
    colorbar;

    % Set axis limits to match data range (ensures heatmap fills the box)
    xlim([min(log10(h1_unique)), max(log10(h1_unique))]);
    ylim([min(log10(h2_unique)), max(log10(h2_unique))]);

    % Mark selected bandwidth with red star (1-SE rule)
    plot(log10(kde2d.lcv.h1se(1)), ...
         log10(kde2d.lcv.h1se(2)), ...
         'r*', 'MarkerSize', 20, 'LineWidth', 2);

    % Add annotation
    text(log10(kde2d.lcv.h1se(1)), ...
         log10(kde2d.lcv.h1se(2)) + 0.1, ...
         '1-SE', 'Color', 'w', 'FontSize', 12, ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');

    xlabel('log_{10}(h_1)', 'FontSize', 14);
    ylabel('log_{10}(h_2)', 'FontSize', 14);
    title('(e) 2D Bandwidth Selection via LCV', 'FontSize', 16, 'FontWeight', 'bold');
    set(gca, 'FontSize', 12, 'YDir', 'normal');
else
    % Single bandwidth case
    text(0.5, 0.5, 'Single bandwidth (no selection needed)', ...
        'HorizontalAlignment', 'center', 'FontSize', 14);
    axis off;
    title('(e) 2D Bandwidth Selection (Heatmap)', 'FontSize', 16, 'FontWeight', 'bold');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel (f): 3D Bandwidth Selection (Volume Rendering)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(2, 3, 6);
hold on; box on;

if ~isempty(kde3d.lcv)
    % Get LCV scores and bandwidth list
    lcv_scores = kde3d.lcv.lcv_m;
    h_list = kde3d.lcv.hlist;

    % For 3D anisotropic bandwidth, reshape LCV scores to 3D grid
    % h_list is [1000×3] for 10×10×10 grid
    n_h = round(size(h_list, 1)^(1/3));  % Should be 10
    LCV_3d = reshape(lcv_scores, n_h, n_h, n_h);

    % Get unique bandwidth values for each dimension
    h1_values = unique(h_list(:,1));
    h2_values = unique(h_list(:,2));
    h3_values = unique(h_list(:,3));

    % Convert to log10 scale for consistency with panels (d) and (e)
    log10_h1 = log10(h1_values);
    log10_h2 = log10(h2_values);
    log10_h3 = log10(h3_values);

    % Create 3D grid with log10 bandwidth values
    [H1, H2, H3] = meshgrid(log10_h1, log10_h2, log10_h3);

    % Flatten for scatter3
    h1_flat = H1(:);
    h2_flat = H2(:);
    h3_flat = H3(:);
    lcv_flat = LCV_3d(:);

    % Normalize LCV for coloring and transparency
    lcv_norm = (lcv_flat - min(lcv_flat)) / (max(lcv_flat) - min(lcv_flat) + eps);
    alpha_data = 0.05 + 0.6 * lcv_norm;  % Higher LCV = more visible

    % Plot volume as scatter3 with size variation
    marker_size = 10 + 30 * lcv_norm;  % Larger markers for higher LCV
    s = scatter3(h1_flat, h2_flat, h3_flat, marker_size, lcv_flat, 'filled');
    s.AlphaData = alpha_data;
    s.MarkerFaceAlpha = 'flat';

    % Mark selected bandwidth with large red star
    h_selected = kde3d.lcv.h1se;
    plot3(log10(h_selected(1)), log10(h_selected(2)), log10(h_selected(3)), ...
        'r*', 'MarkerSize', 25, 'LineWidth', 4);

    % Set axis labels (linear scale showing log10 values, consistent with panels d and e)
    xlabel('log_{10}(h_1)', 'FontSize', 14);
    ylabel('log_{10}(h_2)', 'FontSize', 14);
    zlabel('log_{10}(h_3)', 'FontSize', 14);
    title('(f) 3D Bandwidth Selection via LCV', 'FontSize', 16, 'FontWeight', 'bold');
    set(gca, 'FontSize', 12);
    view(45, 30);
    grid on;
    colormap(gca, jet);
    cb = colorbar;
    ylabel(cb, 'LCV Score', 'FontSize', 12);
    set(gca, 'Color', [0.95 0.95 0.95]);
    axis tight;
else
    % Single bandwidth case
    text(0.5, 0.5, 'Single bandwidth (no selection needed)', ...
        'HorizontalAlignment', 'center', 'FontSize', 14);
    axis off;
    title('(f) 3D Bandwidth Selection', 'FontSize', 16, 'FontWeight', 'bold');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Add Main Title
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sgtitle('Kernel Density Estimation with Automatic Bandwidth Selection (1-SE Rule)', ...
    'FontSize', 18, 'FontWeight', 'bold');

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
pngPath = fullfile(figDir, 'fig2_fastkde_matlab.png');
exportgraphics(fig, pngPath, 'Resolution', 300);
fprintf('  - Saved full composite PNG: %s\n', pngPath);

% Save as MATLAB figure for editing
figPath = fullfile(figDir, 'fig2_fastkde_matlab.fig');
savefig(fig, figPath);
fprintf('  - Saved FIG: %s\n', figPath);

% % Save individual panels for manuscript (doc/fig/)
% fprintf('  - Saving individual panels to doc/fig/...\n');
% panel_labels = {'a', 'b', 'c', 'd', 'e', 'f'};
% for i = 1:6
%     % Get subplot axes handle
%     ax = subplot(2, 3, i);
% 
%     % Export individual panel
%     panelPath = fullfile(docFigDir, sprintf('fig2_panel_%s.png', panel_labels{i}));
%     exportgraphics(ax, panelPath, 'Resolution', 300);
%     fprintf('    Panel (%s): %s\n', panel_labels{i}, panelPath);
% end

% Also save the full composite to doc/fig/ for reference
compositePath = fullfile(docFigDir, 'fig2_kde1d.png');
exportgraphics(fig, compositePath, 'Resolution', 300);
fprintf('  - Saved full composite to doc/fig/: %s\n', compositePath);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Summary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n');
fprintf('================================================================================\n');
fprintf('Figure 2 Generation Complete!\n');
fprintf('================================================================================\n\n');

fprintf('Summary:\n');
fprintf('  - 1D KDE: %d samples, selected h = %.4f\n', length(x), kde.h);
fprintf('  - 2D KDE: %d samples, selected h = [%.4f, %.4f]\n', ...
    size(x2d,1), kde2d.h(1), kde2d.h(2));
fprintf('  - Figure saved to: %s\n', figDir);
fprintf('\n');

