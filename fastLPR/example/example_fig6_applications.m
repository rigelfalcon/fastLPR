%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code to generate Figure 6: Real-World Applications (qEEG and MRI)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script reproduces Figure 6 from the fastLPR paper, demonstrating:
%   - Panel (a): 2D qEEG log-spectrum regression (age × frequency)
%   - Panel (b): 3D MRI T1 brain image regression (spatial coordinates)
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

% Add utility path
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'utility'));

fprintf('\n');
fprintf('================================================================================\n');
fprintf('Figure 6: Real-World Applications (qEEG and MRI)\n');
fprintf('================================================================================\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel (a): 2D qEEG Log-Spectrum Regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nPanel (a): 2D qEEG Log-Spectrum Regression...\n');

% Load qEEG data
dataFile = fullfile(fileparts(mfilename('fullpath')),'/../data' , 'data_qeeg.csv');
if ~isfile(dataFile)
    error(['qEEG data file not found: ', dataFile, '\n', ...
           'Run data/extract_qeeg_subset.m to generate it.']);
end

data_table = readtable(dataFile);
fprintf('  - Loaded %d qEEG samples\n', height(data_table));

% Prepare data: X = [age (log10), frequency], y = log-spectrum
X_qeeg = [data_table.age, data_table.freq];
y_qeeg = real(data_table.log10_10);

% Normalize predictors(no need)
% X_mean = mean(X_qeeg);
% X_std = std(X_qeeg);
% X_qeeg_norm = (X_qeeg - X_mean) ./ X_std;

fprintf('  - Age range: [%.1f, %.1f] years\n', 10^min(X_qeeg(:,1)), 10^max(X_qeeg(:,1)));
fprintf('  - Frequency range: [%.2f, %.2f] Hz\n', min(X_qeeg(:,2)), max(X_qeeg(:,2)));

% Regression with bandwidth selection
opt_qeeg.order = 0;  % Local linear
opt_qeeg.dstd = 10;
opt_qeeg.kernel_type = 'gaussian';

hlist_qeeg = get_hlist([5, 5], [0.1, 1; 0.1, 2], @logspace);
fprintf('  - Performing regression with %d bandwidth combinations...\n', size(hlist_qeeg, 1));

tic;
regs_qeeg = cv_fastlpr(X_qeeg, y_qeeg, hlist_qeeg, opt_qeeg);
time_qeeg = toc;

fprintf('  - Computation time: %.2f seconds\n', time_qeeg);
fprintf('  - Selected bandwidth: h = [%.4f, %.4f]\n', ...
    regs_qeeg.gcv_yhat.h1se(1), regs_qeeg.gcv_yhat.h1se(2));

% Create grid for visualization
n_grid = 50;
age_grid = linspace(min(X_qeeg(:,1)), max(X_qeeg(:,1)), n_grid);
freq_grid = linspace(min(X_qeeg(:,2)), max(X_qeeg(:,2)), n_grid);
[Age_grid, Freq_grid] = ndgrid(age_grid, freq_grid);

% Get predictions
Y_pred_qeeg = regs_qeeg.fpp_yhat(Age_grid, Freq_grid);

% % Convert back to original scale
% Age_grid = Age_grid * X_std(1) + X_mean(1);
% Freq_grid = Freq_grid * X_std(2) + X_mean(2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel (b): 3D MRI T1 Brain Image Regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nPanel (b): 3D MRI T1 Brain Image Regression...\n');

% Load MRI T1 data
mriFile = fullfile(fileparts(mfilename('fullpath')), '..', 'data', ...
    'subjectimage_T1.mat');
if ~isfile(mriFile)
    error(['MRI data file not found: ', mriFile]);
end

load(mriFile, 'Cube');
fprintf('  - Loaded MRI T1 data: %s\n', mat2str(size(Cube)));

% Create scattered data from non-zero voxels
[I, J, K] = ind2sub(size(Cube), find(Cube > 0));
X_mri_full = [I, J, K];
y_mri_full = single(Cube(Cube > 0));

% Subsample to reasonable size for computational efficiency
rng(42);
n_samples = min([100000, length(y_mri_full)]);

idx_sample = randperm(length(y_mri_full), n_samples);
X_mri = X_mri_full(idx_sample, :);
y_mri = y_mri_full(idx_sample);

fprintf('  - Total non-zero voxels: %d\n', length(y_mri_full));
fprintf('  - Using %d scattered samples for regression\n', n_samples);
fprintf('  - Spatial range: [%d, %d] × [%d, %d] × [%d, %d]\n', ...
    min(X_mri(:,1)), max(X_mri(:,1)), ...
    min(X_mri(:,2)), max(X_mri(:,2)), ...
    min(X_mri(:,3)), max(X_mri(:,3)));

% Normalize predictors
% X_mri_mean = mean(X_mri);
% X_mri_std = std(X_mri);
% X_mri_norm = (X_mri - X_mri_mean) ./ X_mri_std;

% Regression with bandwidth selection (use smaller grid for 3D)
opt_mri.order = 0;  % Local constant (order 2 not supported for 3D)
opt_mri.dstd = 1;
opt_mri.kernel_type = 'gaussian';
opt_mri.N = [64, 64, 64];  % Grid size for FFT

% Use fixed bandwidth for faster computation
hlist_mri = [0.03, 0.03, 0.03];
fprintf('  - Performing regression with bandwidth h = [%.3f, %.3f, %.3f]...\n', ...
    hlist_mri(1), hlist_mri(2), hlist_mri(3));

tic;
regs_mri = cv_fastlpr(X_mri, y_mri, hlist_mri, opt_mri);
time_mri = toc;

fprintf('  - Computation time: %.2f seconds\n', time_mri);
fprintf('  - Selected bandwidth: h = [%.4f, %.4f, %.4f]\n', ...
    regs_mri.gcv_yhat.h1se(1), regs_mri.gcv_yhat.h1se(2), regs_mri.gcv_yhat.h1se(3));

% Create grid for visualization (higher resolution for better quality)
grid_res = 50;  % Increased from 25 for better slice panel resolution
x_range = [min(X_mri(:,1)), max(X_mri(:,1))];
y_range = [min(X_mri(:,2)), max(X_mri(:,2))];
z_range = [min(X_mri(:,3)), max(X_mri(:,3))];

[X_grid, Y_grid, Z_grid] = meshgrid(...
    linspace(x_range(1), x_range(2), grid_res), ...
    linspace(y_range(1), y_range(2), grid_res), ...
    linspace(z_range(1), z_range(2), grid_res));

x_grid = [X_grid(:), Y_grid(:), Z_grid(:)];

% Get predictions
y_pred_mri = regs_mri.fpp_yhat(x_grid);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create Figure with 2 Panels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nCreating figure...\n');

fig = figure('Position', [100, 100, 1400, 600], 'Color', 'w');

% Set default font
set(groot, 'defaultAxesFontName', 'Arial');
set(groot, 'defaultTextFontName', 'Arial');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel (a): qEEG 2D Surface
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot('Position', [0.08, 0.15, 0.38, 0.75]);
surf(Age_grid, Freq_grid, Y_pred_qeeg, 'EdgeColor', 'none', 'FaceAlpha', 0.9);
hold on;
% Add heatmap at the bottom
z_bottom = min(Y_pred_qeeg(:)) - 0.5;
surf(Age_grid, Freq_grid, ones(size(Y_pred_qeeg))*z_bottom, Y_pred_qeeg, ...
    'EdgeColor', 'none', 'FaceAlpha', 0.8);
hold off;
xlabel('Age (log_{10} years)', 'FontSize', 14);
ylabel('Frequency (Hz)', 'FontSize', 14);
zlabel('Log-spectrum', 'FontSize', 14);
title('(a) 2D qEEG Regression', 'FontSize', 16, 'FontWeight', 'bold');
colormap(gca, 'parula');
cb1 = colorbar;
ylabel(cb1, 'Log-spectrum', 'FontSize', 12);
set(gca, 'FontSize', 12);
view(-37.5, 30);
grid on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel (b): MRI 3D Volume (Scatter3 with Transparency + 3-Axis Slicing)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot('Position', [0.56, 0.15, 0.38, 0.75]);
hold on;

% Reshape predictions back to 3D grid
y_pred_mri_3d = reshape(y_pred_mri, grid_res, grid_res, grid_res);

% Normalize intensity for coloring and transparency
y_norm = (y_pred_mri - min(y_pred_mri)) / (max(y_pred_mri) - min(y_pred_mri) + eps);

% Apply sigmoid transformation for better contrast (beta=5)
sigmoid = @(x, beta) 1 ./ (1 + exp(-beta * (x - 0.5)));

% Set low values to transparent (threshold at 0.1)
y_norm(y_norm<0.1)=0;

% Compute alpha and marker size based on transformed values
alpha_data= 1./(1+exp(-y_norm));
marker_size = 5 + 10 * y_norm;

% Plot 3D scatter with variable size and transparency
s = scatter3(x_grid(:,1), x_grid(:,2), x_grid(:,3), marker_size, y_pred_mri, 'filled');
s.AlphaData = alpha_data;
s.MarkerFaceAlpha = 'flat';

hold off;

xlabel('x (normalized)', 'FontSize', 14);
ylabel('y (normalized)', 'FontSize', 14);
zlabel('z (normalized)', 'FontSize', 14);
title('(b) 3D MRI T1 Regression', 'FontSize', 16, 'FontWeight', 'bold');
grid on;
box on;
view(124, 22);
axis equal;
colormap(gca, jet);
cb2 = colorbar;
ylabel(cb2, 'Intensity', 'FontSize', 12);
set(gca, 'FontSize', 12);
set(gca, 'Color', [0.95 0.95 0.95]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save Figure
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
pngPath = fullfile(figDir, 'fig6_applications_matlab.png');
exportgraphics(fig, pngPath, 'Resolution', 300);
fprintf('  - Saved full composite PNG: %s\n', pngPath);

% Save as MATLAB figure
figPath = fullfile(figDir, 'fig6_applications_matlab.fig');
savefig(fig, figPath);
fprintf('  - Saved FIG: %s\n', figPath);

% Save individual panels for manuscript (doc/fig/)
fprintf('  - Saving individual panels to doc/fig/...\n');
% Panel names: (a) qEEG 2D Regression, (b) MRI 3D Regression
panel_names = {'a_qeeg', 'b_mri_3d'};
% Get all axes handles (exclude colorbar etc)
allaxes = findall(fig, 'type', 'axes');
plotaxes = allaxes(arrayfun(@(ax) ~strcmp(get(ax, 'Tag'), 'Colorbar'), allaxes));
% Sort by position (left to right): sort by x ascending
pos = cell2mat(arrayfun(@(ax) get(ax, 'Position'), plotaxes, 'UniformOutput', false));
[~, sortIdx] = sortrows([pos(:,1)]);
plotaxes = plotaxes(sortIdx);

for i = 1:min(2, length(plotaxes))
    panelPath = fullfile(docFigDir, sprintf('fig6_%s.png', panel_names{i}));
    exportgraphics(plotaxes(i), panelPath, 'Resolution', 300);
    fprintf('    Panel (%c): %s\n', 'a'+i-1, panelPath);
end

% Also save full composite to doc/fig/ for reference
compositePath = fullfile(docFigDir, 'fig6_qeeg.png');
exportgraphics(fig, compositePath, 'Resolution', 300);
fprintf('  - Saved full composite to doc/fig/: %s\n', compositePath);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Summary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n');
fprintf('================================================================================\n');
fprintf('Figure 6 Generation Complete!\n');
fprintf('================================================================================\n\n');

fprintf('Summary:\n');
fprintf('  Panel (a) - qEEG 2D Regression:\n');
fprintf('    - Samples: %d\n', size(X_qeeg, 1));
fprintf('    - Computation time: %.2f sec\n', time_qeeg);
fprintf('    - Selected bandwidth: h = [%.4f, %.4f]\n', ...
    regs_qeeg.gcv_yhat.h1se(1), regs_qeeg.gcv_yhat.h1se(2));
fprintf('  Panel (b) - MRI 3D Regression:\n');
fprintf('    - Samples: %d\n', size(X_mri, 1));
fprintf('    - Computation time: %.2f sec\n', time_mri);
fprintf('    - Selected bandwidth: h = [%.4f, %.4f, %.4f]\n', ...
    regs_mri.gcv_yhat.h1se(1), regs_mri.gcv_yhat.h1se(2), regs_mri.gcv_yhat.h1se(3));
fprintf('  Figure saved to: %s\n', figDir);

fprintf('\n');

