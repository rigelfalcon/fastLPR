%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code to generate the qEEG figure (fig_qeeg) for the fastLPR paper.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% qEEG cross-spectral normative modeling (Manuscript Section 4).
%   - Data: data_qeeg_cross_only.csv (N = 66505, complex-valued response)
%   - Native complex-valued local polynomial regression (order = 1)
%   - GCV-based bandwidth selection with the 1-SE rule, effective DoF tracking
%   - Prediction and pointwise confidence bands on a dense grid
%
% Five-panel figure:
%   (a) Raw data scatter on (age, frequency), colored by |y|
%   (b) GCV bandwidth selection surface over the (h1, h2) grid, 1-SE marker
%   (c) Fitted real-part surface Re(m_hat)
%   (d) Fitted imaginary-part surface Im(m_hat)
%   (e) 95% confidence band at the f = 10 Hz slice (real top, imaginary bottom)
%
% Self-contained (no external dependencies except fastLPR).
%
% Copyright (c) 2024-2025 Ying Wang, Min Li
% SPDX-License-Identifier: GPL-3.0-or-later
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear all; close all;

% Add fastLPR utility functions to path
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'utility'));

fprintf('\n');
fprintf('================================================================================\n');
fprintf('qEEG Cross-Spectral Normative Modeling\n');
fprintf('================================================================================\n\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load and explore the data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Loading data...\n');
dataFile = fullfile(fileparts(mfilename('fullpath')), '..', 'data', ...
    'data_qeeg_cross_only.csv');
if ~isfile(dataFile)
    error('qEEG data file not found: %s', dataFile);
end
qeeg = readtable(dataFile);
% readtable may read the complex column either as text (older MATLAB) or as a
% native complex double (newer MATLAB). Handle both robustly.
x = [qeeg.age, qeeg.freq];
yraw = qeeg.riemlogm10_1;
if iscell(yraw)
    y = cellfun(@str2num, yraw);          % text -> native complex
elseif istable(yraw) || isstring(yraw)
    y = arrayfun(@(s) str2num(char(s)), yraw);
else
    y = yraw;                              % already numeric (complex) double
end
y = y(:);
fprintf('  - Observations: %d\n', size(x, 1));
fprintf('  - Real part range: [%.3f, %.3f]\n', min(real(y)), max(real(y)));
fprintf('  - Imaginary part range: [%.3f, %.3f]\n', min(imag(y)), max(imag(y)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bandwidth selection and model fitting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nFitting complex-valued LPR (order = 1, GCV bandwidth selection)...\n');
hlist = get_hlist([9, 9], [1e-3, 2; 0.05, 2], @logspace);
opt = struct('order', 1, 'calc_dof', true, 'dstd', 1, 'seed', 42, ...
    'verbose', false);

t0 = tic;
result = cv_fastlpr(x, y, hlist, opt);
elapsed = toc(t0);

h1se = result.gcv_yhat.h1se(:).';
hmin = result.gcv_yhat.hmin(:).';
dof = result.gcv_yhat.df_m(result.gcv_yhat.idmin(1));  % DoF at GCV minimum
fprintf('  - Selected bandwidth (1-SE): [%.4f, %.4f]\n', h1se(1), h1se(2));
fprintf('  - Selected bandwidth (min):  [%.4f, %.4f]\n', hmin(1), hmin(2));
fprintf('  - Effective DoF: %.1f\n', dof);
fprintf('  - Computation time: %.1f seconds\n', elapsed);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Prediction and confidence bands on a dense grid
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nPredicting on 100 x 100 evaluation grid...\n');
n_grid = 100;
age_grid  = linspace(min(x(:,1)), max(x(:,1)), n_grid);
freq_grid = linspace(min(x(:,2)), max(x(:,2)), n_grid);
[Age, Freq] = ndgrid(age_grid, freq_grid);
x_eval = [Age(:), Freq(:)];
pred = fastlpr_predict(result, x_eval);
re_mat = reshape(real(pred), n_grid, n_grid);
im_mat = reshape(imag(pred), n_grid, n_grid);

% Pointwise standard error via the local-polynomial expression used for the
% confidence bands: se^2 = sigma^2 * nu / (|H| * s_0), evaluated at each point.
% (See Manuscript Section 4.)
resid = y - fastlpr_predict(result, x);
sig2 = mean(abs(resid).^2);
nu = 0.079577471546;          % Gaussian kernel, d = 2, order = 1
prod_h = prod(h1se);
s0_eval = max(real(result.fpp_s0(Age(:), Freq(:))), 1e-10);
se_eval = sqrt(sig2 .* nu ./ (prod_h .* s0_eval));
se_mat = reshape(se_eval, n_grid, n_grid);
zval = norminv(0.975);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Build the 5-panel figure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nCreating figure...\n');
fig = figure('Position', [50, 50, 1800, 1000], 'Color', 'w');
tl = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel (a): raw scatter colored by |y|
nexttile(1);
absy = abs(y);
rng(0);
sub = randperm(size(x, 1), min(20000, size(x, 1)));
scatter(x(sub, 1), x(sub, 2), 3, absy(sub), 'filled');
colormap(gca, parula); colorbar;
xlabel('log_{10}(age)'); ylabel('Frequency (Hz)');
title('(a) Raw data, colored by |y|', 'FontWeight', 'bold');

% Panel (b): GCV bandwidth selection surface
nexttile(2);
h1_u = unique(hlist(:, 1));
h2_u = unique(hlist(:, 2));
gcv_grid = nan(numel(h1_u), numel(h2_u));
gcv_m = result.gcv_yhat.gcv_m(:);
for k = 1:size(hlist, 1)
    [~, i1] = min(abs(h1_u - hlist(k, 1)));
    [~, i2] = min(abs(h2_u - hlist(k, 2)));
    gcv_grid(i1, i2) = gcv_m(k);
end
imagesc(log10(h1_u), log10(h2_u), gcv_grid.');
set(gca, 'YDir', 'normal'); colormap(gca, parula); colorbar; hold on;
plot(log10(hmin(1)), log10(hmin(2)), 'o', 'MarkerFaceColor', 'b', ...
    'MarkerEdgeColor', 'b', 'MarkerSize', 10, 'DisplayName', 'GCV min');
plot(log10(h1se(1)), log10(h1se(2)), 'p', 'MarkerFaceColor', 'r', ...
    'MarkerEdgeColor', 'r', 'MarkerSize', 16, 'DisplayName', '1-SE');
xlabel('log_{10}(h_1)'); ylabel('log_{10}(h_2)');
title('(b) GCV bandwidth surface', 'FontWeight', 'bold');
legend('Location', 'northeast'); hold off;

% Panel (c): fitted real-part surface
nexttile(4);
contourf(Age, Freq, re_mat, 30, 'LineColor', 'none'); hold on;
contour(Age, Freq, re_mat, 10, 'k', 'LineWidth', 0.4);
colormap(gca, parula); colorbar;
xlabel('log_{10}(age)'); ylabel('Frequency (Hz)');
title('(c) Fitted real part Re(m)', 'FontWeight', 'bold'); hold off;

% Panel (d): fitted imaginary-part surface
nexttile(5);
contourf(Age, Freq, im_mat, 30, 'LineColor', 'none'); hold on;
contour(Age, Freq, im_mat, 10, 'k', 'LineWidth', 0.4);
colormap(gca, parula); colorbar;
xlabel('log_{10}(age)'); ylabel('Frequency (Hz)');
title('(d) Fitted imag part Im(m)', 'FontWeight', 'bold'); hold off;

% Panel (e): 95% CI band at f = 10 Hz slice (real top, imag bottom)
[~, jf] = min(abs(freq_grid - 10));
ag = age_grid(:);
re_slice = re_mat(:, jf);
im_slice = im_mat(:, jf);
se_slice = se_mat(:, jf);

nexttile(3);
fill([ag; flipud(ag)], ...
     [re_slice + zval * se_slice; flipud(re_slice - zval * se_slice)], ...
     [0.2, 0.2, 0.8], 'FaceAlpha', 0.25, 'EdgeColor', 'none'); hold on;
plot(ag, re_slice, 'k-', 'LineWidth', 2);
xlabel('log_{10}(age)'); ylabel('Re(m)');
title('(e) 95% CI at f = 10 Hz (real)', 'FontWeight', 'bold'); hold off;

nexttile(6);
fill([ag; flipud(ag)], ...
     [im_slice + zval * se_slice; flipud(im_slice - zval * se_slice)], ...
     [0.8, 0.2, 0.2], 'FaceAlpha', 0.25, 'EdgeColor', 'none'); hold on;
plot(ag, im_slice, 'k-', 'LineWidth', 2);
xlabel('log_{10}(age)'); ylabel('Im(m)');
title('95% CI at f = 10 Hz (imag)', 'FontWeight', 'bold'); hold off;

title(tl, 'qEEG Cross-Spectral Normative Modeling (fastLPR)', ...
    'FontWeight', 'bold', 'FontSize', 16);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save figure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nSaving figure...\n');
figDir = fullfile(fileparts(mfilename('fullpath')), '..', 'fig', 'reproduced');
if ~exist(figDir, 'dir')
    mkdir(figDir);
end
pngPath = fullfile(figDir, 'fig_qeeg.png');
exportgraphics(fig, pngPath, 'Resolution', 300);
fprintf('  - Saved PNG: %s\n', pngPath);
figPath = fullfile(figDir, 'fig_qeeg.fig');
savefig(fig, figPath);
fprintf('  - Saved FIG: %s\n', figPath);

fprintf('\nExample completed successfully!\n');
