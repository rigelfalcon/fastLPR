%% Generate Standardized Cross-Validation Reference Data for R/Python Comparison
% This script generates comprehensive reference data for cross-validation testing
% All output files follow a consistent structure for easy comparison
%
% Output format for each test case:
% - x: input data (n x dx)
% - y: response data (n x dy) [for regression only]
% - hlist: bandwidth list (dh x dx)
% - gcv_m or lcv_m: cross-validation scores (dh x 1)
% - h_selected: selected bandwidth (1 x dx)
% - yhat_mean: predicted mean (n x 1) [for regression]
% - elapsed: MATLAB computation time
% - mse: mean squared error values (dh x 1)
% - pdof_m: pseudo-degrees of freedom (dh x 1)
%
% Usage:
%   cd('path/to/jss-code/fastLPR');
%   run('tests/generate_refs.m');

clear; clc;

% Get the directory of this script and set up paths
script_dir = fileparts(mfilename('fullpath'));
fastlpr_root = fileparts(script_dir);  % fastLPR/
cd(fastlpr_root);
run('fastlpr_setup.m');

% Output directory for reference files (in same folder as this script)
out_dir = fullfile(script_dir, 'refs');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

fprintf('================================================================================\n');
fprintf('GENERATING STANDARDIZED CROSS-VALIDATION REFERENCE DATA\n');
fprintf('Output directory: %s\n', out_dir);
fprintf('================================================================================\n\n');

rng(42); % Fixed seed for reproducibility

%% Test Configuration - Sample sizes match JSS paper examples
% KDE sample sizes (from example_kde.m)
n_kde_1d = 1400;   % fig2: 1000+400 bimodal
n_kde_2d = 2000;   % fig2: 2x1000 clusters
n_kde_3d = 2000;   % fig2

% LPR sample sizes (from example_boundary, example_hetero)
n_lpr_1d = 500;    % fig3: boundary comparison
n_lpr_2d = 1200;   % fig5: heteroscedastic 2D

% Complex and heteroscedastic (from example_complex, example_hetero)
n_complex = 10000; % fig4: 100x100 grid
n_hetero = 10000;  % fig5: 1D heteroscedastic

n_bw = 20;         % Number of bandwidths

%% ============================================================================
%% TEST 1: 1D KDE (order=0)
%% ============================================================================
fprintf('>>> TEST 1: 1D KDE (order=0) <<<\n');

x = randn(n_kde_1d, 1);
hlist = logspace(log10(0.05), log10(1.0), n_bw)';
opt = struct('order', 0);

tic;
kde = cv_fastkde(x, hlist, opt);
elapsed = toc;

ref_1d_kde = struct();
ref_1d_kde.x = x;
ref_1d_kde.hlist = hlist;
ref_1d_kde.lcv_m = kde.lcv.lcv_m;
ref_1d_kde.h_selected = kde.h;
ref_1d_kde.id1se = kde.lcv.id1se;
ref_1d_kde.fhat_at_x = kde.fpp(x);  % Density estimates at data points
ref_1d_kde.elapsed = elapsed;

save(fullfile(out_dir, 'ref_1d_kde.mat'), '-struct', 'ref_1d_kde');
fprintf('  Saved: ref_1d_kde.mat (n=%d, elapsed: %.3f sec)\n\n', n_kde_1d, elapsed);

%% ============================================================================
%% TEST 2: 2D KDE (order=0)
%% ============================================================================
fprintf('>>> TEST 2: 2D KDE (order=0) <<<\n');

x = randn(n_kde_2d, 2);
hlist = zeros(n_bw, 2);
h_vals = logspace(log10(0.1), log10(1.0), n_bw);
hlist(:, 1) = h_vals';
hlist(:, 2) = h_vals';
opt = struct('order', 0);

tic;
kde = cv_fastkde(x, hlist, opt);
elapsed = toc;

ref_2d_kde = struct();
ref_2d_kde.x = x;
ref_2d_kde.hlist = hlist;
ref_2d_kde.lcv_m = kde.lcv.lcv_m;
ref_2d_kde.h_selected = kde.h;
ref_2d_kde.id1se = kde.lcv.id1se;
ref_2d_kde.fhat_at_x = kde.fpp(x);  % Density estimates at data points
ref_2d_kde.elapsed = elapsed;

save(fullfile(out_dir, 'ref_2d_kde.mat'), '-struct', 'ref_2d_kde');
fprintf('  Saved: ref_2d_kde.mat (n=%d, elapsed: %.3f sec)\n\n', n_kde_2d, elapsed);

%% ============================================================================
%% TEST 3: 3D KDE (order=0) - smaller bandwidth grid for speed
%% IMPORTANT: Use flag_power2=false to match R's 3D default (R disables
%% power-of-2 padding for 3D to avoid L³ memory explosion, ~21x speedup)
%% ============================================================================
fprintf('>>> TEST 3: 3D KDE (order=0) <<<\n');

x = randn(n_kde_3d, 3);
n_bw_3d = 10;  % Smaller for 3D
hlist = zeros(n_bw_3d, 3);
h_vals = logspace(log10(0.2), log10(1.0), n_bw_3d);
hlist(:, 1) = h_vals';
hlist(:, 2) = h_vals';
hlist(:, 3) = h_vals';
% Use flag_power2=false to match R's 3D default behavior
opt = struct('order', 0, 'flag_power2', false);

tic;
kde = cv_fastkde(x, hlist, opt);
elapsed = toc;

ref_3d_kde = struct();
ref_3d_kde.x = x;
ref_3d_kde.hlist = hlist;
ref_3d_kde.lcv_m = kde.lcv.lcv_m;
ref_3d_kde.h_selected = kde.h;
ref_3d_kde.id1se = kde.lcv.id1se;
ref_3d_kde.fhat_at_x = kde.fpp(x);  % Density estimates at data points
ref_3d_kde.elapsed = elapsed;

save(fullfile(out_dir, 'ref_3d_kde.mat'), '-struct', 'ref_3d_kde');
fprintf('  Saved: ref_3d_kde.mat (n=%d, elapsed: %.3f sec)\n\n', n_kde_3d, elapsed);

%% ============================================================================
%% TEST 4: 1D Local Linear (order=1)
%% ============================================================================
fprintf('>>> TEST 4: 1D Local Linear (order=1) <<<\n');

rng(42);
x = sort(rand(n_lpr_1d, 1));
y_true = sin(2*pi*x);
y = y_true + 0.2*randn(n_lpr_1d, 1);
hlist = logspace(log10(0.01), log10(0.5), n_bw)';
opt = struct('order', 1, 'calc_dof', true);

tic;
regs = cv_fastlpr(x, y, hlist, opt);
elapsed = toc;

ref_1d_order1 = struct();
ref_1d_order1.x = x;
ref_1d_order1.y = y;
ref_1d_order1.y_true = y_true;
ref_1d_order1.hlist = hlist;
ref_1d_order1.gcv_m = regs.gcv_yhat.gcv_m;
ref_1d_order1.mse = regs.gcv_yhat.mse;
ref_1d_order1.pdof_m = regs.gcv_yhat.pdof_m;
ref_1d_order1.h1se = regs.gcv_yhat.h1se;
ref_1d_order1.id1se = regs.gcv_yhat.id1se;
ref_1d_order1.yhat_mean = regs.yhat;
ref_1d_order1.elapsed = elapsed;
ref_1d_order1.dof_random_vectors = regs.dof_random_vectors;  % For Python/R reproducibility

save(fullfile(out_dir, 'ref_1d_order1.mat'), '-struct', 'ref_1d_order1');
fprintf('  Saved: ref_1d_order1.mat (elapsed: %.3f sec)\n\n', elapsed);

%% ============================================================================
%% TEST 5: 2D Local Linear (order=1)
%% ============================================================================
fprintf('>>> TEST 5: 2D Local Linear (order=1) <<<\n');

rng(42);
x = rand(n_lpr_2d, 2);
y_true = sin(2*pi*x(:,1)) .* cos(2*pi*x(:,2));
y = y_true + 0.2*randn(n_lpr_2d, 1);
hlist = zeros(n_bw, 2);
h_vals = logspace(log10(0.05), log10(0.5), n_bw);
hlist(:, 1) = h_vals';
hlist(:, 2) = h_vals';
opt = struct('order', 1, 'calc_dof', true);

tic;
regs = cv_fastlpr(x, y, hlist, opt);
elapsed = toc;

ref_2d_order1 = struct();
ref_2d_order1.x = x;
ref_2d_order1.y = y;
ref_2d_order1.y_true = y_true;
ref_2d_order1.hlist = hlist;
ref_2d_order1.gcv_m = regs.gcv_yhat.gcv_m;
ref_2d_order1.mse = regs.gcv_yhat.mse;
ref_2d_order1.pdof_m = regs.gcv_yhat.pdof_m;
ref_2d_order1.h1se = regs.gcv_yhat.h1se;
ref_2d_order1.id1se = regs.gcv_yhat.id1se;
ref_2d_order1.yhat_mean = regs.yhat;
ref_2d_order1.elapsed = elapsed;
ref_2d_order1.dof_random_vectors = regs.dof_random_vectors;  % For Python/R reproducibility

save(fullfile(out_dir, 'ref_2d_order1.mat'), '-struct', 'ref_2d_order1');
fprintf('  Saved: ref_2d_order1.mat (elapsed: %.3f sec)\n\n', elapsed);

%% ============================================================================
%% TEST 6: 1D Local Quadratic (order=2)
%% ============================================================================
fprintf('>>> TEST 6: 1D Local Quadratic (order=2) <<<\n');

rng(42);
x = sort(rand(n_lpr_1d, 1));
y_true = sin(2*pi*x);
y = y_true + 0.2*randn(n_lpr_1d, 1);
hlist = logspace(log10(0.01), log10(0.5), n_bw)';
opt = struct('order', 2, 'calc_dof', true);

tic;
regs = cv_fastlpr(x, y, hlist, opt);
elapsed = toc;

ref_1d_order2 = struct();
ref_1d_order2.x = x;
ref_1d_order2.y = y;
ref_1d_order2.y_true = y_true;
ref_1d_order2.hlist = hlist;
ref_1d_order2.gcv_m = regs.gcv_yhat.gcv_m;
ref_1d_order2.mse = regs.gcv_yhat.mse;
ref_1d_order2.pdof_m = regs.gcv_yhat.pdof_m;
ref_1d_order2.h1se = regs.gcv_yhat.h1se;
ref_1d_order2.id1se = regs.gcv_yhat.id1se;
ref_1d_order2.yhat_mean = regs.yhat;
ref_1d_order2.elapsed = elapsed;
ref_1d_order2.dof_random_vectors = regs.dof_random_vectors;  % For Python/R reproducibility

save(fullfile(out_dir, 'ref_1d_order2.mat'), '-struct', 'ref_1d_order2');
fprintf('  Saved: ref_1d_order2.mat (elapsed: %.3f sec)\n\n', elapsed);

%% ============================================================================
%% TEST 7: 2D Local Quadratic (order=2)
%% ============================================================================
fprintf('>>> TEST 7: 2D Local Quadratic (order=2) <<<\n');

rng(42);
x = rand(n_lpr_2d, 2);
y_true = sin(2*pi*x(:,1)) .* cos(2*pi*x(:,2));
y = y_true + 0.2*randn(n_lpr_2d, 1);
hlist = zeros(n_bw, 2);
h_vals = logspace(log10(0.05), log10(0.5), n_bw);
hlist(:, 1) = h_vals';
hlist(:, 2) = h_vals';
opt = struct('order', 2, 'calc_dof', true);

tic;
regs = cv_fastlpr(x, y, hlist, opt);
elapsed = toc;

ref_2d_order2 = struct();
ref_2d_order2.x = x;
ref_2d_order2.y = y;
ref_2d_order2.y_true = y_true;
ref_2d_order2.hlist = hlist;
ref_2d_order2.gcv_m = regs.gcv_yhat.gcv_m;
ref_2d_order2.mse = regs.gcv_yhat.mse;
ref_2d_order2.pdof_m = regs.gcv_yhat.pdof_m;
ref_2d_order2.h1se = regs.gcv_yhat.h1se;
ref_2d_order2.id1se = regs.gcv_yhat.id1se;
ref_2d_order2.yhat_mean = regs.yhat;
ref_2d_order2.elapsed = elapsed;
ref_2d_order2.dof_random_vectors = regs.dof_random_vectors;  % For Python/R reproducibility

save(fullfile(out_dir, 'ref_2d_order2.mat'), '-struct', 'ref_2d_order2');
fprintf('  Saved: ref_2d_order2.mat (elapsed: %.3f sec)\n\n', elapsed);

%% ============================================================================
%% TEST 8: Complex Data (1D order=1)
%% ============================================================================
fprintf('>>> TEST 8: Complex Data (1D order=1) <<<\n');

rng(42);
x = sort(rand(n_complex, 1));
y_true_real = sin(2*pi*x);
y_true_imag = cos(2*pi*x);
y = complex(y_true_real + 0.2*randn(n_complex, 1), ...
            y_true_imag + 0.2*randn(n_complex, 1));
hlist = logspace(log10(0.01), log10(0.5), n_bw)';
opt = struct('order', 1, 'calc_dof', true);

tic;
regs = cv_fastlpr(x, y, hlist, opt);
elapsed = toc;

ref_complex = struct();
ref_complex.x = x;
ref_complex.y = y;
ref_complex.hlist = hlist;
ref_complex.gcv_m = regs.gcv_yhat.gcv_m;
ref_complex.mse = regs.gcv_yhat.mse;
ref_complex.pdof_m = regs.gcv_yhat.pdof_m;
ref_complex.h1se = regs.gcv_yhat.h1se;
ref_complex.id1se = regs.gcv_yhat.id1se;
ref_complex.yhat_mean = regs.yhat;
ref_complex.elapsed = elapsed;
ref_complex.dof_random_vectors = regs.dof_random_vectors;  % For Python/R reproducibility

save(fullfile(out_dir, 'ref_complex.mat'), '-struct', 'ref_complex');
fprintf('  Saved: ref_complex.mat (elapsed: %.3f sec)\n\n', elapsed);

%% ============================================================================
%% TEST 9: Heteroscedastic 1D (mean + variance, order=1)
%% Following example_hetero.m pattern: two-step estimation
%% ============================================================================
fprintf('>>> TEST 9: Heteroscedastic 1D (mean + variance, order=1) <<<\n');

rng(42);
x = 2*2*(rand(n_hetero, 1) - 0.5);  % Uniform in [-2, 2] like example
y_true = x.^3;  % Cubic mean function like example
var_true = 1 + 4*exp(-x.^2);  % Variance depends on x like example
y = y_true + sqrt(var_true).*randn(n_hetero, 1);
hlist = logspace(log10(0.01), log10(1), n_bw)';

% Step 1: Estimate mean
opt_mean = struct('order', 1, 'calc_dof', true, 'dstd', 0);
tic;
regs_mean = cv_fastlpr(x, y, hlist, opt_mean);
elapsed_mean = toc;

% Step 2: Estimate variance from residuals^2
residuals = y - regs_mean.yhat;
opt_var = struct('order', 1, 'calc_dof', true, 'y_type_out', 'variance', 'dstd', 1);
tic;
regs_var = cv_fastlpr(x, residuals.^2, hlist, opt_var);
elapsed_var = toc;

elapsed = elapsed_mean + elapsed_var;

ref_hetero_1d = struct();
ref_hetero_1d.x = x;
ref_hetero_1d.y = y;
ref_hetero_1d.y_true = y_true;
ref_hetero_1d.var_true = var_true;
ref_hetero_1d.hlist = hlist;
% Mean estimation results
ref_hetero_1d.gcv_m = regs_mean.gcv_yhat.gcv_m;
ref_hetero_1d.mse = regs_mean.gcv_yhat.mse;
ref_hetero_1d.pdof_m = regs_mean.gcv_yhat.pdof_m;
ref_hetero_1d.h1se_mean = regs_mean.gcv_yhat.h1se;
ref_hetero_1d.id1se_mean = regs_mean.gcv_yhat.id1se;
ref_hetero_1d.yhat_mean = regs_mean.yhat;
ref_hetero_1d.dof_random_vectors_mean = regs_mean.dof_random_vectors;
% Variance estimation results  
ref_hetero_1d.gcv_var = regs_var.gcv_yhat.gcv_m;
ref_hetero_1d.h1se_var = regs_var.gcv_yhat.h1se;
ref_hetero_1d.id1se_var = regs_var.gcv_yhat.id1se;
ref_hetero_1d.yhat_var = regs_var.yhat;
ref_hetero_1d.dof_random_vectors_var = regs_var.dof_random_vectors;
ref_hetero_1d.elapsed = elapsed;

save(fullfile(out_dir, 'ref_hetero_1d.mat'), '-struct', 'ref_hetero_1d');
fprintf('  Saved: ref_hetero_1d.mat (elapsed: %.3f sec)\n\n', elapsed);

%% ============================================================================
%% TEST 10: Heteroscedastic 2D (mean + variance, order=1)
%% Following example_hetero.m pattern
%% ============================================================================
fprintf('>>> TEST 10: Heteroscedastic 2D (mean + variance, order=1) <<<\n');

rng(44);  % Same seed as example
n_hetero_2d = 1200;  % Same as example
x2d = 2*2*(rand(n_hetero_2d, 2) - 0.5);  % Uniform in [-2, 2] x [-2, 2]
y_true_2d = x2d(:,1).^3 + x2d(:,2).^3;  % Cubic mean
var_true_2d = 1 + 4*exp(-(x2d(:,1).^2 + x2d(:,2).^2));  % Variance depends on r^2
y2d = y_true_2d + sqrt(var_true_2d).*randn(n_hetero_2d, 1);
hlist2d = zeros(100, 2);  % 10x10 grid
h_vals = logspace(log10(0.05), log10(0.5), 10);
[H1, H2] = meshgrid(h_vals, h_vals);
hlist2d(:, 1) = H1(:);
hlist2d(:, 2) = H2(:);

% Step 1: Estimate mean
opt_mean_2d = struct('order', 1, 'calc_dof', true, 'dstd', 0);
tic;
regs_mean_2d = cv_fastlpr(x2d, y2d, hlist2d, opt_mean_2d);
elapsed_mean_2d = toc;

% Step 2: Estimate variance from residuals^2
residuals_2d = y2d - regs_mean_2d.yhat;
opt_var_2d = struct('order', 1, 'calc_dof', true, 'y_type_out', 'variance', 'dstd', 1);
tic;
regs_var_2d = cv_fastlpr(x2d, residuals_2d.^2, hlist2d, opt_var_2d);
elapsed_var_2d = toc;

elapsed_2d = elapsed_mean_2d + elapsed_var_2d;

ref_hetero_2d = struct();
ref_hetero_2d.x = x2d;
ref_hetero_2d.y = y2d;
ref_hetero_2d.y_true = y_true_2d;
ref_hetero_2d.var_true = var_true_2d;
ref_hetero_2d.hlist = hlist2d;
% Mean estimation results
ref_hetero_2d.gcv_m = regs_mean_2d.gcv_yhat.gcv_m;
ref_hetero_2d.h1se_mean = regs_mean_2d.gcv_yhat.h1se;
ref_hetero_2d.id1se_mean = regs_mean_2d.gcv_yhat.id1se;
ref_hetero_2d.yhat_mean = regs_mean_2d.yhat;
ref_hetero_2d.dof_random_vectors_mean = regs_mean_2d.dof_random_vectors;
% Variance estimation results
ref_hetero_2d.gcv_var = regs_var_2d.gcv_yhat.gcv_m;
ref_hetero_2d.h1se_var = regs_var_2d.gcv_yhat.h1se;
ref_hetero_2d.id1se_var = regs_var_2d.gcv_yhat.id1se;
ref_hetero_2d.yhat_var = regs_var_2d.yhat;
ref_hetero_2d.dof_random_vectors_var = regs_var_2d.dof_random_vectors;
ref_hetero_2d.elapsed = elapsed_2d;

save(fullfile(out_dir, 'ref_hetero_2d.mat'), '-struct', 'ref_hetero_2d');
fprintf('  Saved: ref_hetero_2d.mat (elapsed: %.3f sec)\n\n', elapsed_2d);

%% Summary
fprintf('================================================================================\n');
fprintf('STANDARDIZED REFERENCE DATA GENERATION COMPLETE\n');
fprintf('================================================================================\n');
fprintf('Generated files in %s:\n', out_dir);
fprintf('  - ref_1d_kde.mat       (1D KDE)\n');
fprintf('  - ref_2d_kde.mat       (2D KDE)\n');
fprintf('  - ref_3d_kde.mat       (3D KDE)\n');
fprintf('  - ref_1d_order1.mat    (1D Local Linear)\n');
fprintf('  - ref_2d_order1.mat    (2D Local Linear)\n');
fprintf('  - ref_1d_order2.mat    (1D Local Quadratic)\n');
fprintf('  - ref_2d_order2.mat    (2D Local Quadratic)\n');
fprintf('  - ref_complex.mat      (Complex Data)\n');
fprintf('  - ref_hetero_1d.mat    (1D Heteroscedastic mean+var)\n');
fprintf('  - ref_hetero_2d.mat    (2D Heteroscedastic mean+var)\n');
fprintf('================================================================================\n');
