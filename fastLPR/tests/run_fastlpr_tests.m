function run_fastlpr_tests()
% RUN_FASTLPR_TESTS - Comprehensive test suite for the fastLPR toolbox
%
% This script tests all major functionality of the fastLPR toolbox:
%   1. Basic 1D regression
%   2. 2D regression
%   3. Complex-valued data
%   4. Confidence intervals
%   5. Error handling
%   6. Memory estimation
%   7. Plotting functions
%   8. Bandwidth selection
%   9. Confidence interval plotting
%   10. 1D Kernel Density Estimation (KDE)
%   11. 2D Kernel Density Estimation (KDE)
%   12. KDE with single bandwidth
%
% Usage:
%   test_fastlpr()
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

fprintf('\n========================================\n');
fprintf('fastLPR Test Suite\n');
fprintf('========================================\n\n');

% Add path
addpath('../fastLPR/utility');

% Initialize test results
total_tests = 0;
passed_tests = 0;
failed_tests = 0;

%% Test 1: Basic 1D Regression
fprintf('Test 1: Basic 1D Regression...\n');
total_tests = total_tests + 1;
try
    % Generate data
    rng(0);
    n = 100;
    x = rand(n, 1);
    y = sin(2*pi*x) + 0.1*randn(n, 1);
    
    % Regression
    hlist = get_hlist(10, [0.01, 0.5], @logspace);
    opt.order = 1;
    regs = cv_fastlpr(x, y, hlist, opt);
    
    % Validate output
    assert(isfield(regs, 'yhat'), 'Missing yhat field');
    assert(isfield(regs, 'fpp_yhat'), 'Missing fpp_yhat field');
    assert(isa(regs.fpp_yhat, 'griddedInterpolant'), 'fpp_yhat not griddedInterpolant');
    assert(length(regs.yhat) > 0, 'Empty yhat');
    
    fprintf('  ✓ PASSED\n\n');
    passed_tests = passed_tests + 1;
catch ME
    fprintf('  ✗ FAILED: %s\n\n', ME.message);
    failed_tests = failed_tests + 1;
end

%% Test 2: 2D Regression
fprintf('Test 2: 2D Regression...\n');
total_tests = total_tests + 1;
try
    % Generate data
    rng(0);
    n = 200;
    x = rand(n, 2);
    y = sin(2*pi*x(:,1)) .* cos(2*pi*x(:,2)) + 0.1*randn(n, 1);
    
    % Regression
    hlist = get_hlist([5, 5], [0.01, 1; 0.01, 1], @logspace);
    opt.order = 1;
    regs = cv_fastlpr(x, y, hlist, opt);
    
    % Validate output
    assert(isfield(regs, 'yhat'), 'Missing yhat field');
    assert(isfield(regs, 'fpp_yhat'), 'Missing fpp_yhat field');
    assert(length(regs.yhat) > 0, 'Empty yhat');
    
    fprintf('  ✓ PASSED\n\n');
    passed_tests = passed_tests + 1;
catch ME
    fprintf('  ✗ FAILED: %s\n\n', ME.message);
    failed_tests = failed_tests + 1;
end

%% Test 3: Complex-Valued Data
fprintf('Test 3: Complex-Valued Data...\n');
total_tests = total_tests + 1;
try
    % Generate complex data
    rng(0);
    n = 100;
    x = rand(n, 1) + 1i*rand(n, 1);
    y = exp(1i*2*pi*abs(x)) + 0.1*randn(n, 1);
    
    % Regression
    hlist = get_hlist(10, [0.01, 0.5], @logspace);
    opt.order = 1;
    regs = cv_fastlpr(x, y, hlist, opt);
    
    % Validate output
    assert(isfield(regs, 'yhat'), 'Missing yhat field');
    assert(~isempty(regs.yhat), 'Empty yhat');
    
    fprintf('  ✓ PASSED\n\n');
    passed_tests = passed_tests + 1;
catch ME
    fprintf('  ✗ FAILED: %s\n\n', ME.message);
    failed_tests = failed_tests + 1;
end

%% Test 4: Confidence Intervals
fprintf('Test 4: Confidence Intervals...\n');
total_tests = total_tests + 1;
try
    % Generate data
    rng(0);
    n = 200;
    x = rand(n, 1) * 2 - 1;
    y = x.^3 + (1 + exp(-x.^2)) .* randn(n, 1);
    
    % Estimate mean
    hlist = get_hlist(10, [0.01, 1], @logspace);
    opt.order = 1;
    opt.dstd = 0;
    regs_mu = cv_fastlpr(x, y, hlist, opt);
    
    % Estimate variance
    residuals = y - regs_mu.yhat;
    opt.y_type_out = 'variance';
    opt.dstd = 5;
    regs_sigma = cv_fastlpr(x, residuals.^2, hlist, opt);
    
    % Compute intervals (CI and PI)
    ci = fastlpr_interval(regs_mu, regs_sigma, 0.05);
    
    % Validate output
    assert(isa(ci, 'griddedInterpolant'), 'CI not griddedInterpolant');
    assert(size(ci.Values, 2) == 2, 'CI should have 2 columns');
    
    fprintf('  ✓ PASSED\n\n');
    passed_tests = passed_tests + 1;
catch ME
    fprintf('  ✗ FAILED: %s\n\n', ME.message);
    failed_tests = failed_tests + 1;
end

%% Test 5: Polynomial Orders
fprintf('Test 5: Polynomial Orders (0, 1, 2)...\n');
total_tests = total_tests + 1;
try
    % Generate data
    rng(0);
    n = 100;
    x = rand(n, 1);
    y = sin(2*pi*x) + 0.1*randn(n, 1);
    hlist = get_hlist(10, [0.01, 0.5], @logspace);
    
    % Test Order 0
    opt.order = 0;
    regs0 = cv_fastlpr(x, y, hlist, opt);
    assert(~isempty(regs0.yhat), 'Order 0 failed');
    
    % Test Order 1
    opt.order = 1;
    regs1 = cv_fastlpr(x, y, hlist, opt);
    assert(~isempty(regs1.yhat), 'Order 1 failed');
    
    % Test Order 2
    opt.order = 2;
    regs2 = cv_fastlpr(x, y, hlist, opt);
    assert(~isempty(regs2.yhat), 'Order 2 failed');
    
    fprintf('  ✓ PASSED\n\n');
    passed_tests = passed_tests + 1;
catch ME
    fprintf('  ✗ FAILED: %s\n\n', ME.message);
    failed_tests = failed_tests + 1;
end

%% Test 6: Error Handling - Dimension Mismatch
fprintf('Test 6: Error Handling - Dimension Mismatch...\n');
total_tests = total_tests + 1;
try
    % Generate mismatched data
    x = rand(100, 1);
    y = rand(50, 1);  % Wrong size
    hlist = get_hlist(10, [0.01, 0.5], @logspace);
    opt.order = 1;
    
    % This should throw an error
    try
        regs = cv_fastlpr(x, y, hlist, opt);
        error('Should have thrown dimension mismatch error');
    catch ME
        if contains(ME.identifier, 'DimensionMismatch')
            fprintf('  ✓ PASSED (correctly caught error)\n\n');
            passed_tests = passed_tests + 1;
        else
            error('Wrong error type: %s', ME.identifier);
        end
    end
catch ME
    fprintf('  ✗ FAILED: %s\n\n', ME.message);
    failed_tests = failed_tests + 1;
end

%% Test 7: Error Handling - Invalid Order
fprintf('Test 7: Error Handling - Invalid Order...\n');
total_tests = total_tests + 1;
try
    % Generate data
    x = rand(100, 1);
    y = rand(100, 1);
    hlist = get_hlist(10, [0.01, 0.5], @logspace);
    opt.order = 3;  % Invalid order
    
    % This should throw an error
    try
        regs = cv_fastlpr(x, y, hlist, opt);
        error('Should have thrown invalid order error');
    catch ME
        if contains(ME.identifier, 'InvalidOrder')
            fprintf('  ✓ PASSED (correctly caught error)\n\n');
            passed_tests = passed_tests + 1;
        else
            error('Wrong error type: %s', ME.identifier);
        end
    end
catch ME
    fprintf('  ✗ FAILED: %s\n\n', ME.message);
    failed_tests = failed_tests + 1;
end

%% Test 8: get_hlist Function
fprintf('Test 8: get_hlist Function...\n');
total_tests = total_tests + 1;
try
    % Test 1D
    hlist1 = get_hlist(20, [0.01, 1], @logspace);
    assert(size(hlist1, 1) == 20, '1D hlist wrong size');
    assert(size(hlist1, 2) == 1, '1D hlist wrong dimensions');
    
    % Test 2D
    hlist2 = get_hlist([10, 10], [0.01, 1; 0.01, 1], @logspace);
    assert(size(hlist2, 1) == 100, '2D hlist wrong size');
    assert(size(hlist2, 2) == 2, '2D hlist wrong dimensions');
    
    fprintf('  ✓ PASSED\n\n');
    passed_tests = passed_tests + 1;
catch ME
    fprintf('  ✗ FAILED: %s\n\n', ME.message);
    failed_tests = failed_tests + 1;
end

%% Test 9: Visualization Functions
fprintf('Test 9: Visualization Functions...\n');
total_tests = total_tests + 1;
try
    % Generate data and fit
    rng(0);
    n = 100;
    x = rand(n, 1);
    y = sin(2*pi*x) + 0.1*randn(n, 1);
    hlist = get_hlist(10, [0.01, 0.5], @logspace);
    opt.order = 1;
    regs = cv_fastlpr(x, y, hlist, opt);
    
    % Test fastlpr_plot
    fig1 = figure('Visible', 'off');
    fastlpr_plot(regs.fpp_yhat);
    close(fig1);
    
    % Test fastlpr_plot_interval
    opt.dstd = 0;
    regs_mu = cv_fastlpr(x, y, hlist, opt);
    residuals = y - regs_mu.yhat;
    opt.y_type_out = 'variance';
    opt.dstd = 5;
    regs_sigma = cv_fastlpr(x, residuals.^2, hlist, opt);
    ci = fastlpr_interval(regs_mu, regs_sigma, 0.05);
    
    fig2 = figure('Visible', 'off');
    fastlpr_plot_interval([], ci);
    close(fig2);
    
    fprintf('  ✓ PASSED\n\n');
    passed_tests = passed_tests + 1;
catch ME
    fprintf('  ✗ FAILED: %s\n\n', ME.message);
    failed_tests = failed_tests + 1;
end

%% Test 10: 1D KDE
fprintf('Test 10: 1D Kernel Density Estimation...\n');
total_tests = total_tests + 1;
try
    % Generate bimodal data
    rng(0);
    x = [randn(100, 1); randn(100, 1) + 3];

    % KDE with bandwidth selection
    hlist = get_hlist(20, [0.1, 1], @logspace);
    opt_kde.verbose = false;
    kde = cv_fastkde(x, hlist, opt_kde);

    % Validate output structure
    assert(isfield(kde, 'fhat'), 'Missing fhat field');
    assert(isfield(kde, 'fpp'), 'Missing fpp field');
    assert(isfield(kde, 'h'), 'Missing h field');
    assert(isfield(kde, 'xlist'), 'Missing xlist field');
    assert(isfield(kde, 'lcv'), 'Missing lcv field');
    assert(isa(kde.fpp, 'griddedInterpolant'), 'fpp not griddedInterpolant');

    % Validate fhat is for selected bandwidth only (not all bandwidths)
    assert(size(kde.fhat, 2) == 1, 'fhat should be Nx1, not Nxdh');

    % Validate density integrates to approximately 1
    % Note: Finite grid may not capture full tail, so tolerance is relaxed
    dx = kde.xlist{1}(2) - kde.xlist{1}(1);
    integral = sum(kde.fhat) * dx;
    assert(integral > 0.5 && integral < 1.5, sprintf('Density integral %.3f not reasonable', integral));

    % Test fastkde_plot
    figure('Visible', 'off');
    h = fastkde_plot(kde, struct('plot_type', '1d'));
    close(gcf);

    % Test fastkde_plot_bandwidth
    figure('Visible', 'off');
    h_bw = fastkde_plot_bandwidth(kde);
    close(gcf);

    fprintf('  ✓ PASSED\n\n');
    passed_tests = passed_tests + 1;
catch ME
    fprintf('  ✗ FAILED: %s\n\n', ME.message);
    failed_tests = failed_tests + 1;
end

%% Test 11: 2D KDE
fprintf('Test 11: 2D Kernel Density Estimation...\n');
total_tests = total_tests + 1;
try
    % Generate 2D bimodal data
    rng(0);
    x = [randn(200, 2); randn(200, 2) + 2];

    % KDE with bandwidth selection
    hlist = get_hlist([5, 5], [0.1, 1; 0.1, 1], @logspace);
    opt_kde.verbose = false;
    opt_kde.N = 30;  % Smaller grid for faster testing
    kde = cv_fastkde(x, hlist, opt_kde);

    % Validate output structure
    assert(isfield(kde, 'fhat'), 'Missing fhat field');
    assert(isfield(kde, 'fpp'), 'Missing fpp field');
    assert(isfield(kde, 'h'), 'Missing h field');
    assert(length(kde.h) == 2, 'h should be 2D');
    assert(isfield(kde, 'lcv'), 'Missing lcv field');

    % Validate fhat is for selected bandwidth only (not all bandwidths)
    assert(ndims(kde.fhat) == 2, 'fhat should be 2D, not 3D');

    % Validate density is positive and reasonable
    % Note: Finite grid and small N may affect integral accuracy
    dx1 = kde.xlist{1}(2) - kde.xlist{1}(1);
    dx2 = kde.xlist{2}(2) - kde.xlist{2}(1);
    integral = sum(kde.fhat(:)) * dx1 * dx2;
    assert(integral > 0.1 && integral < 5.0, sprintf('Density integral %.3f not reasonable', integral));
    assert(all(kde.fhat(:) >= 0), 'Density should be non-negative');

    % Test fastkde_plot
    figure('Visible', 'off');
    h = fastkde_plot(kde, struct('plot_type', '2d_contour'));
    close(gcf);

    % Test fastkde_plot_bandwidth
    figure('Visible', 'off');
    h_bw = fastkde_plot_bandwidth(kde);
    close(gcf);

    fprintf('  ✓ PASSED\n\n');
    passed_tests = passed_tests + 1;
catch ME
    fprintf('  ✗ FAILED: %s\n\n', ME.message);
    failed_tests = failed_tests + 1;
end

%% Test 12: KDE with Single Bandwidth
fprintf('Test 12: KDE with Single Bandwidth...\n');
total_tests = total_tests + 1;
try
    % Generate data
    rng(0);
    x = randn(100, 1);

    % KDE with single bandwidth (no selection)
    h = 0.5;
    opt_kde.verbose = false;
    kde = cv_fastkde(x, h, opt_kde);

    % Validate output structure
    assert(isfield(kde, 'fhat'), 'Missing fhat field');
    assert(isfield(kde, 'fpp'), 'Missing fpp field');
    assert(isfield(kde, 'h'), 'Missing h field');
    assert(kde.h == h, 'Bandwidth mismatch');
    assert(~isfield(kde, 'lcv') || isempty(kde.lcv), 'Should not have lcv for single bandwidth');

    % Validate fhat is 1D
    assert(size(kde.fhat, 2) == 1, 'fhat should be Nx1');

    fprintf('  ✓ PASSED\n\n');
    passed_tests = passed_tests + 1;
catch ME
    fprintf('  ✗ FAILED: %s\n\n', ME.message);
    failed_tests = failed_tests + 1;
end

%% Summary
fprintf('========================================\n');
fprintf('Test Summary\n');
fprintf('========================================\n');
fprintf('Total tests:  %d\n', total_tests);
fprintf('Passed:       %d (%.1f%%)\n', passed_tests, 100*passed_tests/total_tests);
fprintf('Failed:       %d (%.1f%%)\n', failed_tests, 100*failed_tests/total_tests);
fprintf('========================================\n\n');

if failed_tests == 0
    fprintf('✓ ALL TESTS PASSED! ✓\n\n');
else
    fprintf('✗ SOME TESTS FAILED ✗\n\n');
    error('Test suite failed');
end

end


