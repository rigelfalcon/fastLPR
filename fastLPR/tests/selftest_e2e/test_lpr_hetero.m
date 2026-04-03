classdef test_lpr_hetero < matlab.unittest.TestCase
    % test_lpr_hetero - Unit tests for heteroscedastic regression (mean + variance)
    %
    % Run with: >> results = runtests('test_lpr_hetero')
    %
    % Uses test_data_utils for reusable test data generation.
    %
    % Author: Ying Wang, Min Li
    % Copyright (c) 2020-2025 fastLPR Development Team
    % License: GNU General Public License v3.0

    properties
        utils  % Test data utilities
    end

    methods (TestClassSetup)
        function setupPath(testCase)
            % Navigate from selftest_e2e -> tests -> fastLPR root
            testDir = fileparts(mfilename('fullpath'));
            testsDir = fileparts(testDir);
            fastlprRoot = fileparts(testsDir);
            addpath(fullfile(fastlprRoot, 'utility'));
            addpath(fullfile(testsDir, 'generators'));
            testCase.utils = test_data_utils();
        end
    end

    methods (Test)
        function test_mean_estimation(testCase)
            % Unified with Python/R: n=500, h=[0.01,0.5], nh=10
            data = testCase.utils.generate_hetero_1d(500, 42);
            hlist = testCase.utils.generate_hlist([0.01, 0.5], 10, 1);
            opt = struct('order', 1, 'calc_dof', true, 'y_type_out', 'mean');
            regs = cv_fastlpr(data.x, data.y, hlist, opt);
            testCase.verifyNotEmpty(regs.yhat);
            testCase.verifyEqual(length(regs.yhat), 500);
        end

        function test_variance_positive(testCase)
            % Different heteroscedastic pattern
            rng(42);
            x = sort(rand(500, 1));
            var_true = 0.1 + 0.5*x;
            y = x.^2 + sqrt(var_true).*randn(500, 1);
            hlist = testCase.utils.generate_hlist([0.02, 0.4], 10, 1);
            % Step 1: Mean estimation
            opt_mean = struct('order', 1, 'y_type_out', 'mean');
            regs_mean = cv_fastlpr(x, y, hlist, opt_mean);
            % Step 2: Compute squared residuals
            residuals_sq = (y - regs_mean.yhat).^2;
            % Step 3: Variance estimation
            opt_var = struct('order', 1, 'y_type_out', 'variance');
            regs_var = cv_fastlpr(x, residuals_sq, hlist, opt_var);
            testCase.verifyTrue(all(regs_var.yhat >= 0));
        end

        function test_mean_accuracy(testCase)
            % Unified with Python: n=500, h=[0.01,0.3], nh=15, heteroscedastic noise
            rng(42);
            n = 500;
            x = sort(rand(n, 1));
            y_true = sin(2*pi*x);
            var_true = 0.1 + 0.3*x.^2;
            y = y_true + sqrt(var_true).*randn(n, 1);
            hlist = testCase.utils.generate_hlist([0.01, 0.3], 15, 1);
            opt = struct('order', 1, 'y_type_out', 'mean');
            regs = cv_fastlpr(x, y, hlist, opt);
            mse = mean((regs.yhat - y_true).^2);
            testCase.verifyLessThan(mse, 0.05);
        end

        function test_hetero_2d_variance(testCase)
            % 2D variance estimation (from Python)
            rng(42);
            n = 400;
            x = rand(n, 2);
            y_true = sin(2*pi*x(:,1)) .* cos(2*pi*x(:,2));
            var_true = 0.1 + 0.2*x(:,1).^2 + 0.2*x(:,2).^2;
            y = y_true + sqrt(var_true).*randn(n, 1);
            hlist = testCase.utils.generate_hlist([0.05, 0.5], 5, 2);
            % Step 1: Mean estimation
            opt_mean = struct('order', 1, 'y_type_out', 'mean');
            regs_mean = cv_fastlpr(x, y, hlist, opt_mean);
            % Step 2: Compute squared residuals
            residuals_sq = (y - regs_mean.yhat).^2;
            % Step 3: Variance estimation
            opt_var = struct('order', 1, 'y_type_out', 'variance');
            regs_var = cv_fastlpr(x, residuals_sq, hlist, opt_var);
            testCase.verifyNotEmpty(regs_var.yhat, 'Variance estimate should not be empty');
            testCase.verifyEqual(length(regs_var.yhat), n);
            testCase.verifyTrue(all(regs_var.yhat >= 0), 'Variance should be non-negative');
        end

        function test_reproducibility(testCase)
            % Unified with Python/R: n=200, h=[0.1,0.5], nh=5
            % DOF estimation now uses dof_seed (default: 42) for reproducibility
            rng(42);
            n = 200;
            x = sort(rand(n, 1));
            var_true = 0.1 + 0.3*x.^2;
            y = sin(2*pi*x) + sqrt(var_true).*randn(n, 1);
            hlist = testCase.utils.generate_hlist([0.1, 0.5], 5, 1);

            % Step 1: Mean estimation (run twice - dof_seed ensures reproducibility)
            opt_mean = struct('order', 1, 'y_type_out', 'mean');
            regs_mean1 = cv_fastlpr(x, y, hlist, opt_mean);
            regs_mean2 = cv_fastlpr(x, y, hlist, opt_mean);

            % Verify mean estimation is reproducible
            testCase.verifyEqual(regs_mean1.yhat, regs_mean2.yhat, 'AbsTol', 1e-10, ...
                'Mean estimation should be reproducible');

            % Step 2: Compute squared residuals
            residuals_sq = (y - regs_mean1.yhat).^2;

            % Step 3: Variance estimation (run twice - dof_seed ensures reproducibility)
            opt_var = struct('order', 1, 'y_type_out', 'variance');
            regs_var1 = cv_fastlpr(x, residuals_sq, hlist, opt_var);
            regs_var2 = cv_fastlpr(x, residuals_sq, hlist, opt_var);

            testCase.verifyEqual(regs_var1.yhat, regs_var2.yhat, 'AbsTol', 1e-10, ...
                'Variance estimation should be reproducible');
        end
    end
end
