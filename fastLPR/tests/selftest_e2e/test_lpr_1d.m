classdef test_lpr_1d < matlab.unittest.TestCase
    % test_lpr_1d - Unit tests for 1D local polynomial regression
    %
    % Run with: >> results = runtests('test_lpr_1d')
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
        function test_order0_nadaraya_watson(testCase)
            data = testCase.utils.generate_sin_1d(500, 0.2, 42);
            hlist = testCase.utils.generate_hlist([0.01, 0.5], 10, 1);
            opt = struct('order', 0);
            regs = cv_fastlpr(data.x, data.y, hlist, opt);
            testCase.verifyNotEmpty(regs.yhat);
            testCase.verifyGreaterThan(regs.gcv_yhat.h1se, 0);
        end

        function test_order1_local_linear(testCase)
            data = testCase.utils.generate_sin_1d(500, 0.2, 42);
            hlist = testCase.utils.generate_hlist([0.01, 0.5], 10, 1);
            opt = struct('order', 1, 'calc_dof', true);
            regs = cv_fastlpr(data.x, data.y, hlist, opt);
            testCase.verifyNotEmpty(regs.yhat);
            testCase.verifyTrue(isfield(regs.gcv_yhat, 'gcv_m'));
        end

        function test_order2_local_quadratic(testCase)
            data = testCase.utils.generate_sin_1d(500, 0.2, 42);
            hlist = testCase.utils.generate_hlist([0.01, 0.5], 10, 1);
            opt = struct('order', 2, 'calc_dof', true);
            regs = cv_fastlpr(data.x, data.y, hlist, opt);
            testCase.verifyNotEmpty(regs.yhat);
        end

        function test_gcv_bandwidth_selection(testCase)
            data = testCase.utils.generate_sin_1d(500, 0.2, 42);
            hlist = testCase.utils.generate_hlist([0.01, 0.5], 20, 1);
            opt = struct('order', 1, 'calc_dof', true);
            regs = cv_fastlpr(data.x, data.y, hlist, opt);
            testCase.verifyGreaterThan(regs.gcv_yhat.h1se, hlist(1));
            testCase.verifyLessThan(regs.gcv_yhat.h1se, hlist(end));
        end

        function test_prediction_accuracy(testCase)
            data = testCase.utils.generate_sin_1d(500, 0.1, 42);
            hlist = testCase.utils.generate_hlist([0.01, 0.3], 15, 1);
            opt = struct('order', 1);
            regs = cv_fastlpr(data.x, data.y, hlist, opt);
            mse = mean((regs.yhat - data.y_true).^2);
            testCase.verifyLessThan(mse, 0.05);
        end

        function test_reproducibility(testCase)
            data = testCase.utils.generate_sin_1d(200, 0.2, 42);
            hlist = testCase.utils.generate_hlist([0.05, 0.3], 5, 1);
            opt = struct('order', 1);
            regs1 = cv_fastlpr(data.x, data.y, hlist, opt);
            regs2 = cv_fastlpr(data.x, data.y, hlist, opt);
            testCase.verifyEqual(regs1.yhat, regs2.yhat, 'AbsTol', 1e-14);
        end

        function test_single_bandwidth(testCase)
            % Test with fixed bandwidth (from Python)
            rng(42);
            n = 100;
            x = sort(rand(n, 1));
            y = sin(2*pi*x) + 0.1*randn(n, 1);
            h = 0.2;
            opt = struct('order', 1);
            regs = cv_fastlpr(x, y, h, opt);
            testCase.verifyNotEmpty(regs.yhat, 'yhat should not be empty');
            testCase.verifyEqual(length(regs.yhat), n);
        end

        function test_linear_function(testCase)
            % Test with linear trend (from Python)
            rng(42);
            n = 150;
            x = sort(rand(n, 1));
            y_true = 2*x + 1;
            y = y_true + 0.05*randn(n, 1);
            hlist = testCase.utils.generate_hlist([0.2, 0.8], 5, 1);
            opt = struct('order', 1);
            regs = cv_fastlpr(x, y, hlist, opt);
            mse = mean((regs.yhat - y_true).^2);
            testCase.verifyLessThan(mse, 0.01, 'MSE should be small for linear function');
        end

        function test_order1_reduces_bias(testCase)
            % Compare order 0 vs 1 (from R)
            rng(42);
            n = 200;
            x = sort(rand(n, 1));
            y_true = sin(2*pi*x);
            y = y_true + 0.05*randn(n, 1);
            hlist = testCase.utils.generate_hlist([0.3, 0.8], 5, 1);
            opt0 = struct('order', 0);
            opt1 = struct('order', 1);
            regs0 = cv_fastlpr(x, y, hlist, opt0);
            regs1 = cv_fastlpr(x, y, hlist, opt1);
            mse0 = mean((regs0.yhat - y_true).^2);
            mse1 = mean((regs1.yhat - y_true).^2);
            % Order 1 should have lower or similar MSE (allow for variance)
            testCase.verifyLessThan(mse1, mse0 * 1.5, ...
                'Order 1 should have lower or similar MSE than order 0');
        end

        function test_boundary_behavior(testCase)
            % Edge handling test (from R)
            rng(42);
            n = 200;
            x = sort(rand(n, 1));
            y_true = sin(2*pi*x);
            y = y_true + 0.1*randn(n, 1);
            hlist = testCase.utils.generate_hlist([0.1, 0.5], 5, 1);
            opt = struct('order', 1);
            regs = cv_fastlpr(x, y, hlist, opt);
            % Check that fitted values don't have extreme outliers
            yhat_range = [min(regs.yhat), max(regs.yhat)];
            y_range = [min(y), max(y)];
            % Fitted values should be within reasonable range of observed data
            testCase.verifyGreaterThanOrEqual(yhat_range(1), y_range(1) - 1);
            testCase.verifyLessThanOrEqual(yhat_range(2), y_range(2) + 1);
        end
    end
end
