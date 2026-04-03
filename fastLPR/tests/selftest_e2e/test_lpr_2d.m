classdef test_lpr_2d < matlab.unittest.TestCase
    % test_lpr_2d - Unit tests for 2D local polynomial regression
    %
    % Run with: >> results = runtests('test_lpr_2d')
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
        function test_order0_2d(testCase)
            % Test 2D Nadaraya-Watson (order 0)
            data = testCase.utils.generate_sincos_2d(500, 0.2, 42);
            hlist = testCase.utils.generate_hlist([0.05, 0.5], 10, 2);
            opt = struct('order', 0, 'calc_dof', true);
            regs = cv_fastlpr(data.x, data.y, hlist, opt);
            testCase.verifyNotEmpty(regs.yhat);
            testCase.verifyEqual(length(regs.gcv_yhat.h1se), 2);
        end

        function test_order1_2d(testCase)
            data = testCase.utils.generate_sincos_2d(500, 0.2, 42);
            hlist = testCase.utils.generate_hlist([0.05, 0.5], 10, 2);
            opt = struct('order', 1, 'calc_dof', true);
            regs = cv_fastlpr(data.x, data.y, hlist, opt);
            testCase.verifyNotEmpty(regs.yhat);
            testCase.verifyEqual(length(regs.gcv_yhat.h1se), 2);
        end

        function test_order2_2d(testCase)
            data = testCase.utils.generate_sincos_2d(500, 0.2, 42);
            hlist = testCase.utils.generate_hlist([0.05, 0.5], 10, 2);
            opt = struct('order', 2, 'calc_dof', true);
            regs = cv_fastlpr(data.x, data.y, hlist, opt);
            testCase.verifyNotEmpty(regs.yhat);
        end

        function test_anisotropic_data(testCase)
            % Test with anisotropic data and different bandwidth ranges per dimension
            rng(42);
            x = [rand(500, 1)*0.5, rand(500, 1)*2];
            y = x(:,1).^2 + x(:,2) + 0.1*randn(500, 1);
            % Different ranges for each dimension
            hlist = testCase.utils.generate_hlist([0.02, 0.3; 0.1, 0.8], 10, 2);
            opt = struct('order', 1);
            regs = cv_fastlpr(x, y, hlist, opt);
            testCase.verifyNotEmpty(regs.yhat);
        end

        function test_gcv_bandwidth_selection(testCase)
            % GCV selection test (from R)
            rng(42);
            n = 400;
            x = 2*rand(n, 2) - 1;  % [-1, 1]^2
            y_true = sin(pi*x(:,1)) .* cos(pi*x(:,2));
            y = y_true + 0.1*randn(n, 1);
            hlist = testCase.utils.generate_hlist([0.1, 0.6], 5, 2);
            opt = struct('order', 1, 'N', [50, 50]);
            regs = cv_fastlpr(x, y, hlist, opt);
            % Check GCV results
            testCase.verifyTrue(isfield(regs, 'gcv_yhat'), 'Should have gcv_yhat');
            testCase.verifyTrue(isfield(regs.gcv_yhat, 'gcv_m'), 'Should have gcv_m');
            testCase.verifyTrue(isfield(regs.gcv_yhat, 'h1se'), 'Should have h1se');
            testCase.verifyEqual(length(regs.gcv_yhat.h1se), 2, ...
                'h1se should have 2 elements for 2D');
            % Selected bandwidth should be in range
            testCase.verifyTrue(all(regs.gcv_yhat.h1se >= 0.1), ...
                'Bandwidth should be >= 0.1');
            testCase.verifyTrue(all(regs.gcv_yhat.h1se <= 0.6), ...
                'Bandwidth should be <= 0.6');
        end
    end
end
