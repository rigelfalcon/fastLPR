classdef test_lpr_complex < matlab.unittest.TestCase
    % test_lpr_complex - Unit tests for complex-valued regression
    %
    % Run with: >> results = runtests('test_lpr_complex')
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
        function test_complex_1d(testCase)
            data = testCase.utils.generate_complex_1d(500, 0.2, 42);
            hlist = testCase.utils.generate_hlist([0.01, 0.5], 10, 1);
            opt = struct('order', 1, 'calc_dof', true);
            regs = cv_fastlpr(data.x, data.y, hlist, opt);
            testCase.verifyNotEmpty(regs.yhat);
            testCase.verifyTrue(~isreal(regs.yhat) || all(imag(regs.yhat) == 0));
        end

        function test_complex_bandwidth_selection(testCase)
            % Test with complex exponential function
            rng(42);
            x = sort(rand(500, 1));
            y = exp(1i*2*pi*x) + 0.1*(randn(500, 1) + 1i*randn(500, 1));
            hlist = testCase.utils.generate_hlist([0.01, 0.5], 15, 1);
            opt = struct('order', 1);
            regs = cv_fastlpr(x, y, hlist, opt);
            testCase.verifyGreaterThan(regs.gcv_yhat.h1se, hlist(1));
            testCase.verifyLessThan(regs.gcv_yhat.h1se, hlist(end));
        end

        function test_complex_gcv_real(testCase)
            data = testCase.utils.generate_complex_1d(500, 0.1, 42);
            hlist = testCase.utils.generate_hlist([0.01, 0.5], 10, 1);
            opt = struct('order', 1, 'calc_dof', true);
            regs = cv_fastlpr(data.x, data.y, hlist, opt);
            testCase.verifyTrue(isreal(regs.gcv_yhat.gcv_m));
        end

        function test_complex_correlation(testCase)
            % Real/imag correlation (from Python)
            rng(42);
            n = 200;
            x = sort(rand(n, 1));
            y_real = sin(2*pi*x);
            y_imag = cos(2*pi*x);
            y_true = complex(y_real, y_imag);
            noise = 0.1 * (randn(n, 1) + 1i*randn(n, 1));
            y = y_true + noise;
            hlist = testCase.utils.generate_hlist([0.05, 0.5], 5, 1);
            opt = struct('order', 1);
            regs = cv_fastlpr(x, y, hlist, opt);
            % Check real and imaginary parts separately
            corr_real = corr(real(regs.yhat), y_real);
            corr_imag = corr(imag(regs.yhat), y_imag);
            testCase.verifyGreaterThan(corr_real, 0.9, ...
                'Real part correlation should be > 0.9');
            testCase.verifyGreaterThan(corr_imag, 0.9, ...
                'Imaginary part correlation should be > 0.9');
        end

        function test_complex_single_bandwidth(testCase)
            % Fixed bandwidth (from Python)
            rng(42);
            n = 100;
            x = sort(rand(n, 1));
            y = sin(2*pi*x) + 1i*cos(2*pi*x);
            y = y + 0.1*(randn(n, 1) + 1i*randn(n, 1));
            h = 0.15;
            opt = struct('order', 1);
            regs = cv_fastlpr(x, y, h, opt);
            testCase.verifyNotEmpty(regs.yhat, 'yhat should not be empty');
            testCase.verifyTrue(~isreal(regs.yhat) || all(imag(regs.yhat) == 0), ...
                'Output should be complex');
        end
    end
end
