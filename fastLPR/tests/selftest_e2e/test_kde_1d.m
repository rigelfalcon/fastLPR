classdef test_kde_1d < matlab.unittest.TestCase
    % test_kde_1d - Unit tests for 1D kernel density estimation
    %
    % Run with: >> results = runtests('test_kde_1d')
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
        function test_basic_functionality(testCase)
            data = testCase.utils.generate_kde(500, 1, 42);
            hlist = testCase.utils.generate_hlist([0.05, 1.0], 10, 1);
            opt = struct('order', 0);
            kde = cv_fastkde(data.x, hlist, opt);
            testCase.verifyNotEmpty(kde.fhat, 'fhat should not be empty');
            testCase.verifyGreaterThan(kde.h, 0, 'Bandwidth should be positive');
        end

        function test_density_positive(testCase)
            data = testCase.utils.generate_kde(500, 1, 42);
            kde = cv_fastkde(data.x, [0.2]);
            testCase.verifyTrue(all(kde.fhat(:) > 0), 'Density should be positive');
        end

        function test_bandwidth_selection(testCase)
            data = testCase.utils.generate_kde(500, 1, 42);
            hlist = testCase.utils.generate_hlist([0.05, 1.0], 20, 1);
            kde = cv_fastkde(data.x, hlist);
            testCase.verifyGreaterThan(kde.h, hlist(1));
            testCase.verifyLessThan(kde.h, hlist(end));
        end

        function test_reproducibility(testCase)
            data = testCase.utils.generate_kde(200, 1, 42);
            hlist = testCase.utils.generate_hlist([0.1, 0.5], 5, 1);
            kde1 = cv_fastkde(data.x, hlist);
            kde2 = cv_fastkde(data.x, hlist);
            testCase.verifyEqual(kde1.h, kde2.h, 'AbsTol', 1e-14);
        end

        function test_bimodal_distribution(testCase)
            % Test with bimodal data (from Python)
            rng(42);
            n = 400;
            x1 = randn(n/2, 1) - 2;
            x2 = randn(n/2, 1) + 2;
            x = [x1; x2];
            hlist = testCase.utils.generate_hlist([0.1, 1.5], 10, 1);
            opt = struct('order', 0);
            kde = cv_fastkde(x, hlist, opt);
            testCase.verifyNotEmpty(kde.fhat, 'fhat should not be empty');
            testCase.verifyTrue(all(kde.fhat(:) >= 0), 'Density should be non-negative');
        end

        function test_single_bandwidth(testCase)
            % Test with fixed bandwidth (from Python)
            rng(42);
            n = 200;
            x = randn(n, 1);
            h = 0.5;
            opt = struct('order', 0);
            kde = cv_fastkde(x, h, opt);
            testCase.verifyNotEmpty(kde.fhat, 'fhat should not be empty');
            testCase.verifyEqual(length(kde.fhat), n);
        end

        function test_density_integrates_to_1(testCase)
            % Verify integral approximately equals 1 (from R)
            rng(42);
            x = randn(500, 1);
            hlist = testCase.utils.generate_hlist([0.2, 0.5], 5, 1);
            opt = struct('order', 0, 'N', 200);
            kde = cv_fastkde(x, hlist, opt);
            % Numerical integration using trapezoidal rule
            dx = kde.xlist{1}(2) - kde.xlist{1}(1);
            integral_val = double(trapz(kde.xlist{1}, kde.fhat));
            testCase.verifyEqual(integral_val, 1.0, 'RelTol', 0.1, ...
                'Density should integrate to approximately 1');
        end

        function test_uniform_distribution(testCase)
            % Test with uniform data (from R)
            rng(42);
            x = rand(200, 1);  % Uniform [0, 1]
            hlist = testCase.utils.generate_hlist([0.05, 0.2], 5, 1);
            opt = struct('order', 0);
            kde = cv_fastkde(x, hlist, opt);
            % Density should be roughly uniform (close to 1)
            fhat_mean = mean(kde.fhat);
            testCase.verifyTrue(abs(fhat_mean - 1.0) < 0.3, ...
                'Uniform density should be close to 1');
        end
    end
end
