classdef test_kde_2d < matlab.unittest.TestCase
    % test_kde_2d - Unit tests for 2D kernel density estimation
    %
    % Run with: >> results = runtests('test_kde_2d')
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
            data = testCase.utils.generate_kde(500, 2, 42);
            hlist = testCase.utils.generate_hlist([0.1, 1.0], 10, 2);
            kde = cv_fastkde(data.x, hlist);
            testCase.verifyNotEmpty(kde.fhat);
            testCase.verifyEqual(length(kde.h), 2);
        end

        function test_density_positive(testCase)
            data = testCase.utils.generate_kde(300, 2, 42);
            kde = cv_fastkde(data.x, [0.3, 0.3]);
            % Allow small negative values due to numerical precision (NUFFT grid artifacts)
            testCase.verifyTrue(all(kde.fhat(:) > -1e-9));
        end

        function test_anisotropic_bandwidth(testCase)
            % Anisotropic data with different bandwidth ranges
            rng(42);
            x = [randn(500, 1)*0.5, randn(500, 1)*2];
            hlist = testCase.utils.generate_hlist([0.1, 0.5; 0.2, 1.0], 10, 2);
            kde = cv_fastkde(x, hlist);
            testCase.verifyNotEmpty(kde.fhat);
        end

        function test_correlated_data(testCase)
            % Test with correlated 2D data (from Python)
            rng(42);
            n = 300;
            x1 = randn(n, 1);
            x2 = 0.8*x1 + 0.6*randn(n, 1);
            x = [x1, x2];
            hlist = testCase.utils.generate_hlist([0.1, 1.0], 5, 2);
            opt = struct('order', 0);
            kde = cv_fastkde(x, hlist, opt);
            testCase.verifyNotEmpty(kde.fhat, 'fhat should not be empty');
            testCase.verifyTrue(all(kde.fhat(:) >= -1e-9), 'Density should be non-negative');
        end

        function test_single_bandwidth_2d(testCase)
            % Test with fixed bandwidth (from Python)
            rng(42);
            n = 200;
            x = randn(n, 2);
            h = [0.5, 0.5];
            opt = struct('order', 0);
            kde = cv_fastkde(x, h, opt);
            testCase.verifyNotEmpty(kde.fhat, 'fhat should not be empty');
            testCase.verifyNotEmpty(kde.xlist, 'xlist should not be empty');
        end

        function test_bandwidth_selection_2d(testCase)
            % LCV selection test (from R)
            rng(42);
            % Mixture of Gaussians
            n1 = 100; n2 = 100;
            x1 = [randn(n1, 1) - 1, randn(n1, 1) - 1] * 0.3;
            x2 = [randn(n2, 1) + 1, randn(n2, 1) + 1] * 0.5;
            x = [x1; x2];
            hlist = testCase.utils.generate_hlist([0.1, 1.0], 5, 2);
            opt = struct('order', 0);
            kde = cv_fastkde(x, hlist, opt);
            % Check bandwidth selection
            testCase.verifyTrue(all(kde.h >= 0.1), 'Bandwidth should be >= 0.1');
            testCase.verifyTrue(all(kde.h <= 1.0), 'Bandwidth should be <= 1.0');
            testCase.verifyTrue(isfield(kde, 'lcv'), 'Should have LCV results');
        end

        function test_density_integrates_to_1_2d(testCase)
            % Verify integral approximately equals 1 (from R)
            rng(42);
            x = randn(500, 2);
            hlist = testCase.utils.generate_hlist([0.3, 0.6], 3, 2);
            opt = struct('order', 0, 'N', [50, 50]);
            kde = cv_fastkde(x, hlist, opt);
            % Numerical integration using 2D trapezoidal rule
            dx1 = kde.xlist{1}(2) - kde.xlist{1}(1);
            dx2 = kde.xlist{2}(2) - kde.xlist{2}(1);
            integral_val = double(sum(kde.fhat(:)) * dx1 * dx2);
            testCase.verifyEqual(integral_val, 1.0, 'RelTol', 0.15, ...
                'Density should integrate to approximately 1');
        end
    end
end
