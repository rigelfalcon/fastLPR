classdef test_kde_3d < matlab.unittest.TestCase
    % test_kde_3d - Unit tests for 3D kernel density estimation
    %
    % Run with: >> results = runtests('test_kde_3d')
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
            % 3D KDE - single bandwidth, no LCV search
            data = testCase.utils.generate_kde(300, 3, 42);
            h = [0.5, 0.5, 0.5];  % Single bandwidth
            opt = struct('order', 0, 'flag_power2', false);
            kde = cv_fastkde(data.x, h, opt);
            testCase.verifyNotEmpty(kde.fhat);
            testCase.verifyEqual(length(kde.h), 3);
        end

        % ARCHIVED: 2026-01-10 - "test_density_positive" (moved to dev/archive/tests-archive-20260110/matlab/e2e/archived_test_kde_3d.m)
    end
end
