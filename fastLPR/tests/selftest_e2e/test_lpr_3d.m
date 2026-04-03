classdef test_lpr_3d < matlab.unittest.TestCase
    % test_lpr_3d - Unit tests for 3D local polynomial regression
    %
    % Run with: >> results = runtests('test_lpr_3d')
    %
    % Uses test_data_utils for reusable test data generation.
    %
    % Author: Ying Wang, Min Li
    % Copyright (c) 2020-2025 fastLPR Development Team
    % License: GNU General Public License v3.0

    % ARCHIVED: 2026-01-10 - test_order0_3d (moved to dev/archive/tests-archive-20260110/matlab/e2e/archived_test_lpr_3d.m)
    % ARCHIVED: 2026-01-10 - test_single_bandwidth_3d (moved to dev/archive/tests-archive-20260110/matlab/e2e/archived_test_lpr_3d.m)
    % ARCHIVED: 2026-01-10 - test_linear_3d (moved to dev/archive/tests-archive-20260110/matlab/e2e/archived_test_lpr_3d.m)

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
        function test_order1_3d(testCase)
            % Test 3D Local Linear (order 1) - single bandwidth, no GCV
            data = testCase.utils.generate_sincos_3d(300, 0.1, 42);
            h = [0.4, 0.4, 0.4];  % Single bandwidth, no GCV search
            opt = struct('order', 1, 'N', [20, 20, 20], 'calc_dof', false);
            regs = cv_fastlpr(data.x, data.y, h, opt);
            testCase.verifyNotEmpty(regs.yhat);
            testCase.verifyEqual(length(regs.yhat), 300);
            % Reasonable fit - use max absolute error per CLAUDE.md
            max_err = max(abs(regs.yhat - data.y_true));
            testCase.verifyLessThan(max_err, 1.0);  % 3D regression has larger errors
        end
    end
end
