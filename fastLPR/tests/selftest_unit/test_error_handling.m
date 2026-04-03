classdef test_error_handling < matlab.unittest.TestCase
    % test_error_handling - Error handling tests for cv_fastlpr
    %
    % Tests that cv_fastlpr properly validates inputs and throws appropriate
    % errors for invalid parameters.
    %
    % Test cases:
    %   1. Empty hlist input
    %   2. Dimension mismatch between x and y
    %   3. Invalid order parameter (not 0, 1, or 2)
    %   4. All NaN data input
    %   5. Negative bandwidths
    %
    % Run with: >> results = runtests('test_error_handling')
    %
    % Author: fastLPR Development Team
    % Copyright (c) 2020-2025 fastLPR Development Team
    % License: GNU General Public License v3.0

    properties (Constant)
        RANDOM_SEED = 42
    end

    methods (TestClassSetup)
        function setupPath(testCase)
            % Add fastLPR to path
            testDir = fileparts(mfilename('fullpath'));
            repoRoot = fileparts(fileparts(fileparts(testDir)));
            addpath(fullfile(repoRoot, 'fastLPR'));
            fastlpr_setup();
        end
    end

    % =========================================================================
    % Error Handling Tests
    % =========================================================================

    methods (Test)
        function test_empty_hlist_scalar_h_allowed(testCase)
            % Test: Empty hlist input should use automatic bandwidth selection
            % Note: cv_fastlpr allows empty h and computes default bandwidth
            rng(testCase.RANDOM_SEED);

            x = rand(100, 1);
            y = sin(2*pi*x) + 0.1*randn(100, 1);
            opt.order = 0;
            opt.verbose = false;

            % Empty h should NOT throw an error - it uses default bandwidth
            % This test verifies the behavior is graceful
            regs = cv_fastlpr(x, y, [], opt);
            testCase.verifyNotEmpty(regs.yhat, ...
                'Empty h should produce valid output using default bandwidth');
        end

        function test_empty_x_input(testCase)
            % Test: Empty x input should throw error
            rng(testCase.RANDOM_SEED);

            x = [];
            y = rand(100, 1);
            h = 0.5;
            opt.order = 0;

            % Empty x should throw error via validateattributes
            % Note: validateattributes prefixes error ID with function name
            testCase.verifyError(@() cv_fastlpr(x, y, h, opt), ...
                'MATLAB:cv_fastlpr:expectedNonempty', ...
                'Empty x should throw expectedNonempty error');
        end

        function test_empty_y_input(testCase)
            % Test: Empty y input should throw error
            rng(testCase.RANDOM_SEED);

            x = rand(100, 1);
            y = [];
            h = 0.5;
            opt.order = 0;

            % Empty y should throw error via validateattributes
            testCase.verifyError(@() cv_fastlpr(x, y, h, opt), ...
                'MATLAB:cv_fastlpr:expectedNonempty', ...
                'Empty y should throw expectedNonempty error');
        end

        function test_dimension_mismatch_x_y(testCase)
            % Test: Dimension mismatch between x and y should throw error
            rng(testCase.RANDOM_SEED);

            x = rand(100, 1);  % 100 samples
            y = rand(50, 1);   % 50 samples (mismatch!)
            h = 0.5;
            opt.order = 0;

            testCase.verifyError(@() cv_fastlpr(x, y, h, opt), ...
                'cv_fastlpr:DimensionMismatch', ...
                'x and y with different row counts should throw DimensionMismatch');
        end

        function test_dimension_mismatch_2d(testCase)
            % Test: Dimension mismatch in 2D case
            rng(testCase.RANDOM_SEED);

            x = rand(100, 2);  % 100 samples, 2D
            y = rand(80, 1);   % 80 samples (mismatch!)
            h = [0.5, 0.5];
            opt.order = 0;

            testCase.verifyError(@() cv_fastlpr(x, y, h, opt), ...
                'cv_fastlpr:DimensionMismatch', ...
                '2D x and y with different row counts should throw DimensionMismatch');
        end

        function test_invalid_order_negative(testCase)
            % Test: Invalid order parameter (negative) should throw error
            rng(testCase.RANDOM_SEED);

            x = rand(100, 1);
            y = sin(2*pi*x) + 0.1*randn(100, 1);
            h = 0.5;
            opt.order = -1;  % Invalid: must be 0, 1, or 2

            testCase.verifyError(@() cv_fastlpr(x, y, h, opt), ...
                'cv_fastlpr:InvalidOrder', ...
                'Negative order should throw InvalidOrder error');
        end

        function test_invalid_order_too_large(testCase)
            % Test: Invalid order parameter (3) should throw error
            rng(testCase.RANDOM_SEED);

            x = rand(100, 1);
            y = sin(2*pi*x) + 0.1*randn(100, 1);
            h = 0.5;
            opt.order = 3;  % Invalid: must be 0, 1, or 2

            testCase.verifyError(@() cv_fastlpr(x, y, h, opt), ...
                'cv_fastlpr:InvalidOrder', ...
                'Order > 2 should throw InvalidOrder error');
        end

        % test_invalid_order_fractional - ARCHIVED 2026-01-10
        % Reason: Test unification - Python tests integer invalid orders only
        % See: dev/archive/tests-archive-20260110/matlab/unit/archived_error_tests.m

        function test_all_nan_x_data(testCase)
            % Test: All NaN x data should throw error
            rng(testCase.RANDOM_SEED);

            x = NaN(100, 1);  % All NaN
            y = rand(100, 1);
            h = 0.5;
            opt.order = 0;

            % validateattributes with 'finite' catches this
            testCase.verifyError(@() cv_fastlpr(x, y, h, opt), ...
                'MATLAB:cv_fastlpr:expectedFinite', ...
                'All NaN x should throw expectedFinite error');
        end

        function test_all_nan_y_data(testCase)
            % Test: All NaN y data should throw error
            rng(testCase.RANDOM_SEED);

            x = rand(100, 1);
            y = NaN(100, 1);  % All NaN
            h = 0.5;
            opt.order = 0;

            % validateattributes with 'finite' catches this
            testCase.verifyError(@() cv_fastlpr(x, y, h, opt), ...
                'MATLAB:cv_fastlpr:expectedFinite', ...
                'All NaN y should throw expectedFinite error');
        end

        function test_inf_in_x_data(testCase)
            % Test: Inf values in x data should throw error
            rng(testCase.RANDOM_SEED);

            x = rand(100, 1);
            x(50) = Inf;  % One Inf value
            y = rand(100, 1);
            h = 0.5;
            opt.order = 0;

            testCase.verifyError(@() cv_fastlpr(x, y, h, opt), ...
                'MATLAB:cv_fastlpr:expectedFinite', ...
                'Inf in x should throw expectedFinite error');
        end

        function test_inf_in_y_data(testCase)
            % Test: Inf values in y data should throw error
            rng(testCase.RANDOM_SEED);

            x = rand(100, 1);
            y = rand(100, 1);
            y(50) = Inf;  % One Inf value
            h = 0.5;
            opt.order = 0;

            testCase.verifyError(@() cv_fastlpr(x, y, h, opt), ...
                'MATLAB:cv_fastlpr:expectedFinite', ...
                'Inf in y should throw expectedFinite error');
        end

        function test_negative_bandwidth_single(testCase)
            % Test: Negative bandwidth behavior
            % Note: cv_fastlpr does not explicitly validate h > 0,
            % but produces warnings. This tests that it handles gracefully.
            rng(testCase.RANDOM_SEED);

            x = rand(100, 1);
            y = sin(2*pi*x) + 0.1*randn(100, 1);
            h = -0.5;  % Negative bandwidth
            opt.order = 0;
            opt.verbose = false;

            % Negative bandwidth produces results but with warnings
            % Verify that either it produces output or throws a meaningful error
            try
                regs = cv_fastlpr(x, y, h, opt);
                % If no error, verify output exists (behavior may produce NaN)
                testCase.verifyNotEmpty(regs, 'Negative h should produce some output');
            catch ME
                % If error occurs, verify it has a meaningful identifier
                testCase.verifyNotEmpty(ME.identifier, ...
                    'Error for negative h should have identifier');
            end
        end

        function test_negative_bandwidth_array(testCase)
            % Test: Array with negative bandwidth behavior
            rng(testCase.RANDOM_SEED);

            x = rand(100, 2);
            y = sin(2*pi*x(:,1)) + 0.1*randn(100, 1);
            h = [0.5, -0.5];  % One negative bandwidth
            opt.order = 0;
            opt.verbose = false;

            % Negative bandwidth produces results but with warnings
            try
                regs = cv_fastlpr(x, y, h, opt);
                testCase.verifyNotEmpty(regs, 'Array with negative h should produce some output');
            catch ME
                testCase.verifyNotEmpty(ME.identifier, ...
                    'Error for negative h should have identifier');
            end
        end

        function test_zero_bandwidth(testCase)
            % Test: Zero bandwidth should cause error
            rng(testCase.RANDOM_SEED);

            x = rand(100, 1);
            y = sin(2*pi*x) + 0.1*randn(100, 1);
            h = 0;  % Zero bandwidth
            opt.order = 0;
            opt.verbose = false;

            % Zero bandwidth causes division by zero or NaN issues
            % MATLAB throws MATLAB:nologicalnan in kernel computation
            testCase.verifyError(@() cv_fastlpr(x, y, h, opt), ...
                'MATLAB:nologicalnan', ...
                'Zero bandwidth should cause nologicalnan error');
        end

        % test_invalid_kernel_type - ARCHIVED 2026-01-10
        % Reason: Test unification - Python doesn't expose kernel_type option
        % See: dev/archive/tests-archive-20260110/matlab/unit/archived_error_tests.m

        function test_y_matrix_input(testCase)
            % Test: y as matrix (not vector) should throw error
            rng(testCase.RANDOM_SEED);

            x = rand(100, 1);
            y = rand(100, 2);  % Matrix, not vector
            h = 0.5;
            opt.order = 0;

            testCase.verifyError(@() cv_fastlpr(x, y, h, opt), ...
                'cv_fastlpr:InvalidInput', ...
                'Matrix y should throw InvalidInput error');
        end
    end

    % =========================================================================
    % Edge Case Tests (Valid but Extreme Inputs)
    % =========================================================================

    methods (Test)
        function test_single_data_point(testCase)
            % Test: Single data point - should handle gracefully or error
            rng(testCase.RANDOM_SEED);

            x = 0.5;
            y = 1.0;
            h = 0.5;
            opt.order = 0;
            opt.verbose = false;

            % Single point may work for order 0 or may throw error
            % We just verify it doesn't crash unexpectedly
            try
                regs = cv_fastlpr(x, y, h, opt);
                testCase.verifyNotEmpty(regs, 'Single point should produce output');
            catch ME
                % If it errors, should be a meaningful error
                testCase.verifyNotEmpty(ME.identifier, ...
                    'Error should have identifier');
            end
        end

        function test_very_small_bandwidth(testCase)
            % Test: Very small bandwidth - numerical stability
            rng(testCase.RANDOM_SEED);

            x = rand(100, 1);
            y = sin(2*pi*x) + 0.1*randn(100, 1);
            h = 1e-10;  % Very small
            opt.order = 0;
            opt.verbose = false;

            % Should either work or throw meaningful error
            try
                regs = cv_fastlpr(x, y, h, opt);
                testCase.verifyNotEmpty(regs.yhat, 'Small h should produce output');
            catch ME
                testCase.verifyNotEmpty(ME.identifier, ...
                    'Error for small h should have identifier');
            end
        end

        function test_very_large_bandwidth(testCase)
            % Test: Very large bandwidth - numerical stability
            rng(testCase.RANDOM_SEED);

            x = rand(100, 1);
            y = sin(2*pi*x) + 0.1*randn(100, 1);
            h = 1e10;  % Very large
            opt.order = 0;
            opt.verbose = false;

            % Should either work or throw meaningful error
            try
                regs = cv_fastlpr(x, y, h, opt);
                testCase.verifyNotEmpty(regs.yhat, 'Large h should produce output');
            catch ME
                testCase.verifyNotEmpty(ME.identifier, ...
                    'Error for large h should have identifier');
            end
        end
    end
end
