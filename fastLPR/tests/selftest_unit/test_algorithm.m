classdef test_algorithm < matlab.unittest.TestCase
    % test_algorithm - Algorithm correctness tests
    %
    % Unified with Python/R test structure (10 tests):
    %   1. test_1d_small - 1D small sample (n=500), fastLPR vs naive NW
    %   2. test_1d_medium - 1D medium sample (n=2000)
    %   3. test_1d_large - 1D large sample (n=10000)
    %   4. test_2d_small - 2D small sample (n=400)
    %   5. test_2d_medium - 2D medium sample (n=1000)
    %   6. test_2d_large - 2D large sample (n=3000)
    %   7. test_complex_1d - Complex data 1D
    %   8. test_order0_vs_order1_smooth - Order 0 vs 1 for smooth data
    %   9. test_deterministic_output - Output is deterministic
    %   10. test_increasing_accuracy_improves_precision - Higher accuracy improves precision
    %
    % Matches:
    %   - Python: fastLPR_py/tests/test_algorithm.py
    %   - R: fastLPR_R/tests/testthat/test-unit-algorithm.R
    %
    % Author: fastLPR Development Team
    % Copyright (c) 2020-2025 fastLPR Development Team
    % License: GNU General Public License v3.0
    %
    % Run with: >> results = runtests('test_algorithm')

    properties (Constant)
        Accuracy = 6;  % Default NUFFT accuracy
        AccuracyComplex = 9;  % Higher accuracy for complex test
        RandomSeed = 42;  % Reproducibility seed
        MseTolerance = 0.05;  % MSE threshold for fastLPR vs naive NW
    end

    methods (TestClassSetup)
        function setupPath(testCase)
            testDir = fileparts(mfilename('fullpath'));
            repoRoot = fileparts(fileparts(fileparts(testDir)));
            addpath(fullfile(repoRoot, 'fastLPR'));
            fastlpr_setup();
        end
    end

    methods (Test)
        %% ================================================================
        %% Section 1: 1D Regression Tests
        %% ================================================================

        function test_1d_small(testCase)
            % UNIT: 1D small sample fastLPR matches naive NW (n=500)
            % Matches: Python test_1d_small, R test
            rng(testCase.RandomSeed);
            n = 500;
            x = abs(2*(rand(n, 1) - 0.5) * 15);
            y_true = besselj(1, x);
            y = y_true + 0.5 * std(y_true) * randn(size(y_true));
            h = 0.5;

            % Naive NW
            yhat_naive = NwSmooth(x, y, h, x);

            % fastLPR
            opt.order = 0;
            opt.accuracy = testCase.Accuracy;
            opt.calc_dof = false;
            opt.nufft_deconv = true;
            opt.verbose = false;
            regs = cv_fastlpr(x, y, h, opt);
            yhat_fast = regs.yhat;

            % Verify accuracy
            mse = mean((yhat_fast - yhat_naive).^2);
            testCase.verifyLessThan(mse, testCase.MseTolerance, ...
                sprintf('MSE=%.2e exceeds threshold %.2e for n=%d', mse, testCase.MseTolerance, n));
        end

        function test_1d_medium(testCase)
            % UNIT: 1D medium sample fastLPR matches naive NW (n=2000)
            % Matches: Python test_1d_medium, R test
            rng(testCase.RandomSeed);
            n = 2000;
            x = abs(2*(rand(n, 1) - 0.5) * 15);
            y_true = besselj(1, x);
            y = y_true + 0.5 * std(y_true) * randn(size(y_true));
            h = 0.5;

            % Naive NW
            yhat_naive = NwSmooth(x, y, h, x);

            % fastLPR
            opt.order = 0;
            opt.accuracy = testCase.Accuracy;
            opt.calc_dof = false;
            opt.nufft_deconv = true;
            opt.verbose = false;
            regs = cv_fastlpr(x, y, h, opt);
            yhat_fast = regs.yhat;

            % Verify accuracy
            mse = mean((yhat_fast - yhat_naive).^2);
            testCase.verifyLessThan(mse, testCase.MseTolerance, ...
                sprintf('MSE=%.2e exceeds threshold %.2e for n=%d', mse, testCase.MseTolerance, n));
        end

        function test_1d_large(testCase)
            % UNIT: 1D large sample fastLPR matches naive NW (n=10000)
            % Matches: Python test_1d_large, R test
            rng(testCase.RandomSeed);
            n = 10000;
            x = abs(2*(rand(n, 1) - 0.5) * 15);
            y_true = besselj(1, x);
            y = y_true + 0.5 * std(y_true) * randn(size(y_true));
            h = 0.5;

            % Naive NW
            yhat_naive = NwSmooth(x, y, h, x);

            % fastLPR
            opt.order = 0;
            opt.accuracy = testCase.Accuracy;
            opt.calc_dof = false;
            opt.nufft_deconv = true;
            opt.verbose = false;
            regs = cv_fastlpr(x, y, h, opt);
            yhat_fast = regs.yhat;

            % Verify accuracy
            mse = mean((yhat_fast - yhat_naive).^2);
            testCase.verifyLessThan(mse, testCase.MseTolerance, ...
                sprintf('MSE=%.2e exceeds threshold %.2e for n=%d', mse, testCase.MseTolerance, n));
        end

        %% ================================================================
        %% Section 2: 2D Regression Tests
        %% ================================================================

        function test_2d_small(testCase)
            % UNIT: 2D small sample fastLPR matches naive NW (n=400)
            % Matches: Python test_2d_small, R test
            rng(testCase.RandomSeed);
            n = 400;
            x = rand(n, 2) * 4 - 2;  % [-2, 2]^2
            y_true = sin(pi * x(:,1)) .* cos(pi * x(:,2));
            y = y_true + 0.2 * randn(n, 1);
            h = [0.3, 0.3];

            % Naive NW
            yhat_naive = NwSmooth(x, y, h, x);

            % fastLPR
            opt.order = 0;
            opt.accuracy = testCase.Accuracy;
            opt.calc_dof = false;
            opt.nufft_deconv = true;
            opt.verbose = false;
            regs = cv_fastlpr(x, y, h, opt);
            yhat_fast = regs.yhat;

            % Verify accuracy
            mse = mean((yhat_fast - yhat_naive).^2);
            testCase.verifyLessThan(mse, testCase.MseTolerance, ...
                sprintf('2D MSE=%.2e exceeds threshold %.2e for n=%d', mse, testCase.MseTolerance, n));
        end

        function test_2d_medium(testCase)
            % UNIT: 2D medium sample fastLPR matches naive NW (n=1000)
            % Matches: Python test_2d_medium, R test
            rng(testCase.RandomSeed);
            n = 1000;
            x = rand(n, 2) * 4 - 2;
            y_true = sin(pi * x(:,1)) .* cos(pi * x(:,2));
            y = y_true + 0.2 * randn(n, 1);
            h = [0.3, 0.3];

            % Naive NW
            yhat_naive = NwSmooth(x, y, h, x);

            % fastLPR
            opt.order = 0;
            opt.accuracy = testCase.Accuracy;
            opt.calc_dof = false;
            opt.nufft_deconv = true;
            opt.verbose = false;
            regs = cv_fastlpr(x, y, h, opt);
            yhat_fast = regs.yhat;

            % Verify accuracy
            mse = mean((yhat_fast - yhat_naive).^2);
            testCase.verifyLessThan(mse, testCase.MseTolerance, ...
                sprintf('2D MSE=%.2e exceeds threshold %.2e for n=%d', mse, testCase.MseTolerance, n));
        end

        function test_2d_large(testCase)
            % UNIT: 2D large sample fastLPR matches naive NW (n=3000)
            % Matches: Python test_2d_large, R test
            rng(testCase.RandomSeed);
            n = 3000;
            x = rand(n, 2) * 4 - 2;
            y_true = sin(pi * x(:,1)) .* cos(pi * x(:,2));
            y = y_true + 0.2 * randn(n, 1);
            h = [0.3, 0.3];

            % Naive NW
            yhat_naive = NwSmooth(x, y, h, x);

            % fastLPR
            opt.order = 0;
            opt.accuracy = testCase.Accuracy;
            opt.calc_dof = false;
            opt.nufft_deconv = true;
            opt.verbose = false;
            regs = cv_fastlpr(x, y, h, opt);
            yhat_fast = regs.yhat;

            % Verify accuracy
            mse = mean((yhat_fast - yhat_naive).^2);
            testCase.verifyLessThan(mse, testCase.MseTolerance, ...
                sprintf('2D MSE=%.2e exceeds threshold %.2e for n=%d', mse, testCase.MseTolerance, n));
        end

        %% ================================================================
        %% Section 3: Complex-valued Data Tests
        %% ================================================================

        function test_complex_1d(testCase)
            % UNIT: Complex-valued 1D regression
            % Matches: Python test_complex_1d, R test
            rng(testCase.RandomSeed);
            n = 500;
            x = rand(n, 1) * 2 * pi;
            y_true = exp(1i * x);  % Complex signal: e^(ix)
            noise = 0.1 * (randn(n, 1) + 1i * randn(n, 1));
            y = y_true + noise;
            h = 0.3;

            % Naive NW (works with complex data)
            yhat_naive = NwSmooth(x, y, h, x);

            % fastLPR with higher accuracy for complex
            opt.order = 0;
            opt.accuracy = testCase.AccuracyComplex;
            opt.calc_dof = false;
            opt.nufft_deconv = true;
            opt.verbose = false;
            regs = cv_fastlpr(x, y, h, opt);
            yhat_fast = regs.yhat;

            % Verify accuracy (use abs for complex error)
            mse = mean(abs(yhat_fast - yhat_naive).^2);
            testCase.verifyLessThan(mse, testCase.MseTolerance, ...
                sprintf('Complex MSE=%.2e exceeds threshold %.2e', mse, testCase.MseTolerance));
        end

        %% ================================================================
        %% Section 4: Consistency Tests
        %% ================================================================

        function test_order0_vs_order1_smooth(testCase)
            % UNIT: Order 0 and 1 give similar results for smooth data
            % Matches: Python test_order0_vs_order1_smooth, R test
            rng(testCase.RandomSeed);
            n = 500;
            x = rand(n, 1);
            y_true = sin(2 * pi * x);
            y = y_true + 0.1 * randn(n, 1);
            h = 0.15;

            % Order 0 (Nadaraya-Watson)
            opt0.order = 0;
            opt0.accuracy = testCase.Accuracy;
            opt0.calc_dof = false;
            opt0.verbose = false;
            regs0 = cv_fastlpr(x, y, h, opt0);

            % Order 1 (Local Linear)
            opt1.order = 1;
            opt1.accuracy = testCase.Accuracy;
            opt1.calc_dof = false;
            opt1.verbose = false;
            regs1 = cv_fastlpr(x, y, h, opt1);

            % Both should give finite values
            testCase.verifyTrue(all(isfinite(regs0.yhat)), ...
                'Order 0 should give finite values');
            testCase.verifyTrue(all(isfinite(regs1.yhat)), ...
                'Order 1 should give finite values');

            % Correlation should be high for smooth data
            corr_val = corr(regs0.yhat, regs1.yhat);
            testCase.verifyGreaterThan(corr_val, 0.9, ...
                sprintf('Order 0 and 1 correlation %.3f should be > 0.9', corr_val));
        end

        function test_deterministic_output(testCase)
            % UNIT: Same inputs give same outputs
            % Matches: Python test_deterministic_output, R test
            rng(testCase.RandomSeed);
            n = 200;
            x = rand(n, 1);
            y = sin(2 * pi * x) + 0.1 * randn(n, 1);
            h = 0.2;

            opt.order = 1;
            opt.accuracy = testCase.Accuracy;
            opt.calc_dof = false;
            opt.verbose = false;

            % Run twice with same inputs
            regs1 = cv_fastlpr(x, y, h, opt);
            regs2 = cv_fastlpr(x, y, h, opt);

            % Output should be identical
            max_diff = max(abs(regs1.yhat - regs2.yhat));
            testCase.verifyLessThan(max_diff, 1e-12, ...
                sprintf('Output should be deterministic, max diff=%.2e', max_diff));
        end

        function test_increasing_accuracy_improves_precision(testCase)
            % UNIT: Higher accuracy improves precision
            % Matches: Python test_increasing_accuracy_improves_precision, R test
            rng(testCase.RandomSeed);
            n = 500;
            x = rand(n, 1);
            y = sin(2 * pi * x);  % No noise for precision test
            h = 0.2;

            % Compute with different accuracy levels
            opt.order = 0;
            opt.calc_dof = false;
            opt.verbose = false;

            opt.accuracy = 4;
            regs4 = cv_fastlpr(x, y, h, opt);

            opt.accuracy = 6;
            regs6 = cv_fastlpr(x, y, h, opt);

            opt.accuracy = 9;
            regs9 = cv_fastlpr(x, y, h, opt);

            % Higher accuracy should be closer to highest accuracy result
            err_4 = max(abs(regs4.yhat - regs9.yhat));
            err_6 = max(abs(regs6.yhat - regs9.yhat));

            % Either err_6 < err_4, or both are very small (< 0.001)
            condition = (err_6 < err_4) || (err_4 < 0.001 && err_6 < 0.001);
            testCase.verifyTrue(condition, ...
                sprintf('acc=6 (%.2e) should be <= acc=4 (%.2e) or both < 0.001', err_6, err_4));
        end
    end
end
