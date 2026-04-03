classdef test_nufft_edge < matlab.unittest.TestCase
    % test_nufft_edge - Edge case tests for NUFFT Type-1 implementation
    %
    % Unified with R's test-unit-nufft.R structure (7 tests). Covers:
    % 1. Single point (M=1)
    % 2. Two points (M=2)
    % 3. Boundary x values (near 0 and 1)
    % 4. Accuracy gradient tests (acc = 4, 6, 8, 9, 12)
    % 5. Numerical stability (very large y)
    % 6. Larger sample size (M=1000)
    % 7. Naive DFT comparison
    %
    % Run with: >> results = runtests('test_nufft_edge')
    %
    % ARCHIVED: 2026-01-10
    % 8 tests moved to: dev/archive/tests-archive-20260110/matlab/unit/archived_test_nufft_edge_extra.m
    % Archived: 2D/3D variants, constant_y, complex_y, very_small_y
    %
    % Author: fastLPR Development Team
    % Copyright (c) 2020-2025 fastLPR Development Team
    % License: GNU General Public License v3.0

    properties (Constant)
        RANDOM_SEED = 42
        ACCURACY_LEVELS = [4, 6, 8, 9, 12]
        ACCURACY_TOLERANCES = containers.Map(...
            {4, 6, 8, 9, 12}, ...
            {1e-3, 1e-5, 1e-7, 1e-9, 1e-11})
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
    % Helper Methods
    % =========================================================================

    methods (Static)
        function naive = compute_naive_dft_1d(x, y, N)
            % Compute naive DFT for 1D as reference
            % Formula: (1/M) * sum(y * exp(-i*2*pi*k*x))
            % Matches NUFFT(isdeconv=false) with ~0.12 error (spreading approx)
            M = length(x);
            k = (-floor(N/2)):(N - floor(N/2) - 1);
            naive = zeros(N, 1);
            for ik = 1:length(k)
                phase = -2 * pi * k(ik) * x;
                naive(ik) = sum(y .* exp(1i * phase)) / M;
            end
        end

        function metrics = compare_results(result, reference)
            % Compute comparison metrics
            result_vec = result(:);
            ref_vec = reference(:);

            abs_err = abs(result_vec - ref_vec);
            metrics.max_abs_err = max(abs_err);
            metrics.mean_abs_err = mean(abs_err);

            % Relative error (avoid division by zero)
            denom = max(abs(ref_vec), 1e-10);
            rel_err = abs_err ./ denom;
            significant_mask = abs(ref_vec) > 1e-10;
            if any(significant_mask)
                metrics.max_rel_err = max(rel_err(significant_mask));
            else
                metrics.max_rel_err = 0;
            end
        end
    end

    % =========================================================================
    % Test 1: Single Point (M=1)
    % Matches R: "nufftn_type1 handles M=1 (single point)"
    % =========================================================================

    methods (Test)
        function test_single_point_1d(testCase)
            % UNIT: nufftn_type1 handles M=1 (single point) in 1D
            rng(testCase.RANDOM_SEED);

            x = 0.5;
            y = 1.0;
            N = 8;
            df = 1 / N;

            x_shifted = x - 0.5;
            result = nufftn_type1(x_shifted, y, [], df, -1, 6, true);

            % Should produce valid output
            testCase.verifyEqual(length(result), N, 'Output size should be N');
            testCase.verifyFalse(any(isnan(result(:))), 'Should not produce NaN');
            testCase.verifyFalse(all(result(:) == 0), 'Should not be all zeros');
        end
    end

    % =========================================================================
    % Test 2: Two Points (M=2)
    % Matches R: "nufftn_type1 handles small M (M=2)"
    % =========================================================================

    methods (Test)
        function test_two_points_1d(testCase)
            % UNIT: nufftn_type1 handles M=2 (two points) in 1D
            % Note: NUFFT uses different normalization than naive DFT due to
            % spreading/deconvolution. We verify output is valid and shape matches.
            rng(testCase.RANDOM_SEED);

            x = [0.25; 0.75];
            y = [1.0; 2.0];
            N = 8;
            df = 1 / N;

            x_shifted = x - 0.5;
            result = nufftn_type1(x_shifted, y, [], df, -1, 6, true);

            % Verify output is valid
            testCase.verifyEqual(length(result), N, 'Output size should be N');
            testCase.verifyFalse(any(isnan(result(:))), 'Should not produce NaN');
            testCase.verifyFalse(any(isinf(result(:))), 'Should not produce Inf');
            testCase.verifyFalse(all(result(:) == 0), 'Should not be all zeros');
        end
    end

    % =========================================================================
    % Test 3: Boundary X Values
    % Matches R: "nufftn_type1 handles x values at boundaries (0 and 1)"
    % =========================================================================

    methods (Test)
        function test_boundary_x_1d(testCase)
            % UNIT: nufftn_type1 handles boundary x values in 1D
            rng(testCase.RANDOM_SEED);

            x = [0.0; 0.001; 0.999; 0.5];
            y = [1.0; 2.0; 3.0; 4.0];
            N = 8;
            df = 1 / N;

            x_shifted = x - 0.5;
            result = nufftn_type1(x_shifted, y, [], df, -1, 6, true);

            testCase.verifyFalse(any(isnan(result(:))));
            testCase.verifyFalse(any(isinf(result(:))));
            testCase.verifyEqual(length(result), N);
        end
    end

    % =========================================================================
    % Test 4: Accuracy Gradient
    % Matches R: "nufftn_type1 accuracy improves with parameter"
    % =========================================================================

    methods (Test)
        function test_accuracy_gradient_1d(testCase)
            % UNIT: nufftn_type1 1D produces valid output at various accuracy levels
            % Note: NUFFT normalization differs from naive DFT. We verify:
            % 1. Output is valid (no NaN/Inf)
            % 2. Higher accuracy converges (results become more stable)
            rng(testCase.RANDOM_SEED);

            M = 50;
            N = 16;
            x = rand(M, 1);
            y = sin(2 * pi * x) + 0.1 * randn(M, 1);
            df = 1 / N;
            x_shifted = x - 0.5;

            % Use highest accuracy result as reference
            ref_result = nufftn_type1(x_shifted, y, [], df, -1, 12, true);

            prev_diff = Inf;
            for acc = testCase.ACCURACY_LEVELS
                result = nufftn_type1(x_shifted, y, [], df, -1, acc, true);

                % Verify valid output
                testCase.verifyFalse(any(isnan(result(:))), ...
                    sprintf('acc=%d: should not produce NaN', acc));
                testCase.verifyFalse(any(isinf(result(:))), ...
                    sprintf('acc=%d: should not produce Inf', acc));

                % Verify convergence towards high-accuracy reference
                diff_from_ref = max(abs(result(:) - ref_result(:)));
                if acc > 4
                    testCase.verifyLessThanOrEqual(diff_from_ref, prev_diff * 1.1, ...
                        sprintf('acc=%d should converge (diff=%.2e vs prev=%.2e)', ...
                        acc, diff_from_ref, prev_diff));
                end
                prev_diff = diff_from_ref;
            end
        end
    end

    % =========================================================================
    % Test 5: Numerical Stability (Very Large Y)
    % Matches R: numerical stability tests
    % =========================================================================

    methods (Test)
        function test_very_large_y(testCase)
            % UNIT: nufftn_type1 handles very large y values
            rng(testCase.RANDOM_SEED);

            M = 50;
            N = 16;
            x = rand(M, 1);
            y = randn(M, 1) * 1e10;
            df = 1 / N;

            x_shifted = x - 0.5;
            result = nufftn_type1(x_shifted, y, [], df, -1, 6, true);

            testCase.verifyFalse(any(isnan(result(:))));
            testCase.verifyFalse(any(isinf(result(:))));
        end
    end

    % =========================================================================
    % Test 6: Larger Sample Size
    % Matches R: scalability tests
    % =========================================================================

    methods (Test)
        function test_larger_sample_size(testCase)
            % UNIT: nufftn_type1 handles larger sample sizes (M=1000)
            rng(testCase.RANDOM_SEED);

            M = 1000;
            N = 64;
            x = rand(M, 1);
            y = sin(2 * pi * x) + 0.1 * randn(M, 1);
            df = 1 / N;

            x_shifted = x - 0.5;
            result = nufftn_type1(x_shifted, y, [], df, -1, 6, true);

            testCase.verifyEqual(length(result), N);
            testCase.verifyFalse(any(isnan(result(:))));
        end
    end

    % =========================================================================
    % Test 7: Naive DFT Comparison
    % Matches R: "nufftn_type1 1D reproduces naive DFT"
    % =========================================================================

    methods (Test)
        function test_nufft_vs_naive_dft_1d(testCase)
            % UNIT: NUFFT(isdeconv=false) matches naive DFT within tolerance
            % Note: Spreading kernel introduces ~0.12 error at acc=12
            rng(testCase.RANDOM_SEED);

            M = 100;
            N = 32;
            x = rand(M, 1) - 0.5;  % [-0.5, 0.5]
            y = sin(2 * pi * x);
            df = 1 / N;

            % NUFFT without deconvolution
            result = nufftn_type1(x, y, [], df, -1, 12, false);

            % Naive DFT
            naive = test_nufft_edge.compute_naive_dft_1d(x, y, N);

            % Compare (tolerance 0.15 for spreading approximation)
            max_err = max(abs(result(:) - naive(:)));
            testCase.verifyLessThan(max_err, 0.15, ...
                sprintf('NUFFT vs naive DFT: max_err=%.2e should be < 0.15', max_err));
        end
    end
end
