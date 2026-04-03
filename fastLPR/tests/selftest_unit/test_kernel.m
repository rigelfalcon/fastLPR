classdef test_kernel < matlab.unittest.TestCase
    % test_kernel - Unit tests for kernel_function.m
    %
    % Matches Python test_kernel.py and R test-rcpp-kernel.R test cases.
    % Covers:
    % 1. Basic value tests (origin values for 1D/2D/3D)
    % 2. Normalization tests (integral should be ~1)
    % 3. Symmetry tests (K(x) = K(-x))
    % 4. Positivity tests (all values >= 0)
    % 5. Bandwidth edge cases (small/large bandwidth)
    % 6. Multi-dimensional tests (anisotropic bandwidth)
    %
    % Run with: >> results = runtests('test_kernel')
    %
    % Author: fastLPR Development Team
    % Copyright (c) 2020-2025 fastLPR Development Team
    % License: GNU General Public License v3.0

    properties (Constant)
        TOL_STRICT = 1e-12
        TOL_NUMERICAL = 1e-6
        TOL_NORMALIZATION = 0.02  % 2% tolerance for numerical integration (unified across languages)
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
    % Section 1: Basic Value Tests (Gaussian at Origin)
    % =========================================================================

    methods (Test)
        function test_gaussian_at_origin_1d(testCase)
            % UNIT: Gaussian kernel at origin should be 1/sqrt(2*pi)
            % Matches: Python test_gaussian_at_origin, R "1D kernel matches analytical"
            x = 0;
            K = kernel_function(x, 0, 1, 'gaussian');
            expected = 1/sqrt(2*pi);
            testCase.verifyEqual(K, expected, 'RelTol', testCase.TOL_STRICT, ...
                '1D Gaussian at origin should be 1/sqrt(2*pi)');
        end

        function test_gaussian_at_origin_2d(testCase)
            % UNIT: 2D Gaussian kernel at origin
            % Note: MATLAB kernel_function uses 1/sqrt(2*pi) normalization
            % (not dimension-dependent), then divides by det(h)
            x = [0, 0];
            K = kernel_function(x, [0, 0], [1, 1], 'gaussian');
            % MATLAB: K = (1/sqrt(2*pi)) * exp(0) / det([1,1]) = 1/sqrt(2*pi)
            expected = 1/sqrt(2*pi);
            testCase.verifyEqual(K, expected, 'RelTol', testCase.TOL_NUMERICAL, ...
                '2D Gaussian at origin with MATLAB normalization');
        end

        function test_gaussian_at_origin_3d(testCase)
            % UNIT: 3D Gaussian kernel at origin
            % Note: MATLAB kernel_function uses 1/sqrt(2*pi) normalization
            x = [0, 0, 0];
            K = kernel_function(x, [0, 0, 0], [1, 1, 1], 'gaussian');
            % MATLAB: K = (1/sqrt(2*pi)) * exp(0) / det([1,1,1]) = 1/sqrt(2*pi)
            expected = 1/sqrt(2*pi);
            testCase.verifyEqual(K, expected, 'RelTol', testCase.TOL_NUMERICAL, ...
                '3D Gaussian at origin with MATLAB normalization');
        end

        function test_gaussian_decay(testCase)
            % UNIT: Gaussian kernel should decay with distance from center
            % Matches: Python test_gaussian_decay
            x = (0:0.5:3)';
            K = kernel_function(x, 0, 1, 'gaussian');

            % Should be monotonically decreasing
            for i = 1:length(K)-1
                testCase.verifyGreaterThan(K(i), K(i+1), ...
                    'Gaussian should decay monotonically');
            end
        end

        function test_gaussian_tail_behavior(testCase)
            % UNIT: Gaussian tail should be very small at 5 sigma
            % Matches: Python test_gaussian_tail_behavior
            x = 5;  % 5 sigma
            K = kernel_function(x, 0, 1, 'gaussian');
            testCase.verifyLessThan(K, 1e-5, ...
                'Gaussian at 5 sigma should be < 1e-5');
        end
    end

    % =========================================================================
    % Section 2: Normalization Tests
    % =========================================================================

    methods (Test)
        function test_gaussian_normalization_1d(testCase)
            % UNIT: 1D Gaussian kernel should integrate to ~1
            % Matches: Python test_gaussian_normalization_1d, R "kernel integrates to approximately 1"

            % Create fine grid for numerical integration
            % Unified: 1001 points, [-5, 5] range (matches Python/R)
            x = linspace(-5, 5, 1001)';
            dx = x(2) - x(1);
            K = kernel_function(x, 0, 1, 'gaussian');
            integral = sum(K) * dx;

            testCase.verifyEqual(integral, 1.0, 'RelTol', testCase.TOL_NORMALIZATION, ...
                '1D Gaussian should integrate to 1');
        end

        function test_gaussian_normalization_with_bandwidth(testCase)
            % UNIT: Gaussian normalization should hold for different bandwidths
            % Matches: Python test_gaussian_normalization_with_bandwidth

            bandwidths = [0.5, 1.0, 2.0];
            for h = bandwidths
                % Unified: 1001 points, [-5h, 5h] range (matches Python/R)
                x = linspace(-5*h, 5*h, 1001)';
                dx = x(2) - x(1);
                K = kernel_function(x, 0, h, 'gaussian');
                integral = sum(K) * dx;

                testCase.verifyEqual(integral, 1.0, 'RelTol', testCase.TOL_NORMALIZATION, ...
                    sprintf('Gaussian with h=%g should integrate to 1', h));
            end
        end

        function test_epanechnikov_normalization_1d(testCase)
            % UNIT: 1D Epanechnikov kernel should integrate to ~1
            % Matches: Python test_epanechnikov_normalization_1d

            % Unified: 1001 points, [-1.5, 1.5] range (Epan has compact support)
            x = linspace(-1.5, 1.5, 1001)';
            dx = x(2) - x(1);
            K = kernel_function(x, 0, 1, 'epan');

            % Epanechnikov has compact support [-1, 1]
            integral = sum(K) * dx;
            testCase.verifyEqual(integral, 1.0, 'RelTol', testCase.TOL_NORMALIZATION, ...
                '1D Epanechnikov should integrate to 1');
        end
    end

    % =========================================================================
    % Section 3: Symmetry Tests
    % =========================================================================

    methods (Test)
        function test_gaussian_symmetry_1d(testCase)
            % UNIT: 1D Gaussian kernel should be symmetric: K(x) = K(-x)
            % Matches: Python test_gaussian_symmetry_1d, R "kernel is symmetric"

            x_pos = [0.5; 1.0; 2.0; 3.0];
            x_neg = -x_pos;

            K_pos = kernel_function(x_pos, 0, 1, 'gaussian');
            K_neg = kernel_function(x_neg, 0, 1, 'gaussian');

            testCase.verifyEqual(K_pos, K_neg, 'AbsTol', testCase.TOL_STRICT, ...
                '1D Gaussian should be symmetric');
        end

        function test_gaussian_symmetry_2d(testCase)
            % UNIT: 2D Gaussian kernel should be symmetric
            % Matches: Python test_gaussian_symmetry_2d

            x_pos = [1, 0.5; 0.5, 1; 1, 1];
            x_neg = -x_pos;

            K_pos = kernel_function(x_pos, [0, 0], [1, 1], 'gaussian');
            K_neg = kernel_function(x_neg, [0, 0], [1, 1], 'gaussian');

            testCase.verifyEqual(K_pos, K_neg, 'AbsTol', testCase.TOL_STRICT, ...
                '2D Gaussian should be symmetric');
        end

        function test_gaussian_symmetry_3d(testCase)
            % UNIT: 3D Gaussian kernel should be symmetric
            % Matches: Python test_gaussian_symmetry_3d

            x_pos = [1, 0.5, 0.3; 0.5, 1, 0.5; 1, 1, 1];
            x_neg = -x_pos;

            K_pos = kernel_function(x_pos, [0, 0, 0], [1, 1, 1], 'gaussian');
            K_neg = kernel_function(x_neg, [0, 0, 0], [1, 1, 1], 'gaussian');

            testCase.verifyEqual(K_pos, K_neg, 'AbsTol', testCase.TOL_STRICT, ...
                '3D Gaussian should be symmetric');
        end

        function test_epanechnikov_symmetry(testCase)
            % UNIT: Epanechnikov kernel should be symmetric
            % Matches: Python test_epanechnikov_symmetry

            x_pos = [0.3; 0.5; 0.8];
            x_neg = -x_pos;

            K_pos = kernel_function(x_pos, 0, 1, 'epan');
            K_neg = kernel_function(x_neg, 0, 1, 'epan');

            testCase.verifyEqual(K_pos, K_neg, 'AbsTol', testCase.TOL_STRICT, ...
                'Epanechnikov should be symmetric');
        end

        function test_gaussian_radial_symmetry_2d(testCase)
            % UNIT: 2D Gaussian with equal bandwidth should have radial symmetry
            % Matches: Python test_gaussian_radial_symmetry_2d

            % Points at same distance from origin
            r = 1.0;
            angles = [0, pi/4, pi/2, 3*pi/4, pi];
            x = r * [cos(angles)', sin(angles)'];

            K = kernel_function(x, [0, 0], [1, 1], 'gaussian');

            % All values should be equal
            testCase.verifyEqual(K, repmat(K(1), size(K)), 'AbsTol', testCase.TOL_STRICT, ...
                '2D Gaussian should have radial symmetry');
        end
    end

    % =========================================================================
    % Section 4: Positivity Tests
    % =========================================================================

    methods (Test)
        function test_gaussian_positivity(testCase)
            % UNIT: Gaussian kernel values should be non-negative
            % Matches: Python test_gaussian_positivity, R "kernel values are non-negative"

            rng(42);
            x = randn(100, 1);
            K = kernel_function(x, 0, 1, 'gaussian');

            testCase.verifyGreaterThanOrEqual(min(K), 0, ...
                'Gaussian values should be non-negative');
        end

        function test_gaussian_positivity_2d(testCase)
            % UNIT: 2D Gaussian kernel values should be non-negative
            % Matches: Python test_gaussian_positivity_2d

            rng(42);
            x = randn(100, 2);
            K = kernel_function(x, [0, 0], [1, 1], 'gaussian');

            testCase.verifyGreaterThanOrEqual(min(K), 0, ...
                '2D Gaussian values should be non-negative');
        end

        function test_epanechnikov_positivity(testCase)
            % UNIT: Epanechnikov kernel values should be non-negative
            % Matches: Python test_epanechnikov_positivity

            x = linspace(-2, 2, 101)';
            K = kernel_function(x, 0, 1, 'epan');

            testCase.verifyGreaterThanOrEqual(min(K), 0, ...
                'Epanechnikov values should be non-negative');
        end

        function test_kernel_maximum_at_origin(testCase)
            % UNIT: Gaussian kernel should have maximum at origin
            % Matches: R "kernel maximum is at origin"

            x = linspace(-3, 3, 101)';
            K = kernel_function(x, 0, 1, 'gaussian');

            [~, idx] = max(K);
            testCase.verifyEqual(x(idx), 0, 'AbsTol', 0.1, ...
                'Gaussian maximum should be at origin');
        end
    end

    % =========================================================================
    % Section 5: Bandwidth Edge Cases
    % =========================================================================

    methods (Test)
        function test_very_small_bandwidth(testCase)
            % UNIT: Kernel should handle very small bandwidth without NaN
            % Matches: Python test_very_small_bandwidth, R "handles small bandwidth"

            h = 0.01;
            x = linspace(-0.1, 0.1, 21)';
            K = kernel_function(x, 0, h, 'gaussian');

            testCase.verifyFalse(any(isnan(K)), ...
                'Small bandwidth should not produce NaN');
            testCase.verifyFalse(any(isinf(K)), ...
                'Small bandwidth should not produce Inf');
        end

        function test_very_large_bandwidth(testCase)
            % UNIT: Kernel should handle very large bandwidth
            % Matches: Python test_very_large_bandwidth, R "handles large bandwidth"

            h = 100;
            x = linspace(-10, 10, 101)';
            K = kernel_function(x, 0, h, 'gaussian');

            testCase.verifyFalse(any(isnan(K)), ...
                'Large bandwidth should not produce NaN');

            % With large bandwidth, kernel should be nearly constant
            testCase.verifyLessThan(std(K)/mean(K), 0.01, ...
                'Large bandwidth should give nearly constant kernel');
        end

        function test_bandwidth_scaling(testCase)
            % UNIT: Larger bandwidth should give wider (lower peak) kernel
            % Matches: Python test_bandwidth_scaling

            x = 0;
            h1 = 1;
            h2 = 2;

            K1 = kernel_function(x, 0, h1, 'gaussian');
            K2 = kernel_function(x, 0, h2, 'gaussian');

            % Larger bandwidth -> smaller peak (due to normalization)
            testCase.verifyGreaterThan(K1, K2, ...
                'Smaller bandwidth should give larger peak');
        end

        function test_multidimensional_bandwidth(testCase)
            % UNIT: Different bandwidths per dimension should work
            % Matches: Python test_multidimensional_bandwidth, R "anisotropic bandwidth"

            x = [0, 0];
            h = [1, 2];  % Different bandwidth per dimension

            K = kernel_function(x, [0, 0], h, 'gaussian');

            testCase.verifyFalse(isnan(K), ...
                'Anisotropic bandwidth should not produce NaN');
            testCase.verifyGreaterThan(K, 0, ...
                'Anisotropic bandwidth should give positive value');
        end
    end

    % =========================================================================
    % Section 6: Multi-dimensional Tests
    % =========================================================================

    methods (Test)
        function test_gaussian_1d_shape(testCase)
            % UNIT: 1D Gaussian should return column vector
            % Matches: Python test_gaussian_1d_shape

            x = rand(50, 1);
            K = kernel_function(x, 0, 1, 'gaussian');

            testCase.verifySize(K, [50, 1], ...
                '1D output should match input shape');
        end

        function test_gaussian_2d_shape(testCase)
            % UNIT: 2D Gaussian should return column vector for scattered data
            % Matches: Python test_gaussian_2d_scattered

            x = rand(50, 2);
            K = kernel_function(x, [0, 0], [1, 1], 'gaussian');

            testCase.verifySize(K, [50, 1], ...
                '2D scattered output should be column vector');
        end

        function test_gaussian_3d_shape(testCase)
            % UNIT: 3D Gaussian should return column vector for scattered data
            % Matches: R "3D kernel produces correct output dimensions"

            x = rand(50, 3);
            K = kernel_function(x, [0, 0, 0], [1, 1, 1], 'gaussian');

            testCase.verifySize(K, [50, 1], ...
                '3D scattered output should be column vector');
        end
    end

    % =========================================================================
    % Section 7: Epanechnikov Compact Support
    % =========================================================================

    methods (Test)
        function test_epanechnikov_compact_support(testCase)
            % UNIT: Epanechnikov should have compact support [-1, 1]
            % Matches: Python test_epanechnikov_compact_support

            x_inside = [-0.5; 0; 0.5];
            x_outside = [-2; 2];

            K_inside = kernel_function(x_inside, 0, 1, 'epan');
            K_outside = kernel_function(x_outside, 0, 1, 'epan');

            testCase.verifyGreaterThan(K_inside, eps, ...
                'Epanechnikov inside support should be positive');

            % Note: MATLAB implementation uses eps instead of 0 for outside
            testCase.verifyLessThan(K_outside, 0.01, ...
                'Epanechnikov outside support should be ~0');
        end
    end

    % =========================================================================
    % Section 8: Error Handling
    % =========================================================================

    methods (Test)
        function test_invalid_kernel_type(testCase)
            % UNIT: Invalid kernel type should error
            % Matches: Python test_invalid_kernel_type

            x = [0; 1; 2];
            testCase.verifyError(@() kernel_function(x, 0, 1, 'invalid_kernel'), ...
                'fastLPR:kernel:UnknownType');
        end

        function test_polynomial_without_order(testCase)
            % UNIT: Polynomial kernel without order should error
            % Matches: Python test_polynomial_without_order

            x = [0; 1; 2];
            testCase.verifyError(@() kernel_function(x, 0, 1, 'polynomial'), ...
                'fastLPR:kernel:MissingOrder');
        end
    end

    % =========================================================================
    % Section 9: Reproducibility
    % =========================================================================

    methods (Test)
        function test_deterministic_output(testCase)
            % UNIT: Same inputs should give same outputs
            % Matches: Python test_deterministic_output

            x = [0; 0.5; 1; 1.5; 2];

            K1 = kernel_function(x, 0, 1, 'gaussian');
            K2 = kernel_function(x, 0, 1, 'gaussian');

            testCase.verifyEqual(K1, K2, 'AbsTol', testCase.TOL_STRICT, ...
                'Kernel output should be deterministic');
        end
    end

end
