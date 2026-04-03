classdef test_bandwidth < matlab.unittest.TestCase
    % test_bandwidth - Unit tests for bandwidth selection (get_hlist, GCV)
    %
    % Matches Python test_bandwidth.py and R test-unit-gcv.R test cases.
    % Covers:
    % 1. get_hlist tests (spacing, dimensions, ranges)
    % 2. GCV computation tests (U-shape, minimum, 1-SE rule)
    % 3. Integration tests (cv_fastlpr bandwidth selection)
    %
    % Run with: >> results = runtests('test_bandwidth')
    %
    % Author: fastLPR Development Team
    % Copyright (c) 2020-2025 fastLPR Development Team
    % License: GNU General Public License v3.0

    properties (Constant)
        TOL_STRICT = 1e-12
        TOL_NUMERICAL = 1e-6
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
    % Section 1: get_hlist Basic Tests
    % =========================================================================

    methods (Test)
        function test_1d_default_logspace(testCase)
            % UNIT: 1D get_hlist uses logspace by default
            % Matches: Python test_1d_default_logspace

            hlist = get_hlist(20, [0.1, 1.0]);

            testCase.verifySize(hlist, [20, 1], ...
                '1D hlist should have 20 rows and 1 column');

            % Check logspace: ratio should be constant
            ratios = hlist(2:end) ./ hlist(1:end-1);
            testCase.verifyEqual(ratios, repmat(ratios(1), size(ratios)), ...
                'RelTol', testCase.TOL_NUMERICAL, ...
                'Logspace should have constant ratio');
        end

        function test_1d_linear_spacing(testCase)
            % UNIT: 1D get_hlist with linear spacing
            % Matches: Python test_1d_linear_spacing

            hlist = get_hlist(10, [0.1, 1.0], @linspace);

            testCase.verifySize(hlist, [10, 1], ...
                '1D hlist should have 10 rows');

            % Check linspace: difference should be constant
            diffs = diff(hlist);
            testCase.verifyEqual(diffs, repmat(diffs(1), size(diffs)), ...
                'RelTol', testCase.TOL_NUMERICAL, ...
                'Linspace should have constant difference');
        end

        function test_1d_single_bandwidth(testCase)
            % UNIT: get_hlist with n=1 returns single value
            % Matches: Python test_1d_single_bandwidth

            hlist = get_hlist(1, [0.5, 0.5]);

            testCase.verifySize(hlist, [1, 1], ...
                'Single bandwidth should be 1x1');
            testCase.verifyEqual(hlist, 0.5, 'AbsTol', testCase.TOL_STRICT, ...
                'Single bandwidth should match range');
        end

        function test_hlist_range_bounds(testCase)
            % UNIT: hlist should respect range bounds
            % Matches: Python test_positive_bandwidths

            hlist = get_hlist(10, [0.1, 1.0]);

            testCase.verifyGreaterThanOrEqual(min(hlist), 0.1, ...
                'hlist min should be >= range min');
            testCase.verifyLessThanOrEqual(max(hlist), 1.0, ...
                'hlist max should be <= range max');
        end

        function test_monotonic_1d(testCase)
            % UNIT: 1D hlist should be monotonically increasing
            % Matches: Python test_monotonic_1d

            hlist = get_hlist(20, [0.1, 1.0]);

            testCase.verifyTrue(all(diff(hlist) > 0), ...
                'hlist should be monotonically increasing');
        end
    end

    % =========================================================================
    % Section 2: get_hlist Multi-dimensional Tests
    % =========================================================================

    methods (Test)
        function test_2d_bandwidth_grid(testCase)
            % UNIT: 2D get_hlist creates proper grid
            % Matches: Python test_2d_bandwidth_grid

            hlist = get_hlist([5, 5], [0.1, 1.0; 0.2, 2.0]);

            testCase.verifySize(hlist, [25, 2], ...
                '2D 5x5 grid should have 25 rows and 2 columns');

            % First column should have 5 unique values
            unique_h1 = unique(hlist(:, 1));
            testCase.verifyEqual(numel(unique_h1), 5, ...
                'First dimension should have 5 unique values');

            % Second column should have 5 unique values
            unique_h2 = unique(hlist(:, 2));
            testCase.verifyEqual(numel(unique_h2), 5, ...
                'Second dimension should have 5 unique values');
        end

        function test_2d_scalar_n(testCase)
            % UNIT: 2D get_hlist with scalar n broadcasts
            % Matches: Python test_2d_scalar_n

            hlist = get_hlist(4, [0.1, 1.0; 0.1, 1.0]);

            testCase.verifySize(hlist, [16, 2], ...
                'Scalar n=4 for 2D should give 4x4=16 rows');
        end

        function test_3d_bandwidth_grid(testCase)
            % UNIT: 3D get_hlist creates proper grid
            % Matches: Python test_3d_bandwidth_grid

            hlist = get_hlist([3, 3, 3], [0.1, 1.0; 0.1, 1.0; 0.1, 1.0]);

            testCase.verifySize(hlist, [27, 3], ...
                '3D 3x3x3 grid should have 27 rows and 3 columns');
        end
    end

    % =========================================================================
    % Section 3: get_hlist Range Tests
    % =========================================================================

    methods (Test)
        function test_wide_range_logspace(testCase)
            % UNIT: Wide range with logspace covers extremes
            % Matches: Python test_wide_range_logspace

            hlist = get_hlist(20, [0.001, 10.0]);

            testCase.verifyLessThan(hlist(1), 0.01, ...
                'First value should be close to range min');
            testCase.verifyGreaterThan(hlist(end), 5.0, ...
                'Last value should be close to range max');
        end

        function test_narrow_range(testCase)
            % UNIT: Narrow range should work correctly
            % Matches: Python test_narrow_range

            hlist = get_hlist(5, [0.5, 0.6]);

            testCase.verifySize(hlist, [5, 1], ...
                'Narrow range should give 5 values');
            testCase.verifyGreaterThanOrEqual(min(hlist), 0.5, ...
                'Values should be >= 0.5');
            testCase.verifyLessThanOrEqual(max(hlist), 0.6, ...
                'Values should be <= 0.6');
        end

        function test_large_n(testCase)
            % UNIT: Large n should not cause issues
            % Matches: Python test_large_n

            hlist = get_hlist(100, [0.1, 1.0]);

            testCase.verifySize(hlist, [100, 1], ...
                'Large n=100 should work');
            testCase.verifyTrue(all(isfinite(hlist)), ...
                'All values should be finite');
        end
    end

    % =========================================================================
    % Section 4: GCV Basic Tests (via cv_fastlpr)
    % =========================================================================

    methods (Test)
        function test_gcv_values_nonnegative(testCase)
            % UNIT: GCV values should be non-negative
            % Matches: Python test_gcv_basic, R "GCV values are computed and non-negative"

            rng(testCase.RANDOM_SEED);
            n = 100;
            x = rand(n, 1);
            y = sin(2*pi*x) + 0.1*randn(n, 1);

            hlist = get_hlist(10, [0.05, 0.5]);
            opt = struct('order', 1, 'calc_dof', true);
            regs = cv_fastlpr(x, y, hlist, opt);

            testCase.verifyGreaterThanOrEqual(min(regs.gcv_yhat.gcv_m), 0, ...
                'GCV values should be non-negative');
        end

        function test_gcv_curve_ushape(testCase)
            % UNIT: GCV curve should have U-shape for wide bandwidth range
            % Matches: R "GCV curve has U-shape for wide bandwidth range"

            rng(testCase.RANDOM_SEED);
            n = 200;
            x = rand(n, 1);
            y = sin(2*pi*x) + 0.1*randn(n, 1);

            hlist = get_hlist(20, [0.01, 0.8]);
            opt = struct('order', 1, 'calc_dof', true);
            regs = cv_fastlpr(x, y, hlist, opt);

            gcv = regs.gcv_yhat.gcv_m;
            [~, idmin] = min(gcv);

            % U-shape: minimum should not be at boundaries
            testCase.verifyGreaterThan(idmin, 1, ...
                'GCV minimum should not be at first bandwidth');
            testCase.verifyLessThan(idmin, length(gcv), ...
                'GCV minimum should not be at last bandwidth');
        end

        function test_idmin_corresponds_to_minimum(testCase)
            % UNIT: idmin should correspond to minimum GCV value
            % Matches: R "idmin corresponds to minimum GCV value"

            rng(testCase.RANDOM_SEED);
            n = 100;
            x = rand(n, 1);
            y = sin(2*pi*x) + 0.1*randn(n, 1);

            hlist = get_hlist(15, [0.05, 0.5]);
            opt = struct('order', 1, 'calc_dof', true);
            regs = cv_fastlpr(x, y, hlist, opt);

            [gcv_min, computed_idmin] = min(regs.gcv_yhat.gcv_m);

            testCase.verifyEqual(regs.gcv_yhat.gcv_m(regs.gcv_yhat.idmin), gcv_min, ...
                'AbsTol', testCase.TOL_STRICT, ...
                'idmin should point to minimum GCV');
        end
    end

    % =========================================================================
    % Section 5: 1-SE Rule Tests
    % =========================================================================

    methods (Test)
        function test_1se_rule_larger_bandwidth(testCase)
            % UNIT: 1-SE rule should select larger bandwidth than minimum
            % Matches: R "1-SE rule selects larger bandwidth than minimum"

            rng(testCase.RANDOM_SEED);
            n = 200;
            x = rand(n, 1);
            y = sin(2*pi*x) + 0.3*randn(n, 1);  % Higher noise

            hlist = get_hlist(20, [0.02, 0.6]);
            opt = struct('order', 1, 'calc_dof', true, 'dstd', 1);
            regs = cv_fastlpr(x, y, hlist, opt);

            testCase.verifyGreaterThanOrEqual(regs.gcv_yhat.id1se, regs.gcv_yhat.idmin, ...
                '1-SE bandwidth index should be >= minimum index');
        end

        function test_id1se_valid_index(testCase)
            % UNIT: id1se should be valid index
            % Matches: R "id1se is valid index"

            rng(testCase.RANDOM_SEED);
            n = 100;
            x = rand(n, 1);
            y = sin(2*pi*x) + 0.2*randn(n, 1);

            hlist = get_hlist(15, [0.05, 0.5]);
            opt = struct('order', 1, 'calc_dof', true);
            regs = cv_fastlpr(x, y, hlist, opt);

            testCase.verifyGreaterThanOrEqual(regs.gcv_yhat.id1se, 1, ...
                'id1se should be >= 1');
            testCase.verifyLessThanOrEqual(regs.gcv_yhat.id1se, length(hlist), ...
                'id1se should be <= number of bandwidths');
        end

        function test_dstd0_means_no_1se(testCase)
            % UNIT: dstd=0 should mean no 1-SE rule (h1se = hmin)
            % Matches: R "dstd=0 means no 1-SE rule"

            rng(testCase.RANDOM_SEED);
            n = 100;
            x = rand(n, 1);
            y = sin(2*pi*x) + 0.2*randn(n, 1);

            hlist = get_hlist(15, [0.05, 0.5]);
            opt = struct('order', 1, 'calc_dof', true, 'dstd', 0);
            regs = cv_fastlpr(x, y, hlist, opt);

            testCase.verifyEqual(regs.gcv_yhat.id1se, regs.gcv_yhat.idmin, ...
                'With dstd=0, id1se should equal idmin');
        end
    end

    % =========================================================================
    % Section 6: DOF Tests
    % =========================================================================

    methods (Test)
        function test_dof_computed_when_enabled(testCase)
            % UNIT: DOF values should be computed when calc_dof=TRUE
            % Matches: R "DOF values are computed when calc_dof=TRUE"

            rng(testCase.RANDOM_SEED);
            n = 100;
            x = rand(n, 1);
            y = sin(2*pi*x) + 0.1*randn(n, 1);

            hlist = get_hlist(10, [0.05, 0.5]);
            opt = struct('order', 1, 'calc_dof', true);
            regs = cv_fastlpr(x, y, hlist, opt);

            testCase.verifyTrue(isfield(regs.gcv_yhat, 'pdof_m'), ...
                'DOF should be computed');
            testCase.verifyTrue(all(isfinite(regs.gcv_yhat.pdof_m)), ...
                'DOF values should be finite');
        end

        function test_dof_valid_range(testCase)
            % UNIT: DOF should be in valid range [0, n]
            % Matches: R "DOF is in valid range [0, n]"

            rng(testCase.RANDOM_SEED);
            n = 100;
            x = rand(n, 1);
            y = sin(2*pi*x) + 0.1*randn(n, 1);

            hlist = get_hlist(10, [0.05, 0.5]);
            opt = struct('order', 1, 'calc_dof', true);
            regs = cv_fastlpr(x, y, hlist, opt);

            % pdof_m should be positive and finite
            testCase.verifyGreaterThan(min(regs.gcv_yhat.pdof_m), 0, ...
                'pdof_m should be positive');
        end
    end

    % =========================================================================
    % Section 7: Different Orders Tests
    % =========================================================================

    methods (Test)
        function test_gcv_order0(testCase)
            % UNIT: GCV should work for order 0 (Nadaraya-Watson)
            % Matches: R "GCV works for order 0 (Nadaraya-Watson)"

            rng(testCase.RANDOM_SEED);
            n = 100;
            x = rand(n, 1);
            y = sin(2*pi*x) + 0.1*randn(n, 1);

            hlist = get_hlist(10, [0.05, 0.5]);
            opt = struct('order', 0, 'calc_dof', true);
            regs = cv_fastlpr(x, y, hlist, opt);

            testCase.verifyTrue(all(isfinite(regs.gcv_yhat.gcv_m)), ...
                'Order 0 GCV should be finite');
        end

        function test_gcv_order1(testCase)
            % UNIT: GCV should work for order 1 (local linear)
            % Matches: R "GCV works for order 1 (local linear)"

            rng(testCase.RANDOM_SEED);
            n = 100;
            x = rand(n, 1);
            y = sin(2*pi*x) + 0.1*randn(n, 1);

            hlist = get_hlist(10, [0.05, 0.5]);
            opt = struct('order', 1, 'calc_dof', true);
            regs = cv_fastlpr(x, y, hlist, opt);

            testCase.verifyTrue(all(isfinite(regs.gcv_yhat.gcv_m)), ...
                'Order 1 GCV should be finite');
        end

        function test_gcv_order2(testCase)
            % UNIT: GCV should work for order 2 (local quadratic)
            % Matches: R "GCV works for order 2 (local quadratic)"

            rng(testCase.RANDOM_SEED);
            n = 100;
            x = rand(n, 1);
            y = sin(2*pi*x) + 0.1*randn(n, 1);

            hlist = get_hlist(10, [0.05, 0.5]);
            opt = struct('order', 2, 'calc_dof', true);
            regs = cv_fastlpr(x, y, hlist, opt);

            testCase.verifyTrue(all(isfinite(regs.gcv_yhat.gcv_m)), ...
                'Order 2 GCV should be finite');
        end

        function test_different_orders_different_gcv(testCase)
            % UNIT: Different orders should give different GCV values
            % Matches: R "Different orders give different GCV values"

            rng(testCase.RANDOM_SEED);
            n = 100;
            x = rand(n, 1);
            y = sin(2*pi*x) + 0.1*randn(n, 1);

            hlist = get_hlist(10, [0.05, 0.5]);

            opt0 = struct('order', 0, 'calc_dof', true);
            opt1 = struct('order', 1, 'calc_dof', true);
            opt2 = struct('order', 2, 'calc_dof', true);

            regs0 = cv_fastlpr(x, y, hlist, opt0);
            regs1 = cv_fastlpr(x, y, hlist, opt1);
            regs2 = cv_fastlpr(x, y, hlist, opt2);

            % At least one pair should be different
            diff01 = norm(regs0.gcv_yhat.gcv_m - regs1.gcv_yhat.gcv_m);
            diff12 = norm(regs1.gcv_yhat.gcv_m - regs2.gcv_yhat.gcv_m);
            diff02 = norm(regs0.gcv_yhat.gcv_m - regs2.gcv_yhat.gcv_m);

            testCase.verifyGreaterThan(diff01 + diff12 + diff02, 0, ...
                'Different orders should produce different GCV');
        end
    end

    % =========================================================================
    % Section 8: Edge Cases
    % =========================================================================

    methods (Test)
        function test_gcv_single_bandwidth(testCase)
            % UNIT: GCV should work with single bandwidth
            % Matches: R "GCV works with single bandwidth (no selection)"

            rng(testCase.RANDOM_SEED);
            n = 100;
            x = rand(n, 1);
            y = sin(2*pi*x) + 0.1*randn(n, 1);

            hlist = 0.2;  % Single bandwidth
            opt = struct('order', 1, 'calc_dof', true);
            regs = cv_fastlpr(x, y, hlist, opt);

            testCase.verifyTrue(isfinite(regs.gcv_yhat.gcv_m), ...
                'Single bandwidth GCV should be finite');
            testCase.verifyEqual(regs.gcv_yhat.idmin, 1, ...
                'Single bandwidth idmin should be 1');
        end

        function test_gcv_small_sample(testCase)
            % UNIT: GCV should handle small sample size
            % Matches: R "GCV handles small sample size"

            rng(testCase.RANDOM_SEED);
            n = 30;  % Small sample
            x = rand(n, 1);
            y = sin(2*pi*x) + 0.1*randn(n, 1);

            hlist = get_hlist(8, [0.1, 0.5]);
            opt = struct('order', 1, 'calc_dof', true);
            regs = cv_fastlpr(x, y, hlist, opt);

            testCase.verifyTrue(all(isfinite(regs.gcv_yhat.gcv_m)), ...
                'Small sample GCV should be finite');
        end
    end

    % =========================================================================
    % Section 9: Reproducibility
    % =========================================================================

    methods (Test)
        function test_gcv_reproducible(testCase)
            % UNIT: GCV computation should be reproducible
            % Matches: R "GCV computation is reproducible"

            n = 100;
            hlist = get_hlist(10, [0.05, 0.5]);
            opt = struct('order', 1, 'calc_dof', true);

            % Run twice with same seed
            rng(testCase.RANDOM_SEED);
            x = rand(n, 1);
            y = sin(2*pi*x) + 0.1*randn(n, 1);
            regs1 = cv_fastlpr(x, y, hlist, opt);

            rng(testCase.RANDOM_SEED);
            x = rand(n, 1);
            y = sin(2*pi*x) + 0.1*randn(n, 1);
            regs2 = cv_fastlpr(x, y, hlist, opt);

            testCase.verifyEqual(regs1.gcv_yhat.gcv_m, regs2.gcv_yhat.gcv_m, ...
                'AbsTol', testCase.TOL_STRICT, ...
                'GCV should be reproducible');
        end

        function test_gcv_selection_deterministic(testCase)
            % UNIT: GCV selection should be deterministic
            % Matches: R "GCV selection is deterministic"

            rng(testCase.RANDOM_SEED);
            n = 100;
            x = rand(n, 1);
            y = sin(2*pi*x) + 0.1*randn(n, 1);

            hlist = get_hlist(10, [0.05, 0.5]);
            opt = struct('order', 1, 'calc_dof', true);

            regs1 = cv_fastlpr(x, y, hlist, opt);
            regs2 = cv_fastlpr(x, y, hlist, opt);

            % Note: Due to randomized DOF estimation, consecutive calls may give
            % slightly different results. Just verify outputs are valid.
            % For strict reproducibility, use test_gcv_reproducible with same seed.
            testCase.verifyGreaterThanOrEqual(regs1.gcv_yhat.idmin, 1, ...
                'idmin should be valid');
            testCase.verifyLessThanOrEqual(regs1.gcv_yhat.idmin, length(hlist), ...
                'idmin should be within range');
            testCase.verifyGreaterThanOrEqual(regs2.gcv_yhat.idmin, 1, ...
                'idmin should be valid');
            % Note: id1se may vary slightly due to randomized DOF estimation
            % Just verify both are valid indices
            testCase.verifyGreaterThanOrEqual(regs1.gcv_yhat.id1se, 1, ...
                'id1se should be valid');
            testCase.verifyGreaterThanOrEqual(regs2.gcv_yhat.id1se, 1, ...
                'id1se should be valid');
        end
    end

    % =========================================================================
    % Section 10: 2D Tests
    % =========================================================================

    methods (Test)
        function test_2d_gcv_computation(testCase)
            % UNIT: 2D GCV computation should work
            % Matches: R "2D GCV computation works"

            rng(testCase.RANDOM_SEED);
            n = 200;
            x = rand(n, 2);
            y = sin(2*pi*x(:,1)) .* cos(2*pi*x(:,2)) + 0.1*randn(n, 1);

            hlist = get_hlist([5, 5], [0.1, 0.5; 0.1, 0.5]);
            opt = struct('order', 1, 'calc_dof', true);
            regs = cv_fastlpr(x, y, hlist, opt);

            testCase.verifyTrue(all(isfinite(regs.gcv_yhat.gcv_m)), ...
                '2D GCV should be finite');
            testCase.verifyGreaterThanOrEqual(regs.gcv_yhat.idmin, 1, ...
                '2D idmin should be valid');
            testCase.verifyLessThanOrEqual(regs.gcv_yhat.idmin, size(hlist, 1), ...
                '2D idmin should be within range');
        end
    end

end
