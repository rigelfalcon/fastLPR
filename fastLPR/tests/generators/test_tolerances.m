classdef test_tolerances
    % test_tolerances - Centralized tolerance constants for fastLPR tests
    %
    % This class provides unified tolerance constants matching R's helper-utils.R
    % and Python's conftest.py for cross-language test consistency.
    %
    % Usage:
    %   tol = test_tolerances();
    %   assert(max_err < tol.CROSSLANG.bw_maxerr, 'BW error too large');
    %
    % Author: fastLPR Development Team
    % Copyright (c) 2020-2025 fastLPR Development Team
    % License: GNU General Public License v3.0

    properties (Constant)
        % Strict tolerance for exact numerical matching (machine precision)
        % Use for: Pure MATLAB internal consistency tests
        STRICT = 1e-12

        % Numerical tolerance for algorithm comparisons
        % Use for: MATLAB-to-MATLAB comparisons, self-consistency tests
        NUMERICAL = 1e-6

        % Cross-language tolerance for R/Python vs MATLAB comparisons
        % Per CLAUDE.md: BW MaxErr < 0.02, Mean MaxErr < 0.05
        CROSSLANG = struct(...
            'bw_maxerr', 0.02, ...       % Bandwidth selection threshold
            'mean_maxerr', 0.05, ...     % Mean estimate max error
            'var_maxerr', 0.05, ...      % Variance estimate max error
            'lcv_relerr', 0.01, ...      % LCV relative error (1%)
            'gcv_relerr', 0.05, ...      % GCV relative error (5%)
            'speed_ratio', 8.0 ...       % Max allowed speed ratio
        )

        % NUFFT accuracy level tolerances
        % Expected max error for each NUFFT accuracy parameter
        ACCURACY = struct(...
            'acc4', 1e-3, ...
            'acc6', 1e-5, ...
            'acc8', 1e-7, ...
            'acc9', 1e-9, ...
            'acc12', 1e-11 ...
        )
    end

    methods (Static)
        function tol = get_accuracy_tol(accuracy)
            % get_accuracy_tol - Get tolerance for NUFFT accuracy level
            %
            % Parameters:
            %   accuracy : int (4, 6, 8, 9, or 12)
            %
            % Returns:
            %   tol : double - Expected max error tolerance

            tols = test_tolerances();
            switch accuracy
                case 4
                    tol = tols.ACCURACY.acc4;
                case 6
                    tol = tols.ACCURACY.acc6;
                case 8
                    tol = tols.ACCURACY.acc8;
                case 9
                    tol = tols.ACCURACY.acc9;
                case 12
                    tol = tols.ACCURACY.acc12;
                otherwise
                    error('test_tolerances:invalid_accuracy', ...
                        'Unsupported accuracy level: %d (use 4, 6, 8, 9, or 12)', accuracy);
            end
        end

        function assert_bw_match(actual, expected, msg)
            % assert_bw_match - Assert bandwidth selection matches within tolerance
            %
            % Parameters:
            %   actual   : double - Actual bandwidth
            %   expected : double - Expected bandwidth
            %   msg      : char (optional) - Error message

            if nargin < 3
                msg = 'Bandwidth mismatch';
            end

            tols = test_tolerances();
            max_err = abs(actual - expected);
            assert(max_err < tols.CROSSLANG.bw_maxerr, ...
                '%s: |%.6f - %.6f| = %.6f >= %.4f', ...
                msg, actual, expected, max_err, tols.CROSSLANG.bw_maxerr);
        end

        function assert_mean_match(actual, expected, msg)
            % assert_mean_match - Assert mean prediction matches within tolerance
            %
            % Parameters:
            %   actual   : array - Actual predictions
            %   expected : array - Expected predictions
            %   msg      : char (optional) - Error message

            if nargin < 3
                msg = 'Mean prediction mismatch';
            end

            tols = test_tolerances();
            max_err = max(abs(actual(:) - expected(:)));
            assert(max_err < tols.CROSSLANG.mean_maxerr, ...
                '%s: MaxErr = %.6f >= %.4f', ...
                msg, max_err, tols.CROSSLANG.mean_maxerr);
        end

        function assert_gcv_match(actual, expected, msg)
            % assert_gcv_match - Assert GCV score matches within relative tolerance
            %
            % Parameters:
            %   actual   : double - Actual GCV score
            %   expected : double - Expected GCV score
            %   msg      : char (optional) - Error message

            if nargin < 3
                msg = 'GCV score mismatch';
            end

            tols = test_tolerances();
            rel_err = abs(actual - expected) / max(abs(expected), 1e-10);
            assert(rel_err < tols.CROSSLANG.gcv_relerr, ...
                '%s: RelErr = %.6f >= %.4f', ...
                msg, rel_err, tols.CROSSLANG.gcv_relerr);
        end
    end
end
