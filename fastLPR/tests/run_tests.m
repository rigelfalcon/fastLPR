function run_tests()
%RUN_TESTS Execute the full fastLPR MATLAB test suite using matlab.unittest.
%
%   Usage (from repo root):
%       >> cd fastLPR/tests
%       >> run_tests
%
%   Or using runtests:
%       >> results = runtests('fastLPR/tests')
%
%   Tests included:
%       - test_kde_1d   : 1D kernel density estimation
%       - test_kde_2d   : 2D kernel density estimation
%       - test_kde_3d   : 3D kernel density estimation
%       - test_lpr_1d   : 1D local polynomial regression (orders 0, 1, 2)
%       - test_lpr_2d   : 2D local polynomial regression
%       - test_lpr_complex : Complex-valued regression
%       - test_lpr_hetero  : Heteroscedastic (mean + variance)
%
%   Author: Ying Wang, Min Li
%   Copyright (c) 2020-2025 fastLPR Development Team
%   License: GNU General Public License v3.0

fprintf('\n');
fprintf('================================================================================\n');
fprintf('fastLPR MATLAB Test Suite (matlab.unittest)\n');
fprintf('================================================================================\n\n');

% Get test directory
testDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(testDir));

% Setup path
run(fullfile(repoRoot, 'fastLPR', 'fastlpr_setup.m'));

% Run all unittest classes
fprintf('Running matlab.unittest test classes...\n\n');

import matlab.unittest.TestSuite
import matlab.unittest.TestRunner
import matlab.unittest.plugins.DiagnosticsValidationPlugin

% Create test suite from folder (including subfolders for organized test structure)
suite = TestSuite.fromFolder(testDir, 'IncludingSubfolders', true);

% Create runner with verbose output
runner = TestRunner.withTextOutput('Verbosity', 2);

% Run tests
results = runner.run(suite);

% Summary
fprintf('\n');
fprintf('================================================================================\n');
fprintf('TEST SUMMARY\n');
fprintf('================================================================================\n');
fprintf('Total:  %d\n', numel(results));
fprintf('Passed: %d\n', sum([results.Passed]));
fprintf('Failed: %d\n', sum([results.Failed]));
fprintf('================================================================================\n\n');

% Also run legacy tests for backward compatibility
fprintf('Running legacy tests...\n\n');
run_fastlpr_tests();

fprintf('\n================================================================================\n');
fprintf('All Tests Complete!\n');
fprintf('================================================================================\n\n');

end
