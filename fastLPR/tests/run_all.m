function run_all()
%RUN_ALL Execute the full fastLPR MATLAB test suite.
%   RUN_ALL() adds the toolbox to the path (if needed) and calls all test
%   functions to validate regression, KDE, plotting, and cross-validation.
%
%   Usage:
%       run_all();
%
%   This wrapper is intended for automated CI scripts as well as manual
%   verification prior to publishing. The function requires no inputs and
%   prints a summary of the test results to the MATLAB command window.
%
%   Tests included:
%       1. run_fastlpr_tests() - Core functionality tests
%       2. test_fastlpr_vs_naive_nw() - Compare fastLPR vs naive NW
%       3. test_generate_matlab_reference() - Generate Python cross-validation data

fprintf('\n');
fprintf('================================================================================\n');
fprintf('fastLPR MATLAB Test Suite\n');
fprintf('================================================================================\n\n');

% Setup path
fastlpr_setup();

% Run core tests
fprintf('Running core functionality tests...\n');
run_fastlpr_tests();

% Run fastLPR vs naive NW comparison
fprintf('\n\nRunning fastLPR vs Naive NW comparison tests...\n');
test_fastlpr_vs_naive_nw();

% Generate MATLAB reference data for Python cross-validation
fprintf('\n\nGenerating MATLAB reference data for Python cross-validation...\n');
test_generate_matlab_reference();

fprintf('\n');
fprintf('================================================================================\n');
fprintf('All Tests Complete!\n');
fprintf('================================================================================\n\n');

end
