function run_all()
%RUN_ALL Execute the full fastLPR MATLAB test suite.
%   RUN_ALL() adds the toolbox to the path (if needed) and calls the core
%   test suite to validate regression, KDE, plotting, and cross-validation.
%
%   Usage:
%       run_all();

fprintf('\n');
fprintf('================================================================================\n');
fprintf('fastLPR MATLAB Test Suite\n');
fprintf('================================================================================\n\n');

% Setup path
fastlpr_setup();

% Run core tests
fprintf('Running core functionality tests...\n');
run_fastlpr_tests();

fprintf('\n');
fprintf('================================================================================\n');
fprintf('All Tests Complete!\n');
fprintf('================================================================================\n\n');

end
