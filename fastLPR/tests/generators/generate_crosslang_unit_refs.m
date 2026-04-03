%% Generate Cross-Language Unit Test Reference Data
% This script generates NUFFT reference data for R/Python unit tests
%
% Output files:
% - ref_nufft_1d.mat: 1D NUFFT reference
%
% Usage:
%   cd('path/to/jss-code/fastLPR');
%   fastlpr_setup;
%   run('tests/generators/generate_crosslang_unit_refs.m');

clear; clc;

% Get the directory of this script and set up paths
script_dir = fileparts(mfilename('fullpath'));
tests_dir = fileparts(script_dir);  % tests/
fastlpr_root = fileparts(tests_dir);  % fastLPR/
cd(fastlpr_root);
run('fastlpr_setup.m');

% Output directory for reference files
out_dir = fullfile(tests_dir, 'refs', 'crosslang_unit');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

fprintf('================================================================================\n');
fprintf('GENERATING CROSS-LANGUAGE UNIT TEST REFERENCE DATA\n');
fprintf('Output directory: %s\n', out_dir);
fprintf('================================================================================\n\n');

rng(42); % Fixed seed for reproducibility

%% ============================================================================
%% NUFFT 1D Reference
%% ============================================================================
fprintf('>>> NUFFT 1D Reference <<<\n');

M = 100;  % Number of non-uniform points
N = 32;   % Grid size
acc = 9;  % Accuracy parameter

% Generate test data
x = rand(M, 1) - 0.5;  % x in [-0.5, 0.5]
y = sin(2 * pi * x) + 0.1 * randn(M, 1);

% Frequency spacing
df = 1.0 / N;

% Run MATLAB NUFFT
% Signature: nufftn_type1(x, y, M, df, iflag, acc, isdeconv)
Yq = nufftn_type1(x, y, N, df, -1, acc, true);

% Save reference
ref_nufft_1d = struct();
ref_nufft_1d.x = x;
ref_nufft_1d.y = y;
ref_nufft_1d.N = N;
ref_nufft_1d.acc = acc;
ref_nufft_1d.df = df;
ref_nufft_1d.Yq = Yq;
ref_nufft_1d.M = M;

save(fullfile(out_dir, 'ref_nufft_1d.mat'), '-struct', 'ref_nufft_1d');
fprintf('  Saved: ref_nufft_1d.mat (M=%d, N=%d, acc=%d)\n', M, N, acc);

fprintf('\n================================================================================\n');
fprintf('DONE: All reference files generated\n');
fprintf('================================================================================\n');
