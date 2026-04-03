%% benchmark_direct.m - Benchmark Direct KDE/NW methods and generate ground truth
%
% Runs DirectKDE and DirectNW benchmarks.
% Also saves ground truth files for other methods to compare against.
%
% Data: Uniform [0,1] + zscore (consistent with all other benchmarks)
% N ranges: 32 to 65,536 (2^16) for O(N^2) methods
%
% Output:
%   - benchmark/data/direct_matlab_benchmark.csv (timing results)
%   - benchmark/data/ground_truth/gt_d{d}_N{N}.mat (ground truth for accuracy)
%
% Author: Ying Wang, Min Li
% Create Time: 2025-12-21
% Updated: 2025-12-25 (unified data, ground truth files)

clear; clc;

%% Setup paths (run from repo root)
REPO_ROOT = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(fullfile(REPO_ROOT, 'fastLPR'));
addpath(fullfile(REPO_ROOT, 'fastLPR', 'utility'));

%% Configuration
N_RUNS = 1;  % Quick run for initial benchmark
MAX_TIME_PER_RUN = 300;  % Skip larger N if single run exceeds this (5 minutes)

% N values: Limited to 2^16 (65,536) for O(N^2) methods
N_VALUES = 2.^(5:16);  % 32 to 65,536
D_VALUES = [1, 2, 3];
SEED = 42;
H0 = 0.3;  % Base bandwidth constant: h_N = H0 * N^(-1/(d+4))
NOISE_STD = 0.1;

% Grid vs data_point evaluation mode (for grid accuracy support)
EVAL_MODE = getenv('EVAL_MODE');
if isempty(EVAL_MODE)
    EVAL_MODE = 'data_point';  % Default: evaluate at data points
end
fprintf('EVAL_MODE: %s\n', EVAL_MODE);

% Output paths
SCRIPT_DIR = fileparts(mfilename('fullpath'));
BENCHMARK_DIR = fullfile(SCRIPT_DIR, '..', '..');
OUTPUT_FILE = fullfile(BENCHMARK_DIR, 'data', 'direct_matlab_benchmark.csv');
GT_DIR = fullfile(BENCHMARK_DIR, 'data', 'ground_truth');

% Create directories
if ~exist(fullfile(BENCHMARK_DIR, 'data'), 'dir')
    mkdir(fullfile(BENCHMARK_DIR, 'data'));
end
if ~exist(GT_DIR, 'dir')
    mkdir(GT_DIR);
end

% Options: NO BLOCK PROCESSING
opt = struct();
opt.block_size = 1e10;  % Disable blocking

%% Warm-up (JIT compilation)
fprintf('Warming up JIT...\n');
try
    dummy_x = randn(100, 1);
    dummy_y = sin(dummy_x) + 0.1*randn(100, 1);
    dummy_h = 0.3;
    DirectKDE(dummy_x, dummy_h, dummy_x, opt);
    DirectNW(dummy_x, dummy_y, dummy_h, dummy_x, opt);
    clear dummy_x dummy_y dummy_h;
catch
    % Ignore warm-up errors
end

%% Initialize output
results = {};
fprintf('\n============================================================\n');
fprintf('DirectKDE/DirectNW Benchmark + Ground Truth Generation\n');
fprintf('============================================================\n');
fprintf('N_RUNS: %d\n', N_RUNS);
fprintf('N range: %d to %d\n', min(N_VALUES), max(N_VALUES));
fprintf('Dimensions: %s\n', mat2str(D_VALUES));
fprintf('Ground truth dir: %s\n', GT_DIR);
fprintf('Output CSV: %s\n\n', OUTPUT_FILE);

skip_kde = containers.Map({1, 2, 3}, {false, false, false});
skip_nw = containers.Map({1, 2, 3}, {false, false, false});

%% Run benchmarks
for d = D_VALUES
    fprintf('\n=== Dimension d=%d ===\n', d);

    for N = N_VALUES
        % Variable bandwidth: h_N = H0 * N^(-1/(d+4))
        h_N = H0 * N^(-1/(d+4));

        % ========== Generate data: Uniform [0,1] + zscore ==========
        % This is the UNIFIED data format for all benchmarks
        rng(SEED);
        x_orig = rand(N, d);
        [x_zs, x_mu, x_sigma] = zscore(x_orig);

        % Response for LPR
        if d == 1
            y_true = sin(2*pi*x_orig);
        else
            y_true = sin(2*pi*mean(x_orig, 2));
        end
        y = y_true + NOISE_STD * randn(N, 1);

        % Determine evaluation points based on EVAL_MODE
        if strcmp(EVAL_MODE, 'grid')
            x_eval = make_eval_grid(N, d, x_zs);  % Grid based on data range
            gt_file = fullfile(GT_DIR, sprintf('gt_d%d_N%d_grid.mat', d, N));
            fprintf('  Grid mode: evaluating at %d grid points (range: [%.2f, %.2f])\n', ...
                size(x_eval, 1), min(x_eval(:)), max(x_eval(:)));
        else
            x_eval = x_zs;  % Evaluate at data points (default)
            gt_file = fullfile(GT_DIR, sprintf('gt_d%d_N%d.mat', d, N));
        end

        % Skip if ground truth file already exists (Req 1.4 cache behavior)
        if exist(gt_file, 'file')
            fprintf('  -> Ground truth exists, skipping: %s\n', gt_file);
            continue;
        end

        %% DirectKDE benchmark
        if ~skip_kde(d)
            fprintf('DirectKDE d=%d N=%6d: ', d, N);
            try
                times = zeros(N_RUNS, 1);
                kde_gt = [];
                dbg_kde = [];

                for run = 1:N_RUNS
                    tic;
                    [kde_gt, dbg_kde] = DirectKDE(x_zs, h_N, x_eval, opt);
                    times(run) = toc;

                    % Early exit if too slow
                    if times(run) > MAX_TIME_PER_RUN
                        fprintf('(slow, filling remaining) ');
                        times((run+1):N_RUNS) = times(run);
                        break;
                    end
                end

                time_sec = median(times);
                time_min = min(times);
                time_max = max(times);
                time_std = std(times);

                % Use theoretical memory from dbg structure (O(N*M) baseline)
                mem_mb = dbg_kde.theoretical_mem_mb;

                % DirectKDE is ground truth, so accuracy_vs_direct = 0
                acc = 0.0;

                results{end+1} = {'DirectKDE', 'KDE', 'MATLAB', d, N, time_sec, mem_mb, acc, time_min, time_max, time_std, 'success'};
                fprintf('%.3fs (std=%.3f), %.1f MB\n', time_sec, time_std, mem_mb);

                % Skip larger N if too slow
                if time_sec > MAX_TIME_PER_RUN
                    skip_kde(d) = true;
                    fprintf('  (Skipping larger N for DirectKDE d=%d)\n', d);
                end
            catch ME
                fprintf('ERROR: %s\n', ME.message);
                results{end+1} = {'DirectKDE', 'KDE', 'MATLAB', d, N, NaN, NaN, NaN, NaN, NaN, NaN, 'error'};
                kde_gt = [];
            end
        else
            kde_gt = [];
        end

        %% DirectNW benchmark
        if ~skip_nw(d)
            fprintf('DirectNW  d=%d N=%6d: ', d, N);
            try
                times = zeros(N_RUNS, 1);
                nw_gt = [];
                dbg_nw = [];

                for run = 1:N_RUNS
                    tic;
                    [nw_gt, ~, dbg_nw] = DirectNW(x_zs, y, h_N, x_eval, opt);
                    times(run) = toc;

                    % Early exit if too slow
                    if times(run) > MAX_TIME_PER_RUN
                        fprintf('(slow, filling remaining) ');
                        times((run+1):N_RUNS) = times(run);
                        break;
                    end
                end

                time_sec = median(times);
                time_min = min(times);
                time_max = max(times);
                time_std = std(times);

                % Use theoretical memory from dbg structure (O(N*M) baseline)
                mem_mb = dbg_nw.theoretical_mem_mb;

                % DirectNW is ground truth, so accuracy_vs_direct = 0
                acc = 0.0;

                results{end+1} = {'DirectNW', 'LPR', 'MATLAB', d, N, time_sec, mem_mb, acc, time_min, time_max, time_std, 'success'};
                fprintf('%.3fs (std=%.3f), %.1f MB\n', time_sec, time_std, mem_mb);

                % Skip larger N if too slow
                if time_sec > MAX_TIME_PER_RUN
                    skip_nw(d) = true;
                    fprintf('  (Skipping larger N for DirectNW d=%d)\n', d);
                end
            catch ME
                fprintf('ERROR: %s\n', ME.message);
                results{end+1} = {'DirectNW', 'LPR', 'MATLAB', d, N, NaN, NaN, NaN, NaN, NaN, NaN, 'error'};
                nw_gt = [];
            end
        else
            nw_gt = [];
        end

        %% Save ground truth if both KDE and NW succeeded
        if ~isempty(kde_gt) && ~isempty(nw_gt)
            if strcmp(EVAL_MODE, 'grid')
                % Grid mode: include x_grid (the evaluation points)
                x_grid = x_eval;
                save(gt_file, 'x_orig', 'x_zs', 'y', 'y_true', 'x_grid', 'kde_gt', 'nw_gt', ...
                     'h_N', 'd', 'N', 'x_mu', 'x_sigma', 'SEED', 'H0', 'NOISE_STD', 'EVAL_MODE');
            else
                % Data point mode: standard save
                save(gt_file, 'x_orig', 'x_zs', 'y', 'y_true', 'kde_gt', 'nw_gt', ...
                     'h_N', 'd', 'N', 'x_mu', 'x_sigma', 'SEED', 'H0', 'NOISE_STD');
            end
            fprintf('  -> Ground truth saved: %s\n', gt_file);
        end
    end
end

%% Save results to CSV
fprintf('\n============================================================\n');
fprintf('Saving results to: %s\n', OUTPUT_FILE);

% Create header
header = 'method,task,lang,d,N,time_sec,mem_mb,accuracy_vs_direct,time_min,time_max,time_std,status';

% Open file
fid = fopen(OUTPUT_FILE, 'w');
fprintf(fid, '%s\n', header);

% Write data rows
for i = 1:length(results)
    row = results{i};
    fprintf(fid, '%s,%s,%s,%d,%d,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%s\n', ...
        row{1}, row{2}, row{3}, row{4}, row{5}, row{6}, row{7}, row{8}, row{9}, row{10}, row{11}, row{12});
end

fclose(fid);
fprintf('Total rows: %d\n', length(results));
fprintf('Done!\n');

%% ========== Local Functions ==========

function x_grid = make_eval_grid(~, d, x_zs)
    % Generate uniform evaluation grid based on DATA RANGE with 5% margin
    % Uses ndgrid for column-major ordering (first dim varies fastest)
    %
    % Note: We use data-based range because KDE/LPR values are only meaningful
    % within the data range. Extrapolation beyond data range produces incorrect values.
    %
    % Grid size: Fixed at 10000 total points (M_per_dim = ceil(10000^(1/d)))
    % - 1D: 10000 points
    % - 2D: 100x100 = 10000 points
    % - 3D: 22x22x22 = 10648 points
    %
    % Args:
    %   ~: (unused) sample size
    %   d: dimension
    %   x_zs: (N, d) z-scored data to determine grid range
    %
    % Returns:
    %   x_grid: (M_total, d) grid points covering data range with 5% margin

    M_TOTAL = 10000;
    M_per_dim = ceil(M_TOTAL^(1/d));

    % Compute range from data (NO margin - stay within interpolatable range)
    % Removing 5% margin fixes edge interpolation errors in fastKDE/fastLPR
    x_min = min(x_zs, [], 1);  % (1, d)
    x_max = max(x_zs, [], 1);  % (1, d)
    % margin = 0.05 * (x_max - x_min);  % REMOVED: causes edge extrapolation errors
    % x_min = x_min - margin;
    % x_max = x_max + margin;

    if d == 1
        x_grid = linspace(x_min, x_max, M_per_dim)';
    else
        % Create axes based on data range
        axes = cell(1, d);
        for i = 1:d
            axes{i} = linspace(x_min(i), x_max(i), M_per_dim);
        end

        % Use ndgrid (column-major: first dimension varies fastest)
        grids = cell(1, d);
        [grids{1:d}] = ndgrid(axes{:});

        % Flatten to (M_total, d) matrix
        M_total = M_per_dim^d;
        x_grid = zeros(M_total, d);
        for i = 1:d
            x_grid(:, i) = grids{i}(:);  % (:) flattens column-major
        end
    end
end
