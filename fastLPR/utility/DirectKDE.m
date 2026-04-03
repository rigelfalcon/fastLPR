function [f_hat, dbg] = DirectKDE(x, h, grid, opt)
% DIRECTKDE - Direct kernel density estimation (O(N*M) baseline)
%
% Computes kernel density estimate using direct pairwise summation.
% This is a reference implementation used for:
%   1. Accuracy ground truth for fast KDE methods
%   2. Speedup comparison baseline
%   3. Validation of NUFFT-accelerated methods
%
% For large datasets, use cv_fastkde with NUFFT acceleration (O(N + M log M)).
%
% Syntax:
%   f_hat = DirectKDE(x, h)
%   f_hat = DirectKDE(x, h, grid)
%   f_hat = DirectKDE(x, h, grid, opt)
%   [f_hat, dbg] = DirectKDE(...)
%
% Inputs:
%   x    - Data points (N x d matrix)
%   h    - Bandwidth (scalar or 1 x d vector)
%   grid - Evaluation points (M x d matrix, default: x)
%   opt  - Options structure (reserved for future use)
%
% Outputs:
%   f_hat - Density estimates at grid points (M x 1)
%   dbg   - Debug structure with computation details
%
% Algorithm:
%   Gaussian KDE: f_hat(x) = (1/N) * sum_i K((x - x_i) / h) / prod(h)
%   where K(u) = (2*pi)^(-d/2) * exp(-0.5 * ||u||^2)
%
% Complexity: O(N * M) time, O(N * M) memory (true baseline, no optimization)
%
% Author: Ying Wang, Min Li
% Create Time: 2025-12-19
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

%% Input validation and defaults
[N, d] = size(x);

% Default options
if nargin < 4 || isempty(opt)
    opt = struct();
end

% Default grid = data points
if nargin < 3 || isempty(grid)
    grid = x;
end
M = size(grid, 1);

% Normalize bandwidth to row vector
h = h(:)';  % Ensure row vector
if numel(h) == 1
    h = repmat(h, 1, d);  % Broadcast scalar to d dimensions
end

%% Note: No internal zscore - caller should provide pre-standardized data

%% Compute KDE - direct O(N*M) computation without any optimization
f_hat = compute_kde_direct(x, h, grid, d);

%% Debug output
if nargout > 1
    dbg.N = N;
    dbg.M = M;
    dbg.d = d;
    dbg.h = h;
    % Theoretical peak memory: D_sq (M x N) + kernel_vals (M x N) = 2 * M * N * 8 bytes
    dbg.theoretical_mem_mb = 2 * M * N * 8 / (1024^2);
end

end

%% Helper function: compute KDE directly (no blocking)
function f_hat = compute_kde_direct(x, h, grid, d)
% Compute KDE using direct O(N*M) pairwise computation
% This is the true baseline implementation without any optimization

N = size(x, 1);
M = size(grid, 1);

% Scale by bandwidth
x_scaled = x ./ h;
grid_scaled = grid ./ h;

if d == 1
    % 1D case: simple vectorized computation
    D_sq = (grid_scaled - x_scaled').^2;  % (M x N)
else
    % Multi-D case: use pdist2 for efficiency
    D_sq = pdist2(grid_scaled, x_scaled, 'squaredeuclidean');  % (M x N)
end

% Gaussian kernel with proper d-dimensional normalization
% K(u) = (2*pi)^(-d/2) * exp(-0.5 * ||u||^2)
% f_hat = (1/N) * sum(K) / prod(h)
norm_factor = (2*pi)^(-d/2) / prod(h);
kernel_vals = exp(-0.5 * D_sq);  % (M x N)

% Sum over data points and normalize
f_hat = norm_factor * sum(kernel_vals, 2) / N;  % (M x 1)

end
