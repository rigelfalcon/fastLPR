function [yq, L, dbg] = DirectNW(x, y, h, xq, opt)
% DIRECTNW - Direct Nadaraya-Watson kernel regression (O(N*M) baseline)
%
% Computes kernel-weighted local average (Nadaraya-Watson estimator).
% This is a reference implementation used for:
%   1. Accuracy ground truth for fast LPR methods
%   2. Speedup comparison baseline
%   3. Validation of NUFFT-accelerated methods
%
% For large datasets, use cv_fastlpr with NUFFT acceleration (O(N + M log M)).
%
% Syntax:
%   yq = DirectNW(x, y, h)
%   yq = DirectNW(x, y, h, xq)
%   yq = DirectNW(x, y, h, xq, opt)
%   [yq, L, dbg] = DirectNW(...)
%
% Inputs:
%   x   - Training predictors (N x d matrix)
%   y   - Training responses (N x 1 vector)
%   h   - Bandwidth (scalar or 1 x d vector)
%   xq  - Evaluation points (M x d matrix, default: x)
%   opt - Options structure (reserved for future use)
%
% Outputs:
%   yq  - Smoothed values at evaluation points (M x 1)
%   L   - Smoother matrix (M x N, optional)
%   dbg - Debug structure with computation details
%
% Algorithm:
%   Nadaraya-Watson estimator: yq(x) = sum(K((x-xi)/h) * yi) / sum(K((x-xi)/h))
%   where K is the Gaussian kernel.
%
% Complexity: O(N * M) time, O(N * M) memory (true baseline, no optimization)
%
% See also: DirectKDE, cv_fastlpr
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
if nargin < 5 || isempty(opt)
    opt = struct();
end

% Default evaluation points = training points
if nargin < 4 || isempty(xq)
    xq = x;
end
M = size(xq, 1);

% Normalize bandwidth to row vector
h = h(:)';  % Ensure row vector
if numel(h) == 1
    h = repmat(h, 1, d);  % Broadcast scalar to d dimensions
end

%% Note: No internal zscore - caller should provide pre-standardized data

%% Compute NW regression - direct O(N*M) computation without any optimization
[yq, L, kernel_sums] = compute_nw_direct(x, y, h, xq, d);

%% Debug output
if nargout > 2
    dbg.N = N;
    dbg.M = M;
    dbg.d = d;
    dbg.h = h;
    dbg.kernel_sums = kernel_sums;
    % Theoretical peak memory: D_sq (N x M) + Dkn (N x M) = 2 * N * M * 8 bytes
    dbg.theoretical_mem_mb = 2 * N * M * 8 / (1024^2);
end

end

%% Helper function: compute NW directly (no blocking)
function [yq, L, kernel_sums, weighted_sums] = compute_nw_direct(x, y, h, xq, d)
% Compute Nadaraya-Watson using direct O(N*M) pairwise computation
% This is the true baseline implementation without any optimization

N = size(x, 1);
M = size(xq, 1);

% Scale by bandwidth
x_scaled = x ./ h;
xq_scaled = xq ./ h;

if d == 1
    % 1D case: simple vectorized computation
    D_sq = (x_scaled - xq_scaled').^2;  % (N x M)
else
    % Multi-D case: use pdist2 for efficiency
    D_sq = pdist2(x_scaled, xq_scaled, 'squaredeuclidean');  % (N x M)
end

% Gaussian kernel (normalization cancels in NW ratio)
% K(u) = (2*pi)^(-d/2) * exp(-0.5 * ||u||^2)
% For NW, we can drop the constant since it cancels
Dkn = exp(-0.5 * D_sq);  % (N x M)

% Kernel sums: sum over data points for each query point
kernel_sums = sum(Dkn, 1)';  % (M x 1)

% Weighted sums: sum(K * y) for each query point
weighted_sums = Dkn' * y;  % (M x 1)

% NW estimate
yq = weighted_sums ./ kernel_sums;

% Optional: smoother matrix L (yq = L * y)
if nargout > 1
    L = Dkn ./ kernel_sums';  % (N x M), normalized weights
    L = L';  % (M x N) so that yq = L * y
end

end
