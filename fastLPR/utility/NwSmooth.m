function [yq,L,dbg]=NwSmooth(x,y,h,xq)
% NWSMOOTH - Nadaraya-Watson kernel smoothing (reference implementation)
%
% Computes kernel-weighted local average (Nadaraya-Watson estimator).
% This is a simple O(N*M) implementation used for validation and small datasets.
% For large datasets, use fastLPR_* functions with NUFFT acceleration.
%
% Syntax:
%   yq = NwSmooth(x, y, h)
%   yq = NwSmooth(x, y, h, xq)
%   [yq, L, dbg] = NwSmooth(...)
%
% Inputs:
%   x  - Training predictors (n x dx matrix)
%   y  - Training responses (n x 1 vector)
%   h  - Bandwidth (scalar or 1 x dx vector)
%   xq - Evaluation points (optional, default: x)
%
% Outputs:
%   yq  - Smoothed values at evaluation points
%   L   - Smoother matrix (optional, nq x n)
%   dbg - Debug structure with kernel weights
%
% Algorithm:
%   Nadaraya-Watson estimator: yq(x) = sum(K((x-xi)/h) * yi) / sum(K((x-xi)/h))
%   where K is the Gaussian kernel.
%
% Note:
%   This is a reference implementation with O(N*M) complexity.
%   For large datasets, use cv_fastlpr with NUFFT acceleration (O(N + M log M)).
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

% Standardize training data
[x,x_mean,x_std]=zscore(x);

% Set default evaluation points
if  nargin<4||isempty(xq)
    xq=x;
end

% Standardize evaluation points using training statistics
xq=(xq-x_mean)./x_std;

% Get dimensions
[n,dx]=size(x);
[nq,~]=size(xq);

% OPTIMIZATION: Use pdist2 for efficient pairwise distance computation
% This is much faster than manual broadcasting for large datasets
if dx == 1
    % 1D case: simple vectorized computation
    h_scalar = h(1);
    D = (x - xq') / h_scalar;  % (n x nq) distance matrix
    Dkn = (1/sqrt(2*pi)) * exp(-0.5 * D.^2);  % Gaussian kernel
else
    % Multi-D case: use pdist2 with 'euclidean' and scale by bandwidth
    % Scale coordinates by bandwidth
    x_scaled = x ./ h;
    xq_scaled = xq ./ h;

    % Compute squared Euclidean distances efficiently
    D_sq = pdist2(x_scaled, xq_scaled, 'squaredeuclidean');  % (n x nq)

    % Apply Gaussian kernel
    Dkn = (1/sqrt(2*pi))^dx * exp(-0.5 * D_sq);
end

% Nadaraya-Watson estimator: weighted average
% yq(j) = sum_i(K(x_i, xq_j) * y_i) / sum_i(K(x_i, xq_j))
kernel_sums = sum(Dkn, 1);  % (1 x nq)
yq = (Dkn' * y) ./ kernel_sums';  % (nq x 1)

% Debug output: kernel weight sums
dbg.s = kernel_sums;

% Optional: return smoother matrix L (yq = L * y)
if nargout > 1
    L = Dkn ./ kernel_sums;  % (n x nq), normalized weights
    L = L';  % (nq x n) so that yq = L * y
end
end

%% Helper functions

function u=epan_kernel(u,b)
% Epanechnikov kernel (not currently used, kept for reference)
u=u./b;
u=max(0,3/4*(1-sum(u.^2,3)));
end

function u=gaussian_kernel(u,b)
% Gaussian kernel: K(u) = (1/sqrt(2*pi)) * exp(-0.5 * u^2)
u=u./b;
u=(1/sqrt(2*pi))*exp(-0.5*sum((u.^2),3));
end


