function [Hlist] = get_hlist(n, range, spacing)
% GET_HLIST Generate bandwidth candidates for cross-validation.
%
%   HLIST = GET_HLIST(N, RANGE) creates a grid of N bandwidth values
%   per dimension using logarithmic spacing (default).
%
%   HLIST = GET_HLIST(N, RANGE, SPACING) uses the specified spacing function.
%
%   Parameters (Unified API v2.0):
%   ----------
%   n : int or vector
%       Number of bandwidth candidates per dimension.
%       Scalar: same number of points for all dimensions.
%       Vector: specify number of points per dimension.
%       Typical values: 20 for 1D, [15, 15] for 2D.
%
%   range : array
%       Bandwidth range [h_min, h_max] per dimension.
%       Shape: (n_dims, 2) or (1, 2) for 1D.
%       Rule of thumb:
%       - h_min = 0.1 * std(x): fine details
%       - h_max = 1.0 * std(x): smooth trend
%
%   spacing : function handle, optional
%       Spacing function: @(lo, hi, num) -> vector.
%       Default: @logspace (logarithmic spacing, better for CV).
%       Use @linspace for linear spacing.
%
%   Returns:
%   --------
%   Hlist : array
%       Bandwidth grid with shape (prod(n), n_dims).
%       Each row is a bandwidth vector [h1, h2, ..., hd].
%
%   Examples:
%   ---------
%   % 1D bandwidth grid (default logspace):
%   hlist = get_hlist(20, [0.1, 1.0]);
%
%   % 2D bandwidth grid:
%   hlist = get_hlist([10, 10], [0.1, 1.0; 0.1, 1.0]);
%
%   % Linear spacing:
%   hlist = get_hlist(10, [0.5, 0.6], @linspace);
%
%   See also: cv_fastlpr, cv_fastkde

%   Copyright (c) 2024-2025 Ying Wang, Min Li
%   SPDX-License-Identifier: GPL-3.0-or-later

% ============================================================
% Set default spacing to logspace (v2.0 change)
% ============================================================
if nargin < 3 || isempty(spacing)
    spacing = @logspace;
end

% ============================================================
% Main implementation
% ============================================================

% Ensure range is 2D matrix
if size(range, 1) == 1 && size(range, 2) == 2
    range = range(:)';  % Ensure row vector for 1D case
end

dims = size(range, 1);

% Broadcast n to match dims if scalar
if isscalar(n)
    n = repmat(n, 1, dims);
end

if length(n) ~= dims
    error('n must have length equal to the number of rows in range.');
end

% Generate 1D grids for each dimension
grids = cell(1, dims);
for d = 1:dims
    lo = range(d, 1);
    hi = range(d, 2);
    nd = n(d);

    if nd > 1
        if isequal(spacing, @logspace)
            grids{d} = logspace(log10(lo), log10(hi), nd);
        else
            grids{d} = spacing(lo, hi, nd);
        end
    else
        grids{d} = lo;
    end
end

% Generate n-dimensional grid using get_ndgrid with 'list' format
% Returns (prod(n) x d) matrix where each row is one bandwidth combination
Hlist = get_ndgrid(grids, 'list');

end
