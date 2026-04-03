function grid=get_ndgrid_scatter(x,type,N,space)
% GET_NDGRID_SCATTER - Create grid from scattered data bounds
%
% Generates a regular grid covering the range of scattered data points.
% Useful for creating evaluation grids for regression.
%
% Syntax:
%   grid = get_ndgrid_scatter(x)
%   grid = get_ndgrid_scatter(x, type, N, space)
%
% Inputs:
%   x     - Scattered data points (T x d matrix or {d x 1} cell array)
%   type  - Output format (default: 'cell')
%           See get_ndgrid for format options
%   N     - Number of grid points per dimension (default: size(x,1))
%           Can be scalar (same for all dims) or d x 1 vector
%   space - Spacing function (default: @linspace)
%           Options: @linspace, @logspace
%
% Output:
%   grid - Regular grid covering range of x, in requested format
%
% Example:
%   % Create grid from scattered data
%   x = rand(100, 2);  % 100 random points in 2D
%
%   % Create 50x50 grid covering data range
%   grid = get_ndgrid_scatter(x, 'array', 50);
%   % Result: 50x50x2 array
%
% Algorithm:
%   1. Find min and max of data in each dimension
%   2. Create linearly/logarithmically spaced vectors
%   3. Generate grid using get_ndgrid
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

%% Input validation and defaults
if nargin<2 || isempty(type)
    type='cell';
end
if nargin<3 || isempty(N)
    N=size(x,1);  % Default: same number of grid points as data points
end
if nargin<4 || isempty(space)
    space=@linspace;
end

%% Convert cell input to matrix if needed
if iscell(x)
    x = cell2mat2(x(:)');
end

%% Create grid vectors covering data range
% Find min and max in each dimension
% Create N evenly-spaced points between min and max
x=multispace(min(x,[],1),max(x,[],1),N,space).';

%% Generate grid
grid=get_ndgrid(x,type);

end


