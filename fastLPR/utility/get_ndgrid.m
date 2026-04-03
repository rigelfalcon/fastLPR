function grid=get_ndgrid(x,type)
% GET_NDGRID - Create multi-dimensional grid from vectors
%
% Generates an N-dimensional grid from coordinate vectors, similar to ndgrid
% but with flexible output formats.
%
% Syntax:
%   grid = get_ndgrid(x)
%   grid = get_ndgrid(x, type)
%
% Inputs:
%   x    - Grid vectors, either:
%          * (N x d) matrix: N points in d dimensions
%          * {d x 1} cell array: one vector per dimension
%   type - Output format (default: 'cell'):
%          * 'cell' - Cell array of grid matrices {d x 1}
%          * 'array' - Single (N1 x N2 x ... x Nd x d) array
%          * 'list' - List of all grid points (prod(N) x d) matrix
%          * 'cellList' - Cell array of column vectors {d x 1}
%
% Output:
%   grid - Grid in requested format
%
% Example:
%   % Create 2D grid
%   x = {linspace(0,1,10), linspace(0,2,20)};
%
%   % Cell format: {10x20, 10x20}
%   grid_cell = get_ndgrid(x, 'cell');
%
%   % Array format: 10x20x2
%   grid_array = get_ndgrid(x, 'array');
%
%   % List format: 200x2 (all grid points)
%   grid_list = get_ndgrid(x, 'list');
%
% Notes:
%   - Uses MATLAB's ndgrid internally
%   - 'array' format is memory-efficient for dense grids
%   - 'list' format is useful for scattered data operations
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

%% Convert input to cell array if needed
if ismatrix(x) && ~iscell(x)
    % Matrix input: (N x d) -> {d x 1} cell array
    N=size(x,1);
    dx=size(x,2);
    x=mat2cell(x,N(1),ones(1,dx));
else
    % Cell input: already in correct format
    dx=numel(x);
end

%% Create grid using ndgrid
grid=cell(dx,1);
[grid{1:dx}]=ndgrid(x{:});

%% Convert to requested output format
if strcmp(type,'array')
    % Array format: (N1 x N2 x ... x Nd x d)
    grid=permute(grid,dx+1:-1:1);
    grid=cell2mat2(grid);  % Concatenate along last dimension

elseif strcmp(type,'list') && ~iscell(x)
    % List format: (prod(N) x d) matrix
    grid=permute(grid,dx+1:-1:1);
    grid=cell2mat(grid);
    grid=reshape(grid,[],dx);

elseif strcmp(type,'list') && iscell(x)
    % List format from cell input
    grid=cellfun(@(x) reshape(x,[],1),grid,'UniformOutput',false);
    grid=cat(2,grid{:});

elseif strcmp(type,'cellList')
    % Cell list format: {d x 1} of column vectors
    grid=cellfun(@(x) reshape(x,[],1),grid,'UniformOutput',false);
end

end
