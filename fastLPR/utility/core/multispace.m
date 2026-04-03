function x=multispace(x_min,x_max,N,space,type,istranspose)
% MULTISPACE - Generate multi-dimensional grid vectors
%
% Creates linearly or logarithmically spaced vectors for multiple dimensions.
% Similar to linspace/logspace but handles multiple dimensions at once.
%
% Syntax:
%   x = multispace(x_min, x_max, N)
%   x = multispace(x_min, x_max, N, space)
%   x = multispace(x_min, x_max, N, space, type)
%   x = multispace(x_min, x_max, N, space, type, istranspose)
%
% Inputs:
%   x_min - Minimum values (scalar or m x 1 vector)
%   x_max - Maximum values (scalar or m x 1 vector)
%   N     - Number of points (scalar or m x 1 vector)
%   space - Spacing function (default: @linspace)
%           Options: @linspace, @logspace
%   type  - Output type (default: 'array')
%           * 'array' - Return m x N matrix (if all N are equal)
%           * 'cell'  - Return m x 1 cell array
%   istranspose - Transpose output (default: false)
%
% Output:
%   x - Grid vectors, either:
%       * (m x N) matrix if type='array' and all N are equal
%       * {m x 1} cell array otherwise
%
% Example:
%   % Create 2D grid: x in [0,1] with 10 points, y in [0,2] with 20 points
%   x = multispace([0;0], [1;2], [10;20]);
%   % Result: 2x10 matrix (uses N=10 for both since type='array')
%
%   % Create with different N per dimension
%   x = multispace([0;0], [1;2], [10;20], @linspace, 'cell');
%   % Result: {2x1} cell array: {[1x10], [1x20]}
%
% Notes:
%   - Scalar inputs are broadcast to match the largest dimension
%   - For logspace, x_min and x_max are treated as powers of 10
%   - If N differs across dimensions, output is always cell array
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

%% Input validation and defaults
if nargin<4 || isempty(space)
    space=@linspace;
end
if nargin<5 || isempty(type)
    type='array';
end
if nargin<6 || isempty(istranspose)
    istranspose=false;
end

%% Handle logspace
% For logspace, convert to log10 scale
if strcmp(func2str(space),'logspace')
    x_min=log10(x_min);
    x_max=log10(x_max);
end

%% Broadcast inputs to same size
% Find maximum dimension
m=max(max(numel(x_min),numel(x_max)),numel(N));

% Broadcast scalars to vectors
x_min=broadcast(x_min,m);
x_max=broadcast(x_max,m);
[N,flag_broad]=broadcast(N,m);

%% Generate grid vectors
% If all N are equal and type='array', return matrix
% Otherwise, return cell array
if (flag_broad ||checksame(N)) && strcmp(type,'array')
    % All dimensions have same N: return matrix
    x=ones(m,N(1),class(x_min));
    for i=1:m
        x(i,:)=space(x_min(i),x_max(i),N(i));
    end
    if istranspose
        x = x.';  % Transpose to (N x m)
    end
else
    % Different N per dimension: return cell array
    x=cell(m,1);
    for i=1:m
        x{i}=space(x_min(i),x_max(i),N(i));
        if istranspose
            x{i}= x{i}.';  % Transpose each vector
        end
    end
end

end

%% Helper functions

function [x,flag_broad]=broadcast(x,m)
% Broadcast scalar to vector of length m
flag_broad=isscalar(x);
if flag_broad
    x=repmat(x,[m,1]);
end
end

function  flag_same=checksame(x)
% Check if all elements are the same
flag_same=~sum(abs(diff(x)));
end