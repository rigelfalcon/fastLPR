function [y,xgrid]=predict_griddedInterpolant(fpp,x,N,iValue)
% PREDICT_GRIDDEDINTERPOLANT - Predict using griddedInterpolant object
%
% Evaluates griddedInterpolant at specified points, handles grid and
% scattered data, supports multi-response selection.
%
% Syntax:
%   y = predict_griddedInterpolant(fpp)
%   y = predict_griddedInterpolant(fpp, x)
%   [y, xgrid] = predict_griddedInterpolant(fpp, x, N, iValue)
%
% Inputs:
%   fpp    - griddedInterpolant object
%   x      - Evaluation points (cell array for grid, matrix for scattered)
%            Default: fpp.GridVectors (original grid)
%   N      - Grid size for each dimension (optional)
%   iValue - Response index to extract (optional)
%            If fpp.Values is multi-response, select specific response
%
% Outputs:
%   y     - Interpolated values
%   xgrid - Evaluation grid (formatted)
%
% Example:
%   fpp = griddedInterpolant({1:10, 1:10}, rand(10,10));
%   y = predict_griddedInterpolant(fpp, {1:0.1:10, 1:0.1:10});
%
% Notes:
%   - Handles both grid (cell) and scattered (matrix) input
%   - Supports multi-response data (last dimension of Values)
%   - Uses spline interpolation from griddedInterpolant
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

%% Input validation
if nargin<2 || isempty(x)
    x=fpp.GridVectors;
end
if iscell(x)
    [dx]=numel(x);
    x=x(:)';
else
    [~,dx]=size(x);
end
%% Extract specific response if needed
sz=size(fpp.Values);
dim_value=ndims(fpp.Values);
ndcolon(1:dx) = {':'};
if nargin<4
    iValue=[];
end
if dim_value>dx && sz(end)>1 && ~isempty(iValue)
    fpp.Values=fpp.Values(ndcolon{:},iValue);  % Select response
end

%% Evaluate interpolant
if iscell(x)  && nargin>2 && ~isempty(N)
    % Grid input with specified size
    glist=get_ndgrid_scatter(x,'cellList',N);
    y=fpp(glist{:});
else
    % Scattered input or default grid
    glist=x;
    y=fpp(glist);
end

%% Format output grid if requested
if nargin>2
    if iscell(x) && nargin>2 && ~isempty(N)
        % Grid with specified size
        xgrid=get_ndgrid_scatter(x,'cell',N);
    elseif iscell(x)  && (nargin<3 || isempty(N))
        % Grid with default size
        xgrid=get_ndgrid(x,'cell');
        N=cellfun(@(x) length(x),x);
    else
        % Scattered points
        xgrid=x;
    end
    y=reshape(y,[N(:);length(iValue)]');  % Reshape to grid
end

end




%     N=cellfun(@(x) length(x),glist);
