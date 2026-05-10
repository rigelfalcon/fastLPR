function [fpp,yhat]=fastlpr_gridinterp(regs,mq,xlist,method,opt)
% FASTLPR_GRIDINTERP - Interpolate grid values to scattered data points
%
% Creates an interpolant from grid values and optionally evaluates it at
% the original scattered data points. Used for DOF estimation and prediction.
%
% Syntax:
%   fpp = fastlpr_gridinterp(regs, mq)
%   [fpp, yhat] = fastlpr_gridinterp(regs, mq, xlist, method, opt)
%
% Inputs:
%   regs   - Regression structure containing:
%            .xlist - Grid vectors (cell array)
%            .xraw  - Original scattered data points
%            .N     - Grid size per dimension
%            .dx    - Number of dimensions
%            .Tx    - Number of data points
%   mq     - Grid values (N1 x N2 x ... x Nd x ...) array
%            Can be multi-dimensional for multiple responses/bandwidths
%   xlist  - Grid vectors (default: regs.xlist)
%   method - Interpolation method (default: 'griddedInterpolant')
%   opt    - Interpolation options (default: spline with linear extrapolation)
%            .Method - Interpolation method ('linear', 'spline', 'cubic')
%            .ExtrapolationMethod - Extrapolation method ('linear', 'nearest', 'none')
%
% Outputs:
%   fpp  - griddedInterpolant object for fast evaluation
%   yhat - Interpolated values at original data points (if requested)
%          Size: (Tx x ...) matching mq dimensions beyond dx
%
% Example:
%   % Interpolate regression estimates to data points
%   [fpp, yhat] = fastlpr_gridinterp(regs, m_grid);
%
%   % Predict at new points
%   x_new = rand(100, 2);
%   y_new = fpp(x_new(:,1), x_new(:,2));
%
% Notes:
%   - Uses MATLAB's griddedInterpolant for fast interpolation
%   - Automatically handles GPU arrays (converts to CPU)
%   - Spline interpolation provides smooth estimates
%   - Linear extrapolation prevents extreme values outside data range
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

%% Input validation and defaults
if nargin<3 || isempty(xlist)
    xlist=regs.xlist;
end
if nargin<4 || isempty(method)
    method ='griddedInterpolant';
end
if nargin<5 || isempty(opt)
    % Default: linear interpolation with linear extrapolation
    opt.Method='linear';
    opt.ExtrapolationMethod ='linear';
end

%% Ensure grid vectors have unique points
% At very large N (>8M), floating-point precision can cause duplicate grid points
% when standardizing x values. griddedInterpolant requires unique sample points.
for d = 1:length(xlist)
    grid_d = xlist{d};
    [grid_unique, idx_unique] = unique(grid_d, 'stable');
    if length(grid_unique) < length(grid_d)
        warning('fastLPR:gridinterp:DuplicatePoints', ...
            'Dimension %d had %d duplicate grid points (removed).', ...
            d, length(grid_d) - length(grid_unique));
        xlist{d} = grid_unique;
        % Also need to subsample mq along this dimension
        if d == 1
            mq = mq(idx_unique, :);
        elseif d == 2
            mq = mq(:, idx_unique, :);
        elseif d == 3
            mq = mq(:, :, idx_unique, :);
        end
        % Update regs.N if needed
        if isfield(regs, 'N')
            regs.N(d) = length(grid_unique);
        end
    end
end

%% Convert GPU arrays to CPU
% griddedInterpolant does not support GPU arrays
if isa(mq,'gpuArray')
    mq=gather(mq);
end

%% Create interpolant from grid values
sz=size(mq);

% Reshape mq to (N1 x N2 x ... x Nd x prod(remaining dims))
% This handles multiple responses/bandwidths in extra dimensions
mq_reshaped = reshape(mq,[regs.N, prod(sz(regs.dx+1:end))]);

switch method
    case{'griddedInterpolant'}
        % Create griddedInterpolant object
        % This allows fast evaluation at arbitrary points
        fpp = griddedInterpolant(xlist, mq_reshaped, ...
                                 opt.Method, opt.ExtrapolationMethod);
    otherwise
        error('fastLPR:gridinterp:UnknownMethod', ...
            'Unknown interpolation method: %s. Use ''griddedInterpolant''.', method);
end

%% Evaluate at original scattered data points (if requested)
if nargout>1
    % Convert xraw to cell array for griddedInterpolant evaluation
    xrawc=mat2cell(regs.xraw,regs.Tx,ones(1,regs.dx));

    % Evaluate interpolant at data points
    yhat=squeeze(fpp(xrawc{:}));

    % Reshape to match original dimensions
    yhat=reshape(yhat,[regs.Tx,sz(regs.dx+1:end),1]);
end

end