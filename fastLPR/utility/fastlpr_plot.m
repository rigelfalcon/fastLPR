function varargout = fastlpr_plot(fpp, x, N, opt, varargin)
% FASTLPR_PLOT - Visualize regression results from fastLPR
%
% This function creates publication-quality visualizations of local polynomial
% regression results. It handles both 1D (line plots) and 2D (surface plots)
% cases, with support for custom transformations and plotting functions.
%
% Syntax:
%   fastlpr_plot(fpp)
%   fastlpr_plot(fpp, x)
%   fastlpr_plot(fpp, x, N)
%   fastlpr_plot(fpp, x, N, opt)
%   fastlpr_plot(fpp, x, N, opt, 'Property', Value, ...)
%   h = fastlpr_plot(...)
%   [h, y] = fastlpr_plot(...)
%
% Inputs:
%   fpp - griddedInterpolant object from cv_fastlpr
%         Typically: regs.fpp_yhat
%
%   x   - Evaluation points (optional, default: use grid from fpp)
%         For 1D: vector or cell {x1}
%         For 2D: cell array {x1, x2}
%
%   N   - Number of evaluation points per dimension (optional, default: auto)
%         Scalar: same resolution for all dimensions
%         Vector: different resolution per dimension
%
%   opt - Options structure (optional) with fields:
%         .iValue: Index of value to plot (default: 1)
%                  For multi-response, select which response to plot
%         .xfun: Cell array of transformation functions for x-axes (default: {})
%                Example: {[], @log10} applies log10 to second axis
%         .yfun: Transformation function for y-axis (default: [])
%                Example: @log10 for log-scale y-axis
%         .plotfun: Custom plotting function (default: [] = auto)
%                   For 1D: default is @plot
%                   For 2D: default is @surf, can use @contour, @contour3, etc.
%
%   'Property', Value - Additional plot properties passed to plotting function
%                       Examples: 'LineWidth', 2, 'Color', 'r', 'FaceAlpha', 0.8
%
% Outputs:
%   h   - Handle to plot object (optional)
%   y   - Evaluated values at grid points (optional)
%
% Examples:
%   % Example 1: Simple 1D plot
%   x = rand(500, 1) * 4 - 2;
%   y = sin(2*pi*x) + 0.2*randn(500, 1);
%   hlist = get_hlist(20, [0.01, 1], @logspace);
%   opt.order = 1;
%   regs = cv_fastlpr(x, y, hlist, opt);
%   
%   figure; hold on;
%   scatter(x, y, 'k.');
%   fastlpr_plot(regs.fpp_yhat, [], [], [], 'LineWidth', 2, 'Color', 'r');
%
%   % Example 2: 2D surface plot
%   x = rand(1000, 2);
%   y = sin(2*pi*x(:,1)) .* cos(2*pi*x(:,2)) + 0.1*randn(1000, 1);
%   hlist = get_hlist([10, 10], [0.01, 1; 0.01, 1], @logspace);
%   opt.order = 1;
%   regs = cv_fastlpr(x, y, hlist, opt);
%   
%   figure;
%   fastlpr_plot(regs.fpp_yhat);
%   xlabel('X1'); ylabel('X2'); zlabel('Y');
%
%   % Example 3: 2D contour plot
%   opt_plot.plotfun = @contour;
%   figure;
%   fastlpr_plot(regs.fpp_yhat, [], [], opt_plot);
%
%   % Example 4: Log-scale x-axis
%   x = 10.^(rand(500, 1) * 3 - 1);  % Log-uniform from 0.1 to 100
%   y = log10(x) + 0.2*randn(500, 1);
%   hlist = get_hlist(20, [0.01, 1], @logspace);
%   opt.order = 1;
%   regs = cv_fastlpr(x, y, hlist, opt);
%   
%   opt_plot.xfun = {@log10};  % Apply log10 to x-axis
%   figure; hold on;
%   scatter(log10(x), y, 'k.');
%   fastlpr_plot(regs.fpp_yhat, [], [], opt_plot, 'LineWidth', 2);
%   xlabel('log10(X)'); ylabel('Y');
%
%   % Example 5: Custom resolution
%   figure;
%   fastlpr_plot(regs.fpp_yhat, [], 200);  % 200 points per dimension
%
% Notes:
%   - For 1D: Creates line plot with plot()
%   - For 2D: Creates surface plot with surf() by default
%   - Supports custom plotting functions via opt.plotfun
%   - Supports axis transformations via opt.xfun and opt.yfun
%   - Additional plot properties can be passed as name-value pairs
%
% See also: cv_fastlpr, fastlpr_plot_interval, plot, surf, contour
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com, 
%                Min Li, minli.231314@gmail.com 
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

%% Input validation
narginchk(1, inf);

% Validate fpp
if ~isa(fpp, 'griddedInterpolant')
    error('fastlpr_plot:InvalidInput', ...
        'First input must be a griddedInterpolant object (e.g., regs.fpp_yhat).');
end

% Set default x (use grid from fpp)
if nargin < 2 || isempty(x)
    x = fpp.GridVectors;
end

% Set default N (use length of x)
if nargin < 3 || isempty(N)
    if iscell(x)
        N = cellfun(@(c) length(c), x);
        N = N(:)';
    else
        N = size(x, 1);
    end
end

% Set default options
if nargin < 4 || isempty(opt)
    opt = struct();
end

% Set default option values
[opt] = set_defaults(opt, 'iValue', 1);
[opt] = set_defaults(opt, 'xfun', []);
[opt] = set_defaults(opt, 'yfun', []);
[opt] = set_defaults(opt, 'plotfun', []);

%% Determine dimensionality
if iscell(x)
    dx = numel(x);
    x = x(:)';
else
    [~, dx] = size(x);
end

% Check dimension support
if dx > 2
    error('fastlpr_plot:UnsupportedDimension', ...
        'Only 1D and 2D plots are supported. Got %d dimensions.', dx);
end

dy = length(opt.iValue);

%% Evaluate function on grid
[y, xgrid] = predict_griddedInterpolant(fpp, x, N, opt.iValue);

%% Apply transformations to x and y
% Apply transformation functions to x-axes (e.g., log scale)
for i = 1:numel(opt.xfun)
    if ~isempty(opt.xfun{i})
        xgrid{i} = opt.xfun{i}(xgrid{i});
    end
end

% Apply transformation function to y-axis
if ~isempty(opt.yfun)
    y = opt.yfun(y);
end

%% Create plot based on dimensionality
if dx == 1
    % 1D plot: line plot
    if iscell(xgrid)
        s = plot(xgrid{1}, y, varargin{:});
    else
        s = plot(xgrid, y, varargin{:});
    end
    
elseif dx == 2
    % 2D plot: surface or contour plot
    for i = 1:dy
        if iscell(xgrid)
            if notemptyfield(opt, 'plotfun')
                % Use custom plotting function
                if isequal(opt.plotfun, @contour3)
                    [~, s] = opt.plotfun(xgrid{:}, y(:,:,i), varargin{:});
                else
                    s = opt.plotfun(xgrid{:}, y(:,:,i), varargin{:});
                    colormap("jet");
                end
            else
                % Default: surface plot
                s = surf(xgrid{:}, y(:,:,i), varargin{:});
                colormap("jet");
            end
        else
            % Scattered data (not on regular grid)
            if notemptyfield(opt, 'plotfun')
                s = opt.plotfun(xgrid(:,1), xgrid(:,2), y(:,i), varargin{:});
            else
                s = scatter3(xgrid(:,1), xgrid(:,2), y(:,i), varargin{:});
            end
        end
    end
end

%% Return outputs if requested
if nargout == 1
    varargout{1} = s;
elseif nargout == 2
    varargout{1} = s;
    varargout{2} = y;
end

end

