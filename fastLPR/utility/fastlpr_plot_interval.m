function fastlpr_plot_interval(x, ci)
% FASTLPR_PLOT_CI - Visualize interval bands for regression estimates
%
% This function creates shaded bands for local polynomial regression
% estimates. It handles both 1D (filled area) and 2D (transparent surface) cases.
% The intervals are typically computed using fastlpr_interval().
%
% Syntax:
%   fastlpr_plot_interval([], ci)
%   fastlpr_plot_interval(x, ci)
%
% Inputs:
%   x   - Evaluation points (optional, default: use grid from ci)
%         For 1D: vector of x-coordinates
%         For 2D: not used (set to [])
%         Typically set to [] to use grid from ci object
%
%   ci  - Confidence interval object from fastlpr_interval()
%         griddedInterpolant with last dimension containing [lower, upper] bounds
%
% Outputs:
%   None (creates plot in current axes)
%
% Examples:
%   % Example 1: 1D intervals
%   x = rand(500, 1) * 4 - 2;
%   y_true = x.^3;
%   sigma_true = 1 + 4*exp(-x.^2);  % Heteroscedastic variance
%   y = y_true + randn(500, 1) .* sqrt(sigma_true);
%   
%   % Estimate mean
%   hlist = get_hlist(20, [0.01, 1], @logspace);
%   opt.order = 1;
%   opt.dstd = 0;
%   regs_mu = cv_fastlpr(x, y, hlist, opt);
%   
%   % Estimate variance
%   residuals = y - regs_mu.yhat;
%   opt.y_type_out = 'variance';
%   opt.dstd = 10;
%   regs_sigma = cv_fastlpr(x, residuals.^2, hlist, opt);
%   
%   % Compute and plot intervals
%   ci = fastlpr_interval(regs_mu, regs_sigma, 0.05);
%   
%   figure; hold on;
%   scatter(x, y, 'k.', 'DisplayName', 'Data');
%   fastlpr_plot(regs_mu.fpp_yhat, [], [], [], 'LineWidth', 2, 'Color', 'r', 'DisplayName', 'Mean');
%   fastlpr_plot_interval([], ci);  % Add interval band
%   legend('Location', 'best');
%   xlabel('X'); ylabel('Y');
%   title('1D Regression with 95% Confidence Intervals');
%
%   % Example 2: 2D intervals
%   x = rand(1000, 2);
%   y = sin(2*pi*x(:,1)) .* cos(2*pi*x(:,2)) + 0.2*randn(1000, 1);
%   
%   hlist = get_hlist([10, 10], [0.01, 1; 0.01, 1], @logspace);
%   opt.order = 1;
%   
%   % Estimate mean
%   regs_mu = cv_fastlpr(x, y, hlist, opt);
%   
%   % Estimate variance
%   residuals = y - regs_mu.yhat;
%   opt.y_type_out = 'variance';
%   opt.dstd = 10;
%   regs_sigma = cv_fastlpr(x, residuals.^2, hlist, opt);
%   
%   % Compute intervals
%   ci = fastlpr_interval(regs_mu, regs_sigma, 0.05);
%   
%   % Plot mean surface
%   figure; hold on;
%   fastlpr_plot(regs_mu.fpp_yhat);
%   
%   % Add interval surfaces
%   fastlpr_plot_interval([], ci);
%   xlabel('X1'); ylabel('X2'); zlabel('Y');
%   title('2D Regression with 95% Confidence Intervals');
%
%   % Example 3: Multiple confidence levels
%   ci_95 = fastlpr_interval(regs_mu, regs_sigma, 0.05);  % 95%
%   ci_99 = fastlpr_interval(regs_mu, regs_sigma, 0.01);  % 99%
%   
%   figure; hold on;
%   scatter(x, y, 'k.');
%   fastlpr_plot(regs_mu.fpp_yhat, [], [], [], 'LineWidth', 2, 'Color', 'r');
%   fastlpr_plot_interval([], ci_95);  % Inner band (95%)
%   fastlpr_plot_interval([], ci_99);  % Outer band (99%)
%
% Notes:
%   - For 1D: Creates filled area between lower and upper bounds
%   - For 2D: Creates transparent surfaces for lower and upper bounds
%   - The shaded region is green with 20% transparency (1D) or 40% (2D)
%   - Always set x = [] to use the grid from ci object (recommended)
%   - The intervals should be computed using fastlpr_interval()
%
% Visualization Details:
%   1D: Uses fill() to create shaded area
%       - Color: green
%       - FaceAlpha: 0.2 (20% opacity)
%       - EdgeAlpha: 0 (no edge)
%   
%   2D: Uses fastlpr_plot() to create surfaces
%       - Color: green
%       - FaceAlpha: 0.4 (40% opacity)
%       - EdgeAlpha: 0.1 (10% edge opacity)
%
% See also: fastlpr_interval, fastlpr_plot, cv_fastlpr, fill, surf
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com, 
%                Min Li, minli.231314@gmail.com 
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

%% Input validation
narginchk(2, 2);

% Validate ci
if ~isa(ci, 'griddedInterpolant')
    error('fastlpr_plot_interval:InvalidInput', ...
        'Input ci must be a griddedInterpolant object from fastlpr_interval().');
end

% Get dimensionality
dx = numel(ci.GridVectors);

% Check dimension support
if dx > 2
    error('fastlpr_plot_interval:UnsupportedDimension', ...
        'Only 1D and 2D intervals are supported. Got %d dimensions.', dx);
end

%% Create interval plot based on dimensionality
if dx == 1
    % 1D intervals: filled area
    
    if isobject(ci) && isempty(x)
        % Extract confidence bounds from griddedInterpolant
        [ci_values, x_grid] = predict_griddedInterpolant(ci, [], [], [1, 2]);
        
        % Create filled area between lower and upper bounds
        % ci_values(:,1) = upper bound, ci_values(:,2) = lower bound
        % IMPORTANT: Set HandleVisibility='off' to prevent "data1" in legend
        fill([x_grid{1}; flipud(x_grid{1})], ...
             [ci_values(:,1); flipud(ci_values(:,2))], ...
             'g', 'FaceAlpha', 0.2, 'EdgeAlpha', 0, 'EdgeColor', 'w', 'HandleVisibility', 'off');
    else
        error('fastlpr_plot_interval:InvalidInput', ...
            'For 1D plots, set x = [] and provide ci from fastlpr_interval().');
    end
    
elseif dx == 2
    % 2D intervals: transparent surfaces
    
    % Plot both lower and upper bound surfaces
    % IMPORTANT: Set HandleVisibility='off' to prevent "data1", "data2" in legend
    opt.iValue = [1, 2];  % Plot both bounds
    fastlpr_plot(ci, [], [], opt, ...
        'FaceColor', 'g', 'FaceAlpha', 0.4, ...
        'EdgeAlpha', 0.1, 'EdgeColor', 'w', 'HandleVisibility', 'off');
end

end

