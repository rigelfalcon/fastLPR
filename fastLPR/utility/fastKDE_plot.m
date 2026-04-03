function varargout = fastkde_plot(kde, opt, varargin)
% FASTKDE_PLOT - Visualize kernel density estimation results from cv_fastkde
%
% This function creates visualizations of KDE results on the current axes.
% It handles both 1D (line plots with histogram) and 2D (contour/surface plots).
% Users should create figures before calling this function.
%
% Syntax:
%   fastkde_plot(kde)
%   fastkde_plot(kde, opt)
%   fastkde_plot(kde, opt, 'Property', Value, ...)
%   h = fastkde_plot(...)
%
% Inputs:
%   kde - KDE structure from cv_fastkde with fields:
%         .fhat: Density estimate at grid points
%         .fpp: griddedInterpolant object
%         .xlist: Grid vectors (cell array)
%         .xraw: Original data points
%         .h: Selected bandwidth
%
%   opt - Options structure (optional) with fields:
%         .plot_type: Type of plot (default: 'auto')
%                     '1d': 1D density with histogram
%                     '2d_contour': 2D contour plot
%                     '2d_surface': 2D surface plot
%                     'auto': Automatically choose based on dimension
%         .show_data: Show raw data points (default: true)
%         .show_histogram: Show histogram for 1D (default: true)
%         .n_bins: Number of histogram bins (default: 30)
%         .colormap: Colormap for 2D plots (default: 'jet')
%         .alpha: Transparency for data points (default: 0.4)
%         .linewidth: Line width for 1D plots (default: 2)
%         .title: Custom title (default: auto)
%         .xlabel: Custom x-label (default: 'x')
%         .ylabel: Custom y-label (default: 'Density' for 1D, 'y' for 2D)
%         .zlabel: Custom z-label for 2D (default: 'Density')
%
%   'Property', Value - Additional plot properties passed to plotting functions
%
% Outputs:
%   h - Structure with handles to plot objects (optional):
%       .ax: Axes handle
%       .density: Handle to density plot
%       .data: Handle to data plot (if shown)
%       .histogram: Handle to histogram (if shown)
%
% Examples:
%   % Example 1: Simple 1D plot
%   x = [randn(100, 1); randn(100, 1) + 3];
%   hlist = get_hlist(20, [0.1, 1], @logspace);
%   kde = cv_fastkde(x, hlist);
%
%   figure; hold on;
%   fastkde_plot(kde);
%
%   % Example 2: 1D plot without histogram
%   opt.show_histogram = false;
%   figure; hold on;
%   fastkde_plot(kde, opt);
%
%   % Example 3: 2D contour plot
%   x = [randn(200, 2); randn(200, 2) + 2];
%   hlist = get_hlist([10, 10], [0.1, 1; 0.1, 1], @logspace);
%   kde = cv_fastkde(x, hlist);
%
%   figure;
%   opt.plot_type = '2d_contour';
%   fastkde_plot(kde, opt);
%
%   % Example 4: Bandwidth selection (separate function)
%   figure;
%   subplot(1, 2, 1);
%   fastkde_plot(kde);
%   subplot(1, 2, 2);
%   fastkde_plot_bandwidth(kde);
%
% See also: cv_fastkde, fastkde_plot_bandwidth, fastlpr_plot, histogram, contour, surf
%
% Author: Ying Wang, Min Li
% Create Time: 2025
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

%% Input validation
narginchk(1, inf);

% Validate kde structure
if ~isstruct(kde) || ~isfield(kde, 'fhat') || ~isfield(kde, 'xlist')
    error('fastkde_plot:InvalidInput', ...
        'First input must be a KDE structure from cv_fastkde.');
end

% Set default options
if nargin < 2 || isempty(opt)
    opt = struct();
end

% Set default option values
opt = set_defaults(opt, 'plot_type', 'auto');
opt = set_defaults(opt, 'show_data', true);
opt = set_defaults(opt, 'show_histogram', true);
opt = set_defaults(opt, 'n_bins', 30);
opt = set_defaults(opt, 'colormap', 'jet');
opt = set_defaults(opt, 'alpha', 0.4);
opt = set_defaults(opt, 'linewidth', 2);
opt = set_defaults(opt, 'title', '');
opt = set_defaults(opt, 'xlabel', 'x');
opt = set_defaults(opt, 'ylabel', '');
opt = set_defaults(opt, 'zlabel', 'Density');

%% Determine dimensionality
dx = length(kde.xlist);

% Auto-detect plot type
if strcmp(opt.plot_type, 'auto')
    if dx == 1
        opt.plot_type = '1d';
    else
        opt.plot_type = '2d_contour';
    end
end

% Set default ylabel
if isempty(opt.ylabel)
    if dx == 1 || contains(opt.plot_type, '2d')
        opt.ylabel = 'Density';
    else
        opt.ylabel = 'y';
    end
end

%% Create plot based on type
switch opt.plot_type
    case '1d'
        h = plot_1d(kde, opt, varargin{:});

    case '2d_contour'
        h = plot_2d_contour(kde, opt, varargin{:});

    case '2d_surface'
        h = plot_2d_surface(kde, opt, varargin{:});

    otherwise
        error('fastkde_plot:InvalidPlotType', ...
            'Invalid plot_type: %s. Valid options: 1d, 2d_contour, 2d_surface, auto', ...
            opt.plot_type);
end

%% Return outputs if requested
if nargout >= 1
    varargout{1} = h;
end

end

%% Helper function: 1D density plot
function h = plot_1d(kde, opt, varargin)
    h = struct();
    h.ax = gca;
    hold on;

    % Plot histogram
    if opt.show_histogram && isfield(kde, 'xraw')
        h.histogram = histogram(kde.xraw, opt.n_bins, 'Normalization', 'pdf', ...
            'FaceColor', [0.7, 0.7, 0.7], 'EdgeColor', 'none', ...
            'FaceAlpha', 0.5, 'DisplayName', 'Histogram');
    end

    % Plot density
    h.density = plot(kde.xlist{1}, kde.fhat, 'b-', 'LineWidth', opt.linewidth, ...
        'DisplayName', sprintf('KDE (h=%.3f)', kde.h), varargin{:});

    % Plot data points
    if opt.show_data && isfield(kde, 'xraw')
        y_pos = min(kde.fhat) - 0.05 * range(kde.fhat);
        h.data = plot(kde.xraw, y_pos * ones(size(kde.xraw)), 'k|', ...
            'MarkerSize', 10, 'DisplayName', 'Data');
    end

    % Labels and formatting
    xlabel(opt.xlabel, 'FontSize', 12);
    ylabel(opt.ylabel, 'FontSize', 12);
    if ~isempty(opt.title)
        title(opt.title, 'FontSize', 13, 'FontWeight', 'bold');
    else
        title('Kernel Density Estimate', 'FontSize', 13, 'FontWeight', 'bold');
    end
    legend('Location', 'best', 'FontSize', 10);
    grid on;
    box on;
end

%% Helper function: 2D contour plot
function h = plot_2d_contour(kde, opt, varargin)
    h = struct();
    h.ax = gca;
    hold on;

    % Create meshgrid
    [X1, X2] = meshgrid(kde.xlist{1}, kde.xlist{2});

    % Plot contour
    h.density = contourf(X1, X2, kde.fhat', 20, 'LineStyle', 'none', varargin{:});
    % h.density = surf(X1, X2, kde.fhat','EdgeColor','none');
    colormap(h.ax, opt.colormap);
    colorbar;

    % Plot data points
    if opt.show_data && isfield(kde, 'xraw')
        h.data = scatter(kde.xraw(:,1), kde.xraw(:,2), 5, 'k', 'filled', ...
            'MarkerFaceAlpha', opt.alpha, 'DisplayName', 'Data');
    end

    % Labels and formatting
    xlabel(sprintf('%s_1', opt.xlabel), 'FontSize', 12);
    ylabel(sprintf('%s_2', opt.xlabel), 'FontSize', 12);
    if ~isempty(opt.title)
        title(opt.title, 'FontSize', 13, 'FontWeight', 'bold');
    else
        title(sprintf('2D KDE (h=[%.2f, %.2f])', kde.h(1), kde.h(2)), ...
            'FontSize', 13, 'FontWeight', 'bold');
    end
    % axis equal tight;
    box on;
end

%% Helper function: 2D surface plot
function h = plot_2d_surface(kde, opt, varargin)
    h = struct();
    h.ax = gca;

    % Create meshgrid
    [X1, X2] = meshgrid(kde.xlist{1}, kde.xlist{2});

    % Plot surface
    h.density = surf(X1, X2, kde.fhat', varargin{:});
    colormap(h.ax, opt.colormap);
    colorbar;
    shading interp;

    % Labels and formatting
    xlabel(sprintf('%s_1', opt.xlabel), 'FontSize', 12);
    ylabel(sprintf('%s_2', opt.xlabel), 'FontSize', 12);
    zlabel(opt.zlabel, 'FontSize', 12);
    if ~isempty(opt.title)
        title(opt.title, 'FontSize', 13, 'FontWeight', 'bold');
    else
        title(sprintf('2D KDE (h=[%.2f, %.2f])', kde.h(1), kde.h(2)), ...
            'FontSize', 13, 'FontWeight', 'bold');
    end
    view(3);
    box on;
end

%% Helper function: set_defaults
function opt = set_defaults(opt, field, value)
    if ~isfield(opt, field) || isempty(opt.(field))
        opt.(field) = value;
    end
end

