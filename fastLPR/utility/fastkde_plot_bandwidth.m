function varargout = fastkde_plot_bandwidth(kde, opt, varargin)
% FASTKDE_PLOT_BANDWIDTH - Visualize bandwidth selection results from cv_fastkde
%
% This function creates visualizations of bandwidth selection (LCV) results
% on the current axes. Users should create figures before calling this function.
%
% Syntax:
%   fastkde_plot_bandwidth(kde)
%   fastkde_plot_bandwidth(kde, opt)
%   fastkde_plot_bandwidth(kde, opt, 'Property', Value, ...)
%   h = fastkde_plot_bandwidth(...)
%
% Inputs:
%   kde - KDE structure from cv_fastkde with fields:
%         .h: Selected bandwidth
%         .lcv: LCV results with fields:
%                      .hlist: All bandwidth candidates
%                      .lcv_m: LCV scores
%                      .id1se: Index of selected bandwidth
%
%   opt - Options structure (optional) with fields:
%         .title: Custom title (default: 'Bandwidth Selection (LCV)')
%         .xlabel: Custom x-label (default: auto)
%         .ylabel: Custom y-label (default: auto)
%         .colormap: Colormap for 2D plots (default: 'parula')
%
%   'Property', Value - Additional plot properties passed to plotting functions
%
% Outputs:
%   h - Structure with handles to plot objects (optional):
%       .ax: Axes handle
%       .lcv: Handle to LCV plot
%       .selected: Handle to selected bandwidth marker
%
% Examples:
%   % Example 1: 1D bandwidth selection
%   x = [randn(100, 1); randn(100, 1) + 3];
%   hlist = get_hlist(20, [0.1, 1], @logspace);
%   kde = cv_fastkde(x, hlist);
%   
%   figure;
%   fastkde_plot_bandwidth(kde);
%
%   % Example 2: Combined plot with density and bandwidth
%   figure;
%   subplot(1, 2, 1);
%   fastkde_plot(kde);
%   subplot(1, 2, 2);
%   fastkde_plot_bandwidth(kde);
%
%   % Example 3: 2D bandwidth selection
%   x = [randn(200, 2); randn(200, 2) + 2];
%   hlist = get_hlist([10, 10], [0.1, 1; 0.1, 1], @logspace);
%   kde = cv_fastkde(x, hlist);
%   
%   figure;
%   fastkde_plot_bandwidth(kde);
%
% See also: cv_fastkde, fastkde_plot, fastlpr_plot
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
if ~isstruct(kde) || ~isfield(kde, 'lcv') || isempty(kde.lcv)
    error('fastkde_plot_bandwidth:InvalidInput', ...
        'Input must be a KDE structure from cv_fastkde with lcv field.');
end

% Set default options
if nargin < 2 || isempty(opt)
    opt = struct();
end

opt = set_defaults(opt, 'title', '');
opt = set_defaults(opt, 'xlabel', '');
opt = set_defaults(opt, 'ylabel', '');
opt = set_defaults(opt, 'colormap', 'parula');

%% Get LCV results
lcv = kde.lcv;
dx = length(kde.h);  % Dimension

%% Create plot
h = struct();
h.ax = gca;

if dx == 1
    % 1D: Line plot
    % lcv.hlist is (dh x 1) matrix, extract as vector
    hlist_vec = lcv.hlist(:, 1);
    h_selected = kde.h(1);  % Extract scalar from row vector

    h.lcv = semilogx(hlist_vec, lcv.lcv_m, 'ko-', 'LineWidth', 1.5, ...
        'MarkerSize', 6, 'DisplayName', 'LCV scores', varargin{:});hold on;

    h.selected = semilogx(h_selected, lcv.lcv_m(lcv.id1se), 'r*', ...
        'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', 'Selected');
    % semilogx(h_selected, lcv.lcv_m(lcv.id1se), 'ro', ...
    %     'MarkerSize', 12, 'LineWidth', 2);
    
    % Labels
    if isempty(opt.xlabel)
        xlabel('Bandwidth h', 'FontSize', 12);
    else
        xlabel(opt.xlabel, 'FontSize', 12);
    end
    
    if isempty(opt.ylabel)
        ylabel('Log-Likelihood CV Score', 'FontSize', 12);
    else
        ylabel(opt.ylabel, 'FontSize', 12);
    end
    
    legend('Location', 'best', 'FontSize', 10);
    
else
    % 2D: Heatmap
    n_h = sqrt(length(lcv.lcv_m));
    lcv_matrix = reshape(lcv.lcv_m, [n_h, n_h]);
    h1_vals = unique(lcv.hlist(:,1));
    h2_vals = unique(lcv.hlist(:,2));

    h.lcv = imagesc(log10(h1_vals), log10(h2_vals), (lcv_matrix'), varargin{:});hold on;%symlog
    colormap(h.ax, opt.colormap);
    colorbar;
    hold on;
    
    h.selected = plot(log10(kde.h(1)), log10(kde.h(2)), 'r*', ...
        'MarkerSize', 25, 'LineWidth', 3);
    % plot(log10(kde.h(1)), log10(kde.h(2)), 'ro', ...
    %     'MarkerSize', 18, 'LineWidth', 2.5);
    text(log10(kde.h(1)), log10(kde.h(2))-0.15, ...
        sprintf('h=[%.2f, %.2f]', kde.h(1), kde.h(2)), ...
        'Color', 'r', 'FontSize', 10, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center');
    
    % Labels
    if isempty(opt.xlabel)
        xlabel('log_{10}(h_1)', 'FontSize', 12);
    else
        xlabel(opt.xlabel, 'FontSize', 12);
    end
    
    if isempty(opt.ylabel)
        ylabel('log_{10}(h_2)', 'FontSize', 12);
    else
        ylabel(opt.ylabel, 'FontSize', 12);
    end
    
    axis xy;
end

% Title
if ~isempty(opt.title)
    title(opt.title, 'FontSize', 13, 'FontWeight', 'bold');
else
    title('Bandwidth Selection (LCV)', 'FontSize', 13, 'FontWeight', 'bold');
end

grid on;
box on;

%% Return outputs if requested
if nargout > 0
    varargout{1} = h;
end

end

%% Helper function: set_defaults
function opt = set_defaults(opt, field, value)
    if ~isfield(opt, field) || isempty(opt.(field))
        opt.(field) = value;
    end
end

