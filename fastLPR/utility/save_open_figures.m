function filenames = save_open_figures(outputDir, prefix)
% SAVE_OPEN_FIGURES - Export all open figures as PNG and FIG files
%
% This utility function saves all currently open MATLAB figures to disk
% in both PNG (300 DPI) and FIG formats. Useful for batch exporting
% figures after running multiple examples.
%
% Syntax:
%   filenames = save_open_figures(outputDir)
%   filenames = save_open_figures(outputDir, prefix)
%
% Inputs:
%   outputDir - Directory to save figures (string or char)
%               Will be created if it doesn't exist
%   prefix    - Filename prefix (optional, default: 'figure')
%               Figures will be named: prefix_01.png, prefix_02.png, etc.
%
% Outputs:
%   filenames - Cell array of PNG file paths
%               One entry per saved figure
%
% Examples:
%   % Save all open figures to a directory
%   figure; plot(1:10);
%   figure; surf(peaks);
%   filenames = save_open_figures('output', 'myfig');
%   % Creates: output/myfig_01.png, output/myfig_01.fig,
%   %          output/myfig_02.png, output/myfig_02.fig
%
%   % Save with default prefix
%   filenames = save_open_figures('results');
%   % Creates: results/figure_01.png, results/figure_01.fig, ...
%
% Notes:
%   - PNG files are saved at 300 DPI for publication quality
%   - FIG files preserve full editability in MATLAB
%   - Figures are saved in creation order
%   - Directory is created automatically if it doesn't exist
%
% See also: exportgraphics, savefig, print
%
% Author: Ying Wang, Min Li
% Create Time: 2024
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

if nargin < 2
    prefix = 'figure';
end

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

figHandles = findall(groot, 'Type', 'figure');
figHandles = flipud(figHandles);  % Preserve creation order

filenames = cell(numel(figHandles), 1);

for k = 1:numel(figHandles)
    fig = figHandles(k);
    pngName = sprintf('%s_%02d.png', prefix, k);
    figName = sprintf('%s_%02d.fig', prefix, k);

    pngPath = fullfile(outputDir, pngName);
    figPath = fullfile(outputDir, figName);

    exportgraphics(fig, pngPath, 'Resolution', 300);
    savefig(fig, figPath);

    filenames{k} = pngPath;
end

end
