function output=readsubtable(path_table,key)
% READSUBTABLE - Read selected columns from table file
%
% Reads a table file (CSV, Excel, etc.) with optional column selection.
% Wrapper around readtable with automatic import options detection.
%
% Syntax:
%   output = readsubtable(path_table)
%   output = readsubtable(path_table, key)
%
% Inputs:
%   path_table - Path to table file (CSV, Excel, etc.)
%   key        - Column names to read (string, char, or cell array)
%                If empty or omitted, reads all columns
%
% Output:
%   output - MATLAB table with selected columns
%
% Example:
%   % Read all columns
%   data = readsubtable('data.csv');
%
%   % Read specific columns
%   data = readsubtable('data.csv', {'Age', 'Height'});
%
% Notes:
%   - Uses detectImportOptions for automatic format detection
%   - Supports CSV, Excel, text files
%   - More efficient than reading all then selecting
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

%% Detect import options
opts = detectImportOptions(path_table);

%% Select specific columns if requested
if nargin>1 &&~isempty(key)
    opts.SelectedVariableNames = key;
end

%% Read table
output=readtable(path_table,opts);

end



