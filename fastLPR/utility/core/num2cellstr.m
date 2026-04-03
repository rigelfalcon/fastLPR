function out=num2cellstr(num)
% NUM2CELLSTR - Convert numeric array to cell array of strings
%
% Converts each number to a string, removes spaces, preserves array shape.
%
% Syntax:
%   out = num2cellstr(num)
%
% Input:
%   num - Numeric array (any size)
%
% Output:
%   out - Cell array of strings (same size as input)
%
% Example:
%   num2cellstr([1.5, 2.3; 3.7, 4.1])
%   % Returns: {'1.5', '2.3'; '3.7', '4.1'}
%
% Notes:
%   - Removes all spaces from string representation
%   - Preserves array dimensions
%   - Useful for creating labels or filenames
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

sz=size(num);
out=cellstr(num2str(num(:)));
out = erase(out,' ');  % Remove spaces
out=reshape(out,sz);   % Restore original shape

end