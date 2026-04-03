function tf=notemptyfield(s,field)
% NOTEMPTYFIELD - Check if structure field exists and is non-empty
%
% Returns true only if field exists AND is not empty.
%
% Syntax:
%   tf = notemptyfield(s, field)
%
% Inputs:
%   s     - Structure to check
%   field - Field name (string or char)
%
% Output:
%   tf - true if field exists and is non-empty, false otherwise
%
% Example:
%   s = struct('a', 1, 'b', []);
%   notemptyfield(s, 'a')  % true
%   notemptyfield(s, 'b')  % false (empty)
%   notemptyfield(s, 'c')  % false (doesn't exist)
%
% Notes:
%   - Combines isfield() and ~isempty() checks
%   - Useful for optional structure fields
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

tf=isfield(s,field) && ~isempty(s.(field));

end