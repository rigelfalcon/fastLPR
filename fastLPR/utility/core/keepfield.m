function S=keepfield(S,fields)
% KEEPFIELD - Keep only specified fields in a structure
%
% Removes all fields except those specified, opposite of rmfield.
%
% Syntax:
%   S = keepfield(S, fields)
%
% Inputs:
%   S      - Input structure
%   fields - Field name(s) to keep (string, char, or cell array)
%
% Output:
%   S - Structure with only specified fields
%
% Example:
%   S = struct('a', 1, 'b', 2, 'c', 3);
%   S = keepfield(S, {'a', 'c'});  % Keeps only 'a' and 'c'
%
% Notes:
%   - Opposite of rmfield (removes all except specified)
%   - Useful for cleaning up large structures
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

% Convert to cell array if needed
if ischar(fields)|| isstring(fields)
    fields = {fields};
end

% Find fields to remove (all except specified)
fields = setdiff(fieldnames(S), fields);

% Remove unwanted fields
S = rmfield(S, fields);

end



