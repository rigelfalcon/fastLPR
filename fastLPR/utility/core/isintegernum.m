function result=isintegernum(x)
% ISINTEGERNUM - Check if numeric values are integers
%
% Returns true for elements that are integer-valued (even if stored as double).
% Different from isinteger() which checks data type.
%
% Syntax:
%   result = isintegernum(x)
%
% Input:
%   x - Numeric array
%
% Output:
%   result - Logical array, true where x has integer values
%
% Example:
%   isintegernum([1.0, 1.5, 2.0])  % [true, false, true]
%   isinteger([1.0, 1.5, 2.0])     % false (checks type, not value)
%
% Notes:
%   - Checks if x == round(x)
%   - Works with any numeric type
%   - Useful for determining optimal integer type
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

result = x == round(x);

end
