function y=iscomplex(x,dim)
% ISCOMPLEX - Check if array has non-zero imaginary part
%
% Returns true for columns/elements that have non-zero imaginary parts.
% More strict than MATLAB's built-in isreal (which checks if ANY element is complex).
%
% Syntax:
%   y = iscomplex(x)
%   y = iscomplex(x, dim)
%
% Inputs:
%   x   - Input array (numeric)
%   dim - Dimension along which to check (default: 'all')
%         'all' - Check if all elements are complex
%         1     - Check each column
%         2     - Check each row
%
% Output:
%   y - Logical array indicating complex-valued elements/columns/rows
%
% Example:
%   x = [1+2i, 3; 4+5i, 6];
%   iscomplex(x, 1)  % [true, false] - first column is complex
%
% Notes:
%   - Returns true only if imaginary part is non-zero
%   - Different from MATLAB's isreal: isreal(x) = ~any(iscomplex(x,'all'))
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

if nargin<2 || isempty(dim)
    dim='all';
end

% Check if imaginary part is non-zero along specified dimension
y=all(imag(x)~=0,dim);

end