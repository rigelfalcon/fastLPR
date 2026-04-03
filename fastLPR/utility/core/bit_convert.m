function varargout=bit_convert(accuracy,varargin)
% BIT_CONVERT - Convert arrays to appropriate precision based on accuracy
%
% Converts input arrays to single or double precision based on required
% accuracy (in decimal digits). Used to optimize memory and speed.
%
% Syntax:
%   [out1, out2, ...] = bit_convert(accuracy, in1, in2, ...)
%
% Inputs:
%   accuracy - Required accuracy in decimal digits, or 'single'/'double'
%              < 5: convert to single (23-bit mantissa, ~7 digits)
%              >= 5: convert to double (52-bit mantissa, ~15 digits)
%   varargin - Input arrays to convert
%
% Outputs:
%   varargout - Converted arrays in single or double precision
%
% Example:
%   [x, y] = bit_convert(3, x, y);  % Convert to single (3 < 5)
%   [x, y] = bit_convert(10, x, y); % Convert to double (10 >= 5)
%   [x, y] = bit_convert('single', x, y);  % Explicit single
%
% Notes:
%   - Single precision: 4 bytes, ~7 decimal digits, faster
%   - Double precision: 8 bytes, ~15 decimal digits, more accurate
%   - Threshold of 5 digits balances accuracy and performance
%
% Precision Reference:
%   - log2(eps('single'))  -23 (mantissa bits)
%   - log2(eps('double'))  -52 (mantissa bits)
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

%% Convert based on accuracy requirement
if isnumeric(accuracy)
    % Numeric accuracy: threshold at 5 digits
    if accuracy<5
        % Low accuracy: use single precision (faster, less memory)
        varargout=cellfun(@(x) single(x), varargin,'UniformOutput',false);
    else
        % High accuracy: use double precision (more accurate)
        varargout=cellfun(@(x) double(x), varargin,'UniformOutput',false);
    end
elseif ischar(accuracy)
    % Explicit precision specification
    switch accuracy
        case 'single'
            varargout=cellfun(@(x) single(x), varargin,'UniformOutput',false);
        case 'double'
            varargout=cellfun(@(x) double(x), varargin,'UniformOutput',false);
    end
end

end

%% Precision reference
% log2(eps('single')) = -23  (mantissa bits)
% log2(eps('double')) = -52  (mantissa bits)