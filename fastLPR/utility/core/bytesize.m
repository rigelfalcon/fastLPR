function [bsize,scale] = bytesize( b, unit, type )
% Copyright (c) 2024-2025 Ying Wang, Min Li
% SPDX-License-Identifier: GPL-3.0-or-later
%
% BYTESIZE - Calculate memory size with automatic unit conversion
%
% Computes memory size and converts to appropriate units (B, kB, MB, GB, TB, PB).
% Can calculate size from number of elements or measure actual variable size.
%
% Syntax:
%   bsize = bytesize(b)
%   bsize = bytesize(b, unit)
%   [bsize, scale] = bytesize(b, unit, type)
%   bytesize(b)  % Prints size to console
%
% Inputs:
%   b    - Number of elements, or variable to measure
%          If scalar: number of elements (requires type)
%          If non-scalar: measures actual variable size
%   unit - 'B' for bytes (default), 'b' for bits
%   type - Numeric type: 'double', 'single', 'int8', etc.
%          Only used when b is scalar
%
% Outputs:
%   bsize - Struct with fields: B, kB, MB, GB, TB, PB
%   scale - Recommended unit string (e.g., 'MB' for megabytes)
%
% Example:
%   [bsize, scale] = bytesize(1000, 'B', 'double')
%   % Returns: bsize.kB = 7.8125, scale = 'kB'
%
%   x = rand(1000, 1000);
%   bytesize(x)  % Prints: "7.63 MB"
%
% Notes:
%   - Uses 1024-based units (binary prefixes)
%   - For bits, multiply byte size by 8
%   - Handle class size estimation may be inaccurate
%   - If no output, prints size with recommended unit
%
% Modified from: https://github.com/OHBA-analysis/osl-core
%


if nargin < 3 || isempty(type), type=['']; end
if nargin < 2|| isempty(unit), unit='B'; end

if ~isnumeric(b) || ~isscalar(b)
    w = whos('b');
    b = w.bytes;
else % isnumeric(b) && isscalar(b)
    if ischar(type)
        switch type
            case {'uint64','int64','double'}
                b = 8*b;
            case {'uint32','int32','single'}
                b = 4*b;
            case {'uint16','int16'}
                b = 2*b;
            case {'uint8','int8'}
                
            otherwise
                error('not include')                
        end
    elseif isnumeric(type)
        w = whos('type');
        b = w.bytes*b;
    end
end

if strcmp(unit,'b')
    b=b*8;
end


assert( isnumeric(b) && isscalar(b), 'Bad input bytesize.' );

units = mapfun( @(x) [x unit], {'','k','M','G','T','P'}, false );
scale = units{min( numel(units), 1+floor( log(b)/log(1024) ) )};

for i = 1:numel(units)
    u = units{i};
    bsize.(u) = b / 1024^(i-1);
end

if nargout == 0
    fprintf('%.2f %s\n', bsize.(scale), scale );
end

end
function out = mapfun( fun, val, unif )
%
% out = osl_util.mapfun( fun, val, unif=false )
%
% Use cellfun, arrayfun or structfun depending on the type of input.
% Returns a cell by default; set unif=true if you would like an array.
%
% It is fine if fun doesn't return anything, but then you should not collect the output.
%
% See also: cellfun, arrayfun, structfun
% 
% JH

    if nargin < 3, unif=false; end
    
    if iscell(val)
        map = @cellfun;
    elseif isscalar(val) && isstruct(val)
        map = @structfun;
    else
        map = @arrayfun;
    end
    
    if nargout == 0
        map( fun, val, 'UniformOutput', unif );
    else
        out = map( fun, val, 'UniformOutput', unif );
    end

end
