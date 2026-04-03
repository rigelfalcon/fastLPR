function [type,byte]=min_numerical_type(minval,maxval,redundancy,isunsign,isfloating)
% MIN_NUMERICAL_TYPE - Find smallest numeric type that can represent a range
%
% Determines the most memory-efficient numeric type (uint8, int16, single, etc.)
% that can represent values in the range [minval, maxval].
%
% Syntax:
%   type = min_numerical_type(minval, maxval)
%   [type, byte] = min_numerical_type(minval, maxval, redundancy, isunsign, isfloating)
%
% Inputs:
%   minval     - Minimum value to represent (default: 0)
%   maxval     - Maximum value to represent (required)
%   redundancy - Safety factor for max value (default: 1)
%   isunsign   - Force unsigned type (default: auto from minval>=0)
%   isfloating - Force floating-point type (default: auto from integer check)
%
% Outputs:
%   type - String name of optimal type ('uint8', 'int16', 'single', etc.)
%   byte - Bytes per element for this type
%
% Example:
%   [type, byte] = min_numerical_type(0, 1000)
%   % Returns: type='uint16', byte=2 (can hold 0-65535)
%
% Type Selection Priority:
%   1. Unsigned integers: uint8, uint16, uint32, uint64
%   2. Signed integers: int8, int16, int32, int64
%   3. Floating-point: single, double
%
% Notes:
%   - Selects smallest type where: type_max * redundancy > maxval
%   - Useful for memory optimization in large arrays
%   - Errors if no type can represent the range
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

%% Input validation and defaults
if nargin<1 || isempty(minval)
    minval = 0;
end
if nargin<2 || isempty(maxval)
    error('fastLPR:minType:MissingMaxval', 'maxval is required');
end
if nargin<3 || isempty(redundancy)
    redundancy = 1;
end
if nargin<4 || isempty(isunsign)
    isunsign = minval>=0;  % Auto-detect: unsigned if minval >= 0
end
if nargin<5 || isempty(isfloating)
    isfloating = ~all(isintegernum([minval,maxval]));  % Auto-detect: float if non-integer
end
%% Define available types
unint_types={'uint8'; 'uint16'; 'uint32'; 'uint64'};
int_types={'int8'; 'int16'; 'int32'; 'int64';};
float_types={'single'; 'double';};

%% Select type candidates based on requirements
if isunsign && ~isfloating
    % Unsigned integers preferred, then signed, then float
    types=[unint_types;int_types;float_types];
    types_max_val=cellfun(@(x) double(intmax(x)),[unint_types;int_types]);
    types_max_val=[types_max_val;cellfun(@(x) double(realmax(x)),[float_types])];
elseif ~isunsign && ~isfloating
    % Signed integers preferred, then float
    types=[int_types;float_types];
    types_max_val=cellfun(@(x) double(intmax(x)),[int_types]);
    types_max_val=[types_max_val;cellfun(@(x) double(realmax(x)),[float_types])];
elseif isfloating
    % Only floating-point types
    types=[float_types];
    types_max_val=[cellfun(@(x) double(realmax(x)),[float_types])];
end

%% Find smallest type that can hold maxval with redundancy
% Check: type_max * redundancy > maxval
idx_avalible=(types_max_val*redundancy-maxval)>0;

if ~any(idx_avalible)
    error('fastLPR:minType:Overflow', ...
        'No numeric type can represent maxval=%.2e with redundancy=%.2f', ...
        maxval, redundancy);
end

% Select smallest available type
[~,idx_min]=min(types_max_val(idx_avalible));
type=types(idx_avalible);
type=type{idx_min};

% Get byte size
byte=bytesize(1,'B',type);
byte=byte.B;

end