function idx=gen_loopgroup(num_y,num_y_loop,col_in_y,isflip)
% GEN_LOOPGROUP - Generate index groups for batch processing
%
% Splits a set of items into batches for memory-efficient loop processing.
% Used to process multiple response variables in batches when memory is limited.
%
% Syntax:
%   idx = gen_loopgroup(num_y, num_y_loop)
%   idx = gen_loopgroup(num_y, num_y_loop, col_in_y)
%   idx = gen_loopgroup(num_y, num_y_loop, col_in_y, isflip)
%
% Inputs:
%   num_y      - Total number of items to split
%   num_y_loop - Target number of items per batch (will be adjusted)
%   col_in_y   - Number of columns per item (default: 1)
%                Used for bandwidth-specific responses
%   isflip     - Whether to reverse batch order (default: false)
%                Useful for processing in reverse order
%
% Output:
%   idx - Cell array of index vectors, one per batch
%         Each cell contains indices for that batch
%
% Example:
%   % Split 100 items into batches of ~30
%   idx = gen_loopgroup(100, 30);
%   % Result: {1:33, 34:66, 67:100} (3 batches of ~33 each)
%
% Algorithm:
%   1. Compute number of batches: num_loop = ceil(num_y / num_y_loop)
%   2. Adjust batch size: num_y_loop = ceil(num_y / num_loop)
%   3. Create evenly-sized batches (last batch may be smaller)
%   4. Scale indices by col_in_y if needed
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

%% Input validation and defaults
if nargin<4 || isempty(isflip)
    isflip=false;
end
if nargin<3 || isempty(col_in_y)
    col_in_y=1;
end

% Ensure at least one item per batch
assert(num_y_loop>=1,'fastLPR:loopgroup:InvalidBatchSize', ...
    'Each batch must have at least one item (num_y_loop >= 1)');

%% Compute batch parameters
num_y_loop=floor(num_y_loop);  % Round down to integer

% Compute number of batches needed
num_loop=ceil(num_y/num_y_loop);

% Adjust batch size to distribute items evenly
% This ensures all batches have similar size
num_y_loop=ceil(num_y/num_loop);

%% Create batch boundaries
% up: end index of each batch
% down: start index of each batch
up=num_y_loop:num_y_loop:num_y;
down=1:num_y_loop:num_y;

% Handle edge cases
if isempty(up)
    up(1)=num_y;  % Single batch with all items
end
if up(end)<num_y
    up(end+1)=num_y;  % Extend last batch to include remaining items
end

%% Generate index groups
idx=cell(num_loop,1);
for i=1:num_loop
    idx{i}=down(i):up(i);
end

%% Reverse order if requested
if isflip
    idx=flip(idx,1);  % Reverse batch order
    idx=cellfun(@(x) flip(x),idx,'UniformOutput',false);  % Reverse indices within each batch
end

%% Scale indices for multi-column items
% When each item has multiple columns (e.g., bandwidth-specific responses),
% scale indices to cover all columns
% Example: item 1 with 3 columns -> indices 1:3
%          item 2 with 3 columns -> indices 4:6
idx=cellfun(@(x) col_in_y*(min(x)-1)+1:col_in_y*max(x),idx,'UniformOutput',false);

end

