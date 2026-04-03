function idx_patch=get_patch_index(q,M)
% GET_PATCH_INDEX - Compute linear indices for a sub-patch of an array
%
% Converts multi-dimensional sub-patch bounds to linear indices for efficient
% array indexing. Used for extracting sub-regions from padded arrays.
%
% Syntax:
%   idx = get_patch_index(q, M)
%
% Inputs:
%   q - Sub-patch bounds (dx x 2 matrix)
%       q(:,1) = start indices in each dimension
%       q(:,2) = end indices in each dimension
%   M - Full array size (dM x 1 vector)
%       Size of the full array from which to extract the patch
%
% Output:
%   idx - Linear indices of all points in the sub-patch (column vector)
%
% Example:
%   % Extract patch [10:20, 30:40] from 100x100 array
%   q = [10, 20; 30, 40];
%   M = [100; 100];
%   idx = get_patch_index(q, M);
%   % Result: linear indices of 11x11 = 121 points
%
%   % Use for extraction:
%   A = rand(100, 100);
%   patch = A(idx);  % Extract patch as column vector
%   patch = reshape(patch, [11, 11]);  % Reshape to 2D
%
% Algorithm:
%   1. Start with first dimension indices: q(1,1):q(1,2)
%   2. For each additional dimension i:
%      Add offset: (q(i,1):q(i,2) - 1) * prod(M(1:i-1))
%   3. If dM > dx, replicate for extra dimensions
%
% Notes:
%   - Uses minimal integer type to save memory
%   - Handles multi-dimensional arrays efficiently
%   - Supports extra dimensions beyond patch dimensions
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

%% Determine optimal integer type for memory efficiency
% Use smallest integer type that can represent all indices
dtype=min_numerical_type(1,prod(M));
fun_dtype=str2func(dtype);

% Convert to optimal type
q = fun_dtype(q);
M = fun_dtype(M);

%% Get dimensions
dx=size(q,1);  % Number of patch dimensions
dM=length(M);  % Number of array dimensions

%% Compute linear indices for first dimension
idx_patch=[q(1,1):q(1,2)].';

%% Add offsets for additional dimensions
% For dimension i, offset = (index - 1) * prod(M(1:i-1))
% This converts multi-dimensional indices to linear indices
if dx>1
    for i=2:dx
        % Compute offset for dimension i
        offset = ([q(i,1):q(i,2)]-1)*prod(M(1:i-1));

        % Add offset to all existing indices (Cartesian product)
        idx_patch=idx_patch+offset;
        idx_patch=idx_patch(:);  % Flatten to column vector
    end
end

%% Handle extra dimensions beyond patch dimensions
% If array has more dimensions than patch, replicate patch for all
% combinations of extra dimensions
if dM>dx
    % Compute offsets for extra dimensions
    extra_offset = prod(M(1:dx))*fun_dtype(0:prod(M(dx+1:dM))-1);

    % Add offsets (Cartesian product)
    idx_patch=idx_patch+extra_offset;
end

%% Ensure output is column vector
idx_patch=idx_patch(:);

end

% function idx_patch=get_patch_index(q,M)
% % q is a d*2 matrix represent the subpatch, q(:,1) is start index, q(:,2) is end index
% % M is a d*1 vector denote the whole space with each direction of length
% dx=size(q,1);
% dM=length(M);


% N=q(:,2)-q(:,1)+1;
% idx_patch=multispace(q(:,1),q(:,2),N,[],'cell',true);
% if dM>dx
%     for i=1:dM-dx
%         idx_patch(dx+i)={1:M(dx+i)};
%     end
% end
% idx_patch=get_ndgrid(idx_patch,'cellList');

% % idx_patch=cellfun(@(x) reshape(x,[],1),idx_patch,'UniformOutput',false);
% idx_patch=sub2ind([M(:);1].',idx_patch{:});
% end
