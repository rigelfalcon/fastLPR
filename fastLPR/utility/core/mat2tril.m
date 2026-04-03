function [mattril,onesind]=mat2tril(data,k)
% MAT2TRIL - Extract lower triangular elements from 3D matrix
%
% Extracts lower triangular elements from each 2D slice of a 3D matrix.
% Useful for covariance/correlation matrices where only lower triangle is needed.
%
% Syntax:
%   mattril = mat2tril(data)
%   [mattril, onesind] = mat2tril(data, k)
%
% Inputs:
%   data - 3D matrix (r x c x p)
%   k    - Diagonal offset (default: 0)
%          k=0: main diagonal included
%          k>0: include k diagonals above main
%          k<0: exclude k diagonals including main
%
% Outputs:
%   mattril - Matrix of lower triangular elements (nn x p)
%             where nn = number of elements in lower triangle
%   onesind - Logical index array (r x c x p)
%
% Example:
%   data = repmat(magic(3), [1, 1, 2]);  % 3x3x2
%   mattril = mat2tril(data, 0);  % Extract lower triangle
%   % Returns: 6x2 matrix (6 elements per triangle)
%
% Notes:
%   - Uses MATLAB's tril() for indexing
%   - Efficient for batch processing multiple matrices
%   - Common in covariance/correlation analysis
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

if nargin<2
    k=0;
end
[r,c,p]=size(data);

%% Create lower triangular index
onesind=tril(true(r,c),k);

%% Extract elements
nn=sum(sum(onesind));  % Number of elements in lower triangle
onesind=repmat(onesind,[1,1,p]);  % Replicate for all slices
mattril=reshape(data(onesind),[nn,p]);  % Extract and reshape

end


