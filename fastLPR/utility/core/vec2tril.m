function Atrl=vec2tril(A,ind)
% VEC2TRIL - Reconstruct lower triangular matrices from vectorized form
%
% Converts vectors of lower triangular elements back to full matrices.
% Inverse operation of mat2tril.
%
% Syntax:
%   Atrl = vec2tril(A)
%   Atrl = vec2tril(A, ind)
%
% Inputs:
%   A   - Matrix of vectorized triangular elements (n x m)
%         n = number of elements in lower triangle
%         m = number of matrices
%   ind - Diagonal offset (default: 0)
%         ind=0: main diagonal included
%         ind>0: include ind diagonals above main
%         ind<0: exclude ind diagonals including main
%
% Output:
%   Atrl - 3D array of lower triangular matrices (a x a x m)
%          where a is matrix size (computed from n)
%
% Example:
%   A = [1; 2; 3];  % 3 elements = 2x2 lower triangle
%   Atrl = vec2tril(A, 0);
%   % Returns: 2x2x1 matrix with lower triangle filled
%
% Notes:
%   - Matrix size a computed from: n = (a^2 + a)/2 for ind=0
%   - Inverse of mat2tril
%   - Upper triangle elements are zero
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

if nargin<2
    ind=0;
end
[n,m]=size(A);

%% Compute matrix size from number of elements
a=(-1+sqrt(1+8*n))/2;  % Solve: n = (a^2 + a)/2

%% Get linear indices for lower triangle
switch ind
case -1
    onesind=find(tril(ones(a),-1));
case 1
    onesind=find(tril(ones(a),1));
case 0
    onesind=find(tril(ones(a)));
end

%% Reconstruct matrices
AA=zeros(a,class(A));
Atrl=zeros(a,a,m,class(A));
for i=1:m
    AA(onesind)=A(:,i);  % Fill lower triangle
    Atrl(:,:,i)=AA;      % Store matrix
end

end