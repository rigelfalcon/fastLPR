function [grid,df]=nufftfreqs(N, df)
% NUFFTFREQS - Generate frequency grid for NUFFT
%
% Creates the frequency grid used in Non-Uniform Fast Fourier Transform.
% Frequencies are centered around zero with proper spacing.
%
% Syntax:
%   grid = nufftfreqs(N)
%   [grid, df] = nufftfreqs(N, df)
%
% Inputs:
%   N  - Grid size in each dimension (1 x dx vector)
%   df - Frequency spacing in each dimension (default: ones(1,dx))
%
% Outputs:
%   grid - Frequency grid (prod(N) x dx matrix)
%   df   - Frequency spacing (1 x dx vector)
%
% Example:
%   grid = nufftfreqs([8, 8]);
%   % Returns: 64x2 matrix with frequencies from -4 to 3 in each dim
%
% Notes:
%   - Frequencies range from fix(-N/2) to N-fix(N/2)-1
%   - Centered at zero (DC component)
%   - Scaled by df for non-unit spacing
%   - Used internally by nufftn_type1
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

dx=length(N);
if nargin<2 || isempty(df)
    df=1*ones(1,dx);
end

%% Compute frequency range for each dimension
% Frequencies: fix(-N/2) to N-fix(N/2)-1
% NOTE: Must use 'cell' type to ensure grid is a cell array (required for brace indexing below)
grid=multispace(fix(-N/2),N - fix(N/2)-1,N,@linspace,'cell',true);

% Scale by frequency spacing
for i = 1:dx
    grid{i} = df(i) * grid{i};
end

%% Create multidimensional grid
grid=get_ndgrid(grid);
grid=permute(grid,[dx+1:-1:1]);
grid=cell2mat(grid);

end