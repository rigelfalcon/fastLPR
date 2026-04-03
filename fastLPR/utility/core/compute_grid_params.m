function [Msp, Mr, tau,ratio]=compute_grid_params(N,accuracy)
% COMPUTE_GRID_PARAMS - Compute NUFFT grid parameters
%
% Calculates spreading width, oversampling ratio, and Gaussian width
% for Non-Uniform FFT based on desired accuracy.
%
% Syntax:
%   [Msp, Mr, tau, ratio] = compute_grid_params(N)
%   [Msp, Mr, tau, ratio] = compute_grid_params(N, accuracy)
%
% Inputs:
%   N        - Original grid size (scalar or vector)
%   accuracy - Desired accuracy in decimal digits (default: 8)
%              6: ratio=1 (no oversampling)
%              7-11: ratio=2 (2x oversampling)
%              >11: ratio=3 (3x oversampling)
%
% Outputs:
%   Msp   - Spreading width (number of neighbors)
%   Mr    - Oversampled grid size = ratio * N
%   tau   - Gaussian spreading parameter
%   ratio - Oversampling ratio (1, 2, or 3)
%
% Example:
%   [Msp, Mr, tau, ratio] = compute_grid_params(100, 8);
%   % Returns: Msp=8, Mr=200, tau=..., ratio=2
%
% Notes:
%   - Higher accuracy requires more oversampling
%   - tau controls Gaussian spreading width
%   - Msp is number of grid points affected by each sample
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

if nargin < 2 || isempty(accuracy)
    accuracy=8;
end

%% Determine oversampling ratio based on accuracy
if accuracy<=6
    ratio=1;  % No oversampling
elseif accuracy>6 && accuracy<=11
    ratio=2;  % 2x oversampling
else
    ratio=3;  % 3x oversampling
end

%% Compute grid parameters
% Ensure N is a row vector for consistent output shapes
N = N(:)';
Msp=accuracy*ones(1,length(N));  % Spreading width (1×d row vector)
tau = (pi*Msp./(N.*N*ratio*(ratio-0.5)));  % Gaussian width (1×d row vector)
Mr = ratio*N;  % Oversampled grid size (1×d row vector)

end