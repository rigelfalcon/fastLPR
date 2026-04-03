function knots=scale_knots(knots,N,Fs)
% SCALE_KNOTS - Scale and shift knot locations for NUFFT
%
% Transforms knot locations to [0, 2pi] range required by Greengard's NUFFT.
% Applies frequency scaling, centering, and modulo wrapping.
%
% Syntax:
%   knots = scale_knots(knots, N, Fs)
%
% Inputs:
%   knots - Original knot locations (any range)
%   N     - Grid size
%   Fs    - Sampling frequency
%
% Output:
%   knots - Scaled knot locations in [0, 2pi]
%
% Example:
%   knots = scale_knots([0.1, 0.5, 0.9], 100, 1);
%
% Notes:
%   - Step 1: Scale by Fs/N
%   - Step 2: Center around 0
%   - Step 3: Shift to [-1/2, 1/2]
%   - Step 4: Convert to [0, 2pi] (Greengard convention)
%   - Phase may be affected by centering
%
% Reference:
%   Ferrara, M. (2009). NUFFT, NFFT, USFFT.
%   https://www.mathworks.com/matlabcentral/fileexchange/25135
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

%% Ensure N and Fs are row vectors for proper broadcasting
N = N(:)';
Fs = Fs(:)';

%% Scale by frequency and grid size
knots=knots.*Fs./N;

%% Center around 0
kmin=(min(knots)+max(knots))/2-0.5;
shift=-1/2-kmin;
knots=knots + shift;  % Now in [-1/2, 1/2]

%% Convert to [0, 2pi] convention (Greengard)
knots=mod(2*pi*knots,2*pi);

end


