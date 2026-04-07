function y_ft=fastlpr_nufft(regs,y)
% FASTLPR_NUFFT - Non-Uniform Fast Fourier Transform for scattered data
%
% Computes the Fourier transform of data at non-uniform (scattered) sample points.
% This is a key component enabling O(N + M log M) complexity for kernel regression.
%
% Syntax:
%   y_ft = fastlpr_nufft(regs, y)
%
% Inputs:
%   regs - Regression structure containing:
%          .knot         - Normalized sample positions in [-0.5, 0.5]
%          .hf           - Frequency grid spacing
%          .opt.accuracy - NUFFT accuracy (number of digits)
%          .opt.nufft_deconv - Whether to apply deconvolution correction
%          .dx           - Number of dimensions
%   y    - Data values at sample points (Tx x dy matrix)
%
% Output:
%   y_ft - Fourier transform on uniform grid (L x L x ... x dy array)
%
% Algorithm:
%   Uses Type-1 NUFFT to transform from non-uniform points to uniform grid.
%   The NUFFT uses Gaussian spreading and FFT for fast computation.
%   ifftshift is applied to center the zero frequency.
%
% Reference:
%   Greengard, L., & Lee, J. Y. (2004). Accelerating the Nonuniform Fast
%   Fourier Transform. SIAM Review, 46(3), 443-454.
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

% Apply Type-1 NUFFT: non-uniform points -> uniform grid
% Output from nufftn_type1 has DC at position M/2+1 (centered, after fftshift)
y_ft=nufftn_type1(regs.knot,y,[],regs.hf,-1,regs.opt.accuracy,regs.opt.nufft_deconv);

% Apply ifftshift to move DC back to position 1 (standard FFT convention)
% This is REQUIRED because:
% 1. Kernel (from fastlpr_kdf) uses plain fft() -> DC at position 1
% 2. Convolution multiplies y_ft * kdf -> both must have DC at same position
% 3. Inverse FFT (ifft) expects DC at position 1
% NOTE: The fftshift/ifftshift pair is NOT wasteful - fftshift is needed
% for deconvolution in nufftn_type1, and ifftshift converts to standard convention
for i=1:regs.dx
    y_ft=ifftshift(y_ft,i);
end

end
