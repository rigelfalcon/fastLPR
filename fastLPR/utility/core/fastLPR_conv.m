function [m]=fastlpr_conv(regs,kdf,y,flag_transformed)
% FASTLPR_CONV - Fast convolution using NUFFT for scattered data
%
% Computes convolution of kernel with data using NUFFT (Non-Uniform FFT).
% This is the core operation for local polynomial regression, enabling
% O(N + M log M) complexity instead of naive O(N*M).
%
% Syntax:
%   m = fastlpr_conv(regs, kdf, y)
%   m = fastlpr_conv(regs, kdf, y, flag_transformed)
%
% Inputs:
%   regs            - Regression structure from fastlpr_create
%   kdf             - Kernel density function in Fourier domain
%   y               - Data to convolve (either raw or Fourier-transformed)
%   flag_transformed - Whether y is already Fourier-transformed (optional)
%
% Output:
%   m - Convolution result on evaluation grid
%
% Algorithm:
%   1. Transform y to Fourier domain (if not already)
%   2. Multiply by kernel in Fourier domain (convolution theorem)
%   3. Inverse FFT to get result in spatial domain
%   4. Extract evaluation grid points (remove padding)
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

% Get dimensions
sz=size(y);
dy=sz(end);
Ny=sz(1);

% Determine if y is already Fourier-transformed
if nargin<4|| isempty(flag_transformed)
    if Ny~=regs.L  % L is padded length, N is grid size
        flag_transformed=false;  % y is raw data
    else
        flag_transformed=true;   % y is already transformed
        dy=size(y,regs.dx+1);
    end
end

%% Convolution in Fourier domain
% By convolution theorem: conv(f, g) = ifft(fft(f) .* fft(g))
% Here: kdf = fft(kernel), y_ft = fft(data)

if  ~flag_transformed
    % Transform y to Fourier domain using NUFFT
    y_ft_raw = fastlpr_nufft(regs,y);
    y_ft_permuted = permute(y_ft_raw,[1:regs.dx,regs.dx+2,regs.dx+1]);
    m_ft = kdf.*y_ft_permuted;
    sz=size(m_ft);
    % For multi-D, regs.L is already a vector [L1, L2, ...]
    if regs.dx == 1
        m=reshape(m_ft,[regs.L, regs.dh*dy]);
    else
        m=reshape(m_ft,[regs.L, regs.dh*dy]);
    end
else
    % y is already in Fourier domain
    if ~regs.opt.y_corr_bandwidth
        m_ft = kdf.*permute(y,[1:regs.dx,regs.dx+2,regs.dx+1]);
        sz=size(m_ft);
        if regs.dx == 1
            m=reshape(m_ft,[regs.L, regs.dh*dy]);
        else
            m=reshape(m_ft,[regs.L, regs.dh*dy]);
        end
    else
        % Special case: different bandwidth for each response variable
        m_ft = kdf.*reshape(y,[regs.L, regs.dh, dy/regs.dh]);
        m=reshape(m_ft,[regs.L, dy]);
    end
end

%% Inverse FFT to spatial domain
clear kdf y m_ft

% Apply inverse FFT along each dimension
% NOTE:ifftn() for 2D/3D takes more memory than loop, avoid it
for ix = regs.dx:-1:1
    m = ifft(m, [], ix);
end

% Reshape result
if ~flag_transformed || ~regs.opt.y_corr_bandwidth
    m=reshape(m,[regs.L, regs.dh, dy]);
else
    m=reshape(m,[regs.L, regs.dh, dy/regs.dh]);
end

% For real-valued data, take real part (imaginary part is numerical noise)
if regs.y_isreal
    m=real(m);
end

%% Extract evaluation grid (remove padding)
% Get indices of evaluation grid within padded array
idx_mq=get_patch_index(regs.qout,sz);

m=m(idx_mq);

% Reshape to final output size
if ~flag_transformed || ~regs.opt.y_corr_bandwidth
    m=reshape(m,[regs.N, regs.dh, dy]);
else
    m=reshape(m,[regs.N, regs.dh, dy/regs.dh]);
end

end



