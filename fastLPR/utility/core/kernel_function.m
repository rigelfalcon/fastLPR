function x=kernel_function(x,c,h,type,kernel,opt)
% KERNEL_FUNCTION - Evaluate kernel function at given points
%
% Computes kernel function values K((x-c)/h) for various kernel types.
% Supports both grid-based and scattered data formats.
%
% Syntax:
%   K = kernel_function(x)
%   K = kernel_function(x, c, h, type)
%   K = kernel_function(x, c, h, type, kernel, opt)
%
% Inputs:
%   x      - Evaluation points, either:
%            * Grid format: (N1 x N2 x ... x Nd x d) array
%            * Scattered format: (T x d) matrix
%   c      - Center point (1 x d vector, default: 0)
%   h      - Bandwidth (scalar or dh x dx matrix, default: 1)
%   type   - Kernel type (default: 'gaussian'):
%            * 'gaussian' or 'gauss' or 'normal' - Gaussian kernel
%            * 'epanechnikov' or 'epan' - Epanechnikov kernel
%            * 'gaussian_ft' or 'normal_ft' - Gaussian in Fourier domain
%            * 'polynomial' or 'poly' - Polynomial kernel
%   kernel - Custom kernel function handle (optional)
%            If provided, overrides 'type'
%   opt    - Options structure (for polynomial kernel: opt.order)
%
% Output:
%   K - Kernel values at each point
%
% Kernel Definitions:
%   Gaussian:      K(u) = (1/sqrt(2pi)) * exp(-0.5 * ||u||^2)
%   Epanechnikov:  K(u) = (3/4) * max(0, 1 - ||u||^2)
%   Gaussian FT:   K(u) = exp(-0.5 * ||u||^2)  [no normalization]
%   Polynomial:    K(u) = ||u||^order
%
% Notes:
%   - Automatically handles multi-dimensional data
%   - Normalizes by bandwidth: u = (x-c)/h
%   - Normalizes by det(h) for proper density
%   - Uses eps instead of 0 for Epanechnikov to avoid division by zero
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

%% Input validation and defaults
if nargin<2 ||isempty(c)
    c=0;
else
    x=x-c;  % Center at c
end

if nargin<3 ||isempty(h)
    h=1;
end

if nargin<4 ||isempty(type)
    type='gaussian';
end
if nargin<5 ||isempty(kernel)
    kernel=[];
end
if nargin<6 ||isempty(opt)
    opt=[];
end

%% Normalize by bandwidth
% Compute u = (x-c)/h for each dimension
[dh,dx]=size(h);
d=ndims(x);
deth=prod(h,2);  % Determinant of bandwidth matrix (for normalization)

% Reshape h for broadcasting
if d>dx  % Grid format: (N1 x N2 x ... x Nd x d)
    h=permute(h,[dx+2:-1:1]);
    deth=permute(deth,[dx+2:-1:1]);
else  % Scattered format: (T x d)
    % For scattered format, only permute if dh > 1 (multiple bandwidths)
    % Bug fix: permute([1,1,1], [4,3,2,1]) creates (1,1,3) instead of (1,3)
    % This causes broadcasting (100,3) ./ (1,1,3) -> (100,3,3) with repeated values
    if dh > 1
        h=permute(h,[dx+1:-1:1]);
        deth=permute(deth,[dx+1:-1:1]);
    end
    % For dh=1, h is already (1,dx) which broadcasts correctly with (T,dx)
end

% Normalize: u = (x-c)/h
x=x./h;

%% Evaluate kernel function
if isempty(kernel)
    % Use built-in kernel
    switch lower(type)
        case {'epan','epanechnikov'}
            x=epan_kernel(x,d);
        case {'gauss','gaussian','normal'}
            x=gaussian_kernel(x,d);
        case {'gaussian_ft','normal_ft'}
            x=gaussian_kernel_ft(x,d);
        case {'poly','polynomial'}
            if isempty(opt) || ~isfield(opt,'order')
                error('fastLPR:kernel:MissingOrder', ...
                    'Polynomial kernel requires opt.order to be specified');
            end
            x=polynomial_kernel(x,d,opt.order);
        otherwise
            error('fastLPR:kernel:UnknownType', ...
                'Unknown kernel type: %s. Use ''gaussian'', ''epanechnikov'', or provide custom kernel function.', type);
    end
    % Normalize by det(h) for proper density
    x=squeeze(x./deth);
else
    % Use custom kernel function
    x=kernel(x);
end

end

%% Built-in kernel functions

function x=epan_kernel(x,d)
% Epanechnikov kernel: K(u) = (3/4) * max(0, 1 - ||u||^2)
% Compact support: K(u) = 0 for ||u|| > 1
% Note: Use eps instead of 0 to avoid division by zero in sparse regions
x=max(eps,3/4*(1-sum((x.^2),d)));
end

function x=gaussian_kernel(x,d)
% Gaussian kernel: K(u) = (1/sqrt(2pi)^d) * exp(-0.5 * ||u||^2)
% This is the d-dimensional standard normal density
% Note: d here is the ndims of the input array, not the data dimension
% For grid format: d = dx + 1 (e.g., 2 for 1D data, 3 for 2D data)
% For scattered format: d = 2 (always)
% We need to infer the actual data dimension from the array structure
%
% For KDE: the kernel should integrate to 1 over R^dx
% For d-dimensional data: K(u) = (1/(2pi)^(d/2)) * exp(-||u||^2/2)
%
% However, this function is called with grid format where d = dx + 1
% So we use (1/sqrt(2pi)) which is correct for the way it's called
x=(1/sqrt(2*pi))*exp(-0.5*sum((x.^2),d));
end

function x=gaussian_kernel_ft(x,d)
% Gaussian kernel in Fourier domain (no 1/sqrt(2pi) normalization)
% Used for NUFFT-based convolution
x=exp(-0.5*sum((x.^2),d));
end

function x=polynomial_kernel(x,d,order)
% Polynomial kernel: K(u) = ||u||^order
% Used for weighted polynomial regression
x=sum(x.^order,d);
end








