function [kd,ihbad,dh,dh_input,h,lwp]=fastlpr_kdf(x,h,N,opt)
% FASTLPR_KDF - Compute kernel density function in Fourier domain
%
% Generates the kernel function in Fourier domain for fast convolution.
% For local polynomial regression, computes design matrix elements.
%
% Syntax:
%   [kd,ihbad,dh,dh_input,h,lwp] = fastlpr_kdf(x, h, N, opt)
%
% Inputs:
%   x   - Predictor data (Tx x dx matrix, z-scored)
%   h   - Bandwidth candidates (dh x dx matrix)
%   N   - Grid size (scalar or 1 x dx vector)
%   opt - Options structure with fields:
%         .flag_power2   - Use power-of-2 padding (default: true)
%         .kernel_type   - Kernel type ('gaussian' or 'epanechnikov')
%         .order         - Polynomial order (0, 1, or 2)
%
% Outputs:
%   kd       - Kernel density in Fourier domain
%              Order 0: N x N x ... x dh array
%              Order >= 1: Cell array of ns elements
%   ihbad    - Logical array indicating bandwidths that are too small
%   dh       - Number of valid bandwidths (after removing too-small ones)
%   dh_input - Original number of bandwidths
%   h        - Bandwidth matrix (possibly modified)
%   lwp      - Local polynomial parameters structure (for order >= 1)
%
% Algorithm:
%   1. Create evaluation grid matching data range
%   2. Compute kernel function on grid for each bandwidth
%   3. For order >= 1, compute design matrix elements
%   4. Transform to Fourier domain via FFT
%   5. Remove bandwidths that are too small (kernel sum  0)
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

% Get data dimensions and range
[Tx,dx]=size(x);
x_min=min(x);
x_max=max(x);
x_scale=x_max-x_min;

% Compute padded length L
if opt.flag_power2
    % Power-of-2 padding improves FFT speed and accuracy
    % For multidimensional case, apply element-wise
    L = 2.^nextpow2(2*N-1);
else
    L=N+2;
end

% Compute symmetric padding indices
% This ensures the kernel is centered correctly
qin = zeros(2, dx);
pad_left = floor((L - N) / 2);
qin(1,:) = pad_left + 1;
qin(2,:) = pad_left + N;


%% Process bandwidth parameter
% If no bandwidth provided, use rule-of-thumb (Silverman's rule)
if nargin<3 || isempty(h)
    % Silverman's rule: h = (4/(d+2)/n)^(1/(d+4))
    h=(4/(dx+2)/Tx)^(1/(dx+4))*ones(1,dx,class(x));
elseif (isvector(h)|| isscalar(h) )&& length(h)~=dx
    % Scalar or vector: replicate for each dimension
    h=repmat(h(:),[1,dx]);
elseif size(h,2)~=dx && size(h,1)==dx
    % Transpose if needed
    h=h.';
elseif (isscalar(h) && dx==1) || (ismatrix(h) && dx==size(h,2))
    % Already correct format
else
    error('fastLPR:kdf:InvalidBandwidth', 'Bandwidth h has incorrect dimensions');
end

[dh]=size(h,1);  % Number of bandwidth candidates
dh_input=dh;     % Store original number

%% Generate kernel function in spatial domain
% Create evaluation grid matching data range
xgrid=get_ndgrid_scatter(x,'array',N);

% Get local polynomial estimator functions (for order >= 1)
if opt.order>0
    [lwp.Sxfun,lwp.Txfun,lwp.mfun,lwp.ns,lwp.nt,lwp.mmfun,lwp.mmmfun]=get_lwp_estimator(dx,opt.order);
else
    lwp=[];
end

% Compute kernel function on grid
% For order 0: returns kernel values K(x)
% For order >= 1: returns design matrix elements S_ij(x)
[kd,ihbad]=get_local_polynomial_kernel(xgrid,h,opt.kernel_type,opt.order,dx,lwp);


%% Remove bandwidths that are too small
% A bandwidth is "too small" if the kernel sum is approximately zero,
% meaning no data points fall within the kernel support.
ihbad=ihbad(:);
ndcolon(1:dx) = {':'};

if sum(ihbad) && sum(~ihbad)>0
    % Some bandwidths are too small, remove them
    warning('fastLPR:kdf:SmallBandwidth', ...
        '%d bandwidth(s) are too small for this grid and will be removed', sum(ihbad));

    if isnumeric(kd)
        % Order 0: kd is a numeric array
        kd(ndcolon{:},ihbad)=[];
    else
        % Order >= 1: kd is a cell array
        for i=1:length(kd)
            kd{i}(ndcolon{:},ihbad)=[];
        end
    end
    dh=sum(~ihbad);  % Update number of valid bandwidths

elseif prod(ihbad)
    % All bandwidths are too small, use Silverman's rule
    warning('fastLPR:kdf:AllBandwidthsTooSmall', ...
        'All bandwidths are too small. Using Silverman''s rule: h = %.4f', ...
        (4/(dx+2)/Tx)^(1/(dx+4)));

    h=(4/(dx+2)/Tx)^(1/(dx+4));
    dh=1;
    [kd]=get_local_polynomial_kernel(xgrid,h,opt.kernel_type,opt.order,dx,lwp);
    ihbad=0;
end

%% Transform kernel to Fourier domain
% We use FFT instead of theoretical Fourier transform because:
% 1. Theoretical transform loses phase information
% 2. FFT is exact for discrete convolution
% 3. FFT handles arbitrary kernel shapes

switch opt.order
    case 0
        % Order 0: single kernel function
        kd=get_fourier_transformed_kernel(kd,L,qin,dx);
    otherwise
        % Order >= 1: multiple design matrix elements
        for i=1:lwp.ns
            kd{i}=get_fourier_transformed_kernel(kd{i},L,qin,dx);
        end
end

end

%% Helper Functions

function [kd,ihbad]=get_local_polynomial_kernel(xgrid,h,kernel_type,order,dx,lwp)
% GET_LOCAL_POLYNOMIAL_KERNEL - Compute kernel function for local polynomial regression
%
% For order 0: Returns kernel weights K(x)
% For order >= 1: Returns design matrix elements S_ij(x) = K(x) * x^i * x^j
%
% Inputs:
%   xgrid       - Evaluation grid (N x N x ... x dx array)
%   h           - Bandwidth (dh x dx matrix)
%   kernel_type - Kernel type ('gaussian' or 'epanechnikov')
%   order       - Polynomial order (0, 1, or 2)
%   dx          - Number of dimensions
%   lwp         - Local polynomial parameters (for order >= 1)
%
% Outputs:
%   kd    - Kernel density (order 0) or design matrix elements (order >= 1)
%   ihbad - Logical array indicating bandwidths with zero kernel sum

switch order
    case 0
        % Order 0: Nadaraya-Watson kernel
        kd=kernel_function(xgrid,[],h,kernel_type);
        % Check if kernel sum is too small (bandwidth too small for grid)
        % Use eps*10 threshold to catch underflow without filtering normal values
        ihbad=squeeze(abs(sum(kd,(1:dx))) < eps*10);

    otherwise
        % Order >= 1: Local polynomial kernel
        % Compute base kernel K(x/h)
        kd=kernel_function(xgrid,[],h,kernel_type);

        % Normalize grid by bandwidth (use SIGNED coordinates for symmetric cancellation)
        % CRITICAL: Do NOT use abs() - local polynomial regression requires signed coordinates
        % for cross-term cancellation (e.g., sum(K*x) ≈ 0 for symmetric kernels)
        xgrid=squeeze(xgrid./permute(h,[dx+2:-1:1]));

        % Compute design matrix elements: S_ij = K(x) * x^i * x^j
        kd=lwp.Sxfun(kd,xgrid);

        % Check if kernel sum (kd{1}) is too small (bandwidth too small for grid)
        % CRITICAL: Only check kd{1} (kernel sum), NOT cross terms (kd{2}, etc.)!
        % After removing abs() from coordinate normalization, cross terms are
        % CORRECTLY near machine epsilon due to symmetric cancellation.
        % Checking cross terms would incorrectly mark valid bandwidths as bad.
        ihbad=squeeze(abs(sum(kd{1},(1:dx))) < eps*10);
end
end

function kd=get_fourier_transformed_kernel(kd,L,qin,dx)
% GET_FOURIER_TRANSFORMED_KERNEL - Transform kernel to Fourier domain
%
% Pads kernel to length L and applies FFT along each dimension.
% Padding avoids circular convolution artifacts.
%
% Inputs:
%   kd  - Kernel in spatial domain (N x N x ... array)
%   L   - Padded length (scalar or 1 x dx vector)
%   qin - Padding indices (2 x dx matrix)
%   dx  - Number of dimensions
%
% Output:
%   kd - Kernel in Fourier domain (L x L x ... array)

% Pad kernel to avoid boundary effects (circular convolution)
kd=padarray(kd,qin(1,:).*ones(1,dx),0,'pre');   % Pad before
kd=padarray(kd,(L-qin(2,:)-1).*ones(1,dx),0,'post');  % Pad after

% Apply FFT along each dimension
% NOTE: Uses plain fft() (NOT fftshift(fft(...))), so DC is at position 1
% This matches the output from fastlpr_nufft (which applies ifftshift)
% Both kernel and data have DC at position 1 for convolution
for i=1:dx
    kd=fft(kd,[],i);
end
end










