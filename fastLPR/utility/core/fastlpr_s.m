function regs=fastlpr_s(regs)
% FASTLPR_S - Compute design matrix for local polynomial regression
%
% Computes the design matrix S = X'*W*X for local polynomial regression.
% For order 0 (Nadaraya-Watson), S is a scalar (sum of kernel weights).
% For order >= 1, S is a matrix with elements computed via convolution.
%
% Syntax:
%   regs = fastlpr_s(regs)
%
% Input:
%   regs - Regression structure from fastlpr_create containing:
%          .Tx       - Number of data samples
%          .dtype    - Data type (e.g., 'double', 'single')
%          .opt.order - Polynomial order (0, 1, or 2)
%          .kdf      - Kernel density function in Fourier domain
%          .lwp.ns   - Number of design matrix elements (for order >= 1)
%
% Output:
%   regs - Updated regression structure with:
%          .s - Design matrix elements
%               Order 0: scalar (sum of weights)
%               Order >= 1: cell array of ns elements
%
% Algorithm:
%   Uses NUFFT-based convolution to compute design matrix elements efficiently.
%   For order 0: S = sum(K(x_i - x_j))
%   For order >= 1: S_ij = sum(K(x_k - x_j) * X_k^i * X_k^j)
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

% Create vector of ones for convolution (represents constant function)
Is=ones(regs.Tx,1,regs.dtype);

switch regs.opt.order
    case 0
        % Order 0: Nadaraya-Watson estimator
        % S = sum of kernel weights at each evaluation point
        % Add eps for numerical stability (prevents division by zero)
        [regs.s]=fastlpr_conv(regs,regs.kdf,Is,false)+eps;

    otherwise
        % Order >= 1: Local polynomial regression
        % Compute each element of the design matrix S via convolution
        % S has ns unique elements (lower triangular due to symmetry)
        for i=regs.lwp.ns:-1:1
            regs.s{i}=fastlpr_conv(regs,regs.kdf{i},Is);
        end
end

end