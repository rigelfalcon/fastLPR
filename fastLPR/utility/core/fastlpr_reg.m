function [mq]=fastlpr_reg(regs,y)
% Copyright (c) 2024-2025 Ying Wang, Min Li
% SPDX-License-Identifier: GPL-3.0-or-later
%
% FASTLPR_REG - Local polynomial regression with adaptive regularization
%
% Computes the regression estimate using precomputed density (design matrix).
% For higher-order polynomials (order >= 1), applies adaptive regularization
% based on the determinant of the local design matrix to prevent instability
% in sparse regions.

assert(isfield(regs,'s'),'need precompute density');

switch regs.opt.order
    case 0
        % Order 0: Nadaraya-Watson estimator
        % mq = sum(K * y) / sum(K)
        [t]=fastlpr_conv(regs,regs.kdf,y);

        % Adaptive threshold for numerical stability
        % Prevents division by near-zero in sparse regions
        s_threshold = max(regs.s(:)) * 1e-6;
        mq = t ./ max(regs.s, s_threshold);

    otherwise
        % Order >= 1: Local polynomial regression
        % Solve: S * beta = T, where S is the design matrix

        % Compute convolutions T = sum(K * X^j * y)
        for i=regs.lwp.nt:-1:1
            t{i}=squeeze(fastlpr_conv(regs,regs.kdf{i},y));
        end

        % Apply adaptive regularization to design matrix S
        s_reg = apply_adaptive_regularization(regs);

        % Save S and T matrices if debug flag is set
        if isfield(regs.opt, 'save_st_debug') && regs.opt.save_st_debug
            save(fullfile(tempdir, 'matlab_st_debug.mat'), 's_reg', 't', '-v7');
            fprintf('[fastlpr] Saved S and T matrices to %s\n', ...
                fullfile(tempdir, 'matlab_st_debug.mat'));
        end

        % Solve for regression coefficients using symbolic formula
        mq=regs.lwp.mfun(s_reg,t);
end

end

%% Helper function: Adaptive regularization
function s_reg = apply_adaptive_regularization(regs)
% APPLY_ADAPTIVE_REGULARIZATION - Add regularization only where needed
%
% Strategy: Regularize based on maximum diagonal element
% - Add a small, fixed regularization relative to max(diag(S))
% - This ensures stability without suppressing the fit in high-signal areas
% - Keeps GCV surface smooth because regularization is NOT bandwidth-dependent
%
% This approach is more robust than determinant-based methods because:
% 1. No invalid linear approximation (det(S + lambda*I) ~= det(S) + lambda*trace(S))
% 2. Scales naturally with local signal power
% 3. Simple and computationally efficient

nt = regs.lwp.nt;  % Regularize ALL diagonal elements of the design matrix

% S is stored column-major lower-triangular, so the k-th diagonal element
% S(k,k) is at storage index k + (k-1)*(2*nt - k)/2. (The earlier k*(k+1)/2
% was the row-major upper-triangular formula; for nt>=3 it perturbed an
% off-diagonal moment and missed a true diagonal.)
diag_idx_of = @(k) k + (k-1)*(2*nt - k)/2;

% Find the maximum diagonal element across all spatial points
max_diag = max(abs(regs.s{1}(:)));  % Start with S11
for k = 2:nt
    max_diag = max(max_diag, max(abs(regs.s{diag_idx_of(k)}(:))));
end

% Add a small, fixed regularization relative to the max diagonal
% alpha = 1e-6: regularize at 0.0001% of maximum signal
% This is enough to prevent singularity without biasing the fit
alpha = 1e-6;
lambda_fixed = alpha * max_diag;

% Apply regularization to diagonal elements: S_reg = S + lambda*I
s_reg = regs.s;
for k = 1:nt
    diag_idx = diag_idx_of(k);
    s_reg{diag_idx} = regs.s{diag_idx} + lambda_fixed;
end

end

