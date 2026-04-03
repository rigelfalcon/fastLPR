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

        % DEBUG: Save S and T matrices if debug flag is set
        if isfield(regs.opt, 'save_st_debug') && regs.opt.save_st_debug
            save('temp/matlab_st_debug.mat', 's_reg', 't', '-v7');
            fprintf('[DEBUG] Saved S and T matrices to temp/matlab_st_debug.mat\n');
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

nt = regs.opt.order + 1;  % Size of design matrix (order+1) x (order+1)

% Find the maximum diagonal element across all spatial points
max_diag = max(abs(regs.s{1}(:)));  % Start with S11
for k = 2:nt
    diag_idx = k*(k+1)/2;  % Diagonal index in lower triangular storage
    max_diag = max(max_diag, max(abs(regs.s{diag_idx}(:))));
end

% Add a small, fixed regularization relative to the max diagonal
% alpha = 1e-6: regularize at 0.0001% of maximum signal
% This is enough to prevent singularity without biasing the fit
alpha = 1e-6;
lambda_fixed = alpha * max_diag;

% Apply regularization to diagonal elements: S_reg = S + lambda*I
s_reg = regs.s;
for k = 1:nt
    diag_idx = k*(k+1)/2;
    s_reg{diag_idx} = regs.s{diag_idx} + lambda_fixed;
end

end

%% Debug visualization (optional, for development only)
function debug_visualization(regs, t, mq)
if iscell(t)&&ndims(t{1})==3 && ~isempty(regs.lwp.mmfun) %&&false
    mqq=regs.lwp.mmfun(regs.s,t);
    mqqq=regs.lwp.mmmfun(regs.s,t);
    
    
    figure(98)
    clf
    hold on
    scatter3(regs.x(:,1),regs.x(:,2),regs.y,'.')
    view(90,0)
    
    figure(99)
    clf
    hold on
    [r,c]=numSubplots(regs.lwp.ns+1);
    for i=1:regs.lwp.ns
        subplot(r(1),r(2),i)
        surf(fftshift(abs(regs.kdf{i}(:,:,10))),'FaceLighting','gouraud','FaceAlpha',0.2)
        view(90,0)
    end
    subplot(r(1),r(2),i+1)
    surf(fftshift(abs(mq(:,:,10))),'FaceLighting','gouraud','FaceAlpha',0.2)
    view(90,0)
    
    
    
    figure(100)
    clf
    hold on
    [r,c]=numSubplots(regs.lwp.ns+1);
    for i=1:regs.lwp.ns
        subplot(r(1),r(2),i)
        surf(fftshift(abs(regs.s{i}(:,:,10))),'FaceLighting','gouraud','FaceAlpha',0.2)
        view(90,0)
    end
    subplot(r(1),r(2),i+1)
    surf(fftshift(abs(mq(:,:,10))),'FaceLighting','gouraud','FaceAlpha',0.2)
    view(90,0)
     
    figure(101)
    clf
    hold on
    [r,c]=numSubplots(regs.lwp.nt+1);
    for i=1:regs.lwp.nt
        subplot(r(1),r(2),i)
        surf(fftshift(abs(t{i}(:,:,10))),'FaceLighting','gouraud','FaceAlpha',0.2)
        view(90,0)
    end
    subplot(r(1),r(2),i+1)
    surf(fftshift(abs(mq(:,:,10))),'FaceLighting','gouraud','FaceAlpha',0.2)
    view(90,0)
    
    figure(102)
    clf
    hold on
    [r,c]=numSubplots(length(mqq)+2);
    for i=1:6
        subplot(r(1),r(2),i)
        surf(fftshift(abs(mqq{i}(:,:,10))),'FaceLighting','gouraud','FaceAlpha',0.2)
        view(90,0)
    end
    subplot(r(1),r(2),i+1)
    surf((abs(mqqq{1}(:,:,10))),'FaceLighting','gouraud','FaceAlpha',0.2)
    view(90,0)
    subplot(r(1),r(2),i+2)
    surf((abs(mq(:,:,10))),'FaceLighting','gouraud','FaceAlpha',0.2)
    view(90,0)
    
    figure(103)
    clf
    hold on
    for i=7:11
        subplot(2,4,i-6)
        surf(fftshift(abs(mqq{i}(:,:,10))),'FaceLighting','gouraud','FaceAlpha',0.2)
        view(90,0)
    end
    subplot(2,4,i-6+1)
    surf((abs(mqqq{2}(:,:,10))),'FaceLighting','gouraud','FaceAlpha',0.2)
    view(90,0)
    subplot(2,4,i-6+2)
    surf((abs(mq(:,:,10))),'FaceLighting','gouraud','FaceAlpha',0.2)
    view(90,0)
     
    figure(104)
    clf
    hold on
    for i=1:2
        subplot(2,2,i)
        surf(((mqqq{i}(:,:,10))),'FaceLighting','gouraud','FaceAlpha',0.2)
        view(90,0)
    end
    subplot(2,2,i+1)
    surf(((mq(:,:,10))),'FaceLighting','gouraud','FaceAlpha',0.2)
    view(90,0)
    subplot(2,2,i+2)
    view(90,0)
    view(90,0)
    
    
    
elseif isnumeric(t) && ndims(t)==3 && false
    figure(105)
    subplot(3,1,1)
    surf(t(:,:,10),'FaceLighting','gouraud','FaceAlpha',0.2)
    view(90,0)
    subplot(3,1,2)
    surf(abs(regs.s(:,:,10)),'FaceLighting','gouraud','FaceAlpha',0.2)
    view(90,0)
    subplot(3,1,3)
    surf(abs(mq(:,:,10)),'FaceLighting','gouraud','FaceAlpha',0.2)
    view(90,0)
end

end  % End of debug_visualization function
