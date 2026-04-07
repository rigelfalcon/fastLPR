function [regs]=fastlpr_dof(regs)
% FASTLPR_DOF - Estimate degrees of freedom using Hutchinson's method
%
% Estimates the effective degrees of freedom (DOF) for each bandwidth using
% a randomized trace estimator (Hutchinson's method). The DOF is used for
% GCV bandwidth selection and variance estimation.
%
% Syntax:
%   regs = fastlpr_dof(regs)
%
% Input:
%   regs - Regression structure from fastlpr_s containing:
%          .opt.calc_dof - Whether to compute DOF (boolean)
%          .opt.num_dof_sample - Number of random samples (default: 10)
%          .Ty - Number of data points
%          .dh - Number of bandwidths
%          .dy - Number of responses
%          .y_isreal - Logical array indicating real-valued responses
%
% Output:
%   regs - Updated structure with DOF statistics:
%          .pdof_m - Mean proportion of residual DOF: (1/N)tr(I-H)
%          .pdof_sd - Std of pdof
%          .df_m - Mean degrees of freedom: tr(H)
%          .df_sd - Std of df
%          .pdof_inv_m - Mean GCV penalty: 1/pdof
%          .pdof_inv_sd - Std of GCV penalty
%
% Algorithm (Hutchinson's Method):
%   1. Generate random vectors: p ~ N(0,I)
%   2. Compute fitted values: p_hat = H*p
%   3. Estimate trace: tr(I-H) ~= (1/N) * p'*(p - p_hat) / (p'*p)
%   4. Repeat for multiple samples and compute statistics
%   5. Clamp pdof to [0,1] for numerical stability
%   6. Compute GCV penalty with additional clamping
%
% GCV Penalty Clamping:
%   - pdof estimates (1/N)tr(I-H), theoretically in [0,1]
%   - For GCV penalty 1/pdof, clamp pdof to minimum 0.01
%   - This prevents penalty explosion when pdof -> 0 (overfitting)
%   - Maximum penalty = 1/0.01 = 10,000 (reasonable upper bound)
%
% Complex-Valued Responses:
%   - Complex responses have 2x DOF (real and imaginary parts)
%   - Adjustment: pdof_complex = 2*pdof_real - 1
%
% References:
%   - Hutchinson (1990): A stochastic estimator of the trace of the
%     influence matrix for Laplacian smoothing splines
%   - Girard (1989): A fast 'Monte-Carlo cross-validation' procedure
%     for large least squares problems with noisy data
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

% Formula: (1/n)w'*(w-what) == (1/n)tr(I-A)
if regs.opt.calc_dof
    % Use external random vectors if provided (for exact cross-language matching)
    % Otherwise use dof_seed for reproducible random generation
    if isfield(regs.opt, 'dof_random_vectors') && ~isempty(regs.opt.dof_random_vectors)
        p = regs.opt.dof_random_vectors;
        % Validate dimensions
        if size(p, 1) ~= regs.Ty || size(p, 2) ~= regs.opt.num_dof_sample
            error('fastLPR:dof:invalidVectors', ...
                'dof_random_vectors must be (%d x %d), got (%d x %d)', ...
                regs.Ty, regs.opt.num_dof_sample, size(p, 1), size(p, 2));
        end
    else
        % Use dof_seed for reproducible DOF estimation (like Python's dof_seed)
        % Save current RNG state, set seed, generate vectors, restore state
        rng_state = rng;
        rng(regs.opt.dof_seed);
        p = randn(regs.Ty, regs.opt.num_dof_sample);
        rng(rng_state);  % Restore original RNG state
    end
    % Store the vectors used (for saving to reference files)
    regs.dof_random_vectors = p;
    
    mq=fastlpr_reg(regs,p);%N*ones(1,regs.dx) regs.dh regs.opt.num_dof_sample
    
    mq=reshape(mq,[regs.N, regs.dh*regs.opt.num_dof_sample]);
%     method ='griddedInterpolant';
%     opt.Method='spline';
%     opt.ExtrapolationMethod ='linear';
%     [~,phat]=fastlpr_gridinterp(regs,mq,[],method,opt);
    [~,phat]=fastlpr_gridinterp(regs,mq);
    phat=real(phat);
    phat=reshape(phat,[regs.Ty,regs.dh,regs.opt.num_dof_sample]);
    phat=permute(phat,[1,3,2]);
    
    pdof=(sum(p.*(p-phat))./sum(p.*p));
    pdof=permute(pdof,[2,3,1]);
    pdof=repmat(pdof,1,1,regs.dy);
    if ~regs.y_isreal
        pdof(:,:,~regs.y_isreal)=2*pdof(:,:,~regs.y_isreal)-1; % complex value has 2*df
    end

    % Clamp pdof to valid range [0, 1]
    % pdof estimates (1/N)tr(I-H), which theoretically should be in [0, 1]
    % However, Hutchinson estimator has variance and can give pdof > 1 or pdof < 0
    % This happens especially with very small bandwidths (high DOF, tr(I-H)~0)
    % Clamping prevents negative df and improves numerical stability
    pdof_clamped = max(0, min(1, pdof));  % Clamp to [0, 1]

    % Check if clamping was needed (for diagnostic purposes)
    % Count violations per bandwidth (average across samples)
    pdof_mean = mean(pdof, 1);  % Average across samples
    n_bandwidths_violated = sum(any(pdof_mean < 0 | pdof_mean > 1, 3));

    % Only warn if violations are significant (not just numerical noise)
    if any(pdof_mean(:) < -0.05 | pdof_mean(:) > 1.05)
        max_violation = max(abs([min(pdof_mean(:)), max(pdof_mean(:))-1]));
        warning('fastLPR:dof:outOfRange', ...
            ['pdof out of valid range [0,1] for %d bandwidth(s) (max violation: %.2f).\n' ...
             'This is normal with extreme bandwidths due to Hutchinson estimator variance.\n' ...
             'Values are automatically clamped. Consider using a narrower bandwidth range.'], ...
            n_bandwidths_violated, max_violation);
    end

    df=regs.Ty-regs.Ty*pdof_clamped;

    % Compute GCV penalty with clamping for numerical stability
    %
    % The GCV formula is: GCV = MSE / pdof^2
    % where pdof = (1/N)tr(I-H) is the proportion of residual DOF
    %
    % Problem: When pdof -> 0 (small bandwidth, overfitting):
    % 1. The penalty 1/pdof^2 -> inf (explodes)
    % 2. The variance of 1/pdof^2 -> inf (unstable for 1-SE rule)
    %
    % Solution: Clamp pdof to a reasonable minimum
    % - Use pdof_min = 0.01 (1% residual DOF)
    % - This means at least N/100 residual DOF
    % - For N=1000: at least 10 residual DOF
    % - For N=73508 (qEEG): at least 735 residual DOF
    % - Bandwidths with pdof < 0.01 are considered equally bad (overfitting)
    %
    % This gives maximum penalty = 1/0.01^2 = 10,000 (reasonable upper bound)
    pdof_for_penalty = max(pdof_clamped, 0.01);  % Clamp to minimum 1% residual DOF

    % Compute penalty and its statistics
    penalty = 1 ./ pdof_for_penalty.^2;  % GCV penalty: 1/pdof^2
    regs.pdof_inv_m = permute(mean(penalty, 1), [2,3,1]);
    regs.pdof_inv_sd = permute(std(penalty, [], 1), [2,3,1]);

    % For backward compatibility, also store log-space versions
    % (not used in GCV calculation, but kept for diagnostics)
    regs.log_gcv_penalty_m = permute(mean(log(penalty), 1), [2,3,1]);
    regs.log_gcv_penalty_sd = permute(std(log(penalty), [], 1), [2,3,1]);

    % Store other DOF statistics
    regs.df_m = permute(mean(df, 1), [2,3,1]);
    regs.df_sd = permute(std(df, [], 1), [2,3,1]);
    regs.pdof_m = permute(mean(pdof_clamped, 1), [2,3,1]);  % Use clamped pdof
    regs.pdof_sd = permute(std(pdof_clamped, [], 1), [2,3,1]);  % Use clamped pdof
else
    regs.log_gcv_penalty_m = [];
    regs.log_gcv_penalty_sd = [];
    regs.pdof_inv_sd = [];
    regs.pdof_inv_m = [];
    regs.df_m = [];
    regs.df_sd = [];
    regs.pdof_m = [];
    regs.pdof_sd = [];
end

end
