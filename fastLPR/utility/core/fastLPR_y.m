function [regs]=fastlpr_y(regs)
% FASTLPR_Y - Compute regression estimates for response variables
%
% This is the main computational function that performs the actual regression
% using NUFFT-accelerated convolution. It handles multiple response variables,
% memory management, and both mean and variance estimation.
%
% Syntax:
%   regs = fastlpr_y(regs)
%
% Input:
%   regs - Regression structure from fastlpr_create, must contain:
%          .y, .opt, .dh, .dy, .Ty, .s, .kdf, .y_isreal
%
% Output:
%   regs - Updated regression structure with:
%          .t - Weighted response T = X'*W*y (for each bandwidth)
%          .m - Regression estimate m(x) (for each bandwidth)
%
% Algorithm:
%   1. Prepare response data (handle bandwidth-specific responses)
%   2. Apply log transform for variance estimation if needed
%   3. Estimate memory requirements
%   4. Split responses into batches if memory is limited
%   5. For each batch:
%      a. Compute weighted response T via NUFFT convolution
%      b. Compute regression estimate m = solve(S, T)
%   6. Merge results from all batches
%
% Memory Management:
%   - Estimates memory needed for NUFFT and regression
%   - Splits responses into batches to fit available memory
%   - Uses redundancy factor of 15 (6*2.5) for safety margin
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

%% Prepare response data
% Handle bandwidth-specific responses (different bandwidth for each response)
if regs.opt.y_corr_bandwidth
    dhy=size(regs.y,2);
    if dhy==1
        % Single response: replicate for each bandwidth
        y=repmat(regs.y,[1,regs.dh,1]);
    else
        % Multiple responses: already have bandwidth-specific data
        y=regs.y;
    end
    % Reshape to (Ty x (dh*dy)) for batch processing
    y=reshape(y,[regs.Ty,regs.dh*regs.dy]);
    dy=regs.dh*regs.dy;
else
    % Standard case: same bandwidth for all responses
    y=regs.y;
    dy=regs.dy;
end

%% Apply log transform for variance estimation
% For variance estimation, we estimate log(variance) for numerical stability
if strcmp(regs.opt.y_type_out,'variance')
    yraw=regs.y;
    y=log(regs.y+1/regs.Ty);  % Add small constant to avoid log(0)
end
%% Memory management: split responses into batches
% For multiple responses, estimate memory requirements and split into batches
% to avoid out-of-memory errors
if dy>1
    % Redundancy factor: accounts for temporary arrays and overhead
    % Configurable via opt.memory_redundancy (default: 7)
    % Breakdown: 2x for T and m, 2x for s and kdf, 1.5x for solve temporaries,
    %            1.5x for MATLAB overhead = ~7x total
    if isfield(regs.opt, 'memory_redundancy') && ~isempty(regs.opt.memory_redundancy)
        redundancy = regs.opt.memory_redundancy;
    else
        redundancy = 7;  % Optimized default (was 15)
    end

    % Estimate NUFFT memory requirements
    % NUFFT needs memory for:
    % 1. Index arrays: (2*Msp+1)^dx * Tx * dx elements
    % 2. Data arrays: max of spread data and FFT result
    dtype_nufft=min_numerical_type(0,max(regs.L));  % Integer type for indices
    byte_nufft_xindx=bytesize(((2*regs.Msp+1)^regs.dx)*regs.Tx*(regs.dx),'B',dtype_nufft);
    bsize_nufft_data=bytesize(max(((2*regs.Msp+1)^regs.dx)*regs.Tx*dy,prod(regs.L)*dy),'B',regs.dtype);
    bsize_nufft=byte_nufft_xindx.B+bsize_nufft_data.B;

    % Estimate regression memory requirements
    % Regression needs memory for:
    % 1. Weighted response T: L^dx * dy * dh elements
    % 2. Regression estimate m: L^dx * dy * dh elements
    % 3. Design matrix s: L^dx * dh elements (shared across responses)
    % 4. Kernel kdf: L^dx * dh elements (shared across responses)
    bsize_reg=bytesize((regs.L)^regs.dx*regs.dy*regs.dh,'B',regs.dtype);
    bsize_reg=2*bsize_reg.B;  % Factor of 2 for T and m

    % Total memory per response
    bsize = max(bsize_nufft,bsize_reg)*redundancy;

    % Adjust for complex-valued responses
    % Complex responses need 2x memory (real + imaginary parts)
    % Formula: (num_complex*2 + num_real) / dy
    num_complex = sum(~regs.y_isreal);  % Number of complex responses
    num_real = dy - num_complex;         % Number of real responses
    complex_factor = (num_complex*2 + num_real) / dy;
    bsize = complex_factor * bsize;

    % Verbose logging if requested
    if isfield(regs.opt, 'verbose') && regs.opt.verbose
        fprintf('Memory estimation:\n');
        fprintf('  NUFFT memory: %.2f MB\n', bsize_nufft/1e6);
        fprintf('  Regression memory: %.2f MB\n', bsize_reg/1e6);
        fprintf('  Redundancy factor: %.1f\n', redundancy);
        fprintf('  Complex factor: %.2f\n', complex_factor);
        fprintf('  Total per response: %.2f MB\n', bsize/1e6);
        fprintf('  Number of responses: %d (%d real, %d complex)\n', ...
            dy, num_real, num_complex);
    end

    % Check available memory and move to GPU if needed
    if ~regs.opt.isgpu
        [~,freemem]=test_memory(bsize,true);
    else
        [~,freemem]=test_gpu(bsize,1);
        % Move precomputed arrays to GPU
        regs.s=gpuArray(regs.s);
        regs.kdf=gpuArray(regs.kdf);
    end

    % Sanity check: warn if memory estimate seems unreasonable
    if bsize > freemem * 0.9
        warning('fastLPR:memory:HighUsage', ...
            'Estimated memory (%.2f MB) is close to available memory (%.2f MB). Consider reducing data size or using batching.', ...
            bsize/1e6, freemem/1e6);
    end

    % Split responses into batches that fit in available memory
    if ~regs.opt.y_corr_bandwidth
        % Standard case: split dy responses
        idx=gen_loopgroup(dy,freemem/(bsize/dy));
    else
        % Bandwidth-specific case: split regs.dy responses, replicate for dh bandwidths
        idx=gen_loopgroup(regs.dy,freemem/(bsize/regs.dy),regs.dh);
    end
    num_loop=numel(idx);

    % Verbose logging for batching
    if isfield(regs.opt, 'verbose') && regs.opt.verbose
        fprintf('  Available memory: %.2f MB\n', freemem/1e6);
        fprintf('  Number of batches: %d\n', num_loop);
        if num_loop > 1
            fprintf('  Responses per batch: ');
            for i = 1:num_loop
                fprintf('%d ', length(idx{i}));
            end
            fprintf('\n');
        end
    end
else
    % Single response: no batching needed
    idx={1};
    num_loop=1;
end

%% loop for y regression
if  ~regs.opt.y_corr_bandwidth%regs.opt.calc_dof %  && all y one bandwidth
    for iloop=num_loop:-1:1
        iy=idx{iloop};
        %% y regression (important part)
        % fft y
        y_ft=fastlpr_nufft(regs,y(:,iy));
        % regression
        [mq]=fastlpr_reg(regs,y_ft);clear y_ft
        if ~regs.opt.calc_dof
            % grid interpolation
            [gcv(1).fpp,gcv(1).yhat]=fastlpr_gridinterp(regs,mq,regs.xlist,regs.opt.y_grid_method,regs.opt.y_grid_opt);clear mq
            gcv(2)=fastlpr_gcvmerge(gcv);
            if iloop==1
                gcv=gcv(2);
            end
        else
            % grid interpolation
            [fpp,yhat]=fastlpr_gridinterp(regs,mq,regs.xlist,regs.opt.y_grid_method,regs.opt.y_grid_opt);clear mq
            % gcv
            [gcv(iloop)]=fastlpr_gcv(regs,y(:,iy),yhat,fpp,regs.opt.y_gcv_metric,regs.opt.dstd,iy);
            clear fpp yhat
        end
    end
    if ~regs.opt.calc_dof
        regs.gcv_yhat=[];
        regs.fpp_yhat=gcv.fpp;
        regs.yhat=gcv.yhat;
    else
        regs.gcv_yhat=fastlpr_gcvmerge(gcv);
        regs.fpp_yhat=regs.gcv_yhat.fpp;
        regs.gcv_yhat.fpp=[];
        regs.yhat=regs.gcv_yhat.yhat_1se;
        regs.gcv_yhat.yhat_1se=[];
    end
else % one y one bandwidth
    for iloop=num_loop:-1:1
        iy=idx{iloop};
        % fft y
        if ~regs.opt.isgpu
            y_ft=fastlpr_nufft(regs,y(:,iy));
        else
            y_ft=gather(fastlpr_nufft(regs,gpuArray(y(:,iy))));
        end
        % regression
        [mq]=fastlpr_reg(regs,y_ft);clear y_ft
        % grid interpolation
        [gcv(1).fpp,gcv(1).yhat]=fastlpr_gridinterp(regs,mq,regs.xlist,regs.opt.y_grid_method,regs.opt.y_grid_opt);clear mq
        gcv(2)=fastlpr_gcvmerge(gcv);
        if iloop==1
            gcv=gcv(2);
        end
    end
    regs.gcv_yhat=[];
    regs.fpp_yhat=gcv.fpp;
    regs.yhat=gcv.yhat;
end
% if regs.opt.y_corr_bandwidth
%     regs.yhat=reshape(regs.yhat,[regs.Ty,regs.dh,regs.dy]);
% end
%%
regs.d=[];
if strcmp(regs.opt.y_type_out,'variance')
    d=1./((1/regs.Ty).*sum(yraw.*exp(-regs.yhat)));
    regs.yhat=exp(regs.yhat)./d;
    if ~regs.opt.y_corr_bandwidth
        regs.fpp_yhat.Values=exp(regs.fpp_yhat.Values)./reshape(d,[ones(1,regs.dx),regs.dy]);
    else
        regs.fpp_yhat.Values=exp(regs.fpp_yhat.Values)./reshape(d,[ones(1,regs.dx),regs.dh*regs.dy]);
    end
    regs.d=d;
end



if any(isinf(regs.yhat(:)))|| any(isnan(regs.yhat(:)))
    error('yhat have bad value, try enlarge opt.N')
end
% if any(isinf(regs.fpp_yhat.Values(:)))|| any(isnan(regs.fpp_yhat.Values(:)))
%     error('fpp_yhat.Values have bad value')
% end

%% Store memory information for validation
% This information is useful for validating memory estimation accuracy
if exist('bsize', 'var') && exist('freemem', 'var') && exist('num_loop', 'var')
    regs.memory_info.estimated_mb = bsize / 1e6;
    regs.memory_info.available_mb = freemem / 1e6;
    regs.memory_info.num_batches = num_loop;
    if num_loop > 1
        regs.memory_info.batch_size = length(idx{1});
    else
        regs.memory_info.batch_size = dy;
    end
end
