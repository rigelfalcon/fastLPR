function [gcv]=fastlpr_gcv(regs,y,yhat,fpp,metric,dstd,iy)
% FASTLPR_GCV - Compute Generalized Cross-Validation for bandwidth selection
%
% Computes GCV scores for all bandwidth candidates and selects optimal
% bandwidth using minimum GCV and 1-SE rule.
%
% Syntax:
%   gcv = fastlpr_gcv(regs, y, yhat)
%   gcv = fastlpr_gcv(regs, y, yhat, fpp, metric, dstd, iy)
%
% Inputs:
%   regs   - Regression structure containing:
%            .dh        - Number of bandwidths
%            .h         - Bandwidth matrix (dh x dx)
%            .pdof_inv_m - Mean GCV penalty (1/pdof)
%            .pdof_inv_sd - Std of GCV penalty
%            .ihbad     - Logical array of bad bandwidths
%            .opt.dstd  - Number of std for 1-SE rule
%   y      - Response data (Ty x dy matrix)
%   yhat   - Fitted values for all bandwidths (Ty x dy x dh array)
%   fpp    - griddedInterpolant object (optional)
%   metric - Error metric (default: 'res' for residual sum of squares)
%   dstd   - Number of std for 1-SE rule (default: regs.opt.dstd)
%   iy     - Response indices to process (default: ':' for all)
%
% Output:
%   gcv - Structure containing:
%         .yhat_1se  - Fitted values with 1-SE bandwidth
%         .h1se      - 1-SE bandwidth (dx x dy matrix)
%         .hmin      - Minimum GCV bandwidth (dx x dy matrix)
%         .gcv_m     - Mean GCV scores (dh x dy matrix)
%         .gcv_sd    - Std of GCV scores (dh x dy matrix)
%         .mse       - Mean squared error (dh x dy matrix)
%         .rss       - Residual sum of squares (dh x dy matrix)
%         .pdof_m    - Mean proportion of residual DOF
%         .df_m      - Mean degrees of freedom
%         .fpp       - griddedInterpolant for 1-SE fit
%
% Algorithm:
%   1. Compute MSE for each bandwidth
%   2. Compute GCV = MSE / pdof (with clamped pdof from fastlpr_dof)
%   3. Find bandwidth with minimum GCV
%   4. Apply 1-SE rule: select largest bandwidth within 1 SE of minimum
%
% GCV Formula:
%   GCV(h) = MSE(h) / pdof(h)
%   where pdof = (1/N)tr(I-H) is proportion of residual DOF
%
% 1-SE Rule:
%   - Find minimum GCV: gcv_min
%   - Compute threshold: gcv_min + dstd * SE(gcv_min)
%   - Select largest bandwidth with GCV <= threshold
%   - This prefers smoother fits (larger h) when GCV is similar
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

%% Input validation and defaults
if nargin<7 || isempty(iy)
    iy=':';
end
if nargin<6 || isempty(dstd)
    dstd=regs.opt.dstd;
end
if nargin<5 || isempty(metric)
    metric='res';
end

[Ty,dy]=size(y);
dh=regs.dh;
ihbad=regs.ihbad;

%% Compute error metrics and GCV scores
% Permute yhat to (Ty x dh x dy) for easier indexing
yhat=permute(yhat,[1,3,2]);

switch metric
    case {'res'}
        % Compute residual sum of squares (RSS)
        rss=reshape(sum(abs(y-yhat).^2),dy,regs.dh).';

        % Compute mean squared error (MSE)
        mse=(1/regs.Ty)*rss;

        % Compute GCV with clamped penalty (from fastlpr_dof.m)
        % GCV = MSE / pdof = MSE * (1/pdof)
        % The penalty (1/pdof) is pre-computed with clamping to avoid explosion
        % This prevents GCV from exploding when pdof -> 0 (overfitting)
        gcv_m = mse .* regs.pdof_inv_m(:,iy);
        gcv_sd = mse .* regs.pdof_inv_sd(:,iy);
    otherwise
        error('fastLPR:gcv:UnsupportedMetric', ...
            'Only ''res'' (residual) metric is currently supported');
end

% Check for complete failure
if all(isnan(gcv_m))
    error('fastLPR:gcv:AllNaN', ...
        ['Regression failed for all bandwidths. Try:\n' ...
         '  1. Use larger bandwidth h\n' ...
         '  2. Increase grid points with opt.Nratio\n' ...
         '  3. Check for data quality issues']);
end

% Compute GCV confidence bounds
gcv_up=gcv_m+gcv_sd;
gcv_lo=gcv_m-gcv_sd;

if nargin<4 || isempty(fpp)
    fpp=[];
end

% Get indices of good bandwidths (not too small)
ihgood=find(~ihbad);

%% Find minimum GCV bandwidth
[~,idmingood]=min(gcv_m,[],1);

%% Apply 1-SE rule to select smoothest bandwidth within 1 SE of minimum
% The 1-SE rule prefers larger (smoother) bandwidths when GCV is similar
% This reduces overfitting and improves stability
id1segood=zeros(1,dy);
for iy=1:dy
    % Compute threshold: minimum GCV + dstd * SE
    semin=gcv_m(idmingood(iy),iy)+dstd*gcv_sd(idmingood(iy),iy).';

    % Find all bandwidths with GCV <= threshold
    candidates = find(gcv_m(:,iy)<=semin);

    if isempty(candidates)
        % No candidates found, use minimum GCV bandwidth
        id1segood(iy) = idmingood(iy);
    else
        % Among candidates, select the one with largest bandwidth (sum of h)
        hsum=sum(regs.h(ihgood,:),2);  % Total bandwidth (sum across dimensions)
        hsum_candidates = hsum(candidates);
        [~, max_idx] = max(hsum_candidates);
        id1sey = candidates(max_idx);

        % If multiple bandwidths have same max sum, pick one with smallest GCV
        % (This handles ties in bandwidth sum)
        same_sum_candidates = candidates(hsum_candidates == hsum_candidates(max_idx));
        id1segood(iy) = same_sum_candidates(idxofmin(gcv_m(same_sum_candidates,iy)));
    end
end
%% Extract fitted values for selected bandwidths
if ~isempty(fpp)
    % Update griddedInterpolant to contain only 1-SE fits
    fpp.Values=fpp.Values(regs.ndcolon{:},id1segood+dh*((1:dy)-1));
end

% Extract fitted values at data points for 1-SE bandwidth
yhat_1se=zeros(regs.Tx,dy);
for iy=1:dy
    yhat_1se(:,iy)=squeeze(yhat(:,iy,id1segood(iy)));
end

%% Convert local indices to global bandwidth indices
% idmingood and id1segood are indices within ihgood (good bandwidths)
% Convert to indices within full bandwidth list
idmin=idmingood;
id1se=id1segood;
for iy=1:dy
    idmin(iy)=ihgood(idmingood(iy)).';
    id1se(iy)=ihgood(id1segood(iy)).';
end

% Extract selected bandwidths
hmin=regs.h(idmin,:).';  % Minimum GCV bandwidth (dx x dy)
h1se=regs.h(id1se,:).';  % 1-SE bandwidth (dx x dy)

%% Extract GCV parameters for selected bandwidths
msemin=mse(idmingood);
mse1se=mse(id1segood);
rssmin=rss(idmingood);
rss1se=rss(id1segood);
pdof_m_min=reshape(regs.pdof_m(idmingood,iy),1,[]);
pdof_m_1se=reshape(regs.pdof_m(id1segood,iy),1,[]);
df_m_min=reshape(regs.df_m(idmingood),1,[]);
df_m_1se=reshape(regs.df_m(id1segood),1,[]);

%% Display selected bandwidths (verbose only)
if isfield(regs.opt, 'verbose') && regs.opt.verbose
    for iy=1:dy
        disp(['[fastlpr]: [',num2str(regs.dx),'d] regression with order ',num2str(regs.opt.order) ])
        disp([' - selected bandwidth - [',num2str(find(ismember(regs.h, h1se(:,iy).','row'))),'th] h'])
        disp([' - normalized scale h =[',num2str(h1se(:,iy)'),']'])
        disp([' - original scale h =[',num2str(h1se(:,iy)'.*regs.x_std),']'])
    end
end

%% Package results into output structure
gcv.metric=metric;

% Fitted values
gcv.yhat_1se=yhat_1se;

% Selected bandwidths
gcv.h1se=h1se;  % 1-SE bandwidth (recommended)
gcv.hmin=hmin;  % Minimum GCV bandwidth

% GCV scores
gcv.gcv_m=gcv_m;    % Mean GCV for all bandwidths
gcv.gcv_sd=gcv_sd;  % Std of GCV
gcv.gcv_up=gcv_up;  % Upper confidence bound
gcv.gcv_lo=gcv_lo;  % Lower confidence bound

% Selected bandwidth indices
gcv.idmin=idmin;  % Index of minimum GCV bandwidth
gcv.id1se=id1se;  % Index of 1-SE bandwidth

% Interpolant
gcv.fpp=fpp;

% Error metrics
gcv.rss=rss;        % RSS for all bandwidths
gcv.mse=mse;        % MSE for all bandwidths
gcv.msemin=msemin;  % MSE at minimum GCV
gcv.mse1se=mse1se;  % MSE at 1-SE bandwidth
gcv.rssmin=rssmin;  % RSS at minimum GCV
gcv.rss1se=rss1se;  % RSS at 1-SE bandwidth

% DOF statistics
gcv.pdof_m=regs.pdof_m(:,iy);      % Mean pdof for all bandwidths
gcv.pdof_m_min=pdof_m_min;         % Mean pdof at minimum GCV
gcv.pdof_m_1se=pdof_m_1se;         % Mean pdof at 1-SE bandwidth
gcv.df_m=regs.df_m(:,iy);          % Mean df for all bandwidths
gcv.df_m_min=df_m_min;             % Mean df at minimum GCV
gcv.df_m_1se=df_m_1se;             % Mean df at 1-SE bandwidth

end

%% Helper function

function idx=idxofmin(varargin)
% Find index of minimum value
[~,idx]=min(varargin{:});
end