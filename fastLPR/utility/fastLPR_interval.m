function ci = fastlpr_interval(mu, sigma, alpha, type)
% FASTLPR_INTERVAL - Compute pointwise intervals for regression estimates
%
% Two interval types are supported:
%
%   'confidence' - Approximate asymptotic pointwise confidence interval
%                  for the conditional mean m(x):
%                  CI = m_hat +/- z * se(m_hat)
%                  where se^2 = sigma^2 * nu / (|H| * s_0)
%
%   'prediction' - Pointwise prediction interval for a new observation y|x,
%                  accounting for both noise and estimation uncertainty:
%                  PI = m_hat +/- z * sqrt(sigma^2 + se^2)
%
% Syntax:
%   ci = fastlpr_interval(mu, sigma)
%   ci = fastlpr_interval(mu, sigma, alpha)
%   ci = fastlpr_interval(mu, sigma, alpha, type)
%
% Inputs:
%   mu    - Mean estimate from cv_fastlpr (struct with .fpp_yhat, .fpp_s0, .opt)
%   sigma - Variance estimate from cv_fastlpr (struct with .fpp_yhat) OR numeric
%   alpha - Significance level (default: 0.05)
%   type  - 'confidence' (default) or 'prediction'
%
% Outputs:
%   ci    - griddedInterpolant with bounds in last dimension [upper, lower]
%
% Notes:
%   - Intervals are pointwise (not simultaneous)
%   - The CI is an asymptotic approximation valid at interior points
%   - Boundary behavior is not corrected; no bias correction applied
%   - s_0 (design density proxy) is obtained from mu.fpp_s0, computed by cv_fastlpr
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

%% Input validation
narginchk(2, 4);

if ~isstruct(mu) || ~isfield(mu, 'fpp_yhat')
    error('fastlpr_interval:InvalidMu', ...
        'mu must be a structure with .fpp_yhat from cv_fastlpr.');
end
if ~isa(mu.fpp_yhat, 'griddedInterpolant')
    error('fastlpr_interval:InvalidMu', ...
        'mu.fpp_yhat must be a griddedInterpolant.');
end
if ~isfield(mu, 'fpp_s0') || ~isa(mu.fpp_s0, 'griddedInterpolant')
    error('fastlpr_interval:MissingS0', ...
        'mu.fpp_s0 is missing. Re-run cv_fastlpr with a recent version.');
end

if ~(isstruct(sigma) || isnumeric(sigma))
    error('fastlpr_interval:InvalidSigma', ...
        'sigma must be a struct from cv_fastlpr or numeric.');
end
if isstruct(sigma)
    if ~isfield(sigma, 'fpp_yhat') || ~isa(sigma.fpp_yhat, 'griddedInterpolant')
        error('fastlpr_interval:InvalidSigma', ...
            'sigma.fpp_yhat must be a griddedInterpolant.');
    end
end

if nargin < 3 || isempty(alpha)
    alpha = 0.05;
end
validateattributes(alpha, {'numeric'}, {'scalar', 'positive', '<', 1});

if nargin < 4 || isempty(type)
    type = 'confidence';
end
type = lower(type);
if ~ismember(type, {'confidence', 'prediction'})
    error('fastlpr_interval:InvalidType', ...
        'type must be ''confidence'' or ''prediction''.');
end

%% Precomputed nu constants for Gaussian kernel
% nu = e_1^T M_K^{-1} M_{K^2} M_K^{-1} e_1
nu_table = containers.Map;
nu_table('1_0') = 0.282094791774;  % 1/(2*sqrt(pi))
nu_table('1_1') = 0.282094791774;
nu_table('1_2') = 0.476034961118;  % 27/(32*sqrt(pi))
nu_table('2_0') = 0.079577471546;  % 1/(4*pi)
nu_table('2_1') = 0.079577471546;
nu_table('2_2') = 0.198943678865;
nu_table('3_0') = 0.022448390266;  % 1/(8*pi^{3/2})
nu_table('3_1') = 0.022448390266;
nu_table('3_2') = 0.077166341538;

%% Extract parameters
dx = numel(mu.fpp_yhat.GridVectors);
ell = mu.opt.order;
h = mu.gcv_yhat.h1se;
if isempty(h)
    if isfield(mu, 'h') && ~isempty(mu.h)
        h = mu.h;
    else
        error('fastlpr_interval:NoBandwidth', ...
            'Cannot determine bandwidth. mu must have gcv_yhat.h1se or h field.');
    end
end
prod_h = prod(h);

nu_key = sprintf('%d_%d', dx, ell);
if ~nu_table.isKey(nu_key)
    error('fastlpr_interval:UnsupportedConfig', ...
        'nu not available for d=%d, order=%d.', dx, ell);
end
nu = nu_table(nu_key);

%% Compute z-score
z = norminv(1 - alpha/2);

%% Get mu, sigma, and s_0 values on the grid
mu_val = mu.fpp_yhat.Values;

if isstruct(sigma)
    sigma_val = sigma.fpp_yhat.Values;
else
    sigma_val = sigma;
end

s0_val = mu.fpp_s0.Values;
s0_val = max(s0_val, 1e-10);  % clamp away from zero

%% Grid compatibility check
if isstruct(sigma) && ~isequal(size(mu_val), size(sigma_val))
    error('fastlpr_interval:ShapeMismatch', ...
        'mu and sigma grid shapes do not match: %s vs %s.', ...
        mat2str(size(mu_val)), mat2str(size(sigma_val)));
end
if ~isequal(size(mu_val), size(s0_val))
    error('fastlpr_interval:ShapeMismatch', ...
        'mu grid shape %s does not match s0 shape %s.', ...
        mat2str(size(mu_val)), mat2str(size(s0_val)));
end

%% Compute se^2
sigma_pos = max(sigma_val, 0);
se_sq = sigma_pos .* nu ./ (prod_h .* s0_val);

%% Compute bounds
if strcmp(type, 'confidence')
    % CI = mu +/- z * sqrt(se^2)
    half_width = z .* sqrt(se_sq);
else
    % PI = mu +/- z * sqrt(sigma^2 + se^2)
    half_width = z .* sqrt(sigma_pos + se_sq);
end

%% Build output griddedInterpolant
ci = mu.fpp_yhat;
ci.Values = cat(dx+1, ...
    mu_val + half_width, ...
    mu_val - half_width);

end
