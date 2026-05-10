function [regs] = cv_fastlpr(x, y, h, opt)
% CV_FASTLPR - Fast Local Polynomial Regression with automatic bandwidth selection
%
% This is the main function for fastLPR toolbox. It performs nonparametric
% regression using kernel-weighted local polynomial methods with NUFFT
% (Non-Uniform Fast Fourier Transform) acceleration. The function automatically
% selects the optimal bandwidth via Generalized Cross-Validation (GCV) when
% multiple bandwidth candidates are provided.
%
% Key Features:
%   - Handles scattered and complex-valued data
%   - NUFFT-based fast computation: O(N + M log M) complexity
%   - Automatic bandwidth selection via GCV
%   - Supports local constant (order 0), local linear (order 1), and
%     local quadratic (order 2) regression
%   - Heteroscedastic variance estimation
%   - Multi-dimensional predictors (1D, 2D, or higher)
%
% Syntax:
%   regs = cv_fastlpr(x, y)
%   regs = cv_fastlpr(x, y, h)
%   regs = cv_fastlpr(x, y, h, opt)
%
% Inputs:
%   x   - N x d matrix of predictors (N samples, d dimensions)
%         Can be real or complex-valued. Each row is one observation.
%         For complex-valued predictors, use x = real_part + 1i*imag_part
%
%   y   - N x 1 vector of responses
%         Can be real or complex-valued. Must have same number of rows as x.
%
%   h   - Bandwidth parameter(s) (optional, default: automatic selection)
%         Scalar: same bandwidth for all dimensions
%         1 x d vector: different bandwidth per dimension
%         k x d matrix: grid of k bandwidth combinations for GCV selection
%         Use get_hlist() to generate bandwidth candidates
%
%   opt - Options structure (optional) with fields:
%         .order: Polynomial order (default: 0)
%                 0 = Nadaraya-Watson (local constant)
%                 1 = Local linear regression
%                 2 = Local quadratic regression
%         .kernel_type: Kernel function (default: 'gaussian')
%                       'gaussian' or 'epanechnikov'
%         .dstd: Number of DOF samples for variance estimation (default: 0)
%                Set to 5-20 for heteroscedastic variance estimation
%         .y_type_out: Output type (default: 'mean')
%                      'mean' = estimate conditional mean E[Y|X]
%                      'variance' = estimate conditional variance Var[Y|X]
%         .N: Grid resolution for evaluation (default: auto)
%         .xrange: Evaluation range (default: data range)
%
% Outputs:
%   regs - Structure containing regression results:
%          .yhat: Fitted values at sample points (Nx 1)
%          .fpp_yhat: griddedInterpolant object for prediction at any point
%                     Use: y_pred = regs.fpp_yhat(x_new)
%          .gcv_yhat: GCV results structure (if multiple bandwidths provided)
%                     .h1se: Selected bandwidth (1-SE rule)
%                     .hmin: Bandwidth with minimum GCV
%                     .gcv: GCV values for all bandwidths
%          .xq: Evaluation grid points (cell array for multi-dimensional)
%          .xlist: Grid vectors (cell array)
%          .opt: Options used for regression
%          .xraw: Original predictor data
%          .yraw: Original response data
%
% Examples:
%   % Example 1: 1D regression with automatic bandwidth selection
%   x = rand(500, 1) * 20;
%   y = sin(x) + 0.2*randn(500, 1);
%   hlist = get_hlist(20, [0.01, 1], @logspace);
%   opt.order = 1;  % Local linear
%   regs = cv_fastlpr(x, y, hlist, opt);
%
%   % Plot results
%   figure; hold on;
%   scatter(x, y, 'k.');
%   fastlpr_plot(regs.fpp_yhat);
%
%   % Example 2: 2D regression
%   x = rand(1000, 2);
%   y = sin(2*pi*x(:,1)) .* cos(2*pi*x(:,2)) + 0.1*randn(1000, 1);
%   hlist = get_hlist([10, 10], [0.01, 1; 0.01, 1], @logspace);
%   opt.order = 1;
%   regs = cv_fastlpr(x, y, hlist, opt);
%
%   % Example 3: Complex-valued data
%   x = rand(500, 1) + 1i*rand(500, 1);
%   y = exp(1i*2*pi*abs(x)) + 0.1*randn(500, 1);
%   hlist = get_hlist(20, [0.01, 0.5], @logspace);
%   opt.order = 1;
%   regs = cv_fastlpr(x, y, hlist, opt);
%
%   % Example 4: Heteroscedastic variance estimation
%   % First, estimate mean
%   opt.order = 1;
%   opt.dstd = 0;
%   regs_mu = cv_fastlpr(x, y, hlist, opt);
%
%   % Then, estimate variance
%   residuals = y - regs_mu.yhat;
%   opt.y_type_out = 'variance';
%   opt.dstd = 10;
%   regs_var = cv_fastlpr(x, residuals.^2, hlist, opt);
%
%   % Compute intervals (CI and PI)
%   ci = fastlpr_interval(regs_mu, regs_var, 0.05);
%
% Algorithm:
%   The function implements kernel-weighted local polynomial regression:
%   1. For each evaluation point x0, fit a polynomial using nearby data
%   2. Weights are determined by kernel function K((x-x0)/h)
%   3. NUFFT is used to accelerate convolution operations
%   4. GCV is used to select optimal bandwidth if multiple candidates provided
%
% Computational Complexity:
%   O(N + M log M) where N = number of data points, M = grid size
%   This is much faster than naive O(N*M) implementation
%
% See also: get_hlist, fastlpr_interval, fastlpr_plot, fastlpr_plot_interval
%
% References:
%   [1] Wang, Y., & Li, M. (2024). Fast and Exact Kernel-Weighted Regression
%       for Large-Scale Scattered and Complex-Valued Data.
%       Submitted.
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

%% Input validation
narginchk(2, 4);

% Validate x
validateattributes(x, {'numeric'}, {'2d', 'nonempty', 'finite'}, ...
    'cv_fastlpr', 'x', 1);

% Validate y
validateattributes(y, {'numeric'}, {'nonempty', 'finite'}, ...
    'cv_fastlpr', 'y', 2);

% Ensure y is a column vector
if ~isvector(y)
    error('cv_fastlpr:InvalidInput', ...
        'Input y must be a vector. Got size: [%s]', num2str(size(y)));
end
y = y(:);  % Force column vector

% Check dimension compatibility
if size(x, 1) ~= length(y)
    error('cv_fastlpr:DimensionMismatch', ...
        'Inputs x and y must have the same number of rows.\nx has %d rows, y has %d rows.', ...
        size(x, 1), length(y));
end

% Set defaults for optional inputs
if nargin < 4 || isempty(opt)
    opt = struct();
end
if nargin < 3 || isempty(h)
    h = [];
end

% Validate opt fields if provided
if isfield(opt, 'order')
    if ~ismember(opt.order, [0, 1, 2])
        error('cv_fastlpr:InvalidOrder', ...
            'opt.order must be 0 (Nadaraya-Watson), 1 (local linear), or 2 (local quadratic).\nGot: %d', ...
            opt.order);
    end
end

if isfield(opt, 'kernel_type')
    valid_kernels = {'gaussian', 'epanechnikov'};
    if ~ismember(lower(opt.kernel_type), valid_kernels)
        error('cv_fastlpr:InvalidKernel', ...
            'opt.kernel_type must be ''gaussian'' or ''epanechnikov''.\nGot: %s', ...
            opt.kernel_type);
    end
end

%% Initialization
% Create model structure with data, bandwidth, and options
if isfield(opt, 'profile_timing') && opt.profile_timing
    t_start = tic;
    [regs] = fastlpr_create(x, y, h, opt);
    regs.timing.create = toc(t_start);
    fprintf('[TIMING] fastlpr_create: %.4f s\n', regs.timing.create);
else
    [regs] = fastlpr_create(x, y, h, opt);
end

%% Compute design matrix
% Calculate kernel-weighted design matrix S using NUFFT
if isfield(opt, 'profile_timing') && opt.profile_timing
    t_start = tic;
    [regs] = fastlpr_s(regs);
    regs.timing.s = toc(t_start);
    fprintf('[TIMING] fastlpr_s: %.4f s\n', regs.timing.s);
else
    [regs] = fastlpr_s(regs);
end

%% Estimate degrees of freedom
% Use randomized method for DOF estimation (for GCV and variance)
if isfield(opt, 'profile_timing') && opt.profile_timing
    t_start = tic;
    [regs] = fastlpr_dof(regs);
    regs.timing.dof = toc(t_start);
    fprintf('[TIMING] fastlpr_dof: %.4f s\n', regs.timing.dof);
else
    [regs] = fastlpr_dof(regs);
end

%% Compute fitted values
% Calculate kernel-weighted response y using NUFFT
if isfield(opt, 'profile_timing') && opt.profile_timing
    t_start = tic;
    [regs] = fastlpr_y(regs);
    regs.timing.y = toc(t_start);
    fprintf('[TIMING] fastlpr_y: %.4f s\n', regs.timing.y);
else
    [regs] = fastlpr_y(regs);
end

%% Extract s_0 (zero-th kernel moment) for CI/PI construction
% s_0 = sum_i K_h(x - x_i) = n * f_hat(x), needed for se(m_hat)
try
    if regs.opt.order == 0
        s0_all = regs.s;  % numeric array: (grid..., dh)
    else
        s0_all = regs.s{1};  % first element of cell = zero-th moment
    end
    % Select s_0 at the chosen bandwidth
    if ~isempty(regs.gcv_yhat) && isfield(regs.gcv_yhat, 'id1se')
        id1se_global = regs.gcv_yhat.id1se;
        % Convert global index to local index within good bandwidths
        % s0_all has bad bandwidths removed, so its bandwidth dimension = dh (good only)
        if isfield(regs, 'ihbad') && any(regs.ihbad)
            ihgood = find(~regs.ihbad);
            id1se_local = find(ihgood == id1se_global(1));
            if isempty(id1se_local)
                % id1se points to a removed bandwidth — fall back to closest good one
                [~, id1se_local] = min(abs(ihgood - id1se_global(1)));
            end
        else
            id1se_local = id1se_global(1);
        end
        s0_selected = s0_all(regs.ndcolon{:}, id1se_local);
    else
        % Single bandwidth or no GCV: use first (only) slice
        if ndims(s0_all) > regs.dx || (regs.dx == 1 && size(s0_all, 2) > 1)
            s0_selected = s0_all(regs.ndcolon{:}, 1);
        else
            s0_selected = s0_all;
        end
    end
    s0_selected = squeeze(real(s0_selected));
    % Build griddedInterpolant for s_0 (same pattern as fpp_yhat)
    regs.fpp_s0 = griddedInterpolant(regs.xlist, s0_selected, 'spline', 'linear');
    regs.s0 = s0_selected;
catch ME
    % s_0 extraction failed (e.g., dimension mismatch with complex/multi-response data)
    % CI/PI will not be available for this regression result
    warning('fastLPR:s0extraction', 's_0 extraction failed: %s. CI/PI intervals unavailable.', ME.message);
    regs.fpp_s0 = [];
    regs.s0 = [];
end

%% Compact results
% Remove intermediate variables to save memory
if isfield(opt, 'profile_timing') && opt.profile_timing
    t_start = tic;
    % Save timing before compact (in case compact removes fields)
    timing_backup = regs.timing;
    [regs] = fastlpr_compact(regs);
    timing_backup.compact = toc(t_start);
    regs.timing = timing_backup;
    fprintf('[TIMING] fastlpr_compact: %.4f s\n', regs.timing.compact);
    total_time = regs.timing.create + regs.timing.s + regs.timing.dof + regs.timing.y + regs.timing.compact;
    fprintf('[TIMING] TOTAL: %.4f s\n', total_time);
else
    [regs] = fastlpr_compact(regs);
end

end





