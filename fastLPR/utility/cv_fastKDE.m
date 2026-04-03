function [kde] = cv_fastkde(x, h, opt)
% CV_FASTKDE - Fast Kernel Density Estimation with automatic bandwidth selection
%
% This function performs nonparametric density estimation using kernel
% smoothing with NUFFT (Non-Uniform Fast Fourier Transform) acceleration.
% It automatically selects the optimal bandwidth via Likelihood Cross-
% Validation (LCV) when multiple bandwidth candidates are provided.
%
% Key Features:
%   - Handles scattered and complex-valued data
%   - NUFFT-based fast computation: O(N + M log M) complexity
%   - Automatic bandwidth selection via LCV
%   - Supports 1D and 2D density estimation
%   - Reuses fastLPR infrastructure (order=0 regression with y=1)
%
% Syntax:
%   kde = cv_fastkde(x)
%   kde = cv_fastkde(x, h)
%   kde = cv_fastkde(x, h, opt)
%
% Inputs:
%   x   - N x d matrix of data points (N samples, d dimensions)
%         Can be real or complex-valued. Each row is one observation.
%         For complex-valued data, use x = real_part + 1i*imag_part
%
%   h   - Bandwidth parameter(s) (optional, default: automatic selection)
%         Scalar: same bandwidth for all dimensions
%         1 x d vector: different bandwidth per dimension
%         k x d matrix: grid of k bandwidth combinations for LCV selection
%         Use get_hlist() to generate bandwidth candidates
%
%   opt - Options structure (optional) with fields:
%         .kernel_type: Kernel function (default: 'gaussian')
%                       'gaussian' or 'epanechnikov'
%         .N: Grid resolution for evaluation (default: auto)
%         .xrange: Evaluation range (default: data range)
%         .verbose: Display progress (default: false)
%
% Outputs:
%   kde - Structure containing density estimation results (similar to regs from cv_fastlpr):
%         .h: Selected bandwidth (scalar for 1D, vector for multi-D)
%         .fhat: Density estimate at evaluation grid points for SELECTED bandwidth only
%                (N_grid x 1 for 1D, N_grid x N_grid for 2D)
%                Note: Only contains result for selected bandwidth, not all bandwidths
%         .fpp: griddedInterpolant object for evaluation at any point
%               Use: f_pred = kde.fpp(x_new)
%         .lcv: LCV results structure (if multiple bandwidths provided)
%                      .h1se: Selected bandwidth (1-SE rule)
%                      .hmin: Bandwidth with minimum LCV
%                      .lcv_m: LCV values for all bandwidths
%                      .id1se: Index of selected bandwidth
%         .xlist: Grid vectors (cell array)
%         .opt: Options used for estimation
%         .xraw: Original data points
%
% Examples:
%   % Example 1: 1D density estimation with automatic bandwidth selection
%   x = [randn(100, 1); randn(100, 1) + 3];  % Bimodal distribution
%   hlist = get_hlist(20, [0.01, 1], @logspace);
%   kde = cv_fastkde(x, hlist);
%
%   % Plot results
%   figure; hold on;
%   plot(kde.xlist{1}, kde.fhat, 'b-', 'LineWidth', 2);
%   histogram(x, 30, 'Normalization', 'pdf', 'FaceAlpha', 0.3);
%
%   % Example 2: 2D density estimation
%   x = [randn(200, 2); randn(200, 2) + 2];
%   hlist = get_hlist([10, 10], [0.1, 1; 0.1, 1], @logspace);
%   kde = cv_fastkde(x, hlist);
%
%   % Visualize
%   figure;
%   fastlpr_plot(kde.fpp);
%
% Algorithm:
%   The function implements kernel density estimation:
%   1. For each evaluation point x0, compute f(x0) = (1/n) * sum_i K((x_i-x0)/h)
%   2. Weights are determined by kernel function K((x-x0)/h)
%   3. NUFFT is used to accelerate convolution operations
%   4. LCV is used to select optimal bandwidth if multiple candidates provided
%
% Computational Complexity:
%   O(N + M log M) where N = number of data points, M = grid size
%   This is much faster than naive O(N*M) implementation
%
% See also: cv_fastlpr, get_hlist, fastlpr_plot
%
% References:
%   [1] Wand, M. P., & Jones, M. C. (1994). Kernel Smoothing. CRC press.
%   [2] Wang, Y., & Li, M. (2024). Fast and Exact Kernel-Weighted Regression
%       for Large-Scale Scattered and Complex-Valued Data.
%       Journal of Statistical Software (under review).
%
% Author: Ying Wang, Min Li
% Create Time: 2024
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

%% Input validation
narginchk(1, 3);

% Validate x
validateattributes(x, {'numeric'}, {'2d', 'nonempty', 'finite'}, ...
    'cv_fastkde', 'x', 1);

[n, dx] = size(x);

if n < 2
    error('cv_fastkde:InvalidInput', ...
        'Input x must have at least 2 observations. Got %d observations.', n);
end

% Note: KDE now supports any dimension (1D, 2D, 3D, etc.)
% The underlying fastLPR infrastructure handles multi-dimensional data

% Set defaults for optional inputs
if nargin < 3 || isempty(opt)
    opt = struct();
end
if nargin < 2 || isempty(h)
    h = [];
end

% Validate opt fields if provided
if isfield(opt, 'kernel_type')
    valid_kernels = {'gaussian', 'epanechnikov'};
    if ~ismember(lower(opt.kernel_type), valid_kernels)
        error('cv_fastkde:InvalidKernel', ...
            'opt.kernel_type must be ''gaussian'' or ''epanechnikov''.\nGot: %s', ...
            opt.kernel_type);
    end
end

%% Set KDE-specific options
% KDE is equivalent to order=0 regression with y=1
opt.order = 0;  % Nadaraya-Watson
opt.calc_dof = false;  % Don't need DOF for KDE
opt.y_type_out = 'mean';  % Standard output

% Set default kernel type
opt = set_defaults(opt, 'kernel_type', 'gaussian');
opt = set_defaults(opt, 'verbose', false);

% Grid size N: use adaptive default from fastlpr_create.m
% N = ceil(n^(1/dx)) where n = sample size, dx = dimensions

%% Create dummy response (all ones) for KDE
% KDE: f(x) = (1/n) * sum_i K_h(x - x_i)
% This is equivalent to order=0 regression with y=1, then divide by n
y_dummy = ones(n, 1);

%% Initialization
% Create model structure with data, bandwidth, and options
[regs] = fastlpr_create(x, y_dummy, h, opt);

%% Compute design matrix
% Calculate kernel-weighted design matrix S using NUFFT
% For order=0, this computes sum_i K_h(x - x_i) at each grid point
[regs] = fastlpr_s(regs);

%% Compute density estimates
% KDE formula: f(x) = (1/n) * sum_i K_h(x - x_i)
% regs.s contains sum_i K_h(x - x_i) for each bandwidth
%
% However, the NUFFT-based convolution loses some mass due to the finite
% spreading width (Msp). To correct for this, we normalize so that the
% integral of the density equals 1.
%
% Theoretical: integral[f(x)dx] = sum_x f(x)*dx = 1
% So we normalize: fhat = regs.s / (sum(regs.s)*dx)
%
% This data-driven normalization corrects for NUFFT approximation error
% without needing to increase Msp (which would increase computation).

% Compute grid spacing (product of spacing in each dimension)
dx_grid = prod(cellfun(@(x) x(2)-x(1), regs.xlist));

% Create colon indexing for spatial dimensions
ndcolon = cell(1, dx);
ndcolon(:) = {':'};

% Normalize each bandwidth separately
% Note: Some bandwidths may have been removed (ihbad), so we only normalize good ones
fhat_all = zeros(size(regs.s));
for ih = 1:regs.dh
    % Skip bad bandwidths (they will remain zero)
    if regs.ihbad(ih)
        continue;
    end

    % Extract current bandwidth slice using ndcolon for arbitrary dimensions
    % This works for 1D: s_current = regs.s(:, ih)
    % For 2D: s_current = regs.s(:, :, ih)
    % For 3D: s_current = regs.s(:, :, :, ih), etc.
    s_current = regs.s(ndcolon{:}, ih);

    % Take real part to remove numerical noise from FFT
    % (imaginary part should be negligible for real-valued data)
    s_current = real(s_current);

    % Normalize so integral equals 1
    sum_s_dx = sum(s_current(:)) * dx_grid;

    % Assign normalized values back
    fhat_all(ndcolon{:}, ih) = s_current / sum_s_dx;
end

% Create griddedInterpolant for each bandwidth
[regs.fpp_yhat] = fastlpr_gridinterp(regs, fhat_all, regs.xlist, ...
                                      regs.opt.y_grid_method, regs.opt.y_grid_opt);
regs.yhat = fhat_all;

%% Bandwidth selection via LCV
% If multiple bandwidths provided, select optimal one using LCV
if regs.dh > 1
    [lcv_result] = fastkde_lcv(regs, x);
else
    lcv_result = [];
end

%% Save fields before compacting
% fastlpr_compact removes some fields, so save them first
xraw_save = regs.xraw;
opt_save = regs.opt;
h_save = regs.h;
dh_save = regs.dh;
dx_save = regs.dx;

%% Compact results
% Remove intermediate variables to save memory
[regs] = fastlpr_compact(regs);

%% Prepare output structure
kde = struct();

if ~isempty(lcv_result)
    % Multiple bandwidths: use selected bandwidth
    kde.lcv = lcv_result;
    kde.h = lcv_result.h1se;  % Selected bandwidth (1-SE rule)

    % Extract only the selected bandwidth's density from fpp
    % regs.fpp_yhat contains all bandwidths, we need to extract the 1-SE one
    % Similar to fastlpr_gcv.m line 142
    id1se = lcv_result.id1se;

    % Create a new griddedInterpolant with only the selected bandwidth
    kde.fpp = regs.fpp_yhat;
    if dh_save > 1
        % Extract only the selected bandwidth using ndcolon for arbitrary dimensions
        % For 1D: Values is (N x dh), extract column id1se
        % For 2D: Values is (N1 x N2 x dh), extract slice id1se
        % For 3D: Values is (N1 x N2 x N3 x dh), extract slice id1se, etc.
        ndcolon_save = cell(1, dx_save);
        ndcolon_save(:) = {':'};
        kde.fpp.Values = regs.fpp_yhat.Values(ndcolon_save{:}, id1se);
    end

    % Also extract the selected bandwidth's yhat
    if dh_save > 1
        ndcolon_save = cell(1, dx_save);
        ndcolon_save(:) = {':'};
        kde.fhat = regs.yhat(ndcolon_save{:}, id1se);
    else
        kde.fhat = regs.yhat;
    end
else
    % Single bandwidth
    kde.h = h_save;
    kde.fpp = regs.fpp_yhat;
    kde.fhat = regs.yhat;
end

% Get xlist from fpp.GridVectors (this is the correct grid)
kde.xlist = kde.fpp.GridVectors;

kde.opt = opt_save;
kde.xraw = xraw_save;

end


function [lcv_result] = fastkde_lcv(regs, x)
% FASTKDE_LCV - Compute Likelihood Cross-Validation for bandwidth selection
%
% Computes LCV scores for all bandwidth candidates and selects optimal
% bandwidth using maximum LCV and 1-SE rule.
%
% Syntax:
%   lcv_result = fastkde_lcv(regs, x)
%
% Inputs:
%   regs - Regression structure containing:
%          .dh        - Number of bandwidths
%          .h         - Bandwidth matrix (dh x dx)
%          .yhat      - Density estimates for all bandwidths
%          .ihbad     - Logical array of bad bandwidths
%          .opt       - Options structure
%   x    - Original data points (n x dx matrix)
%
% Output:
%   lcv_result - Structure containing:
%                .fhat_1se  - Density estimate with 1-SE bandwidth
%                .h1se      - 1-SE bandwidth
%                .hmin      - Maximum LCV bandwidth
%                .lcv_m     - Mean LCV scores (dh x 1 vector)
%                .lcv_sd    - Std of LCV scores (dh x 1 vector)
%
% Algorithm:
%   1. Compute leave-one-out density at each data point for each bandwidth
%   2. Compute LCV = sum_i log(f_{-i}(x_i))
%   3. Find bandwidth with maximum LCV
%   4. Apply 1-SE rule: select largest bandwidth within 1 SE of maximum
%
% LCV Formula:
%   LCV(h) = sum_i log(f_{-i}(x_i))
%   where f_{-i} is leave-one-out density estimate
%
% 1-SE Rule:
%   - Find maximum LCV: lcv_max
%   - Compute threshold: lcv_max - 1 * SE(lcv_max)
%   - Select largest bandwidth with LCV >= threshold

[n, dx] = size(x);
dh = regs.dh;

% Note: Bad bandwidths have already been removed by fastlpr_create
% So all bandwidths in regs are valid, and we don't need to check ihbad

%% Compute LCV scores for all bandwidths
lcv_scores = zeros(dh, 1);
lcv_std = zeros(dh, 1);

% Create colon indexing for spatial dimensions
ndcolon = cell(1, dx);
ndcolon(:) = {':'};

for ih = 1:dh

    % Get current bandwidth (always a row vector)
    h_current = regs.h(ih, :);

    % Compute leave-one-out density efficiently using the relationship:
    % f_{-i}(x_i) = (n/(n-1)) * [f(x_i) - (1/n) * K_h(0)]
    %
    % This avoids the O(n^2) loop and uses the already-computed density f(x_i)
    % from the NUFFT-based KDE, achieving O(n + m log m) complexity.
    %
    % Derivation:
    % f(x_i) = (1/n) * sum_j K_h(x_i - x_j)
    % f_{-i}(x_i) = (1/(n-1)) * sum_{j!=i} K_h(x_i - x_j)
    %             = (1/(n-1)) * [sum_j K_h(x_i - x_j) - K_h(0)]
    %             = (1/(n-1)) * [n*f(x_i) - K_h(0)]
    %             = (n/(n-1)) * [f(x_i) - (1/n)*K_h(0)]

    % Get density at data points for this specific bandwidth
    % regs.yhat is (M x dh) for 1D, (M1 x M2 x dh) for 2D, (M1 x M2 x M3 x dh) for 3D, etc.
    % Extract current bandwidth slice using ndcolon for arbitrary dimensions
    fhat_current = regs.yhat(ndcolon{:}, ih);

    % Create temporary griddedInterpolant for this bandwidth
    % regs.xlist is a cell array of grid vectors for each dimension
    fpp_temp = griddedInterpolant(regs.xlist, fhat_current, 'linear', 'linear');

    % Evaluate at data points
    % Convert x columns to cell array for arbitrary dimensions
    x_cell = num2cell(x, 1);  % {x(:,1), x(:,2), ...}
    f_at_data = fpp_temp(x_cell{:});

    % Compute K_h(0) - the kernel value at zero distance
    if strcmp(regs.opt.kernel_type, 'gaussian')
        K0 = 1 / sqrt(2*pi);  % Gaussian kernel at 0
    else  % epanechnikov
        K0 = 0.75;  % Epanechnikov kernel at 0
    end

    % Normalize by bandwidth (product for multi-dimensional)
    K0_normalized = K0 / prod(h_current);

    % Compute leave-one-out density using the efficient formula
    f_loo = (n / (n-1)) * (f_at_data - K0_normalized / n);

    % Avoid log(0)
    f_loo = max(f_loo, 1e-10);

    % LCV score (higher is better)
    log_f_loo = log(f_loo);
    lcv_scores(ih) = sum(log_f_loo);
    lcv_std(ih) = std(log_f_loo) * sqrt(n);  % Standard error
end

%% Find maximum LCV bandwidth
[~, idmax] = max(lcv_scores);

%% Apply 1-SE rule to select smoothest bandwidth within 1 SE of maximum
% The 1-SE rule prefers larger (smoother) bandwidths when LCV is similar
% This reduces overfitting and improves stability
semax = lcv_scores(idmax) - 1 * lcv_std(idmax);

% Find all bandwidths with LCV >= threshold
candidates = find(lcv_scores >= semax);

% Select bandwidth using 1-SE rule
if isempty(candidates)
    % No candidates found, use maximum LCV bandwidth
    id1se = idmax;
else
    % Among candidates, select the one with largest bandwidth (sum of h)
    % This prefers smoother fits when LCV values are similar
    hsum = sum(regs.h, 2);
    hsum_candidates = hsum(candidates);
    [~, max_idx] = max(hsum_candidates);
    id1se = candidates(max_idx);
end

%% Package results
lcv_result = struct();
lcv_result.lcv_m = lcv_scores;
lcv_result.lcv_sd = lcv_std;
lcv_result.idmax = idmax;
lcv_result.id1se = id1se;

% Selected bandwidths (always row vectors)
lcv_result.h1se = regs.h(id1se, :);
lcv_result.hmax = regs.h(idmax, :);
lcv_result.hlist = regs.h;  % All bandwidths for plotting (dh x dx)

% Density estimate with 1-SE bandwidth
% regs.yhat is (M x dh) for 1D, (M1 x M2 x dh) for 2D, etc.
% Extract using ndcolon for arbitrary dimensions
ndcolon = cell(1, dx);
ndcolon(:) = {':'};
lcv_result.fhat_1se = regs.yhat(ndcolon{:}, id1se);

end