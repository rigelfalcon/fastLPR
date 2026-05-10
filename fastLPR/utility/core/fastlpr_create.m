function [regs]=fastlpr_create(x,y,h,opt)
% FASTLPR_CREATE - Initialize regression structure for fast local polynomial regression
%
% Creates and initializes the regression structure containing all parameters,
% grids, and precomputed values needed for fast LPR with NUFFT acceleration.
%
% Syntax:
%   regs = fastlpr_create(x, y)
%   regs = fastlpr_create(x, y, h)
%   regs = fastlpr_create(x, y, h, opt)
%
% Inputs:
%   x   - Predictor data (Tx x dx matrix, can be complex-valued)
%   y   - Response data (Ty x dy matrix or Ty x dh x dy array)
%   h   - Bandwidth candidates (dh x dx matrix, optional)
%   opt - Options structure (optional, see below)
%
% Output:
%   regs - Regression structure containing:
%          Data: .x, .y, .xraw, .Tx, .Ty, .dx, .dy
%          Grid: .N, .L, .xlist, .knot, .qin, .qout
%          Kernel: .h, .kdf, .dh, .lwp
%          Options: .opt
%
% Options (opt structure):
%   .order         - Polynomial order: 0, 1, or 2 (default: 0)
%   .kernel_type   - Kernel: 'gaussian' or 'epanechnikov' (default: 'gaussian')
%   .y_type_out    - Output type: 'mean' or 'variance' (default: 'mean')
%   .N             - Grid size (default: auto based on sample size)
%   .accuracy      - NUFFT accuracy in digits (default: auto)
%   .dstd          - Number of standard errors for 1-SE bandwidth selection rule (default: 0)
%                    The 1-SE rule selects the largest bandwidth h such that:
%                    GCV(h) <= GCV_min + dstd * SE(GCV_min)
%                    dstd=0: minimum GCV; dstd=1: standard 1-SE rule
%                    Set to 5-20 for variance estimation to prefer smoother fits
%
% Algorithm:
%   1. Process complex-valued data (split into real/imaginary)
%   2. Set default options
%   3. Standardize data (z-score)
%   4. Create evaluation grid
%   5. Compute kernel density function in Fourier domain
%   6. Store all parameters in regs structure
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

%% Input validation
if nargin<4||isempty(opt)
    opt=[];
end
if nargin<3||isempty(h)
    h=[];
end

%% Process complex-valued predictors
% Complex predictors are split into real and imaginary parts
% Example: x = [1+2i, 3+4i] becomes x = [1, 3, 2, 4]
x_iscomplex=iscomplex(x,1);  % Check which columns are complex
x=[real(x),imag(x(:,x_iscomplex))];  % Split complex into [real, imag]

%% Get dimensions and validate
[Tx,dx]=size(x);  % Tx = number of samples, dx = number of predictors
if ismatrix(y)
    % Standard case: y is Ty x dy matrix
    [Ty,dy]=size(y);
    opt.y_corr_bandwidth=false;
elseif ndims(y)==3
    % Special case: different bandwidth for each response
    % y is Ty x dh x dy array
    [Ty,dhy,dy]=size(y);
    opt.y_corr_bandwidth=true;
end
assert(Tx==Ty,'fastLPR:create:DimensionMismatch', ...
    'Number of samples in x (%d) must equal number of samples in y (%d)', Tx, Ty);


%% Set default options

% Output type: 'mean' for E[Y|X] or 'variance' for Var[Y|X]
opt=set_defaults(opt,'y_type_out','mean');

% Kernel type: 'gaussian' (recommended) or 'epanechnikov'
opt=set_defaults(opt,'kernel_type','gaussian');

% Polynomial order: 0 (Nadaraya-Watson), 1 (local linear), 2 (local quadratic)
opt=set_defaults(opt,'order',0);

% Grid size N: number of evaluation points per dimension
% Rule of thumb: N ~= Tx^(1/dx) gives good balance of accuracy and speed
% Smaller N is faster but less accurate; larger N is more accurate but slower
opt=set_defaults(opt,'Nratio',1);  % Multiplier for default grid size
opt=set_defaults(opt,'N',ceil(opt.Nratio*(Tx^(1/dx))));
N=opt.N;

% Ensure N is a vector (broadcast scalar to all dimensions)
if isscalar(N)
    N = N * ones(1, dx);
end
opt.N = N;  % Update opt.N to be a vector
opt.Nratio=N(1)/(Tx^(1/dx));

% DEBUG: Print grid size
if isfield(opt, 'verbose') && opt.verbose
    fprintf('[DEBUG fastlpr_create] N = %d (grid size)\n', N);
end

% FFT padding: use power-of-2 length for speed
opt=set_defaults(opt,'flag_power2',true);

% NUFFT accuracy: number of accurate digits
% Auto-adjust based on sample size: fewer samples need higher accuracy
opt=set_defaults(opt,'accuracy',max(6-ceil(log10(Tx)),4));

% DEBUG: Print accuracy parameter
if isfield(opt, 'verbose') && opt.verbose
    fprintf('[DEBUG fastlpr_create] accuracy = %d (Tx=%d, N will be set next)\n', opt.accuracy, Tx);
end

% NUFFT deconvolution: apply correction for spreading kernel
% Only needed for very high accuracy (>6 digits)
opt=set_defaults(opt,'nufft_deconv',opt.accuracy>6);

% Bandwidth selection and degrees of freedom estimation
opt=set_defaults(opt,'dstd',0);  % Number of std for 1-SE rule (0 = minimum GCV, no 1-SE rule)
opt=set_defaults(opt,'calc_dof',true);  % Whether to calculate DOF
opt=set_defaults(opt,'num_dof_sample',10);  % Number of random samples for DOF estimation
opt=set_defaults(opt,'dof_seed',42);  % Random seed for DOF estimation (for reproducibility)


if ~isfield(opt,'y_grid_opt')
    % 'linear','cubic' or 'spline' et al. see griddedInterpolant
    opt.y_grid_opt.Method='linear';
    opt.y_grid_opt.ExtrapolationMethod='linear';
end

opt=set_defaults(opt,'compact',true);

% opt=set_defaults(opt,'x_iscomplex',iscomplex(x,1));
% opt=set_defaults(opt,'select_h',[]);
opt=set_defaults(opt,'y_corr_bandwidth',false);
opt=set_defaults(opt,'y_gcv_metric',[]);
opt=set_defaults(opt,'y_grid_method','griddedInterpolant');
opt=set_defaults(opt,'isgpu',false);


% opt=set_defaults(opt,'y_keep_all',false);
% opt=set_defaults(opt,'inherit_bandwidth',[]);



% opt=set_defaults(opt,'nufft_loop',true);



%% Initialize data types and precision
dtype=class(y);  % Store original data type

% Convert to appropriate precision based on accuracy requirement
% Higher accuracy needs higher precision (double), lower accuracy can use single
[x,y]=bit_convert(opt.accuracy,x,y);

% Handle GPU arrays if needed
if strcmp(dtype,'gpuArray') %#ok<STISA>
    x=gpuArray(x);
    y=gpuArray(y);
end
dtype=class(y);  % Update dtype after conversion

% Create colon indexing for multi-dimensional arrays
ndcolon(1:dx) = {':'};  % Example: for dx=2, ndcolon = {':', ':'}

%% Compute padded length L
if opt.flag_power2
    % Power-of-2 padding improves FFT speed and accuracy
    % Use L = 2^nextpow2(2*N-1) to avoid circular convolution
    % For multidimensional case, apply element-wise
    L = 2.^nextpow2(2*N-1);
else
    % Minimal padding (for testing or special cases)
    L=N+2;
end

% Ensure L is a vector (should already be from above, but double-check)
if isscalar(L)
    L = L * ones(1, dx);
end

%% Standardize data (z-score normalization)
% This ensures bandwidth h has consistent meaning across different scales
% After z-scoring: mean(x) ~= 0, std(x) = 1

xraw=x;  % Save original data
[x,~,x_std]=zscore(x);  % Standardize: x_new = (x - mean(x)) / std(x)

% Get data range after standardization
x_min=min(x);
x_max=max(x);
x_scale=x_max-x_min;  % Range of standardized data
x_delta=x_scale./N;   % Grid spacing

% Center data at origin (for symmetric padding)
x=x-(x_max+x_min)/2;
x_min=min(x);
x_max=max(x);
x_mean=mean(x);

%% Compute padding indices
% qin stores the start and end indices of the evaluation grid within the padded array
% This ensures the evaluation grid is centered in the padded array
qin=zeros(2,dx);

% Use symmetric padding for all N (especially odd N)
% Old method: round(L/2 - N/2) causes asymmetry for odd N
% New method: floor((L-N)/2) ensures symmetric padding
% Example: N=101, L=256 -> pad_left=77, pad_right=78 (symmetric)
pad_left = floor((L - N) / 2);
qin(1,:) = pad_left + 1;      % Start index of evaluation grid
qin(2,:) = pad_left + N;      % End index of evaluation grid


%% Compute kernel density function in Fourier domain
% This is the key step that enables O(N + M log M) complexity
% The kernel is precomputed in Fourier domain for fast convolution
[kdf,ihbad,dh,dh_input,h,lwp]=fastlpr_kdf(x,h,N,opt);

% Validate bandwidth dimensions for multi-response case
if opt.y_corr_bandwidth && dh~=dhy && dhy~=1
    error('fastLPR:create:BandwidthMismatch', ...
        'Number of bandwidths (%d) must match number of responses (%d)', dh, dhy);
end

%% Compute NUFFT knot positions
% Knots are the positions of data points in the frequency domain
% They must be in the range [-0.5, 0.5) for NUFFT

% Create index range for frequency domain (per dimension)
if dx == 1
    % 1D case: simple vectors
    idx_knot = (qin(1):qin(2))';  % Indices of evaluation grid
    idx_range = linspace(-1/2, 1/2-1/L, L)';  % Frequency range [-0.5, 0.5)
    hf = 1/L;

    % Map data points to frequency domain
    if N ~= L
        % With padding: map to evaluation grid range
        knot_scale = idx_range(idx_knot(end)) - idx_range(idx_knot(1));
        knot = (x - x_min) ./ x_scale .* knot_scale + idx_range(idx_knot(1));
    else
        % No padding: map to full range
        knot_scale = idx_range(end) - idx_range(1);
        knot = (x - x_min) ./ x_scale .* knot_scale - 0.5;
    end
else
    % Multidimensional case: cell arrays for each dimension
    idx_knot = cell(1, dx);
    idx_range = cell(1, dx);
    hf = zeros(1, dx);
    knot = zeros(size(x));
    knot_scale = zeros(1, dx);  % Initialize knot_scale for multi-D

    for d = 1:dx
        idx_knot{d} = (qin(1,d):qin(2,d))';
        idx_range{d} = linspace(-1/2, 1/2-1/L(d), L(d))';
        hf(d) = 1/L(d);

        % Map data points to frequency domain for this dimension
        if N(d) ~= L(d)
            % With padding: map to evaluation grid range
            knot_scale(d) = idx_range{d}(idx_knot{d}(end)) - idx_range{d}(idx_knot{d}(1));
            knot(:,d) = (x(:,d) - x_min(d)) ./ x_scale(d) .* knot_scale(d) + idx_range{d}(idx_knot{d}(1));
        else
            % No padding: map to full range
            knot_scale(d) = idx_range{d}(end) - idx_range{d}(1);
            knot(:,d) = (x(:,d) - x_min(d)) ./ x_scale(d) .* knot_scale(d) - 0.5;
        end
    end
end

% Compute NUFFT spreading width based on accuracy requirement
% CRITICAL FIX: Pass L (padded grid size), not Ty (sample size)!
% The NUFFT grid parameters should be based on the FFT grid size, not the number of data points
[Msp, Mr_check, tau_check, ratio_check] = compute_grid_params(L, opt.accuracy);

% DEBUG: Print grid parameters
if isfield(opt, 'verbose') && opt.verbose
    fprintf('[DEBUG fastlpr_create] Grid params: L=%d, Msp=%d, Mr=%d, tau=%.2e, ratio=%d\n', ...
        L, Msp, Mr_check, tau_check, ratio_check);
end

%% Compute output extraction indices
% qout stores the indices to extract the evaluation grid from the padded result
if dx == 1
    qout = (L - N) / 2;
    qout = [ceil(qout) + 1, ceil(qout) + N];
else
    qout = (L - N) / 2;  % Element-wise for vectors
    qout = [ceil(qout)' + 1, ceil(qout)' + N'];  % dx*2 matrix
end

%% Create evaluation grid in original (raw) space
% This grid is used for output and for parametric interpolation methods
xlist = multispace(min(xraw), max(xraw), N);  % Create N-point grid for each dimension

% multispace returns:
% - For 1D (dx=1): (1 x N) matrix
% - For multi-D with same N: (dx x N) matrix
% - For multi-D with different N: {dx x 1} cell array

if dx == 1
    % 1D case: convert to cell
    xlist = {xlist'};  % Transpose to column vector and wrap in cell
elseif ~iscell(xlist)
    % Multi-D with same N: convert matrix (dx x N) to cell array {dx x 1}
    % Each cell contains one row (1 x N) transposed to (N x 1)
    xlist_cell = cell(dx, 1);
    for d = 1:dx
        xlist_cell{d} = xlist(d, :)';  % Extract row d and transpose
    end
    xlist = xlist_cell;
else
    % Multi-D with different N: already cell array, just transpose each
    for d = 1:dx
        xlist{d} = xlist{d}';
    end
end


%% Store all parameters in regression structure
% The regs structure contains all information needed for fast LPR

% Data
regs.xraw=xraw;      % Original predictor data (before standardization)
regs.x=x;            % Standardized predictor data
regs.y=y;            % Response data
regs.Tx=Tx;          % Number of samples
regs.Ty=Ty;          % Number of response samples
regs.dx=dx;          % Number of predictor dimensions
regs.dy=dy;          % Number of response variables
regs.dtype=dtype;    % Data type (single, double, gpuArray)
regs.Msp=Msp;        % NUFFT spreading width

% Grid parameters
regs.L=L;            % Padded length (for FFT)
regs.N=N;            % Grid size (evaluation points per dimension)
regs.ndcolon=ndcolon;% Colon indexing for multi-dimensional arrays

% Standardization parameters
regs.x_min=x_min;    % Minimum of standardized data
regs.x_max=x_max;    % Maximum of standardized data
regs.x_scale=x_scale;% Range of standardized data
regs.x_mean=x_mean;  % Mean of standardized data
regs.x_std=x_std;    % Standard deviation (from zscore)
regs.x_delta=x_delta;% Grid spacing

% Kernel parameters
regs.h=h;            % Bandwidth (dh x dx matrix)
regs.kdf=kdf;        % Kernel density function in Fourier domain
regs.ihbad=ihbad;    % Logical array indicating bad bandwidths
regs.dh=dh;          % Number of valid bandwidths
regs.dh_input=dh_input;  % Original number of bandwidths

% Index parameters
regs.qin=qin;        % Padding indices (start and end of evaluation grid)
regs.qout=qout;      % Output extraction indices

% NUFFT parameters
regs.knot_scale=knot_scale;  % Scaling factor for knot positions
regs.knot=knot;      % NUFFT knot positions (data points in frequency domain)
regs.hf=hf;          % Frequency spacing

% Output grid
regs.xlist=xlist;    % Evaluation grid in original space (cell array)

% Options
regs.opt=opt;        % All options

% Response data properties
regs.y_isreal=~iscomplex(y,1);  % Logical array indicating real-valued responses

% Local polynomial parameters
regs.lwp=lwp;        % Local polynomial estimator functions (for order >= 1)

end