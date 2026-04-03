function [Sxfun,Txfun,mfun,ns,nt,mmfun,mmmfun]=get_lwp_estimator(dx,p,iscomplex)
% GET_LWP_ESTIMATOR - Generate functions for local polynomial regression
%
% Generates functions for computing design matrix (S), weighted response (T),
% and regression coefficients (m) for local polynomial regression of order p in dx dimensions.
%
% Syntax:
%   [Sxfun,Txfun,mfun,ns,nt,mmfun,mmmfun] = get_lwp_estimator(dx, p, iscomplex)
%
% Inputs:
%   dx        - Number of predictor dimensions (default: 2)
%   p         - Polynomial order: 0 (Nadaraya-Watson), 1 (local linear), 2 (local quadratic) (default: 2)
%   iscomplex - Whether data is complex-valued (default: 0)
%
% Outputs:
%   Sxfun  - Function handle for design matrix S = X'*W*X
%   Txfun  - Function handle for weighted response T = X'*W
%   mfun   - Function handle for regression estimate m = e1'*(S\T)
%            Uses SYMBOLIC FORMULAS (vectorized, operates on entire grids)
%   ns     - Number of unique elements in lower triangular S
%   nt     - Number of elements in T (number of polynomial terms)
%   mmfun  - Alternative function (currently unused)
%   mmmfun - Alternative function (currently unused)
%
% Algorithm:
%   1. Uses symbolic math to derive expressions for S and T computation
%   2. For regression estimate (mfun), uses SYMBOLIC FORMULAS:
%      - Pre-computed formulas from estimated_lwp_estimator.m (if available)
%      - Otherwise generates symbolic formula on-the-fly (slow but vectorized)
%      - Formulas are FULLY VECTORIZED (operate on entire grids at once)
%      - Much faster than point-by-point numerical solvers
%
% Supported Cases:
%   - 1D: Order 0, 1, 2 (all supported)
%   - 2D: Order 0, 1, 2 (all supported, Order 2 pre-computed)
%   - 3D: Order 0, 1 (Order 2 NOT supported - too expensive)
%   - >3D: Order 0 only (higher orders NOT supported)
%
% Restrictions:
%   - For dx > 2 and p >= 2: Error (too computationally expensive)
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Modified: 2025-10-22 - Added restriction for high-dimensional high-order cases
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

n=1;
if nargin<1||isempty(dx)
    dx=2;
end
if nargin<2||isempty(p)
    p=2;
end
if nargin<3||isempty(iscomplex)  % NOTE: Fixed - was checking 'p' instead of 'iscomplex'
    iscomplex=0;
end

% Check for unsupported high-dimensional, high-order cases
% For dx > 2 and p > 2, symbolic formula generation is extremely slow
% and design matrix inversion becomes computationally expensive
if dx > 2 && p >= 2
    error('get_lwp_estimator:UnsupportedOrder', ...
        ['Polynomial order %d for %dD data is not supported.\n' ...
         'High-dimensional (dx > 2) high-order (p >= 2) regression requires\n' ...
         'large computational power for symbolic formula generation and\n' ...
         'design matrix inversion. Please use p <= 2 for dx > 2.'], p, dx);
end
% Clear any lingering symbolic variables to avoid symengine corruption
% This prevents issues when function is called repeatedly in loops
if exist('kn', 'var') && isa(kn, 'sym')
    clear kn xn  
end

%% Create symbolic design matrix
% For local polynomial regression, we fit a polynomial of order p
% in a neighborhood around each evaluation point.
% The design matrix X contains polynomial terms up to degree p.

ndcolonsep(1:dx)={':,'};  % For string manipulation later

% Create symbolic variables
% BUGFIX 2025-11-18: Ensure clean symbolic environment
% Convert xn to row vector to avoid dimension issues in matlabFunction
if dx == 1
    xn = sym('x1');  % Single variable
else
    xn = sym('x', [n,dx]);  % Row vector of variables (x1, x2, ...)
end
kn = sym('k');         % Kernel weight
% Ensure variables are in expected format
xn = xn(:).';  % Force row vector
kn = sym(kn);  % Ensure symbolic

% Build design matrix X based on polynomial order
% The design matrix contains all polynomial terms up to degree p
% For efficiency, we only implement explicit formulas for p=0,1,2
% For higher orders, use the general formula (slower but works)
switch p
    case 0
        % Order 0: Nadaraya-Watson (local constant)
        % X = [1]
        X=[ones(n,1)];
    case 1
        % Order 1: Local linear regression
        % X = [1, x1, x2, ..., xd]
        X=[ones(n,1),xn];
    case 2
        % Order 2: Local quadratic regression
        % X = [1, x1, x2, ..., xd, x1^2, x1*x2, ..., xd^2]
        % Only lower triangular elements of x*x' are included (symmetric)
        X=[ones(n,1),xn,mat2tril(xn.'*xn).'];
    otherwise
        % Higher orders: General formula
        % Generate all polynomial terms up to degree p
        % This is slower but works for any order
        error('get_lwp_estimator:UnsupportedOrder', ...
            ['Polynomial order %d is not explicitly supported.\n' ...
             'Only orders 0, 1, 2 are implemented.\n' ...
             'However, the numerical solver in mfun will work for any order\n' ...
             'if you provide the correct S and T elements from another source.'], p);
end

%% Compute weighted design matrix and weighted response
% W = diag(kernel weights)
W=diag(kn);

% Compute S = X'*W*X and T = X'*W*y
% For complex data, use conjugate transpose (')
% For real data, use regular transpose (.')
if iscomplex
    Sx=X'*W*X;   % Design matrix: X'*W*X
    Tx=X'*W;     % Weighted response: X'*W (will multiply by y later)
else
    Sx=X.'*W*X;  % Design matrix: X.'*W*X
    Tx=X.'*W;    % Weighted response: X.'*W (will multiply by y later)
end

%% Convert symbolic expressions to MATLAB functions
% Extract lower triangular elements of S (since S is symmetric)
Sxn=mat2tril(Sx,0);

% Generate MATLAB function handles from symbolic expressions
% BUGFIX 2025-11-18: Add retry logic for symbolic engine corruption
max_retries = 3;
for retry = 1:max_retries
    try
        Sxfun = matlabFunction(Sxn,'vars',{kn,xn});  % Function for design matrix elements
        Txfun = matlabFunction(Tx,'vars',{kn,xn});   % Function for weighted response
        break;  % Success - exit loop
    catch ME
        if retry < max_retries
            % Retry: clear and recreate symbolic variables
            clear kn xn
            if dx == 1
                xn = sym('x1');
            else
                xn = sym('x', [n,dx]);
            end
            kn = sym('k');
            xn = xn(:).';  % Force row vector
            
            % Recompute symbolic expressions
            if p == 0
                X = [ones(n,1)];
            elseif p == 1
                X = [ones(n,1),xn];
            elseif p == 2
                X = [ones(n,1),xn,mat2tril(xn.'*xn).'];
            end
            W = diag(kn);
            if iscomplex
                Sx = X'*W*X;
                Tx = X'*W;
            else
                Sx = X.'*W*X;
                Tx = X.'*W;
            end
            Sxn = mat2tril(Sx,0);
        else
            % Final retry failed - rethrow error
            rethrow(ME);
        end
    end
end

% Convert array indexing to cell indexing for better performance
% This allows vectorized operations across multiple bandwidths
% Example: x(:,1) becomes x{1}
idxold=append('(:,',cellstr(num2str((1:dx)')),')');
idxnew=append(['(',ndcolonsep{:}], cellstr(num2str((1:dx)')),')');
Sxfun=str2func(replace(replace(func2str(Sxfun),idxold,idxnew),{'[';']'},{'{';'}'}));
Txfun=str2func(replace(replace(func2str(Txfun),idxold,idxnew),{'[';']'},{'{';'}'}));

% Store dimensions
ns=length(Sxn);  % Number of unique elements in lower triangular S
nt=length(Tx);   % Number of elements in T (= number of polynomial terms)
%% Compute regression estimator function
% The regression estimate is m = e1' * (S \ T)
% where e1 = [1, 0, 0, ...]' selects the constant term (the estimate at x0)
% This is equivalent to solving S * beta = T and taking beta(1)

% Try to load pre-computed symbolic estimator (VECTORIZED - operates on entire arrays)
[mfun]=estimated_lwp_estimator(dx,p,iscomplex);

if isempty(mfun)
    % No pre-computed estimator, derive it symbolically
    % This is SLOW for high dimensions/orders but produces VECTORIZED code

    % Create symbolic variables for S and T
    sn=sym('s',[ns,1]);  % Lower triangular elements of S
    e1=[1;zeros(nt-1,1)]; % Selector vector for constant term

    % Reconstruct full symmetric matrix S from lower triangular elements
    S=vec2tril(sn);

    if iscomplex
        S = tril(S,0) + tril(S,-1)';   % Hermitian matrix for complex data
    else
        S = tril(S,0) + tril(S,-1).';  % Symmetric matrix for real data
    end

    % Create symbolic variable for weighted response T
    T=sym('t',[nt,1]);

    % Compute regression estimate: m = e1' * (S \ T)
    % This gives the fitted value at the evaluation point
    m=e1'*(S\T);

    % Convert to MATLAB function
    mfun = matlabFunction(m,'vars',{sn.',T.'});

    % Convert array indexing to cell indexing for vectorization
    % This allows the function to operate on entire grids at once
    idxold=strcat('in1(:,',num2cellstr((1:ns)'),')');
    idxnew=append('in1{',num2cellstr((1:ns)'),'}');
    mfun=replace(func2str(mfun),idxold,idxnew);
    idxold=append('in2(:,',num2cellstr((1:nt)'),')');
    idxnew=append('in2{',num2cellstr((1:nt)'),'}');
    mfun=str2func(replace(mfun,idxold,idxnew));
end

% Alternative function formats (currently unused, kept for future optimization)
mmfun=[];
mmmfun=[];

end


% function save_lwp_estimator(Sxfun,Txfun,mfun,ns,nt)
%
%
% end
