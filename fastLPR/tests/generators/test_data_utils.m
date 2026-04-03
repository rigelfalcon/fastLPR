function utils = test_data_utils()
%TEST_DATA_UTILS Reusable test data generation utilities for fastLPR tests.
%
%   Usage:
%       utils = test_data_utils();
%       data = utils.generate_sin_1d(500, 0.2, 42);
%       hlist = utils.generate_hlist([0.01, 0.5], 20, 1);
%
%   Available generators:
%       utils.generate_sin_1d(n, noise, seed)     - 1D sine regression data
%       utils.generate_sincos_2d(n, noise, seed)  - 2D sin*cos regression data
%       utils.generate_kde(n, dim, seed)          - Standard normal KDE data
%       utils.generate_hlist(range, num, dim)     - Bandwidth list for any dim
%       utils.generate_complex_1d(n, noise, seed) - Complex-valued 1D data
%       utils.generate_hetero_1d(n, seed)         - 1D heteroscedastic data
%
%   Author: fastLPR Development Team
%   Copyright (c) 2020-2025 fastLPR Development Team
%   License: GNU General Public License v3.0

    utils = struct();
    utils.generate_sin_1d = @generate_sin_1d;
    utils.generate_sincos_2d = @generate_sincos_2d;
    utils.generate_sincos_3d = @generate_sincos_3d;
    utils.generate_kde = @generate_kde;
    utils.generate_hlist = @generate_hlist;
    utils.generate_complex_1d = @generate_complex_1d;
    utils.generate_hetero_1d = @generate_hetero_1d;
end

function data = generate_sin_1d(n, noise_level, seed)
%GENERATE_SIN_1D Generate 1D sine regression test data.
%
%   data = generate_sin_1d(n, noise_level, seed)
%
%   Inputs:
%       n           - Number of samples (default: 500)
%       noise_level - Standard deviation of noise (default: 0.2)
%       seed        - Random seed for reproducibility (default: 42)
%
%   Outputs:
%       data.x       - Sorted uniform samples in [0, 1]
%       data.y       - Noisy observations
%       data.y_true  - True function values sin(2*pi*x)
%       data.n       - Sample size
%       data.noise   - Noise level used

    if nargin < 1 || isempty(n), n = 500; end
    if nargin < 2 || isempty(noise_level), noise_level = 0.2; end
    if nargin < 3 || isempty(seed), seed = 42; end

    rng(seed);
    x = sort(rand(n, 1));
    y_true = sin(2*pi*x);
    y = y_true + noise_level * randn(n, 1);

    data = struct();
    data.x = x;
    data.y = y;
    data.y_true = y_true;
    data.n = n;
    data.noise = noise_level;
end

function data = generate_sincos_2d(n, noise_level, seed)
%GENERATE_SINCOS_2D Generate 2D sin*cos regression test data.
%
%   data = generate_sincos_2d(n, noise_level, seed)
%
%   Inputs:
%       n           - Number of samples (default: 500)
%       noise_level - Standard deviation of noise (default: 0.2)
%       seed        - Random seed for reproducibility (default: 42)
%
%   Outputs:
%       data.x       - Uniform samples in [0, 1]^2
%       data.y       - Noisy observations
%       data.y_true  - True function values sin(2*pi*x1)*cos(2*pi*x2)
%       data.n       - Sample size
%       data.noise   - Noise level used

    if nargin < 1 || isempty(n), n = 500; end
    if nargin < 2 || isempty(noise_level), noise_level = 0.2; end
    if nargin < 3 || isempty(seed), seed = 42; end

    rng(seed);
    x = rand(n, 2);
    y_true = sin(2*pi*x(:,1)) .* cos(2*pi*x(:,2));
    y = y_true + noise_level * randn(n, 1);

    data = struct();
    data.x = x;
    data.y = y;
    data.y_true = y_true;
    data.n = n;
    data.noise = noise_level;
end

function data = generate_sincos_3d(n, noise_level, seed)
%GENERATE_SINCOS_3D Generate 3D sin*cos*sin regression test data.
%
%   data = generate_sincos_3d(n, noise_level, seed)
%
%   Inputs:
%       n           - Number of samples (default: 500)
%       noise_level - Standard deviation of noise (default: 0.2)
%       seed        - Random seed for reproducibility (default: 42)
%
%   Outputs:
%       data.x       - Uniform samples in [-1, 1]^3
%       data.y       - Noisy observations
%       data.y_true  - True function values sin(pi*x1)*cos(pi*x2)*sin(pi*x3)
%       data.n       - Sample size
%       data.noise   - Noise level used

    if nargin < 1 || isempty(n), n = 500; end
    if nargin < 2 || isempty(noise_level), noise_level = 0.2; end
    if nargin < 3 || isempty(seed), seed = 42; end

    rng(seed);
    x = 2*rand(n, 3) - 1;  % [-1, 1]^3
    y_true = sin(pi*x(:,1)) .* cos(pi*x(:,2)) .* sin(pi*x(:,3));
    y = y_true + noise_level * randn(n, 1);

    data = struct();
    data.x = x;
    data.y = y;
    data.y_true = y_true;
    data.n = n;
    data.noise = noise_level;
end

function data = generate_kde(n, dim, seed)
%GENERATE_KDE Generate standard normal KDE test data.
%
%   data = generate_kde(n, dim, seed)
%
%   Inputs:
%       n    - Number of samples (default: 500)
%       dim  - Dimensionality (default: 1)
%       seed - Random seed for reproducibility (default: 42)
%
%   Outputs:
%       data.x   - Standard normal samples (n x dim)
%       data.n   - Sample size
%       data.dim - Dimensionality

    if nargin < 1 || isempty(n), n = 500; end
    if nargin < 2 || isempty(dim), dim = 1; end
    if nargin < 3 || isempty(seed), seed = 42; end

    rng(seed);
    x = randn(n, dim);

    data = struct();
    data.x = x;
    data.n = n;
    data.dim = dim;
end

function hlist = generate_hlist(range, num, dim)
%GENERATE_HLIST Generate bandwidth list for cross-validation.
%
%   hlist = generate_hlist(range, num, dim)
%
%   Inputs:
%       range - [min, max] bandwidth range (default: [0.01, 0.5])
%       num   - Number of bandwidths (default: 10)
%       dim   - Dimensionality (default: 1)
%               For dim > 1, same range is used for all dimensions.
%               Pass as [d1_min, d1_max; d2_min, d2_max] for different ranges.
%
%   Outputs:
%       hlist - Bandwidth list (num x dim matrix)
%
%   Examples:
%       hlist = generate_hlist([0.01, 0.5], 20, 1)  % 1D: 20x1
%       hlist = generate_hlist([0.05, 0.5], 10, 2)  % 2D: 10x2 (same range)
%       hlist = generate_hlist([0.1, 1.0], 5, 3)    % 3D: 5x3 (same range)

    if nargin < 1 || isempty(range), range = [0.01, 0.5]; end
    if nargin < 2 || isempty(num), num = 10; end
    if nargin < 3 || isempty(dim), dim = 1; end

    % Handle multi-dimensional ranges
    if size(range, 1) == 1
        % Same range for all dimensions
        ranges = repmat(range, dim, 1);
    else
        ranges = range;
    end

    if dim == 1
        hlist = logspace(log10(ranges(1,1)), log10(ranges(1,2)), num)';
    else
        hlist = zeros(num, dim);
        for d = 1:dim
            h_vals = logspace(log10(ranges(d,1)), log10(ranges(d,2)), num);
            hlist(:, d) = h_vals';
        end
    end
end

function data = generate_complex_1d(n, noise_level, seed)
%GENERATE_COMPLEX_1D Generate complex-valued 1D regression test data.
%
%   data = generate_complex_1d(n, noise_level, seed)
%
%   Inputs:
%       n           - Number of samples (default: 500)
%       noise_level - Standard deviation of noise (default: 0.2)
%       seed        - Random seed for reproducibility (default: 42)
%
%   Outputs:
%       data.x       - Sorted uniform samples in [0, 1]
%       data.y       - Complex noisy observations
%       data.y_true  - True complex function (sin + i*cos)
%       data.n       - Sample size
%       data.noise   - Noise level used

    if nargin < 1 || isempty(n), n = 500; end
    if nargin < 2 || isempty(noise_level), noise_level = 0.2; end
    if nargin < 3 || isempty(seed), seed = 42; end

    rng(seed);
    x = sort(rand(n, 1));
    y_real = sin(2*pi*x);
    y_imag = cos(2*pi*x);
    y_true = complex(y_real, y_imag);
    y = complex(y_real + noise_level*randn(n, 1), ...
                y_imag + noise_level*randn(n, 1));

    data = struct();
    data.x = x;
    data.y = y;
    data.y_true = y_true;
    data.n = n;
    data.noise = noise_level;
end

function data = generate_hetero_1d(n, seed)
%GENERATE_HETERO_1D Generate 1D heteroscedastic regression test data.
%
%   data = generate_hetero_1d(n, seed)
%
%   Inputs:
%       n    - Number of samples (default: 500)
%       seed - Random seed for reproducibility (default: 42)
%
%   Outputs:
%       data.x         - Sorted uniform samples in [0, 1]
%       data.y         - Noisy observations
%       data.y_true    - True mean function sin(2*pi*x)
%       data.var_true  - True variance function (0.1 + 0.3*x^2)
%       data.n         - Sample size

    if nargin < 1 || isempty(n), n = 500; end
    if nargin < 2 || isempty(seed), seed = 42; end

    rng(seed);
    x = sort(rand(n, 1));
    y_true = sin(2*pi*x);
    var_true = 0.1 + 0.3*x.^2;
    y = y_true + sqrt(var_true).*randn(n, 1);

    data = struct();
    data.x = x;
    data.y = y;
    data.y_true = y_true;
    data.var_true = var_true;
    data.n = n;
end
