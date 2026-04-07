function [yhat] = fastlpr_predict(regs, x)
% FASTLPR_PREDICT - Predict response values at new locations
%
% Predicts response values at new predictor locations using the fitted
% regression model from cv_fastlpr. Uses fast spline interpolation for
% efficient prediction at arbitrary points.
%
% Syntax:
%   yhat = fastlpr_predict(regs, x)
%
% Inputs:
%   regs - Regression structure from cv_fastlpr containing:
%          .fpp_yhat - griddedInterpolant object with fitted values
%          .opt      - Options structure
%          .d        - Normalization factor (for variance estimation)
%          .dx       - Number of predictor dimensions
%   x    - New predictor locations (Tnew x dx matrix)
%          Can be complex-valued (will be split into real/imaginary parts)
%
% Output:
%   yhat - Predicted response values (Tnew x dy matrix)
%          For mean estimation: predicted mean values
%          For variance estimation: predicted variance values
%
% Example:
%   % Fit model
%   regs = cv_fastlpr(x, y);
%
%   % Predict at new locations
%   x_new = linspace(min(x), max(x), 100)';
%   y_pred = fastlpr_predict(regs, x_new);
%
%   % Plot results
%   plot(x, y, 'o', x_new, y_pred, '-');
%
% Notes:
%   - Handles complex-valued predictors automatically
%   - For variance estimation (opt.y_type_out='variance'):
%     * Applies exp transform to convert from log scale
%     * Applies normalization factor d if available
%   - Uses spline interpolation by default (smooth, continuous derivatives)
%   - Extrapolates linearly outside data range
%   - Much faster than re-running cv_fastlpr for new points
%
% See also: cv_fastlpr, fastlpr_interval, fastlpr_plot
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

% Validate inputs
if ~isfield(regs, 'fpp_yhat')
    error('fastLPR:predict:MissingField', ...
    'Regression structure must contain fpp_yhat field. Run cv_fastlpr first.');
end

opt = regs.opt;
fpp = regs.fpp_yhat;

%% Handle complex-valued predictors
% Split complex predictors into [real, imag] columns
x_iscomplex = iscomplex(x, 1);
x = [real(x), imag(x(:, x_iscomplex))];
dx = size(x, 2);

%% Predict using griddedInterpolant
% This performs fast spline interpolation on the fitted grid
[yhat] = predict_griddedInterpolant(fpp, x);

%% Handle variance estimation
% For variance estimation, the model fits log(variance)
% We need to transform back to original scale and apply normalization
if isfield(opt, 'y_type_out') && strcmp(opt.y_type_out, 'variance')
    % Get normalization factor d from regression structure
    % d is computed in fastlpr_y.m as: d = 1./((1/Ty).*sum(yraw.*exp(-yhat)))
    d = regs.d;
    dy = length(d);

    % Transform back from log scale and apply normalization
    % Formula: variance = exp(log_variance) / d
    yhat = exp(yhat) ./ d;

    % Get normalization factor d from regression structure
    % d is computed in fastlpr_y.m as: d = 1./((1/Ty).*sum(yraw.*exp(-yhat)))
    % if isfield(regs, 'd') && ~isempty(regs.d)
    %     d = regs.d;
    %     dy = length(d);

    %     % Transform back from log scale and apply normalization
    %     % Formula: variance = exp(log_variance) / d
    %     yhat = exp(yhat) ./ d;
    % else
    %     % If d is not available (e.g., compact mode), just use exp transform
    %     % This may give unnormalized variance estimates
    %     warning('fastLPR:predict:MissingNormalization', ...
    %         'Normalization factor d not found. Variance estimates may be unnormalized.');
    %     yhat = exp(yhat);
    % end
end

end
