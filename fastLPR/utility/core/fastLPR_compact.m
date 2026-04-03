function [regs]=fastlpr_compact(regs)
% FASTLPR_COMPACT - Remove intermediate variables to save memory
%
% Removes large intermediate arrays from the regression structure after
% computation is complete. Keeps only essential results for prediction
% and visualization.
%
% Syntax:
%   regs = fastlpr_compact(regs)
%
% Input:
%   regs - Regression structure from cv_fastlpr
%
% Output:
%   regs - Compacted regression structure
%
% Behavior:
%   If opt.compact = true:
%     Keep only: fpp_yhat, yhat, opt, d, gcv_yhat
%     Remove: all intermediate arrays (s, t, kdf, knot, etc.)
%
%   If opt.compact = false:
%     Remove only: x, y (raw data)
%     Keep: all intermediate arrays for debugging
%
% Memory Savings:
%   - Compact mode: ~90% memory reduction
%   - Non-compact mode: ~10% memory reduction
%
% Notes:
%   - Compact mode is recommended for production use
%   - Non-compact mode is useful for debugging and analysis
%   - After compacting, you can still predict at new points
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

if regs.opt.compact
    % Compact mode: keep only essential fields
    % These are sufficient for prediction and visualization
    % dof_random_vectors is kept for cross-validation reproducibility in Python/R
    fields={'fpp_yhat','fpp_s0','s0','yhat','opt','d','gcv_yhat','dof_random_vectors'};
    regs=keepfield(regs,fields);
else
    % Non-compact mode: remove only raw data
    % Keep all intermediate arrays for debugging
    regs=rmfield(regs,{'x','y'});
end

end
