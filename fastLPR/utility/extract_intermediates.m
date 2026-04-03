function intermediates = extract_intermediates(x, y, h, opt)
% Copyright (c) 2024-2025 Ying Wang, Min Li
% SPDX-License-Identifier: GPL-3.0-or-later
%
% EXTRACT_INTERMEDIATES - Extract intermediate values for debugging
% This function is placed in utility/ so it can call private functions
%
% Returns struct with:
%   .s - design matrix
%   .t - T-vector  
%   .mq - grid regression result
%   .yhat_grid - predictions on grid
%   .yhat - predictions at data points

% Step 1: Create regression structure
regs = fastlpr_create(x, y, h, opt);

intermediates.N = regs.N;
intermediates.Tx = regs.Tx;
intermediates.h = regs.h;

% Step 2: Compute design matrix S
regs = fastlpr_s(regs);
intermediates.s = regs.s;

% Step 3: Compute DOF
regs = fastlpr_dof(regs);
if isfield(regs, 'dof')
    intermediates.dof = regs.dof;
end

% Step 4: Compute T-vector and grid regression
% Call fastlpr_reg to get mq
y_ft = regs.y;
mq = fastlpr_reg(regs, y_ft);
intermediates.mq = mq;

% Step 5: Compute fitted values
regs = fastlpr_y(regs);
intermediates.yhat = regs.yhat;

% Extract grid predictions
if isfield(regs, 'fpp_yhat') && isfield(regs.fpp_yhat, 'Values')
    intermediates.yhat_grid = regs.fpp_yhat.Values;
end

end
