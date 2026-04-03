function save_nufft_internals_matlab(x, y, h, opt, output_file)
% Copyright (c) 2024-2025 Ying Wang, Min Li
% SPDX-License-Identifier: GPL-3.0-or-later
%
% SAVE_NUFFT_INTERNALS_MATLAB - Save intermediate NUFFT results from regression pipeline
%
% This function is placed in utility/ so it can access private functions.
% It runs through the regression pipeline step-by-step and saves all intermediates.
%
% Inputs:
%   x           - Predictor data (Nx1 or Nxdx matrix)
%   y           - Response data (Nx1 vector)
%   h           - Bandwidth (scalar or vector)
%   opt         - Options structure
%   output_file - Path to save .mat file
%
% Author: AI-generated for debugging
% Date: 2025-11-15

fprintf('=== MATLAB NUFFT Internal Test (from regression pipeline) ===\n');
fprintf('Data: n=%d\n', size(x, 1));
fprintf('\n--- Testing with h=%.6f, order=%d ---\n', h, opt.order);

% Step 1: Create regression structure (preprocessing)
fprintf('\n--- Step 1: Creating regression structure ---\n');
regs = fastlpr_create(x, y, h, opt);

fprintf('  dx=%d, dy=%d\n', regs.dx, regs.dy);
fprintf('  Tx (sample count)=%d, Ty (sample count)=%d\n', regs.Tx, regs.Ty);
fprintf('  h=%.6f\n', regs.h);
fprintf('  x range: [%.6f, %.6f]\n', min(regs.x(:)), max(regs.x(:)));
fprintf('  y range: [%.6f, %.6f]\n', min(regs.y(:)), max(regs.y(:)));
fprintf('  knot range: [%.6f, %.6f]\n', min(regs.knot(:)), max(regs.knot(:)));
fprintf('  hf: %.6f\n', regs.hf);

% Step 2: Compute kernel density (calls NUFFT internally)
fprintf('\n--- Step 2: Computing kernel density (fastlpr_kdf) ---\n');
[kdf, ihbad, dh, dh_input, h_kdf, lwp] = fastlpr_kdf(regs.x, regs.h, regs.N, regs.opt);
regs.kdf = kdf;
regs.lwp = lwp;

fprintf('  kdf computed, grid size: %s\n', mat2str(size(regs.kdf)));
fprintf('  kdf range: [%.6e, %.6e]\n', min(real(regs.kdf(:))), max(real(regs.kdf(:))));

% Step 3: Design matrix S (calls NUFFT internally)
fprintf('\n--- Step 3: Computing design matrix S ---\n');
S = design_matrix(regs);

fprintf('  S dimensions: [%d x %d]\n', size(S, 1), size(S, 2));
fprintf('  S(1,1): %.6e\n', S(1,1));
fprintf('  S(1,2): %.6e\n', S(1,2));
if size(S, 2) >= 3
    fprintf('  S(1,3): %.6e\n', S(1,3));
end

% Step 4: Regression on grid
fprintf('\n--- Step 4: Computing regression on grid (fastlpr_reg) ---\n');
regs = fastlpr_reg(regs, S);

fprintf('  reg computed, grid size: %s\n', mat2str(size(regs.reg)));
fprintf('  reg(1:5): %.6f %.6f %.6f %.6f %.6f\n', regs.reg(1:5));

% Step 5: Interpolate to get fitted values at data points
fprintf('\n--- Step 5: Interpolating fitted values (fastlpr_y) ---\n');
yhat = fastlpr_y(regs, x);

fprintf('  yhat length: %d\n', length(yhat));
fprintf('  yhat(1:5): %.6f %.6f %.6f %.6f %.6f\n', yhat(1:5));
fprintf('  yhat range: [%.6f, %.6f]\n', min(yhat), max(yhat));

% Save all internals
fprintf('\n--- Saving internal results ---\n');
internal_results = struct();

% Input data
internal_results.x_raw = x;
internal_results.y_raw = y;
internal_results.h = h;
internal_results.opt = opt;

% Preprocessing from fastlpr_create
internal_results.preprocessing.x = regs.x;
internal_results.preprocessing.y = regs.y;
internal_results.preprocessing.knot = regs.knot;
internal_results.preprocessing.hf = regs.hf;
internal_results.preprocessing.N = regs.N;
internal_results.preprocessing.Nratio = regs.Nratio;
internal_results.preprocessing.Tx = regs.Tx;
internal_results.preprocessing.Ty = regs.Ty;
internal_results.preprocessing.dx = regs.dx;
internal_results.preprocessing.dy = regs.dy;
internal_results.preprocessing.x_mean = regs.x_mean;
internal_results.preprocessing.x_std = regs.x_std;

% Kernel density
internal_results.kdf = regs.kdf;

% Design matrix
internal_results.S = S;

% Regression on grid
internal_results.reg = regs.reg;

% Fitted values
internal_results.yhat = yhat;

save(output_file, 'internal_results', '-v6');
fprintf('\nInternal results saved to %s\n', output_file);

end
