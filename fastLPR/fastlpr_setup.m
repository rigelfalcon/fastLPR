function fastlpr_setup()
%FASTLPR_SETUP Add fastLPR toolbox folders to the MATLAB path.
%   FASTLPR_SETUP() adds the utility functions and example scripts shipped
%   with the fastLPR toolbox to the MATLAB search path. Call this function
%   once per MATLAB session before using cv_fastlpr or cv_fastkde.
%
%   Example:
%       fastlpr_setup();
%       regs = cv_fastlpr(rand(200,1), rand(200,1));
%
%   The function preserves the existing MATLAB path and only prepends the
%   fastLPR directories. It can safely be called multiple times.
%
%   See also CV_FASTLPR, CV_FASTKDE, FASTLPR_PLOT, FASTKDE_PLOT.

toolboxRoot = fileparts(mfilename('fullpath'));

% Add root directory (for Contents.m and ver command)
addpath(toolboxRoot);

utilityPath = fullfile(toolboxRoot, 'utility');
corePath = fullfile(toolboxRoot, 'utility', 'core');
examplePath = fullfile(toolboxRoot, 'example');

addpath(utilityPath);
addpath(corePath);
addpath(genpath(examplePath));

fprintf('[fastLPR] Added fastLPR toolbox to MATLAB path.\n');
fprintf('          Type "help fastLPR" to get started, "ver fastLPR" to check version.\n');

end
