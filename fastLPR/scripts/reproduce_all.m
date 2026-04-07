function reproduce_all(outputDir)
%REPRODUCE_ALL Run fastLPR demos and save figure outputs.
%   REPRODUCE_ALL() executes every example script shipped with the toolbox.
%   This is a wrapper that delegates to example/reproduce_all_figures.m.
%
%   REPRODUCE_ALL(OUTPUTDIR) saves artifacts to the specified folder.

if nargin < 1 || isempty(outputDir)
    outputDir = fullfile(fileparts(mfilename('fullpath')), '..', 'artifacts');
end

% Delegate to the canonical reproduction script
exampleDir = fullfile(fileparts(mfilename('fullpath')), '..', 'example');
addpath(exampleDir);
reproduce_all_figures();

end
