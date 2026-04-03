function reproduce_all(outputDir)
%REPRODUCE_ALL Run fastLPR demos and save figure outputs.
%   REPRODUCE_ALL() executes every example script shipped with the toolbox,
%   saving all generated figures (PNG + FIG) to fastLPR/artifacts.
%
%   REPRODUCE_ALL(OUTPUTDIR) saves artifacts to the specified folder.
%
%   The script executes each demo in the base workspace to preserve
%   reproducibility, captures all open figures, and exports both MATLAB FIG
%   and high-resolution PNG files for archival and manuscript use.

if nargin < 1 || isempty(outputDir)
    outputDir = fullfile(fileparts(mfilename('fullpath')), '..', 'artifacts');
end

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

toolboxRoot = fileparts(mfilename('fullpath'));
exampleDir = fullfile(toolboxRoot, '..', 'example');

fastlpr_setup();
addpath(fullfile(toolboxRoot, '..', 'tests'));

fprintf('[fastLPR] Running automated test suite...\n');
run_fastlpr_tests();

demos = {
    struct('id', 'fig2_basic', 'script', 'demo_basic_script.m', ...
           'description', 'Polynomial order comparison (Figure 2)');
    struct('id', 'fig3_kde', 'script', 'demo_kde.m', ...
           'description', '1D/2D KDE panels (Figure 3)');
    struct('id', 'fig4_1d_ci', 'script', 'demo_heteroscedasticity_1d_example.m', ...
           'description', '1D intervals (CI and PI) (Figure 4 a-b)');
    struct('id', 'fig4_2d_ci', 'script', 'demo_heteroscedasticity_2d_example.m', ...
           'description', '2D intervals (CI and PI) (Figure 4 c-d)');
    struct('id', 'fig5_complex', 'script', 'demo_reg_complex_xy_log2d.m', ...
           'description', 'Complex-valued regression (Figure 5)');
    struct('id', 'fig6_qeeg', 'script', 'demo_qeeg_application.m', ...
           'description', 'qEEG application (Figure 6)');
    };

for i = 1:numel(demos)
    demo = demos{i};
    fprintf('[fastLPR] Running %s...\n', demo.description);

    evalin('base', 'FASTLPR_SAVE_FIG = true;');
    runCommand = sprintf( ...
        "cd('%s'); close all; run('%s');", ...
        escape_path(exampleDir), demo.script);
    evalin('base', runCommand);

    saveCommand = sprintf( ...
        "save_open_figures('%s', '%s');", ...
        escape_path(outputDir), demo.id);
    evalin('base', saveCommand);

    evalin('base', 'clear FASTLPR_SAVE_FIG');
    evalin('base', 'close all;');
end

fprintf('[fastLPR] Reproduction complete. Artifacts stored in %s\n', outputDir);

end

function pathOut = escape_path(pathIn)
%ESCAPE_PATH Ensure single quotes in paths are doubled for evalin commands.
pathOut = strrep(pathIn, '''', '''''');
end
