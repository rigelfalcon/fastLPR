function install_fastlpr(varargin)
%INSTALL_FASTLPR Install fastLPR toolbox
%   INSTALL_FASTLPR adds the fastLPR toolbox to the MATLAB path and optionally
%   saves the path for future sessions.
%
%   INSTALL_FASTLPR() - Interactive mode, prompts to save path
%   INSTALL_FASTLPR(true) - Save path automatically without prompt
%   INSTALL_FASTLPR(false) - Do not save path, no prompt
%
%   This function:
%   1. Adds fastLPR root directory to path (for Contents.m and ver command)
%   2. Adds fastLPR/utility to the path (public API)
%   3. Adds fastLPR/example to the path (reproducible examples)
%   4. Does NOT add fastLPR/tests (keep tests separate)
%   5. Optionally saves the path using savepath
%
%   After installation, type 'help fastLPR' to get started.
%
%   Example:
%       cd fastLPR
%       install_fastlpr
%       help fastLPR
%       ver fastLPR
%
%   See also: FASTLPR_SETUP, CV_FASTLPR, CV_FASTKDE

% Copyright (c) 2025 Ying Wang, Min Li, Deirel Paz-Linares, Pedro A. Valdes-Sosa
% Licensed under GPL-3.0

    % Parse input arguments
    if nargin > 0
        savePath = varargin{1};
        if ~islogical(savePath)
            error('Input argument must be true or false');
        end
        interactive = false;
    else
        interactive = true;
        savePath = false;
    end

    % Get installation directory
    install_dir = fileparts(mfilename('fullpath'));

    fprintf('=============================================================\n');
    fprintf('fastLPR Toolbox Installation\n');
    fprintf('=============================================================\n\n');

    % Add toolbox directories
    fprintf('Adding to MATLAB path:\n');
    fprintf('  - %s (root, for Contents.m)\n', install_dir);
    addpath(install_dir);

    fprintf('  - %s (public API)\n', fullfile(install_dir, 'utility'));
    addpath(fullfile(install_dir, 'utility'));

    fprintf('  - %s (core functions)\n', fullfile(install_dir, 'utility', 'core'));
    addpath(fullfile(install_dir, 'utility', 'core'));

    fprintf('  - %s (examples)\n', fullfile(install_dir, 'example'));
    addpath(fullfile(install_dir, 'example'));

    fprintf('\n');
    fprintf('fastLPR toolbox added to MATLAB path.\n');
    fprintf('Type "help fastLPR" or "ver fastLPR" to get started.\n\n');

    % Verify installation
    fprintf('Verifying installation...\n');
    if exist('cv_fastlpr', 'file')
        fprintf('  [OK] cv_fastlpr found\n');
    else
        warning('  [FAIL] cv_fastlpr not found - check installation');
    end

    if exist('cv_fastkde', 'file')
        fprintf('  [OK] cv_fastkde found\n');
    else
        warning('  [FAIL] cv_fastkde not found - check installation');
    end

    fprintf('\n');

    % Optional: Save path
    if interactive
        response = input('Save path for future MATLAB sessions? (y/n): ', 's');
        savePath = strcmpi(response, 'y');
    end

    if savePath
        try
            savepath;
            fprintf('Path saved successfully.\n');
            fprintf('fastLPR will be available in future MATLAB sessions.\n');
        catch ME
            warning('Failed to save path: %s', ME.message);
            fprintf('You may need to run install_fastlpr again in future sessions.\n');
        end
    else
        if ~interactive
            fprintf('Path not saved (non-interactive mode).\n');
        else
            fprintf('Path not saved.\n');
        end
        fprintf('You will need to run install_fastlpr or fastlpr_setup in future sessions.\n');
    end

    fprintf('\n=============================================================\n');
    fprintf('Installation complete!\n');
    fprintf('\nQuick Start:\n');
    fprintf('  help fastLPR              - View toolbox contents\n');
    fprintf('  help cv_fastlpr           - Learn about regression\n');
    fprintf('  help cv_fastkde           - Learn about density estimation\n');
    fprintf('  cd example                - Browse examples\n');
    fprintf('  reproduce_all_figures     - Generate all JSS paper figures\n');
    fprintf('=============================================================\n');
end
