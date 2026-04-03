function [isallow,freemem]=test_memory(bitSize,onlyram)
% TEST_MEMORY - Check if sufficient memory is available
%
% Tests whether requested memory size is available on the system.
% Cross-platform (Windows, Linux, Mac).
%
% Syntax:
%   isallow = test_memory(bitSize)
%   [isallow, freemem] = test_memory(bitSize, onlyram)
%
% Inputs:
%   bitSize - Required memory in bytes (default: 4e9 = 4GB)
%   onlyram - Windows only: check physical RAM only (default: false)
%             false: includes virtual memory
%             true: physical RAM only
%
% Outputs:
%   isallow - true if sufficient memory available
%   freemem - Available memory in bytes
%
% Example:
%   [ok, free] = test_memory(1e9);  % Check for 1GB
%   if ok
%       data = zeros(1e9/8, 1);  % Allocate
%   end
%
% Notes:
%   - Windows: uses memory() function
%   - Linux/Mac: parses 'free' command output
%   - Returns available memory, not total memory
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

if nargin<1
    bitSize=4e9;
end

if nargin<2 ||isempty(onlyram)
    onlyram=false;
end
%% Platform-specific memory check
if ismac
    % Mac platform: use 'free' command
    [r,w] = unix('free | grep Mem');
    stats = str2double(regexp(w, '[0-9]*', 'match'));
    memsize = stats(1);
    freemem = (stats(3))*1024;  % Available memory in bytes

elseif isunix
    % Linux platform: use 'free' command
    [r,w] = unix('free | grep Mem');
    stats = str2double(regexp(w, '[0-9]*', 'match'));
    memsize = stats(1)*1024;
    freemem = (stats(3))*1024;  % Available memory in bytes

elseif ispc
    % Windows platform: use memory() function
    [userview,systemview]=memory;
    if onlyram
        freemem=systemview.PhysicalMemory.Available;
    else
        freemem=userview.MemAvailableAllArrays;  % Includes virtual
    end
else
    error('fastLPR:testMemory:UnsupportedPlatform', ...
        'Platform not supported');
end

%% Check if sufficient memory available
if freemem>bitSize
    isallow=true;
else
    isallow=false;
end

end
