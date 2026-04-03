function [isallow,freemem]=test_gpu(bitSize,count)
% TEST_GPU - Check if sufficient GPU memory is available
%
% Tests whether requested memory size is available on specified GPU device.
%
% Syntax:
%   isallow = test_gpu(bitSize)
%   [isallow, freemem] = test_gpu(bitSize, count)
%
% Inputs:
%   bitSize - Required memory in bytes (default: 4e9 = 4GB)
%   count   - GPU device number (default: 1)
%
% Outputs:
%   isallow - true if sufficient GPU memory available
%   freemem - Available GPU memory in bytes (0 if no GPU)
%
% Example:
%   [ok, free] = test_gpu(1e9, 1);  % Check GPU 1 for 1GB
%   if ok
%       data = gpuArray(zeros(1e9/8, 1));
%   end
%
% Notes:
%   - Returns false if no GPU available
%   - Falls back to GPU 1 if specified device fails
%   - Uses gpuDevice() to query memory
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

if nargin<1 || isempty(bitSize)
    bitSize=4e9;
end
if nargin<2
    count=1;
end

%% Check GPU availability and memory
if gpuDeviceCount>0
    try
        gpuInfo=gpuDevice(count);
    catch
        fprintf('gpuDevice(%d) failed, using GPU 1 instead.\n', count);
        gpuInfo=gpuDevice(1);
    end
    freemem=gpuInfo.AvailableMemory;
    if freemem>bitSize
        isallow=true;
    else
        isallow=false;
    end
else
    % No GPU available
    isallow=false;
    freemem=0;
end

end
