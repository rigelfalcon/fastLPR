function [gcvout]=fastlpr_gcvmerge(gcvin)
% FASTLPR_GCVMERGE - Merge GCV results from multiple batches
%
% Combines GCV results from multiple response variable batches into a single
% structure. Used when responses are processed in batches for memory efficiency.
%
% Syntax:
%   gcvout = fastlpr_gcvmerge(gcvin)
%
% Input:
%   gcvin - Array of GCV structures from fastlpr_gcv, one per batch
%           Each structure contains results for a subset of responses
%
% Output:
%   gcvout - Merged GCV structure containing results for all responses
%
% Algorithm:
%   1. Identify field types:
%      - Unique fields: same for all batches (e.g., 'metric')
%      - Array fields: concatenate across batches (e.g., 'yhat_1se')
%      - fpp field: special handling for griddedInterpolant
%   2. Copy unique fields from first batch
%   3. Concatenate array fields along response dimension
%   4. Merge fpp.Values arrays
%
% Notes:
%   - Assumes all batches have same grid and bandwidths
%   - Only response dimension varies across batches
%   - Preserves all GCV statistics and selected bandwidths
%
% Author: Ying Wang, Min Li
% Create Time: 2021
% Copyright (c): 2020-2025 Ying Wang, yingwangrigel@gmail.com,
%                Min Li, minli.231314@gmail.com
% Joint China-Cuba LAB, UESTC
% License: GNU General Public License v3.0 (see LICENSE file)

%% Determine concatenation dimension
% Find dimension along which to concatenate (response dimension)
if isfield(gcvin,'yhat')
    dimy=ndims(gcvin(1).yhat);
elseif isfield(gcvin,'yhat_1se')
    dimy=ndims(gcvin(1).yhat_1se);
end

%% Classify fields by type
fields_all=fieldnames(gcvin);
fields_fpp={'fpp'};  % griddedInterpolant (special handling)
fields_uni={'metric'};  % Unique fields (same for all batches)
fields_array=setdiff(fields_all,[fields_fpp(:);fields_uni(:)]);  % Array fields (concatenate)

%% Copy unique fields
% These fields are the same for all batches, so just copy from first
for ifield=1:length(fields_uni)
    if isfield(gcvin,fields_uni{ifield})
        gcvout.(fields_uni{ifield})=gcvin.(fields_uni{ifield});
    end
end

%% Concatenate array fields
% These fields vary across batches and need to be concatenated
for ifield=1:length(fields_array)
    gcvout.(fields_array{ifield})=cat(dimy,gcvin.(fields_array{ifield}));
end

%% Merge griddedInterpolant objects
% fpp.Values needs special handling to concatenate along response dimension
num_loop=length(gcvin);

% Start with first batch's fpp
gcvout.(fields_fpp{1})=gcvin(1).(fields_fpp{1});

% Get grid dimensions
dx=length(gcvin(1).(fields_fpp{1}).GridVectors);

% Count total responses across all batches
dy=0;
num_y_loop=zeros(num_loop,1);
for iloop=1:num_loop
    dyi=size(gcvin(iloop).(fields_fpp{1}).Values);
    num_y_loop(iloop)=prod(dyi(dx+1:end));  % Number of responses in this batch
    dy=dy+num_y_loop(iloop);
end

% Allocate merged Values array
% Get grid size for each dimension
N = zeros(1, dx);
for d = 1:dx
    N(d) = length(gcvin(1).(fields_fpp{1}).GridVectors{d});
end
ndcolon(1:dx) = {':'};  % Colon indexing for grid dimensions
gcvout.(fields_fpp{1}).Values=zeros([N, dy]);

% Copy Values from each batch
ky=0;
for iloop=1:num_loop
    iy=ky+[1:num_y_loop(iloop)];  % Response indices for this batch
    gcvout.(fields_fpp{1}).Values(ndcolon{:},iy)=gcvin(iloop).(fields_fpp{1}).Values;
    ky=num_y_loop(iloop)+ky;
end

end