
% Copyright (c) 2024-2025 Ying Wang, Min Li
% SPDX-License-Identifier: GPL-3.0-or-later
%
function x = symlog(x)
% x = sign(x) .* log10(abs(sign(x)*1+x)); %log
x = sign(x) .* log10(1+abs(x)); %log
end
