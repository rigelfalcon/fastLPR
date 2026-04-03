function Yq = nufftn_type1(x, y,Fs, df,iflag,acc,isdeconv,nufft_loop,nufft_chunking)
% Copyright (c) 2024-2025 Ying Wang, Min Li
% SPDX-License-Identifier: GPL-3.0-or-later
%
%nufftn_type1
% x: the N*d dimension non-equispaced locations (knots) where y lies on
% y: a N*dy dimension non-equispaced function values
% M: a dx1 vector denoting number of FFT grid in each dimension
% df: frequency spacing
% iflag: sign for the exponential (small than 0 means :math:`-i`, greater than 0 means :math:`+i`)
% nufft_chunking: memory-efficient mode ('auto', 'on', 'off')
%   'auto': use chunking if estimated memory > 50% of available (default)
%   'on': always use loop-over-neighbors (lowest memory)
%   'off': always use vectorized (fastest, highest memory)

[M,dx]=size(x);
[~,dy]=size(y);
% ndcolon(1:dx) = {':'};

% FIX: Parameter order is (x, y, Fs, df, ...), so parameter 3 is Fs and parameter 4 is df
if nargin<4 || isempty(df)
    N=ceil(M^(1/dx))*ones(dx,1);
    df=1./N;
else
    N=round(1./df);
end
if nargin<3 || isempty(Fs)
    Fs=N;
end
if nargin<5 || isempty(iflag)
    iflag=-1;
end
if nargin<6 || isempty(acc)
    acc=6;
end
if nargin<7 || isempty(isdeconv)
    isdeconv=true;
end
if nargin<8 || isempty(nufft_loop)
    nufft_loop=true;
end
if nargin<9 || isempty(nufft_chunking)
    nufft_chunking='auto';
end
% Fast Non-Uniform Fourier Transform with MATLAB
% Msp denote the number of grid points to which the "spreading" will be
%   accounted for in each direction, depends on the descire accuracy
% Mr is the length of the FFTs(oversampled) Mr=ratio*M
% tau is spacing of grid(ftau)?

[Msp, Mr, tau,R] = compute_grid_params(N, acc);
% Note: compute_grid_params now returns all vectors as row vectors (1×d)

%% Binning mode: accuracy=0 means no spreading, direct accumarray
if acc == 0
    hx = 2*pi./Mr;
    xmod = scale_knots(x,N,Fs);
    m_orig = round(xmod ./ hx);

    % Compute grid indices (mod for periodic boundary)
    Mr_vec = Mr(:)';
    dtype = min_numerical_type(0, max(Mr_vec));
    fun_dtype = str2func(dtype);
    xindx = zeros(M, dx, dtype);
    for d = 1:dx
        xindx(:, d) = fun_dtype(mod(m_orig(:, d), Mr_vec(d)) + 1);
    end

    % Direct binning with accumarray
    sz = Mr_vec;
    if dx == 1
        sz = [Mr_vec, 1];
    end
    mu = bit_convert(underlyingType(y), 0);
    Ftau = zeros([sz, dy]);
    ndcolon(1:dx) = {':'};
    for iy = 1:dy
        Ftau(ndcolon{:}, iy) = accumarray(xindx, y(:, iy), sz, [], mu);
    end

    % FFT (same as spreading mode)
    if iflag < 0
        for ix=dx:-1:1
            Ftau = fftshift(fft(Ftau, [], ix), ix);
        end
        Ftau = Ftau / (M*(R^dx));
    else
        for ix=dx:-1:1
            Ftau = ifftshift(ifft(Ftau, [], ix), ix);
        end
    end

    % Extract output region (no deconvolution needed for binning)
    q=(Mr(:)-N(:))/2;
    q=[ceil(q)+1,ceil(q)+N(:)];
    sz=[Mr(:);dy]';
    idx=get_patch_index(q,sz);
    Yq=Ftau(idx);
    Yq=reshape(Yq,[N(:);dy].');
    return;
end

% Determine whether to use memory-efficient chunking (loop-over-neighbors)
Msp2 = Msp*2+1;
Msp2_prod = prod(Msp2);

% Estimate memory for vectorized approach (in GB)
% Arrays: mpmm(M,Msp2_prod,dx), weight(M,Msp2_prod), ysp(M,Msp2_prod,dy), xindx(M*Msp2_prod,dx)
bytes_per_point = Msp2_prod * dx * 8 + Msp2_prod * 8 + Msp2_prod * dy * 16 + Msp2_prod * dx * 8;
mem_estimate_gb = M * bytes_per_point / 1e9;

% Determine chunking mode
if strcmp(nufft_chunking, 'auto')
    try
        mem_info = memory;
        avail_gb = mem_info.MemAvailableAllArrays / 1e9;
        use_chunking = mem_estimate_gb > 0.5 * avail_gb;
    catch
        % memory() not available, use heuristic: chunk if > 2GB estimated
        use_chunking = mem_estimate_gb > 2.0;
    end
elseif strcmp(nufft_chunking, 'on')
    use_chunking = true;
else  % 'off'
    use_chunking = false;
end

% Construct the convolved grid
% ftau is the grid data after convolved
% hx is spacing of grid(ftau)
% xmod is the the source x location in 0~2pi
% mm is spreading grid(2Msp *1)
% m is the grid index where source xmod should be
% mpmm is m+mm with broadcasting N*Msp matrix denoting the spredding neigbourhood for each source
% indx is make sure all mpmm don't exceed Mr

hx   = 2*pi./Mr;
xmod = scale_knots(x,N,Fs);
% xmod = scale_knots2(x,N,df);
m_orig = round(xmod ./ hx);

if use_chunking
    %% Memory-efficient loop-over-neighbors approach
    % Process one neighbor offset at a time, reducing memory by Msp2_prod times

    % Pre-compute neighbor offsets (Msp2_prod × dx)
    offsets = zeros(Msp2_prod, dx);
    offset_grids = cell(1, dx);
    for i = 1:dx
        offset_grids{i} = round(linspace(-Msp(i), Msp(i), Msp2(i)));
    end
    if dx == 1
        offsets = offset_grids{1}(:);
    elseif dx == 2
        [o1, o2] = ndgrid(offset_grids{1}, offset_grids{2});
        offsets = [o1(:), o2(:)];
    elseif dx == 3
        [o1, o2, o3] = ndgrid(offset_grids{1}, offset_grids{2}, offset_grids{3});
        offsets = [o1(:), o2(:), o3(:)];
    else
        % General case for dx > 3
        grids = cell(1, dx);
        [grids{:}] = ndgrid(offset_grids{:});
        for i = 1:dx
            offsets(:, i) = grids{i}(:);
        end
    end

    % Initialize output
    Mr_vec = Mr(:)';
    sz = Mr_vec;
    if dx == 1
        sz = [Mr_vec, 1];
    end
    Ftau = zeros([sz, dy]);
    ndcolon(1:dx) = {':'};

    % Prepare data types
    dtype = min_numerical_type(0, max(Mr_vec));
    fun_dtype = str2func(dtype);
    mu = bit_convert(underlyingType(y), 0);

    % Loop over each neighbor offset
    for k = 1:Msp2_prod
        off_k = offsets(k, :);  % (1, dx)

        % Fused computation: avoid creating (M, dx) intermediate arrays
        % Compute weight by summing over dimensions (creates only (M, 1) per dim)
        weight_sum = zeros(M, 1);
        xindx_k = zeros(M, dx, dtype);
        for d = 1:dx
            mpmm_d = m_orig(:, d) + off_k(d);           % (M, 1)
            diff_d = xmod(:, d) - hx(d) * mpmm_d;       % (M, 1)
            weight_sum = weight_sum + (diff_d.^2) / (4*tau(d));
            xindx_k(:, d) = fun_dtype(mod(mpmm_d, Mr_vec(d)) + 1);
        end
        weight_k = exp(-weight_sum);  % (M, 1)

        % Compute weighted values
        ysp_k = y .* weight_k;  % (M, dy)

        % Accumulate into Ftau
        for iy = 1:dy
            Ftau(ndcolon{:}, iy) = Ftau(ndcolon{:}, iy) + accumarray(xindx_k, ysp_k(:, iy), sz, [], mu);
        end
    end
    clear m_orig xmod offsets weight_sum xindx_k weight_k ysp_k

else
    %% Original vectorized approach (fastest, uses more memory)
    m = permute(m_orig, [1,3,2]);
    Mr = permute(Mr(:), [3,2,1]);
    hx = permute(hx(:), [3,2,1]);
    xmod = permute(xmod, [1,3,2]);
    mpmm = m;
    for i = 1:dx
        m = mpmm;
        mm = zeros(1, Msp2(i), dx);
        mm(1,:,i) = round(linspace(-Msp(i), Msp(i), Msp2(i)).');
        mpmm = reshape(m + mm, [], 1, dx);
    end
    mpmm = reshape(mpmm, M, Msp2_prod, dx);

    % Compute Gaussian spreading weights using heat kernel
    weight = heat_kernel(xmod - hx .* mpmm, tau);
    y = permute(y, [1,3,2]);
    ysp = y .* weight;
    clear y weight
    ysp = reshape(ysp, [], dy);

    %% xindx
    if nufft_loop
        dtype = min_numerical_type(0, max(Mr));
        fun_dtype = str2func(dtype);
        xindx = fun_dtype(mod(mpmm, Mr) + 1);
        clear mpmm
        xindx = reshape(xindx, [], dx);
    else
        dtype = min_numerical_type(0, max(Mr));
        fun_dtype = str2func(dtype);
        xindx = fun_dtype(mod(mpmm, Mr) + 1);
        clear mpmm
        xindx = reshape(xindx, [], dx);
        if dy > 1
            xindx(:, dx+1) = 1;
        end
        xindx = repmat(xindx, [dy, 1]);
        if dy > 1
            xindx(:, dx+1) = reshape(fun_dtype(1:dy) .* ones(M*Msp2_prod, 1, dtype), [], 1);
        end
    end

    %% accumarray
    mu = bit_convert(underlyingType(ysp), 0);
    if nufft_loop
        sz = Mr(:)';
        Ftau = zeros([sz, dy]);
        if dx == 1
            sz = [Mr(:)', 1];
        end
        ndcolon(1:dx) = {':'};
        for iy = 1:dy
            Ftau(ndcolon{:}, iy) = accumarray(xindx, ysp(:, iy), sz, [], mu);
        end
        clear ysp xindx
    else
        if dx == 1 || dy > 1
            sz = [Mr(:); dy]';
        else
            sz = [Mr(:)]';
        end
        Ftau = accumarray(xindx, ysp(:), sz, [], mu);
        clear ysp xindx
    end

    % Restore Mr for FFT (was permuted)
    Mr = Mr(:)';
end
%% fft
if iflag < 0
    for ix=dx:-1:1
        Ftau = fftshift(fft(Ftau, [], ix), ix);
    end
    Ftau = Ftau / (M*(R^dx));
else
    for ix=dx:-1:1
        Ftau = ifftshift(ifft(Ftau, [], ix), ix);
    end
end

% plot([ifftshift(A.f_tau(:,1))-(ftau(:,1))])
% plot([A.f_tau,ifftshift(ftau)])
% figure(101);clf;plot(real(Ftau)*(M*(R^dx)));
% figure(101);hold on;plot(real(A.F_tau))
% figure(101);clf;plot(real(Ftau));
% figure(101);hold on;plot(real(A.F_tau))

%% index for output
q=(Mr(:)-N(:))/2;
q=[ceil(q)+1,ceil(q)+N(:)];

sz=[Mr(:);dy]';
idx=get_patch_index(q,sz);
Ftau=Ftau(idx);
Ftau=reshape(Ftau,[N(:);dy].');
% plot([A.F_tau,ifftshift(Ftau)])
% plot(abs([(A.F_tau(:,1))/(M*(R^dx)),(Ftau(:,1))]))
%% Deconvolve the grid using convolution theorem, 2004 formula 11
if isdeconv
    Kn=sqrt(prod(tau(:))/(pi^dx))* exp(-sum(permute(tau(:),[2:dx+1,1]) .* nufftfreqs(N).^2,dx+1));
    Kninv=1./Kn;
    Yq =  Kninv .* Ftau;
    clear Kn Kninv Ftau;  % Memory optimization: explicit cleanup
else
    Yq =  Ftau;
    clear Ftau;  % Memory optimization: explicit cleanup
end

end

function x=heat_kernel(x,h)

h=permute(h(:),[3:-1:1]);
x=exp( -sum((x.^2)./(4*h),3));
end
