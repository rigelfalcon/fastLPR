# fastLPR Benchmark Suite

**Version:** 2.3
**Last Updated:** 2026-01-14
**Purpose:** Comprehensive benchmarking for JSS publication

---

## 1. Overview

This benchmark suite compares **fastLPR/fastKDE** against state-of-the-art kernel density estimation (KDE) and local polynomial regression (LPR) methods across MATLAB, Python, and R implementations.

### 1.1 Methods Compared

| Method | Task | Language | Complexity | Reference |
|--------|------|----------|------------|-----------|
| **fastKDE** | KDE | MATLAB/Python/R | O(N + M log M) | This paper |
| **fastLPR** | LPR | MATLAB/Python/R | O(N + M log M) | This paper |
| **ks** | KDE | R | O(N²) exact, O(N + M log M) binned | Duong (2007) JSS |
| **FKSUM** | KDE | R | O(N log N) | Hofmeyr (2021) IEEE TPAMI |
| **locfit** | LPR | R | O(N) | Loader (1999) |
| **npregfast** | LPR | R | O(N) | Sestelo et al. (2017) JSS |
| **StOpt-NW** | LPR | Python | O(N + M log M) | StOpt library |
| **DirectKDE** | KDE | MATLAB | O(N²) | Naive baseline |
| **DirectNW** | LPR | MATLAB | O(N²) | Naive baseline |

> **Note:** 9 distinct algorithms with 13 implementations (fastKDE/fastLPR each have MATLAB, Python, R versions).

### 1.2 Benchmark Parameters

- **Sample sizes (N):** 2^5 to 2^25 (32 to 33,554,432)
- **Dimensions (d):** 1, 2, 3
- **Bandwidth formula:** h_N = H0 × N^(-1/(d+4)), H0 = 0.3 (Silverman's optimal rate)
- **Repetitions:** 3 runs per configuration (median reported)
- **Internal grid size (M_INTERNAL):** Total grid points used inside fastKDE/fastLPR; affects both speed and accuracy
  - Full mode: 16,384 points (d=1/2), 32,768 points (d=3 = 32^3)
  - Quick mode: 1,024 points (all d)
- **Evaluation mode:** `--mode data_point` (at sample points) or `--mode grid` (on GT `x_grid` when available; otherwise fallback grid is power-of-two per dimension)
- **Metrics:** Time (seconds), Memory (MB), Accuracy (MSE vs Direct for N ≤ 65,536)

---

## 2. Directory Structure

```
benchmark/
├── scripts/                        # Benchmark scripts
│   ├── run_all_benchmarks.py       # ★ UNIFIED RUNNER (recommended)
│   ├── matlab/                     # MATLAB: benchmark_fastlpr.m, benchmark_direct.m
│   ├── python/                     # Python: benchmark_fastlpr.py
│   ├── cpp/                        # C++: benchmark_stopt.py, stopt_wrapper.py
│   ├── r/                          # R: benchmark_fastlpr.R, benchmark_ks.R, benchmark_fksum.R,
│   │                               #    benchmark_locfit.R, benchmark_npregfast.R
│   ├── plotting/                   # Figure generation
│   │   └── plot_fig7_benchmark.py  # JSS Figure 7 (3x3 grid: speed/memory/accuracy)
│   └── memory_monitor.py           # Signal-based USS/RSS delta monitoring
├── data/                           # Results output
│   ├── benchmark_results.csv       # ★ Main output
│   └── ground_truth/               # Ground truth .mat files (from benchmark_direct.m)
└── README.md                       # This file
```

---

## 3. Running Benchmarks

### 3.1 Prerequisites

**MATLAB:**
```matlab
cd jss-code
run('./setup.m')
```

**Python:**
```bash
cd fastLPR_py
uv pip install -e .
```

**R:**
```r
install.packages(c("R.matlab", "ks", "FKSUM", "locfit", "npregfast"))
```

### 3.2 Execute Benchmarks

#### Unified Benchmark Runner (Recommended)

The unified benchmark runner (`scripts/run_all_benchmarks.py`) provides consistent execution and memory measurement across all languages using **signal-based USS/RSS delta monitoring**:

```bash
# Quick test (N_RUNS=1, for testing)
uv run python benchmark/scripts/run_all_benchmarks.py --quick --methods fastlpr

# Full benchmark (N_RUNS=3, for publication)
uv run python benchmark/scripts/run_all_benchmarks.py --methods all

# All methods, quick mode
uv run python benchmark/scripts/run_all_benchmarks.py --quick --methods all

# Specific methods
uv run python benchmark/scripts/run_all_benchmarks.py --methods fastlpr ks stopt

# Specific dimensions
uv run python benchmark/scripts/run_all_benchmarks.py --methods fastlpr --dims 1 2
```

**Key Features:**
- **Memory method recorded**: Each row includes `mem_method` so plots can exclude or compare measurement types
- **External monitor (most methods)**: Subprocess prints `READY`/`RUN_START`/`RUN_END`; parent tracks process peak and subtracts baseline
- **In-process RSS (Python + StOpt)**: Python fastKDE/fastLPR and StOpt-NW record RSS peak deltas inside the benchmark process to capture native/C++ allocations without relying on the external monitor
- **Cross-language coverage**: Benchmarks Python, R, MATLAB, and C++ methods with a unified CSV schema

**Memory Measurement:**
```
Memory depends on `mem_method`.
- External monitor: peak(process) - baseline(process) (captured at READY)
- In-process RSS: peak(RSS) - baseline(RSS) (captured before each run)
```

#### Individual Scripts (Alternative)

##### Our Methods (fastKDE/fastLPR)

```bash
# MATLAB
matlab -batch "run('benchmark/scripts/matlab/benchmark_fastlpr.m')"

# Python
uv run python benchmark/scripts/python/benchmark_fastlpr.py

# R
Rscript benchmark/scripts/r/benchmark_fastlpr.R
```

#### KDE Competitors

```bash
# ks (R) - binned mode, d=1,2,3
Rscript benchmark/scripts/r/benchmark_ks.R

# FKSUM (R) - exact O(N log N), d=1,2,3
Rscript benchmark/scripts/r/benchmark_fksum.R
```

#### LPR Competitors

```bash
# locfit (R) - adaptive local fitting, d=1,2,3
Rscript benchmark/scripts/r/benchmark_locfit.R

# npregfast (R) - binning method, 1D ONLY
Rscript benchmark/scripts/r/benchmark_npregfast.R

# StOpt-NW (Python/C++) - NUFFT-based, d=1,2,3
uv run python benchmark/scripts/cpp/benchmark_stopt.py
```

#### Baseline Methods (O(N²))

```bash
# MATLAB Direct - reference baseline for speedup calculation
matlab -batch "run('benchmark/scripts/matlab/benchmark_direct.m')"
```

### 3.3 Generate Figures

```bash
uv run python benchmark/scripts/plotting/plot_fig7_benchmark.py
```

---

## 4. Output Format

### 4.1 CSV Schema

| Column | Type | Description |
|--------|------|-------------|
| method | string | Method name (fastKDE, ks, etc.) |
| task | string | KDE or LPR |
| lang | string | MATLAB, PYTHON, R, or C++ |
| d | int | Dimensionality (1, 2, 3) |
| N | int | Sample size |
| time_sec | float | Median time (seconds) |
| time_min | float | Minimum time across runs |
| time_max | float | Maximum time across runs |
| time_std | float | Standard deviation |
| mem_mb | float | Memory summary (typically median of per-run peaks, in MB) |
| mem_median | float | Median of per-run memory peaks (MB) |
| mem_min | float | Minimum per-run memory peak (MB) |
| mem_max | float | Maximum per-run memory peak (MB) |
| mem_std | float | Std of per-run memory peaks (MB) |
| mem_method | string | Memory measurement method used (e.g., tracemalloc, gc(), MATLAB_internal, RSS) |
| baseline_mb | float | Baseline memory at READY signal (MB), when available |
| accuracy_vs_direct | float | MSE against Direct baseline (N ≤ 65,536 only) |
| acc_mask_applied | bool | In `--mode grid`, whether bounding-box masking was applied for MSE |
| acc_mask_ratio | float | `acc_mask_n_used / acc_mask_n_total` |
| acc_mask_n_total | int | Total grid points (before masking) |
| acc_mask_n_used | int | Grid points used after masking |
| status | string | success, error, timeout, or skip |

### 4.2 Output File

The unified benchmark runner outputs to `data/benchmark_results.csv` with all methods combined.

Note: in `--mode grid`, the default output file is `data/benchmark_results_grid.csv` (used by `plot_fig7_benchmark.py`).

**Memory Measurement:**
- Uses signal-based USS/RSS delta for fair cross-language comparison
- Algorithm Memory = Peak USS - Baseline USS (with RSS fallback)
- Isolates algorithm cost from runtime/interpreter overhead

| Language | Runtime Baseline | Algorithm Memory |
|----------|------------------|------------------|
| Python   | ~141 MB          | ~35-37 MB        |
| R        | ~210 MB          | ~7-18 MB         |
| MATLAB   | ~950 MB          | ~29-36 MB        |

---

## 5. Method Details

### 5.1 Our Methods

**fastKDE/fastLPR** uses Non-Uniform Fast Fourier Transform (NUFFT) with Gaussian gridding for O(N + M log M) complexity, where N is sample size and M is grid size.

#### Python FFT backend (fastLPR_py)

The Python implementation uses `fastLPR_py/src/fastlpr/fft_backend.py` to choose an FFT backend.

- Fallback order: `pyfftw` (only if explicitly enabled) → `scipy.fft` (default) → `numpy.fft` (fallback).
- Rationale: in the benchmark configurations used by this repo (including `accuracy=0` binning mode), FFT is often not the dominant cost, so enabling `pyfftw` may be neutral or slower due to planning/dispatch overhead.

#### NUFFT "accuracy" option vs binned competitors

In the unified runner, the NUFFT accuracy knob is configured by `benchmark/scripts/run_all_benchmarks.py` via `NUFFT_ACCURACY`.

- `accuracy > 0`: NUFFT spreading mode (Gaussian gridding / heat-kernel spreading over a support region).
- `accuracy = 0`: binning mode (no spreading; points are binned onto a grid before FFT).

Important for fairness vs `ks::kde(binned=TRUE)`:
- `ks` uses **linear binning** (each point contributes to 2^d neighboring bins with weights).
- Our current `accuracy=0` path bins each point to a single nearest bin (no linear weights). That is faster per-point but introduces more discretization error.
- If we want `accuracy=0` to match `ks` more closely, we should switch to linear binning weights in the `acc==0` path.

### 5.2 KDE Competitors

| Package | Method | Dimensions | Notes |
|---------|--------|------------|-------|
| **ks** | Linear binning + FFT | d=1,2,3 | Default binned mode O(N + M log M); exact mode O(N²) |
| **FKSUM** | Recursive kernel sums | d=1,2,3 | Exact evaluation in O(N log N) |

### 5.3 LPR Competitors

| Package | Method | Dimensions | Notes |
|---------|--------|------------|-------|
| **locfit** | Local fitting | d=1,2,3 | Adaptive bandwidth; "vertex space" limits for large N |
| **npregfast** | Binning + Fortran | **d=1 only** | O(N) complexity; **uses package default bandwidth** (not H0=0.3) |
| **StOpt-NW** | NUFFT approximation | d=1,2,3 | C++ backend; variable bandwidth formula |

### 5.4 Baseline Methods

**DirectKDE/DirectNW** provide O(N²) naive implementations to demonstrate speedup factors. Limited to N ≤ 65,536 due to computational cost.

### 5.5 Methodological Notes

#### fastKDE vs DirectKDE: Expected ~3% Difference

When comparing fastKDE and DirectKDE at the same bandwidth, there is an expected ~3-4% systematic difference in density values at discrete data points. This is NOT a bug but a result of different normalization strategies:

| Aspect | DirectKDE | cv_fastKDE |
|--------|-----------|------------|
| **Evaluation** | Direct at data points | Grid (200 pts) + interpolation |
| **Normalization** | Analytical: `(2π)^(-d/2)/h` | Numerical: ensures ∫f(x)dx = 1 |
| **NUFFT** | Not used (exact O(N²)) | Uses NUFFT with ~3% mass loss |

**Root Cause:**
1. NUFFT-based convolution loses ~3% mass due to finite spreading width
2. Raw fastKDE values would be ~3% too low
3. Numerical normalization corrects this to ensure proper density (integral = 1)
4. At discrete evaluation points, this makes fastKDE ~3% higher than DirectKDE

**Implications for ISE comparison:**
- DirectKDE ISE is slightly lower because it directly evaluates at data points without normalization
- fastKDE ISE is slightly higher but produces a proper probability density function
- Both are valid KDE implementations with different trade-offs
- The ~18% ISE ratio difference at small N (e.g., N=32) reduces as N increases

**Recommendation:**
Both methods correctly estimate the true density. For applications requiring exact integral = 1, use fastKDE. For minimal point-wise error at data locations, DirectKDE is slightly better but at O(N²) cost.

---

## 6. Accuracy Computation

### 6.1 KDE: Accuracy Metric

The CSV column `accuracy_vs_direct` stores the error **against the Direct baseline** for N ≤ 65,536.

- In `--mode grid`, this is computed as **mean squared error on the evaluation grid**:
  - KDE: `mean((f̂_grid - f_direct_grid)^2)`
  - LPR: `mean((ŷ_grid - ŷ_direct_grid)^2)`
  - To reduce sensitivity to extrapolation, MSE is computed only on grid points inside the `x_zs` bounding box; the `acc_mask_*` columns record the mask diagnostics.
- In `--mode data_point`, this is computed as **mean squared error at the sample points**.

This is a **pointwise MSE** (a proxy for “ISE/MSE”), not a separately reported ISE integral.

### 6.2 LPR: Accuracy Metric

Same `accuracy_vs_direct` definition as above (MSE vs Direct).

#### Important: accuracy depends on internal grid size

For fastKDE/fastLPR, the `accuracy_vs_direct` value depends strongly on the internal computation grid size (`M_INTERNAL`) because the algorithm evaluates on an internal regular grid and then interpolates to the evaluation points.

- Quick mode (`--quick`, `M_INTERNAL=1024`) is meant for smoke testing and can substantially inflate 3D LPR error.
- Full mode uses `M_INTERNAL=16,384` (d=1/2) and `M_INTERNAL=32,768` (d=3) for more stable accuracy.

When comparing methods (e.g., fastLPR vs StOpt-NW) in 3D LPR, ensure you are comparing runs with the same evaluation mode and a sufficiently large `M_INTERNAL`.

### 6.3 Bandwidth Formula

All methods use the same variable bandwidth for fair comparison:

```
h_N = H0 × N^(-1/(d+4))
```

Where:
- H0 = 0.3 (base bandwidth constant)
- d = dimension
- N = sample size

This follows Silverman's optimal rate for Gaussian kernels.

---

## 7. Troubleshooting

### 7.1 Common Issues

| Issue | Solution |
|-------|----------|
| locfit "out of vertex space" | Known limitation for d≥2 and large N; logged as ERROR |
| Memory errors at large N | Expected for N > 16M with limited RAM |
| npregfast d>1 | **Not supported**; 1D only |
| npregfast bandwidth | Uses package default (auto) bandwidth; H0=0.3 formula crashes at large N |
| StOpt import error | Use `uv run python` with Python `>=3.12,<3.14` and ensure `external/StOpt/BUILD_MINGW/bin/` is on `sys.path` (the `StOptReg.pyd` build is Python 3.12) |
| DirectKDE memory ~1MB | Memory measured post-execution; peak is higher during computation |
| First-N timing "hump" (e.g., N=32 slower than N=64) | Usually warmup/JIT/caching effects; the unified runner includes small-dataset warmup, but quick mode (N_RUNS=1) can still show noise |

### 7.2 locfit k-d Tree Limitations

The `locfit` package uses a k-d tree data structure with fixed vertex space allocation. When the tree requires more splits than allocated space, it fails with "newsplit: out of vertex space".

**Maximum N by dimension:**
| Dimension | Max N | Notes |
|-----------|-------|-------|
| d=1 | ~16M | Successfully tested to N=16,777,216 |
| d=2 | ~2K | Fails at N≥4,096 |
| d=3 | ~64 | Fails at N≥128 |

This is a package limitation, not a benchmark bug. The error is logged as `status="error"` in CSV files.

**Workaround:** The `maxk` parameter can be tuned to allow slightly larger N, but practical limits remain due to memory constraints.

### 7.3 Checkpointing

All benchmark scripts support checkpointing:
- Results saved incrementally to CSV
- Resume by re-running the script (skips completed configurations)

---

## 8. References

1. Duong, T. (2007). ks: Kernel Density Estimation and Kernel Discriminant Analysis for Multivariate Data in R. *Journal of Statistical Software*, 21(7).

2. Hofmeyr, D. P. (2021). Fast Exact Evaluation of Univariate Kernel Sums. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 43(2).

3. Sestelo, M., et al. (2017). npregfast: An R Package for Nonparametric Estimation and Inference in Life Sciences. *Journal of Statistical Software*, 82(12).

4. Loader, C. (1999). *Local Regression and Likelihood*. Springer.

---

## 9. Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.3 | 2026-01-14 | Clarified accuracy interpretation (`M_INTERNAL`, eval modes), improved warmup behavior, and updated plotting layout |
| 2.2 | 2026-01-05 | Enhanced memory monitor: USS support, multi-sample baseline; fixed ks binned mode |
| 2.1 | 2025-12-26 | Added unified benchmark runner with signal-based RSS delta monitoring |
| 2.0 | 2025-12-23 | Unified benchmark suite with all methods organized |
| 1.0 | 2025-12-19 | Initial release |

---

## 10. License

GPL-3.0
