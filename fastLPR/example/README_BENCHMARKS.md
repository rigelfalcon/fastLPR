# Benchmark Scripts Organization

This directory contains reproducible benchmark scripts for JSS manuscript tables and figures.

## File Naming Convention

- **Figures:** `example_<topic>.m` (e.g., `example_kde.m`)
- **Tables:** `benchmark_tab<N>_<description>.m` (e.g., `benchmark_tab5_complete.m`)

Where `<N>` is the table/figure number as it appears in the manuscript.

## Manuscript Figures

| Figure # | Script | Description |
|----------|--------|-------------|
| **Figure 2** | `example_kde.m` | 1D/2D/3D KDE with bandwidth selection |
| **Figure 3** | `example_boundary.m` | Boundary bias comparison |
| **Figure 4** | `example_complex.m` | Complex-valued regression |
| **Figure 5** | `example_hetero.m` | Heteroscedastic regression with CI |
| **Figure 6** | `example_qeeg.m` | Real-world applications (qEEG, MRI) |
| **All** | `reproduce_all_figures.m` | Master script to generate all figures |

## Manuscript Tables

### Section 2: Package Description (Documentation Tables)

| Table # | Label | Description | Script |
|---------|-------|-------------|--------|
| Table 1 | `tab:api` | API reference | *(no benchmark - documentation)* |
| Table 2 | `tab:options` | Options structure | *(no benchmark - documentation)* |
| Table 3 | `tab:complexity` | Computational complexity | *(no benchmark - documentation)* |

### Section 3: Examples

| Table # | Label | Script | Description |
|---------|-------|--------|-------------|
| **Table 4** | `tab:kde-comparison` | `fastLPR_R/examples/benchmark_tab4_kde_comparison.R` | Comparison with ks package for 1D KDE |

### Section 4: Performance

| Table # | Label | Scripts | Description |
|---------|-------|---------|-------------|
| **Table 5** | `tab:performance-memory` | `benchmark_tab5_complete.m` | **Unified:** Speed & memory benchmarks (d=1,2,3) × (N=1k,5k,10k,30k,50k) |
| **Table 6** | `tab:highdim` | `benchmark_tab6_highdim.m` | High-dimensional scaling (d=4,5,6,7) |

## Running All Benchmarks

```matlab
% In MATLAB (from repository root)
cd fastLPR/example

% Run table benchmarks
run benchmark_tab5_complete.m    % Table 5: Performance & Memory (unified)
run benchmark_tab6_highdim.m      % Table 6: High-dimensional

% Or run all figures
run reproduce_all_figures.m
```

```r
# In R (from repository root)
setwd("fastLPR_R/examples")
source("benchmark_tab4_kde_comparison.R")  # Table 4
```

## Output Locations

- **Figures:** Generated in current directory during script execution, saved as PNG files
- **Result data:** `benchmark_tab5_complete_results.mat` (MATLAB binary format)

## Table Organization

**Simplified from previous versions:**

- **Previous:** Separate scripts for speed (`benchmark_tab5_performance_memory.m`) and memory (`benchmark_tab5_memory.m`)
- **Current:** Single unified script (`benchmark_tab5_complete.m`) that measures both metrics together
- **Benefit:** Ensures exact same test case for both speed and memory measurements (no discrepancies)

**Table 5 Structure:**
- Each row shows both speed AND memory for the same (d, N) configuration
- Naive methods tested up to N=50,000 (experimentally verified as maximum feasible size)
- For N>50,000: only fastLPR timings reported (naive infeasible due to >75 GB memory requirement)

**Quick Reference: Table Numbers**

The manuscript has **6 tables total**:
- **Tables 1-3:** Documentation (Section 2) - no benchmarks needed
- **Table 4:** KDE comparison with ks package (Section 3) - R script in `fastLPR_R/examples/`
- **Table 5:** Performance & Memory unified (Section 4) - `benchmark_tab5_complete.m`
- **Table 6:** High-dimensional scaling (Section 4) - `benchmark_tab6_highdim.m`

## Benchmark Details

**Table 5 - Performance & Memory:**
- Test matrix: (d=1,2,3) × (N=1,000; 5,000; 10,000; 30,000; 50,000) = 15 cases with naive comparison
- Additional fastLPR-only cases: (d=1,2,3) × (N=100,000; 1,000,000) = 6 cases
- Naive NW feasibility boundary: N≤50,000 (requires up to 18.6 GB RAM)
- N=100,000 naive: Infeasible (MATLAB blocks 75 GB allocation exceeding 63.9 GB safety limit)
- Each row reports: d, N, Naive_time, fastLPR_time, Speedup, Naive_mem, fastLPR_mem
- Maximum speedup achieved: **551.2×** (d=2, N=50,000)
- Maximum memory reduction: **8,800×** (d=2, N=50,000: 18.6 GB → 2.3 MB)

**Table 6 - High-Dimensional:**
- Test dimensions: d=4, 5, 6, 7
- Sample sizes: N=1,000; 5,000; 10,000
- Only fastLPR timings reported (naive infeasible for d>3)

## Important Notes

- All benchmarks use fixed random seeds for reproducibility (seed=42)
- All table values are **actual measured values**, not theoretical estimates
- Speed benchmarks: Median of 10 runs (or 1 run if first run >5 minutes)
- Memory benchmarks: Measured workspace size using `whos()` command
- Table numbers match manuscript numbering for easy cross-reference
- Results saved to `.mat` files for archival and verification

## Archived Scripts

Obsolete or superseded versions moved to `backup/example_obsolete_2025-11-26/`:
- Old Table 5 versions: `benchmark_tab5_unified.m`, `benchmark_tab5_unified_extended.m`
- Old separate scripts: `benchmark_tab5_performance_memory.m`, `benchmark_tab5_memory.m`
- Test scripts: `benchmark_naive_50k.m`, `benchmark_naive_large.m`
- Old generators: `generate_all_performance_tables.m`, `generate_combined_table.m`
- Exploratory: `compare_kde_methods.m`
