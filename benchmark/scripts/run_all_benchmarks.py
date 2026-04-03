# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Unified Benchmark Runner with External psutil RSS Delta Monitoring

Runs ALL 9 benchmark methods with consistent cross-language memory measurement.

Key Features:
- External psutil RSS monitoring for fair cross-language comparison
- Signal-based baseline capture (isolates algorithm memory from runtime overhead)
- Correct N values: 2^5 to 2^25 (32 to 33,554,432)
- --quick flag means N_RUNS=1 (not smaller N range)

Methods Supported:
  fastLPR Methods:
    - fastKDE (MATLAB/Python/R) - O(N + M log M)
    - fastLPR (MATLAB/Python/R) - O(N + M log M)

  Competitor Methods:
    - ks (R) - KDE, O(N^2) exact / O(M) binned
    - FKSUM (R) - KDE, O(N log N), 1D only
    - locfit (R) - LPR, O(N)
    - npregfast (R) - LPR, O(N), 1D only
    - StOpt-NW (C++) - LPR, O(N + M log M)
    - DirectKDE/DirectNW (MATLAB) - O(N^2) baseline

Usage:
    python benchmark/scripts/run_all_benchmarks.py
    python benchmark/scripts/run_all_benchmarks.py --quick   # N_RUNS=1 instead of 10
    python benchmark/scripts/run_all_benchmarks.py --methods fastlpr ks stopt
    python benchmark/scripts/run_all_benchmarks.py --methods all

Author: Ying Wang, Min Li
Updated: 2025-12-26
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat  # For loading ground truth .mat files

# Add wrappers to path
SCRIPT_DIR = Path(__file__).parent.absolute()
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "benchmark" / "scripts"))

from memory_monitor import run_with_memory_monitor, check_psutil, get_memory_metric_info

# Configuration
OUTPUT_DIR = REPO_ROOT / "benchmark" / "data"
OUTPUT_FILE = OUTPUT_DIR / "benchmark_results.csv"  # JSS standard filename
LOG_FILE = OUTPUT_DIR / "benchmark.log"
GT_DIR = OUTPUT_DIR / "ground_truth"  # Ground truth for accuracy computation

# Parameters matching MATLAB/Python/R scripts
H0 = 0.3  # Base bandwidth constant: h_N = H0 * N^(-1/(d+4))
SEED = 42
NOISE_STD = 0.1

# Sample sizes: 2^5 to 2^25
# Dense sampling for N <= 65536 (Direct methods), sparse for N > 65536
N_VALUES_FULL = (
    [2**k for k in range(5, 17)]  # Dense: 32, 64, ..., 65536 (12 values)
    + [
        2**k for k in range(17, 26, 2)
    ]  # Sparse: 131072, 524288, 2097152, 8388608, 33554432 (5 values)
)  # Total: 17 values (was 21)
N_VALUES_QUICK = [2**k for k in range(5, 21)]  # Quick mode: 32 to 1M (2^20), 16 values
N_VALUES_DIRECT = [
    2**k for k in range(5, 17)
]  # [32, 64, ..., 65536] - max for Direct O(N^2)
DIMENSIONS = [1, 2, 3]
MAX_N_PER_DIM = {1: 2**25, 2: 2**23, 3: 2**21}  # d=1:33M, d=2:8M, d=3:2M

# Default N_RUNS (--quick sets to 1)
N_RUNS_DEFAULT = 3  # User agreed to N_RUNS=3 for all languages
N_RUNS_QUICK = 1
N_RUNS_R = 3  # R methods use same N_RUNS as others

# Internal grid size (M_INTERNAL total points for fair comparison)
# Choose powers of two so per-dim grid sizes are FFT-friendly:
# - d=1: 16384
# - d=2: 128^2 = 16384
# - d=3: 32^3 = 32768
M_INTERNAL_FULL = 16384  # Full mode: 16384 points (d=1/2)
M_INTERNAL_FULL_3D = 32768  # Full mode for 3D: 32768 points (32^3)
M_INTERNAL_QUICK = 1024  # Quick mode: 1024 points for faster testing
M_INTERNAL = M_INTERNAL_FULL  # Default to full mode (set by main() based on --quick)


def get_m_internal(d: int) -> int:
    """Get M_INTERNAL based on dimension.

    In quick mode (M_INTERNAL == M_INTERNAL_QUICK), uses M_INTERNAL_QUICK for all dimensions.
    In full mode, uses M_INTERNAL_FULL_3D (32768) for 3D, M_INTERNAL_FULL (16384) for 1D/2D.
    """
    if M_INTERNAL == M_INTERNAL_QUICK:
        return M_INTERNAL_QUICK
    return M_INTERNAL_FULL_3D if d == 3 else M_INTERNAL_FULL


def get_n_runs_r(n_runs: int) -> int:
    """Get N_RUNS for R methods.

    R methods use fewer runs due to PowerShell memory measurement overhead.
    In quick mode (n_runs=1), returns 1.
    In full mode, returns N_RUNS_R (3) instead of N_RUNS_DEFAULT (10).
    """
    if n_runs == N_RUNS_QUICK:
        return N_RUNS_QUICK
    return N_RUNS_R


# NUFFT accuracy parameter (0 = binning mode, matches ks binned)
NUFFT_ACCURACY = 0

# Paths (configurable via environment variables)
PYTHON_PATH = sys.executable
MATLAB_CMD = os.environ.get("MATLAB_CMD", "matlab")
R_PATH = os.environ.get(
    "R_PATH", r"C:\Software\R\R-4.5.1\bin\x64\Rscript.exe"
)  # Default for Windows


def log(msg):
    """Print and log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(full_msg + "\n")


def safe_unlink(path, retries=3, delay=0.2):
    """Safely delete a file with retry for Windows file locking."""
    import time as time_module

    for i in range(retries):
        try:
            if i > 0:
                time_module.sleep(delay)
            path.unlink(missing_ok=True)
            return True
        except PermissionError:
            if i < retries - 1:
                time_module.sleep(delay * 2)
    return False


def run_subprocess_capture(cmd, description: str, timeout: int):
    """Run subprocess without external memory monitor.

    Used for cases where in-process memory sampling is preferred over the
    parent-process monitor (e.g., native/C++ allocations inside Python).

    Returns a run_with_memory_monitor-compatible dict subset.
    """

    import time as time_module

    start = time_module.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "time_sec": None,
            "mem_mb": 0.0,
            "mem_median": 0.0,
            "mem_min": 0.0,
            "mem_max": 0.0,
            "mem_std": 0.0,
            "mem_method": "none",
            "baseline_mb": 0.0,
            "status": "timeout",
            "stdout": "",
            "stderr": "",
        }

    wall_time = time_module.perf_counter() - start
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""

    # Prefer TIME: from stdout if present
    time_sec = None
    for line in stdout.split("\n"):
        line = line.strip()
        if line.startswith("TIME:"):
            try:
                time_sec = float(line.split(":", 1)[1])
                break
            except ValueError:
                pass

    if time_sec is None:
        time_sec = wall_time

    status = (
        "success"
        if completed.returncode == 0
        else f"error: returncode={completed.returncode}"
    )

    if status != "success":
        log(f"[{description}] status={status}")

    return {
        "time_sec": time_sec,
        "mem_mb": 0.0,
        "mem_median": 0.0,
        "mem_min": 0.0,
        "mem_max": 0.0,
        "mem_std": 0.0,
        "mem_method": "none",
        "baseline_mb": 0.0,
        "status": status,
        "stdout": stdout,
        "stderr": stderr,
    }


def make_eval_grid(N: int, d: int, x_zs: np.ndarray = None) -> np.ndarray:
    """Generate fixed power-of-two evaluation grid (DEPRECATED - use GT file loading).

    WARNING: This function is deprecated. Prefer loading x_grid from GT files
    to ensure exact match with ground truth. Only use as fallback for large N.

    Grid size: power-of-two per-dimension (FFT-friendly)
    - 1D: 16384 points
    - 2D: 128x128 = 16384 points
    - 3D: 32x32x32 = 32768 points


    Args:
        N: (unused) Sample size - kept for API compatibility
        d: Dimension
        x_zs: (N, d) z-scored data to determine grid range

    Returns:
        grid: Shape (M_total, d) array of grid points covering data range with 5% margin
    """
    # Fixed power-of-two per-dimension grid (not N-dependent)
    M_per_dim = {1: 16384, 2: 128, 3: 32}[d]

    if x_zs is not None:
        # Compute range from data (NO margin - stay within interpolatable range)
        x_min = x_zs.min(axis=0)
        x_max = x_zs.max(axis=0)
        # margin removed: causes edge extrapolation errors
    else:
        # Fallback: should not happen in normal use
        x_min = np.array([-3.0] * d)
        x_max = np.array([3.0] * d)

    if d == 1:
        return np.linspace(float(x_min), float(x_max), M_per_dim).reshape(-1, 1)
    else:
        axes = [np.linspace(x_min[i], x_max[i], M_per_dim) for i in range(d)]
        mesh = np.meshgrid(*axes, indexing="ij")
        grid = np.stack(mesh, axis=-1).reshape(
            -1, d, order="F"
        )  # Column-major to match MATLAB/R
        return grid


def get_grid_matlab_code(N: int, d: int) -> str:
    """Generate MATLAB code to create evaluation grid (DEPRECATED - use GT file loading).

    WARNING: This function is deprecated. Prefer loading x_grid from GT files
    to ensure exact match with ground truth. Only use as fallback for large N.

    Returns MATLAB code that creates a matrix `x_grid` with shape (M_total, d).
    Grid size: power-of-two per-dimension (not N-dependent).
    Requires `x_zs` variable to be available in MATLAB context.
    """
    # Fixed power-of-two per-dimension grid
    M_per_dim = {1: 16384, 2: 128, 3: 32}[d]
    return f"""
% Grid mode: generate uniform grid (power-of-two per dimension)
% WARNING: Prefer loading x_grid from GT file for exact match
% NO margin - stay within data range for proper interpolation
M_per_dim = {M_per_dim};
x_min = min(x_zs, [], 1);
x_max = max(x_zs, [], 1);
% margin removed: causes edge extrapolation errors

if {d} == 1
    x_grid = linspace(x_min, x_max, M_per_dim)';
else
    grid_axes = cell(1, {d});
    for i = 1:{d}
        grid_axes{{i}} = linspace(x_min(i), x_max(i), M_per_dim);
    end
    [mesh{{1:{d}}}] = ndgrid(grid_axes{{:}});
    x_grid = zeros(M_per_dim^{d}, {d});
    for i = 1:{d}
        x_grid(:, i) = mesh{{i}}(:);
    end
end
"""


def get_grid_r_code(N: int, d: int, gt_dir: str) -> str:
    """Generate R code to load evaluation grid from GT file for grid mode.

    For N <= 65536: Load x_grid directly from GT file (exact match with ground truth).
    For N > 65536: Generate a power-of-two per-dimension evaluation grid.

    Args:
        N: Sample size
        d: Dimension
        gt_dir: Path to ground truth directory

    Returns:
        R code that creates a matrix `x_grid` with shape (M_total, d).
    """
    # Fixed power-of-two per-dimension grid
    M_per_dim = {1: 16384, 2: 128, 3: 32}[d]

    return f'''
# Grid mode: load x_grid from GT file for exact match with ground truth
gt_grid_file <- file.path("{gt_dir}", sprintf("gt_d%d_N%d_grid.mat", {d}, {N}))
x_grid <- NULL
if (file.exists(gt_grid_file)) {{
    if (!requireNamespace("R.matlab", quietly = TRUE)) {{
        install.packages("R.matlab", repos = "https://cloud.r-project.org", quiet = TRUE)
    }}
    gt_grid <- R.matlab::readMat(gt_grid_file)
    if (!is.null(gt_grid[['x_grid']])) {{
        x_grid <- as.matrix(gt_grid[['x_grid']])
    }} else if (!is.null(gt_grid[['x.grid']])) {{
        x_grid <- as.matrix(gt_grid[['x.grid']])
    }}
    if (!is.null(x_grid)) {{
        cat(sprintf("[R] Using x_grid from GT file: %d x %d\\n", nrow(x_grid), ncol(x_grid)), file = stderr())
    }} else {{
        cat("[R] WARNING: x_grid not found in GT file, falling back to generated grid\\n", file = stderr())
    }}
}}
if (is.null(x_grid)) {{
    # Fallback for large N or missing x_grid: generate grid (power-of-two per dimension)
    # NO margin - stay within data range for proper interpolation
    M_per_dim <- {M_per_dim}
    x_min <- apply(x_zs, 2, min)
    x_max <- apply(x_zs, 2, max)
    # margin removed: causes edge extrapolation errors

    if ({d} == 1) {{
        x_grid <- matrix(seq(x_min, x_max, length.out = M_per_dim), ncol = 1)
    }} else {{
        grid_axes <- lapply(1:{d}, function(i) seq(x_min[i], x_max[i], length.out = M_per_dim))
        grid_list <- expand.grid(grid_axes)
        x_grid <- as.matrix(grid_list)
    }}
    cat(sprintf("[R] Generated grid: %d x %d (no GT file)\\n", nrow(x_grid), ncol(x_grid)), file = stderr())
}}
'''


# Global evaluation mode (set by main)
EVAL_MODE = "grid"  # 'grid' (default) or 'data_point'


# Ground truth cache to avoid repeated file loading
_gt_cache = {}


def load_ground_truth(d: int, N: int, mode: str = "data_point") -> dict | None:
    """Load ground truth data from MATLAB .mat file.

    Args:
        d: Dimension (1, 2, or 3)
        N: Sample size (must be ≤ 65536)
        mode: Evaluation mode ('data_point' or 'grid')

    Returns:
        Dictionary with 'kde_gt', 'nw_gt', and 'x_grid' (grid mode only) arrays,
        or None if file not found
    """
    if N > 65536:
        return None  # Ground truth only exists for N ≤ 65536

    cache_key = (d, N, mode)
    if cache_key in _gt_cache:
        return _gt_cache[cache_key]

    # Select file based on mode
    if mode == "grid":
        gt_file = GT_DIR / f"gt_d{d}_N{N}_grid.mat"
    else:
        gt_file = GT_DIR / f"gt_d{d}_N{N}.mat"

    if not gt_file.exists():
        _gt_cache[cache_key] = None
        return None

    try:
        mat = loadmat(str(gt_file))
        gt_data = {
            "kde_gt": mat["kde_gt"].flatten(),
            "nw_gt": mat["nw_gt"].flatten(),
        }
        # In grid mode, also load x_grid for evaluation
        if mode == "grid" and "x_grid" in mat:
            gt_data["x_grid"] = np.asarray(mat["x_grid"], dtype=np.float64)
        _gt_cache[cache_key] = gt_data
        return gt_data
    except Exception as e:
        log(f"  Warning: Failed to load ground truth {gt_file}: {e}")
        _gt_cache[cache_key] = None
        return None


def compute_accuracy_vs_direct_stats(
    result_file: str | None, task: str, d: int, N: int, mode: str = "data_point"
) -> dict:
    """Compute MSE accuracy against Direct ground truth, plus mask diagnostics.

    Returns a dict with:
      - mse: float (np.nan if not available)
      - acc_mask_applied: bool (True only when grid bounding-box mask is evaluated)
      - acc_mask_ratio: float (n_used / n_total)
      - acc_mask_n_total: int (pre-mask length)
      - acc_mask_n_used: int (post-mask length)

    The mask is only relevant for `mode="grid"`: we compute MSE only on grid points inside
    the `x_zs` bounding box to reduce sensitivity to language-specific extrapolation.
    """

    out = {
        "mse": np.nan,
        "acc_mask_applied": False,
        "acc_mask_ratio": np.nan,
        "acc_mask_n_total": 0,
        "acc_mask_n_used": 0,
    }

    if result_file is None or not os.path.exists(result_file):
        return out

    gt = load_ground_truth(d, N, mode=mode)
    if gt is None:
        return out

    try:
        # Load result based on file extension
        if result_file.endswith(".npy"):
            result_hat = np.load(result_file).flatten()
        elif result_file.endswith(".csv"):
            result_hat = np.loadtxt(result_file).flatten()
        elif result_file.endswith(".mat"):
            mat = loadmat(result_file)
            result_hat = mat["result_vec"].flatten()
        elif result_file.endswith(".rds"):
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as tmp:
                tmp_csv = tmp.name
            r_cmd = (
                f'x <- readRDS("{result_file.replace(chr(92), "/")}"); '
                f'write.csv(x, "{tmp_csv.replace(chr(92), "/")}", row.names=FALSE)'
            )
            subprocess.run([R_PATH, "-e", r_cmd], capture_output=True, check=True)
            result_hat = np.loadtxt(tmp_csv, delimiter=",", skiprows=1).flatten()
            os.unlink(tmp_csv)
        else:
            log(f"  Warning: Unknown result file format: {result_file}")
            return out

        gt_key = "kde_gt" if task == "KDE" else "nw_gt"
        gt_vec = gt[gt_key]

        # Validate sizes match (grid mode has fixed GT vector length)
        if len(result_hat) != len(gt_vec):
            return out

        out["acc_mask_n_total"] = int(len(gt_vec))
        out["acc_mask_n_used"] = int(len(gt_vec))
        out["acc_mask_ratio"] = 1.0

        # In grid mode, exclude extrapolation region (x_grid outside x_zs bounds)
        if mode == "grid" and "x_grid" in gt:
            try:
                gt_data_file = GT_DIR / f"gt_d{d}_N{N}.mat"
                if gt_data_file.exists():
                    gt_data_mat = loadmat(str(gt_data_file))
                    x_zs = np.asarray(gt_data_mat["x_zs"], dtype=np.float64)
                    x_grid = np.asarray(gt["x_grid"], dtype=np.float64)
                    x_min = x_zs.min(axis=0)
                    x_max = x_zs.max(axis=0)
                    inside = np.logical_and(x_grid >= x_min, x_grid <= x_max).all(
                        axis=1
                    )
                    n_used = int(inside.sum())
                    out["acc_mask_n_used"] = n_used
                    out["acc_mask_ratio"] = float(n_used / max(1, len(gt_vec)))
                    if n_used == 0:
                        return out
                    result_hat = result_hat[inside]
                    gt_vec = gt_vec[inside]
                    out["acc_mask_applied"] = True
            except Exception:
                # Mask diagnostics are best-effort; keep MSE as full-grid if masking fails.
                out["acc_mask_applied"] = False
                out["acc_mask_ratio"] = 1.0
                out["acc_mask_n_used"] = int(len(gt_vec))

        out["mse"] = float(np.nanmean((result_hat - gt_vec) ** 2))
        return out
    except Exception as e:
        log(f"  Warning: Failed to compute accuracy: {e}")
        return out
    finally:
        # Clean up temp file
        if result_file and os.path.exists(result_file):
            try:
                os.remove(result_file)
            except OSError:
                pass


def compute_accuracy_vs_direct(
    result_file: str | None, task: str, d: int, N: int, mode: str = "data_point"
) -> float:
    """Compute MSE accuracy against Direct ground truth."""

    return float(
        compute_accuracy_vs_direct_stats(result_file, task, d, N, mode=mode).get(
            "mse", np.nan
        )
    )


# ==============================================================================
# PYTHON BENCHMARKS
# ==============================================================================


def run_python_single(N, d, task="KDE", n_runs=1):
    """Run single Python benchmark with READY signal for RSS baseline."""
    # Grid mode: load x_grid from GT file for exact match
    grid_code = ""
    eval_points = "x_zs"
    if EVAL_MODE == "grid":
        grid_code = f"""
# Grid mode: load x_grid from GT file for exact match with ground truth
gt_grid_file = os.path.join(GT_DIR, f'gt_d{{d}}_N{{N}}_grid.mat')
if os.path.exists(gt_grid_file):
    gt_grid = loadmat(gt_grid_file)
    x_grid = np.asarray(gt_grid['x_grid'], dtype=np.float64)
    print(f"[Python] Using x_grid from GT file: {{x_grid.shape}}", file=sys.stderr)
else:
    # Fallback for large N: generate grid (power-of-two per dimension, FFT-friendly)
    # NO margin - stay within data range for proper interpolation
    M_per_dim = {{1: 16384, 2: 128, 3: 32}}[d]
    x_min = x_zs.min(axis=0)
    x_max = x_zs.max(axis=0)
    # margin removed: causes edge extrapolation errors
    if d == 1:
        x_grid = np.linspace(x_min[0], x_max[0], M_per_dim).reshape(-1, 1)
    else:
        grid_axes = [np.linspace(x_min[i], x_max[i], M_per_dim) for i in range(d)]
        mesh = np.meshgrid(*grid_axes, indexing='ij')
        x_grid = np.stack(mesh, axis=-1).reshape(-1, d, order='F')
    print(f"[Python] Generated x_grid (no GT file): {{x_grid.shape}}", file=sys.stderr)
"""
        eval_points = "x_grid"

    # Get M_INTERNAL based on dimension (d=1/2: 16384, d=3: 32768)
    m_internal = get_m_internal(d)

    script = f'''
import sys
import time
import gc
import tracemalloc
import tempfile
import os
import numpy as np

# Print backend info at startup
from scipy import fft
try:
    import multiprocessing
    n_cores = multiprocessing.cpu_count()
except:
    n_cores = 1

# Check if pyfftw is available
try:
    import pyfftw
    fft_backend = "pyfftw (multi-threaded)"
except ImportError:
    fft_backend = "scipy.fft (single-threaded)"

# Note: scipy.fft.set_workers only affects concurrent FFT jobs, not individual FFT parallelization
# Individual FFT is single-threaded in scipy unless using pyfftw/MKL
print(f"[Python] FFT backend: {{fft_backend}}, CPU cores: {{n_cores}}", file=sys.stderr)

sys.path.insert(0, r"{REPO_ROOT / "fastLPR_py" / "src"}")
from fastlpr import cv_fastkde, cv_fastlpr
from scipy.io import loadmat

N = {N}
d = {d}
SEED = {SEED}
H0 = {H0}
NOISE_STD = {NOISE_STD}
N_RUNS = {n_runs}
GT_DIR = r"{GT_DIR}"

# Try to load ground truth data for fair cross-language comparison
# Ground truth files contain MATLAB-generated x_zs, y, h_N
gt_file = os.path.join(GT_DIR, f'gt_d{{d}}_N{{N}}.mat')
if os.path.exists(gt_file):
    gt = loadmat(gt_file)
    x_zs = np.asarray(gt['x_zs'], dtype=np.float64)
    h_N = float(gt['h_N'].flatten()[0])
    if "{task}" != "KDE":
        y = np.asarray(gt['y'], dtype=np.float64)
    print(f"[Python] Using ground truth data from {{gt_file}}", file=sys.stderr)
else:
    # Fallback: generate data (for N > 65536 where no ground truth exists)
    np.random.seed(SEED)
    x_orig = np.random.rand(N, d)
    x_mean = np.mean(x_orig, axis=0)
    x_std = np.std(x_orig, axis=0, ddof=0)
    x_zs = (x_orig - x_mean) / x_std
    h_N = H0 * N ** (-1 / (d + 4))
    if "{task}" != "KDE":
        y_true = np.sin(2 * np.pi * x_orig) if d == 1 else np.sin(2 * np.pi * np.mean(x_orig, axis=1, keepdims=True))
        y = y_true + NOISE_STD * np.random.randn(N, 1)
    print(f"[Python] No ground truth, generated data with seed={{SEED}}", file=sys.stderr)

h = np.array([h_N] * d)
# Fixed M for fair cross-method comparison (same as evaluation grid)
# 3D uses 100000 for better accuracy (22^3=10648 too coarse, 46^3=97336 better)
M_INTERNAL = {m_internal}
grid_size = int(np.ceil(M_INTERNAL ** (1 / d)))
# interp_method='linear' for O(N) complexity and fair cross-language comparison
opt = {{"order": 0, "calc_dof": False, "N": grid_size, "interp_method": "linear", "accuracy": {NUFFT_ACCURACY}}}

{grid_code}

gc.collect()

# Warmup with SMALL data (N=32) - trigger JIT without allocating full buffers
# This ensures actual run allocates new memory that psutil can detect
warmup_n = 32
warmup_x = np.random.rand(warmup_n, d)
warmup_x = (warmup_x - warmup_x.mean(axis=0)) / warmup_x.std(axis=0, ddof=0)
warmup_h = np.array([{H0} * warmup_n ** (-1 / (d + 4))] * d)
warmup_opt = {{"order": 0, "calc_dof": False, "N": warmup_n, "interp_method": "linear", "accuracy": {NUFFT_ACCURACY}}}
if "{task}" == "KDE":
    _ = cv_fastkde(warmup_x, warmup_h, warmup_opt)
else:
    warmup_y = np.sin(2 * np.pi * warmup_x) if d == 1 else np.sin(2 * np.pi * np.mean(warmup_x, axis=1, keepdims=True))
    warmup_y = warmup_y + {NOISE_STD} * np.random.randn(warmup_n, 1)
    _ = cv_fastlpr(warmup_x, warmup_y, warmup_h, warmup_opt)
del warmup_x, warmup_h, warmup_opt
if "{task}" != "KDE":
    del warmup_y

# Double gc + sleep to stabilize memory before baseline capture
gc.collect()
time.sleep(0.1)
gc.collect()

print("READY", flush=True)  # Signal: baseline can be captured now
time.sleep(0.05)  # Allow parent to capture baseline before algorithm starts

times = []
result = None
for run in range(N_RUNS):
    print("RUN_START", flush=True)  # Signal: per-run memory baseline
    t0 = time.perf_counter()
    if "{task}" == "KDE":
        result = cv_fastkde(x_zs, h, opt)
    else:
        result = cv_fastlpr(x_zs, y, h, opt)
    times.append(time.perf_counter() - t0)
    print("RUN_END", flush=True)  # Signal: per-run memory peak recorded

print(f"TIME:{{np.median(times)}}")
print(f"TIME_MIN:{{np.min(times)}}")
print(f"TIME_MAX:{{np.max(times)}}")
print(f"TIME_STD:{{np.std(times)}}")

# Save result for accuracy computation (only for N <= 65536)
if N <= 65536 and result is not None:
    # KDE uses .fpp (density), LPR uses .fpp_yhat (fitted values)
    if "{task}" == "KDE":
        result_vec = result.fpp({eval_points}).flatten()
    else:
        result_vec = result.fpp_yhat({eval_points}).flatten()
    result_file = tempfile.mktemp(suffix=".npy", prefix="bench_py_")
    np.save(result_file, result_vec)
    print(f"RESULT_FILE:{{result_file}}")
'''
    result = run_with_memory_monitor(
        [PYTHON_PATH, "-c", script], f"Python {task} d={d} N={N}", timeout=600
    )
    return _parse_result(result, f"Python {task} d={d} N={N}")


# ==============================================================================
# PYTHON BENCHMARKS (Batched Execution to Avoid Startup Overhead)
# ==============================================================================

# Cache for batched Python results
_PYTHON_BATCH_RESULTS = {}


def run_python_batch(test_list, n_runs=1):
    """Run ALL Python benchmarks in a single Python process.

    Args:
        test_list: List of (N, d, task) tuples to run
        n_runs: Number of runs per test

    Returns:
        Dictionary keyed by (N, d, task) with results
    """
    global _PYTHON_BATCH_RESULTS

    if not test_list:
        return {}

    temp_dir = str(OUTPUT_DIR).replace("\\", "/")
    gt_dir = str(GT_DIR).replace("\\", "/")

    # Build test list for Python
    test_items = [f"    ({N}, {d}, '{task}')" for N, d, task in test_list]
    test_list_str = "tests = [\n" + ",\n".join(test_items) + "\n]"

    # Grid mode
    grid_mode = EVAL_MODE == "grid"
    eval_points_var = "x_grid" if grid_mode else "x_zs"

    script = f'''
import sys
import time
import gc
import os
import threading
import numpy as np
from scipy.io import loadmat

# Print backend info at startup
try:
    import multiprocessing
    n_cores = multiprocessing.cpu_count()
except:
    n_cores = 1

try:
    import pyfftw
    fft_backend = "pyfftw (multi-threaded)"
except ImportError:
    fft_backend = "scipy.fft (single-threaded)"

print(f"[Python Batch] FFT backend: {{fft_backend}}, CPU cores: {{n_cores}}", file=sys.stderr)

sys.path.insert(0, r"{REPO_ROOT / "fastLPR_py" / "src"}")
from fastlpr import cv_fastkde, cv_fastlpr

# Test parameters
SEED = {SEED}
H0 = {H0}
NOISE_STD = {NOISE_STD}
N_RUNS = {n_runs}
temp_dir = r"{temp_dir}"
gt_dir = r"{gt_dir}"
GRID_MODE = {grid_mode}
NUFFT_ACCURACY = {NUFFT_ACCURACY}
M_INTERNAL_FULL = {M_INTERNAL_FULL}
M_INTERNAL_FULL_3D = {M_INTERNAL_FULL_3D}
M_INTERNAL_QUICK = {M_INTERNAL_QUICK}
M_INTERNAL_MODE = {M_INTERNAL}

{test_list_str}
num_tests = len(tests)

# Warmup with small dataset for ALL dimensions (avoid cold-start humps)
np.random.seed(SEED)
for d_warmup in (1, 2, 3):
    x_small = np.random.rand(100, d_warmup)
    x_small_zs = (x_small - x_small.mean(axis=0)) / x_small.std(axis=0, ddof=0)
    y_small = (
        np.sin(2 * np.pi * x_small)
        if d_warmup == 1
        else np.sin(2 * np.pi * x_small.mean(axis=1, keepdims=True))
    ) + NOISE_STD * np.random.randn(100, 1)
    h_small = np.array([0.3] * d_warmup)
    opt_small = {{
        "order": 0,
        "calc_dof": False,
        "N": 20,
        "interp_method": "linear",
        "accuracy": NUFFT_ACCURACY,
    }}
    try:
        cv_fastkde(x_small_zs, h_small, opt_small)
        cv_fastlpr(x_small_zs, y_small, h_small, opt_small)
    except Exception:
        pass
    del x_small, x_small_zs, y_small, h_small, opt_small

# Stabilize memory
gc.collect()
time.sleep(0.1)
gc.collect()

print("READY", flush=True)
time.sleep(0.5)

# Run all tests
for t_idx, (N, d, task) in enumerate(tests):
    # Load or generate data
    gt_file = os.path.join(gt_dir, f'gt_d{{d}}_N{{N}}.mat')
    if os.path.exists(gt_file):
        gt = loadmat(gt_file)
        x_zs = np.asarray(gt['x_zs'], dtype=np.float64)
        h_N = float(gt['h_N'].flatten()[0])
        if task != "KDE":
            y = np.asarray(gt['y'], dtype=np.float64)
        print(f"[Python] Using GT: {{gt_file}}", file=sys.stderr)
    else:
        np.random.seed(SEED)
        x_orig = np.random.rand(N, d)
        x_mean = np.mean(x_orig, axis=0)
        x_std = np.std(x_orig, axis=0, ddof=0)
        x_zs = (x_orig - x_mean) / x_std
        h_N = H0 * N ** (-1 / (d + 4))
        if task != "KDE":
            y_true = np.sin(2 * np.pi * x_orig) if d == 1 else np.sin(2 * np.pi * np.mean(x_orig, axis=1, keepdims=True))
            y = y_true + NOISE_STD * np.random.randn(N, 1)
        print(f"[Python] Generated data N={{N}}", file=sys.stderr)

    # Setup
    h = np.array([h_N] * d)
    # M_INTERNAL: 100000 for 3D (better accuracy), 10000 for 1D/2D
    if M_INTERNAL_MODE == M_INTERNAL_QUICK:
        M_INTERNAL = M_INTERNAL_QUICK
    else:
        M_INTERNAL = M_INTERNAL_FULL_3D if d == 3 else M_INTERNAL_FULL
    grid_size = int(np.ceil(M_INTERNAL ** (1 / d)))
    opt = {{"order": 0, "calc_dof": False, "N": grid_size, "interp_method": "linear", "accuracy": NUFFT_ACCURACY}}

    # Grid mode: load x_grid from GT file
    x_grid = None
    if GRID_MODE:
        gt_grid_file = os.path.join(gt_dir, f'gt_d{{d}}_N{{N}}_grid.mat')
        if os.path.exists(gt_grid_file):
            gt_grid = loadmat(gt_grid_file)
            x_grid = np.asarray(gt_grid['x_grid'], dtype=np.float64)
            print(f"[Python] Using x_grid from GT file: {{x_grid.shape}}", file=sys.stderr)
        else:
            # Fallback: generate grid (power-of-two per dimension, FFT-friendly)
            M_per_dim = {{1: 16384, 2: 128, 3: 32}}[d]
            x_min = x_zs.min(axis=0)
            x_max = x_zs.max(axis=0)
            if d == 1:
                x_grid = np.linspace(x_min[0], x_max[0], M_per_dim).reshape(-1, 1)
            else:
                grid_axes = [np.linspace(x_min[i], x_max[i], M_per_dim) for i in range(d)]
                mesh = np.meshgrid(*grid_axes, indexing='ij')
                x_grid = np.stack(mesh, axis=-1).reshape(-1, d, order='F')
            print(f"[Python] Using fallback grid: {{x_grid.shape}}", file=sys.stderr)
    eval_pts = x_grid if GRID_MODE else x_zs

    # Per-test cleanup before timing
    gc.collect()

    # Timed runs - internal process RSS peak delta (includes NumPy/native memory)
    import psutil

    process = psutil.Process(os.getpid())

    def rss_mb() -> float:
        return float(process.memory_info().rss) / (1024 * 1024)

    def baseline_mb(samples: int = 5, interval: float = 0.01) -> float:
        vals = []
        for _ in range(samples):
            vals.append(rss_mb())
            time.sleep(interval)
        return max(vals) if vals else rss_mb()

    times = []
    mem_deltas = []  # Per-run RSS peak deltas in MB
    result = None
    n_runs_test = N_RUNS if N_RUNS > 1 else (5 if N <= 64 else 1)
    for run in range(n_runs_test):
        gc.collect()
        base = baseline_mb()
        peak = [base]
        stop_evt = threading.Event()

        def poll():
            while not stop_evt.is_set():
                v = rss_mb()
                if v > peak[0]:
                    peak[0] = v
                time.sleep(0.001)

        t_poll = threading.Thread(target=poll, daemon=True)
        t_poll.start()

        print("RUN_START", flush=True)
        t0 = time.perf_counter()
        if task == "KDE":
            result = cv_fastkde(x_zs, h, opt)
        else:
            result = cv_fastlpr(x_zs, y, h, opt)
        times.append(time.perf_counter() - t0)

        stop_evt.set()
        t_poll.join(timeout=1)

        mem_deltas.append(max(0.0, peak[0] - base))
        print("RUN_END", flush=True)

    # Output timing
    print(f"TIME:{{np.median(times)}}")
    print(f"TIME_MIN:{{np.min(times)}}")
    print(f"TIME_MAX:{{np.max(times)}}")
    print(f"TIME_STD:{{np.std(times)}}")

    # Output RSS peak delta statistics
    if mem_deltas:
        print(f"MEM_PYTHON:{{float(np.median(mem_deltas))}}")
        print(f"MEM_PYTHON_MIN:{{float(np.min(mem_deltas))}}")
        print(f"MEM_PYTHON_MAX:{{float(np.max(mem_deltas))}}")
        print(f"MEM_PYTHON_STD:{{float(np.std(mem_deltas))}}")


    # Save result for accuracy computation (only for N <= 65536)
    if N <= 65536 and result is not None:
        if task == "KDE":
            result_vec = result.fpp(eval_pts).flatten()
        else:
            result_vec = result.fpp_yhat(eval_pts).flatten()
        result_file = os.path.join(temp_dir, f"py_result_d{{d}}_N{{N}}_{{task}}.npy")
        np.save(result_file, result_vec)
        print(f"RESULT_FILE:{{result_file}}")

    print(f"TEST_END:{{N}}_{{d}}_{{task}}", flush=True)

    # Cleanup
    del x_zs, h, opt, times, result
    if task != "KDE":
        del y
    if x_grid is not None:
        del x_grid
    gc.collect()
'''

    # Write and execute
    temp_script = OUTPUT_DIR / "temp_python_batch.py"
    temp_script.parent.mkdir(parents=True, exist_ok=True)
    temp_script.write_text(script)

    log(f"[Python Batch] Running {len(test_list)} tests in single Python process...")
    result = run_subprocess_capture(
        [PYTHON_PATH, str(temp_script)],
        f"Python Batch ({len(test_list)} tests)",
        timeout=600 * len(test_list),  # Scale timeout with number of tests
    )
    safe_unlink(temp_script)

    # Parse results
    if result["status"] != "success":
        log(f"[Python Batch] FAILED: {result.get('error', 'Unknown error')}")
        return {}

    # Extract per-test results from stdout
    # Memory is distributed using run_peaks (each test has n_runs peaks)
    results_dict = {}
    lines = result.get("stdout", "").split("\n")
    run_peaks = result.get("run_peaks", [])
    test_index = 0
    current_result = {}

    for line in lines:
        if line.startswith("TIME:"):
            current_result["time_sec"] = float(line.split(":")[1])
        elif line.startswith("TIME_MIN:"):
            current_result["time_min"] = float(line.split(":")[1])
        elif line.startswith("TIME_MAX:"):
            current_result["time_max"] = float(line.split(":")[1])
        elif line.startswith("TIME_STD:"):
            val = line.split(":")[1]
            current_result["time_std"] = (
                0.0 if val.upper() in ("NA", "NAN", "INF") else float(val)
            )
        elif line.startswith("MEM_PYTHON:"):
            current_result["mem_python"] = float(line.split(":")[1])
        elif line.startswith("MEM_PYTHON_MIN:"):
            current_result["mem_python_min"] = float(line.split(":")[1])
        elif line.startswith("MEM_PYTHON_MAX:"):
            current_result["mem_python_max"] = float(line.split(":")[1])
        elif line.startswith("MEM_PYTHON_STD:"):
            current_result["mem_python_std"] = float(line.split(":")[1])
        elif line.startswith("RESULT_FILE:"):
            current_result["result_file"] = line.split(":", 1)[1].strip()
        elif line.startswith("TEST_END:"):
            if test_index < len(test_list):
                N, d, task = test_list[test_index]
                current_result["status"] = "success"

                mem_python = current_result.get("mem_python")
                mem_python_min = current_result.get("mem_python_min")
                mem_python_max = current_result.get("mem_python_max")
                mem_python_std = current_result.get("mem_python_std")

                if mem_python is not None:
                    final_mem = mem_python
                    final_mem_min = (
                        mem_python_min if mem_python_min is not None else mem_python
                    )
                    final_mem_max = (
                        mem_python_max if mem_python_max is not None else mem_python
                    )
                    final_mem_median = mem_python
                    final_mem_std = (
                        mem_python_std if mem_python_std is not None else 0.0
                    )
                    mem_method = "psutil_rss_internal"
                else:
                    final_mem = 0.0
                    final_mem_min = 0.0
                    final_mem_max = 0.0
                    final_mem_median = 0.0
                    final_mem_std = 0.0
                    mem_method = "none"

                current_result["mem_mb"] = final_mem
                current_result["mem_median"] = final_mem_median
                current_result["mem_min"] = final_mem_min
                current_result["mem_max"] = final_mem_max
                current_result["mem_std"] = final_mem_std
                current_result["mem_method"] = mem_method
                current_result["baseline_mb"] = result.get("baseline_mb", 0)

                results_dict[(N, d, task)] = current_result.copy()
                current_result = {}
                test_index += 1

    log(f"[Python Batch] Completed {len(results_dict)}/{len(test_list)} tests")
    return results_dict


def run_python_single_cached(N, d, task="KDE", n_runs=1):
    """Run single Python benchmark using batch mode (avoids startup overhead).

    Uses cached batch results if available, otherwise triggers batch execution.
    """
    global _PYTHON_BATCH_RESULTS

    # Check if batch results are cached
    key = (N, d, task)
    if key in _PYTHON_BATCH_RESULTS:
        result = _PYTHON_BATCH_RESULTS[key]
    else:
        # Batch not run yet - run single test in batch mode
        results = run_python_batch([key], n_runs)
        _PYTHON_BATCH_RESULTS.update(results)
        result = results.get(
            key,
            {
                "status": "error",
                "error": "Batch execution failed",
                "time_sec": 0.0,
                "mem_mb": 0.0,
            },
        )

    # Log result in same format as R/MATLAB
    description = f"Python {task} d={d} N={N}"
    if result.get("status") == "success":
        time_sec = result.get("time_sec", 0.0)
        time_min = result.get("time_min", time_sec)
        time_max = result.get("time_max", time_sec)
        mem_mb = result.get("mem_mb", 0.0)
        mem_min = result.get("mem_min", mem_mb)
        mem_max = result.get("mem_max", mem_mb)
        mem_method = result.get("mem_method", "none")
        log(
            f"  {description}: {time_sec:.2f}s [{time_min:.2f}-{time_max:.2f}], {mem_mb:.1f}MB [{mem_min:.1f}-{mem_max:.1f}] ({mem_method})"
        )
    else:
        log(f"  {description}: {result.get('status', 'error')}")

    return result


# ==============================================================================
# R BENCHMARKS (Batched Execution to Avoid Startup Overhead)
# ==============================================================================

# Cache for batched R results
_R_BATCH_RESULTS = {}


def run_r_batch(test_list, n_runs=1):
    """Run ALL R benchmarks in a single R process.

    Args:
        test_list: List of (N, d, task) tuples to run
        n_runs: Number of runs per test

    Returns:
        Dictionary keyed by (N, d, task) with results
    """
    global _R_BATCH_RESULTS

    if not test_list:
        return {}

    fastlpr_r_dir = str(REPO_ROOT / "fastLPR_R").replace("\\", "/")
    temp_dir = str(OUTPUT_DIR).replace("\\", "/")
    gt_dir = str(GT_DIR).replace("\\", "/")

    # Build test list for R (no trailing comma - R doesn't allow it)
    test_items = [f"  list(N={N}, d={d}, task='{task}')" for N, d, task in test_list]
    test_list_str = "tests <- list(\n" + ",\n".join(test_items) + "\n)"

    # Grid mode
    grid_mode = EVAL_MODE == "grid"
    eval_points_var = "x_grid" if grid_mode else "x_zs"

    # Part 1: Setup and warmup
    # NOTE: Do NOT disable OpenMP for R - Rcpp/RcppArmadillo relies on OpenMP for
    # matrix operations. Disabling it causes 10x slowdown at large N. (2026-01-09 fix)
    r_code_part1 = f'''
# Load fastLPR package
setwd("{fastlpr_r_dir}")
source("setup.R")

# Verify Rcpp
if (!rcpp_available()) {{
    stop("FATAL: Rcpp acceleration not available")
}}

# Test parameters
SEED <- {SEED}
H0 <- {H0}
NOISE_STD <- {NOISE_STD}
N_RUNS <- {n_runs}
temp_dir <- "{temp_dir}"
gt_dir <- "{gt_dir}"
GRID_MODE <- {"TRUE" if grid_mode else "FALSE"}

{test_list_str}
num_tests <- length(tests)

# Warmup with small dataset for ALL dimensions to trigger JIT
set.seed(SEED)
for (d_warmup in 1:3) {{
    x_small <- matrix(runif(100 * d_warmup), nrow=100, ncol=d_warmup)
    x_small_zs <- scale(x_small)
    y_small <- matrix(sin(2*pi*rowMeans(x_small)) + NOISE_STD*rnorm(100), ncol=1)
    h_small <- matrix(0.3, nrow=1, ncol=d_warmup)
    opt_small <- list(order=0, accuracy={NUFFT_ACCURACY}, calc_dof=FALSE, N=20)
    tryCatch({{
        cv_fastkde(x_small_zs, h_small, opt_small)
        cv_fastlpr(x_small_zs, y_small, h_small, opt_small)
    }}, error=function(e) NULL)
}}
rm(x_small, x_small_zs, y_small, h_small, opt_small, d_warmup)

# Stabilize memory
invisible(gc(reset=TRUE, full=TRUE))
Sys.sleep(0.1)

cat("READY\\n")
flush(stdout())
Sys.sleep(0.5)

# Run all tests
'''

    # Continue with part 2...
    return _run_r_batch_continue(
        test_list, r_code_part1, temp_dir, gt_dir, grid_mode, eval_points_var, n_runs
    )


def _run_r_batch_continue(
    test_list, r_code_part1, temp_dir, gt_dir, grid_mode, eval_points_var, n_runs
):
    """Continue R batch execution (split for readability)."""
    # Part 2: Main loop
    r_code_part2 = (
        """
for (t in 1:num_tests) {
    N <- tests[[t]]$N
    d <- tests[[t]]$d
    task <- tests[[t]]$task

    # Load or generate data
    gt_file <- file.path(gt_dir, sprintf("gt_d%d_N%d.mat", d, N))
    if (file.exists(gt_file)) {
        if (!requireNamespace("R.matlab", quietly=TRUE)) {
            install.packages("R.matlab", repos="https://cloud.r-project.org", quiet=TRUE)
        }
        mat <- R.matlab::readMat(gt_file)
        x_orig <- as.matrix(mat[['x.orig']])
        x_zs <- as.matrix(mat[['x.zs']])
        h_N <- as.numeric(mat[['h.N']])
        if (task == "LPR") {
            y <- as.matrix(mat[['y']])
        }
        cat(sprintf("[R] Using GT: %s\\n", gt_file), file=stderr())
    } else {
        set.seed(SEED)
        x_orig <- matrix(runif(N * d), nrow=N, ncol=d)
        x_mean <- colMeans(x_orig)
        x_std <- apply(x_orig, 2, function(col) sd(col)*sqrt((length(col)-1)/length(col)))
        x_zs <- scale(x_orig, center=x_mean, scale=x_std)
        h_N <- H0 * N^(-1/(d+4))
        if (task == "LPR") {
            y_true <- if(d==1) sin(2*pi*x_orig) else matrix(sin(2*pi*rowMeans(x_orig)), ncol=1)
            y <- y_true + NOISE_STD * rnorm(N)
        }
        cat(sprintf("[R] Generated data N=%d\\n", N), file=stderr())
    }

    # Setup
    h <- matrix(rep(h_N, d), nrow=1)
    # M_INTERNAL: 100000 for 3D (better accuracy), 10000 for 1D/2D
    M_INTERNAL <- if (d == 3) """
        + str(M_INTERNAL_FULL_3D)
        + """ else """
        + str(M_INTERNAL_FULL)
        + """
    # In quick mode, use smaller M_INTERNAL
    if ("""
        + str(M_INTERNAL)
        + """ == """
        + str(M_INTERNAL_QUICK)
        + """) M_INTERNAL <- """
        + str(M_INTERNAL_QUICK)
        + """
    grid_size <- ceiling(M_INTERNAL^(1/d))
    opt <- list(order=0, calc_dof=FALSE, N=grid_size, accuracy="""
        + str(NUFFT_ACCURACY)
        + """)
"""
    )

    # Continue with part 3...
    return _run_r_batch_final(
        test_list,
        r_code_part1 + r_code_part2,
        temp_dir,
        gt_dir,
        grid_mode,
        eval_points_var,
        n_runs,
    )


def _run_r_batch_final(
    test_list, r_code_prefix, temp_dir, gt_dir, grid_mode, eval_points_var, n_runs
):
    """Final part of R batch execution."""
    # Part 3: Grid generation and timing
    grid_code = ""
    if grid_mode:
        grid_code = """
    # Grid mode: load x_grid from GT grid file (NOT data_point file)
    # The grid file is gt_d%d_N%d_grid.mat, separate from the data_point file
    x_grid <- NULL
    gt_grid_file <- file.path(gt_dir, sprintf("gt_d%d_N%d_grid.mat", d, N))
    if (file.exists(gt_grid_file)) {
        gt_grid <- R.matlab::readMat(gt_grid_file)
        # R.matlab converts underscores to dots (x_grid -> x.grid)
        if (!is.null(gt_grid[['x.grid']])) {
            x_grid <- as.matrix(gt_grid[['x.grid']])
            cat(sprintf("[R] Using x_grid from GT grid file: %d x %d\\n", nrow(x_grid), ncol(x_grid)), file=stderr())
        }
    }
    if (is.null(x_grid)) {
        # Fallback: generate grid dynamically
        grid_1d <- seq(-3, 3, length.out=ceiling(M_INTERNAL^(1/d)))
        if (d == 1) {
            x_grid <- matrix(grid_1d, ncol=1)
        } else if (d == 2) {
            grid_expand <- expand.grid(grid_1d, grid_1d)
            x_grid <- as.matrix(grid_expand)
        } else {
            grid_expand <- expand.grid(grid_1d, grid_1d, grid_1d)
            x_grid <- as.matrix(grid_expand)
        }
        cat(sprintf("[R] Using fallback grid: %d x %d\\n", nrow(x_grid), ncol(x_grid)), file=stderr())
    }
"""

    r_code_part3 = (
        grid_code
        + f"""

    # Helper function to get R memory usage in MB using gc()
    # gc() tracks memory managed by R's memory manager (more accurate than RSS)
    get_r_mem_mb <- function() {{
        gc_info <- gc(verbose=FALSE)
        # gc() returns matrix: [Ncells, Vcells] x [used, gc trigger, max used]
        # Column 2 is "used (Mb)" - sum of Ncells and Vcells
        return(sum(gc_info[, 2]))
    }}

    # Timing loop - each run has its own RUN_START/RUN_END for memory tracking
    invisible(gc(reset=TRUE, full=TRUE))

    n_runs_test <- if (N_RUNS > 1) N_RUNS else if (N <= 64) 5 else 1

    times <- numeric(n_runs_test)
    mem_deltas <- numeric(n_runs_test)  # Internal R memory delta per run
    result <- NULL
    for (run in 1:n_runs_test) {{
        invisible(gc(reset=TRUE, full=TRUE))
        mem_before <- get_r_mem_mb()
        cat("RUN_START\\n"); flush(stdout())
        t0 <- Sys.time()
        if (task == "KDE") {{
            result <- cv_fastkde(x_zs, h, opt)
        }} else {{
            result <- cv_fastlpr(x_zs, y, h, opt)
        }}
        times[run] <- as.numeric(Sys.time() - t0, units="secs")
        mem_after <- get_r_mem_mb()
        mem_deltas[run] <- max(0, mem_after - mem_before)
        cat("RUN_END\\n"); flush(stdout())
    }}


    # Save results (fix stale result risk: delete old file first, track save success)
    result_file <- file.path(temp_dir, sprintf("r_result_d%d_N%d_%s.rds", d, N, task))
    result_saved <- FALSE
    if (file.exists(result_file)) {{
        unlink(result_file)  # Remove stale file before saving
    }}
    if (N <= 65536 && !is.null(result)) {{
        if (task == "KDE") {{
            result_vec <- as.vector(result$fpp$evaluate({eval_points_var}))
        }} else {{
            if (is.function(result$fpp_yhat)) {{
                result_vec <- as.vector(result$fpp_yhat({eval_points_var}))
            }} else if (is.list(result$fpp_yhat) && !is.null(result$fpp_yhat$evaluate)) {{
                result_vec <- as.vector(result$fpp_yhat$evaluate({eval_points_var}))
            }} else {{
                result_vec <- NULL
            }}
        }}
        if (!is.null(result_vec)) {{
            saveRDS(result_vec, result_file)
            result_saved <- TRUE
        }}
    }}

    # Output timing
    cat(sprintf("TIME:%f\\n", median(times)))
    cat(sprintf("TIME_MIN:%f\\n", min(times)))
    cat(sprintf("TIME_MAX:%f\\n", max(times)))
    cat(sprintf("TIME_STD:%f\\n", if(length(times)>1) sd(times) else 0))

    # Output internal memory measurement (RSS delta)
    cat(sprintf("MEM_R:%f\\n", max(mem_deltas)))
    cat(sprintf("MEM_R_MIN:%f\\n", min(mem_deltas)))
    cat(sprintf("MEM_R_MAX:%f\\n", max(mem_deltas)))

    # Only output RESULT_FILE if we actually saved it this run (not stale)
    if (result_saved && file.exists(result_file)) {{
        cat(sprintf("RESULT_FILE:%s\\n", result_file))
    }}

    cat(sprintf("TEST_END:%d_%d_%s\\n", N, d, task)); flush(stdout())

    # Cleanup
    rm(x_orig, x_zs, h, opt, times, mem_deltas, result)
    if (task == "LPR") rm(y)
    if (exists("x_grid")) rm(x_grid)
    if (exists("result_vec")) rm(result_vec)
    invisible(gc(reset=TRUE, full=TRUE))
}}
"""
    )

    full_r_code = r_code_prefix + r_code_part3

    # Write and execute
    temp_script = OUTPUT_DIR / "temp_r_batch.R"
    temp_script.parent.mkdir(parents=True, exist_ok=True)
    temp_script.write_text(full_r_code)

    log(f"[R Batch] Running {len(test_list)} tests in single R process...")
    result = run_with_memory_monitor(
        [R_PATH, str(temp_script)],
        f"R Batch ({len(test_list)} tests)",
        timeout=600 * len(test_list),  # Scale timeout with number of tests
    )
    safe_unlink(temp_script)

    # Parse results
    if result["status"] != "success":
        log(f"[R Batch] FAILED: {result.get('error', 'Unknown error')}")
        return {}

    # Extract per-test results from stdout
    # Memory is distributed using run_peaks (each test has n_runs peaks)
    results_dict = {}
    lines = result.get("stdout", "").split("\n")
    run_peaks = result.get("run_peaks", [])
    test_index = 0
    current_result = {}

    for line in lines:
        if line.startswith("TIME:"):
            current_result["time_sec"] = float(line.split(":")[1])
        elif line.startswith("TIME_MIN:"):
            current_result["time_min"] = float(line.split(":")[1])
        elif line.startswith("TIME_MAX:"):
            current_result["time_max"] = float(line.split(":")[1])
        elif line.startswith("TIME_STD:"):
            val = line.split(":")[1]
            current_result["time_std"] = (
                0.0 if val.upper() in ("NA", "NAN", "INF") else float(val)
            )
        elif line.startswith("MEM_R:"):
            current_result["mem_r"] = float(line.split(":")[1])
        elif line.startswith("MEM_R_MIN:"):
            current_result["mem_r_min"] = float(line.split(":")[1])
        elif line.startswith("MEM_R_MAX:"):
            current_result["mem_r_max"] = float(line.split(":")[1])
        elif line.startswith("RESULT_FILE:"):
            current_result["result_file"] = line.split(":", 1)[1].strip()
        elif line.startswith("TEST_END:"):
            if test_index < len(test_list):
                N, d, task = test_list[test_index]
                current_result["status"] = "success"

                # Distribute per-run peaks to this test
                # Each test has n_runs peaks at indices [test_index*n_runs : (test_index+1)*n_runs]
                start_idx = test_index * n_runs
                end_idx = start_idx + n_runs
                test_peaks = (
                    run_peaks[start_idx:end_idx] if start_idx < len(run_peaks) else []
                )

                if test_peaks:
                    import statistics

                    mem_median = statistics.median(test_peaks)
                    mem_min = min(test_peaks)
                    mem_max = max(test_peaks)
                    mem_std = (
                        statistics.stdev(test_peaks) if len(test_peaks) > 1 else 0.0
                    )
                else:
                    mem_median = mem_min = mem_max = mem_std = 0.0

                # Use R internal memory measurement consistently (don't mix with JobObject)
                # gc() measures R memory manager allocations (more accurate than RSS)
                mem_r = current_result.get("mem_r")
                mem_r_min = current_result.get("mem_r_min")
                mem_r_max = current_result.get("mem_r_max")
                if mem_r is not None:
                    # Always use R internal measurement for consistency
                    # Even if 0, it's more accurate than JobObject for R
                    final_mem = mem_r
                    final_mem_min = mem_r_min if mem_r_min is not None else mem_r
                    final_mem_max = mem_r_max if mem_r_max is not None else mem_r
                    final_mem_median = mem_r
                    final_mem_std = 0.0
                    mem_method = "gc()"
                else:
                    # Fallback to JobObject only if R internal not available at all
                    final_mem = mem_median
                    final_mem_min = mem_min
                    final_mem_max = mem_max
                    final_mem_median = mem_median
                    final_mem_std = mem_std
                    mem_method = result.get("mem_method", "none")

                current_result["mem_mb"] = final_mem
                current_result["mem_median"] = final_mem_median
                current_result["mem_min"] = final_mem_min
                current_result["mem_max"] = final_mem_max
                current_result["mem_std"] = final_mem_std
                current_result["mem_method"] = mem_method
                current_result["baseline_mb"] = result.get("baseline_mb", 0)

                results_dict[(N, d, task)] = current_result.copy()
                current_result = {}
                test_index += 1

    log(f"[R Batch] Completed {len(results_dict)}/{len(test_list)} tests")
    return results_dict


def run_r_single(N, d, task="KDE", n_runs=1):
    """Run single R benchmark using batch mode (avoids 0.12s startup overhead).

    Uses cached batch results if available, otherwise triggers batch execution.
    """
    global _R_BATCH_RESULTS

    # Check if batch results are cached
    key = (N, d, task)
    if key in _R_BATCH_RESULTS:
        result = _R_BATCH_RESULTS[key]
    else:
        # Batch not run yet - run single test in batch mode
        results = run_r_batch([key], n_runs)
        _R_BATCH_RESULTS.update(results)
        result = results.get(
            key,
            {
                "status": "error",
                "error": "Batch execution failed",
                "time_sec": 0.0,
                "mem_mb": 0.0,
            },
        )

    # Log result in same format as Python/MATLAB (2026-01-09 fix)
    description = f"R {task} d={d} N={N}"
    if result.get("status") == "success":
        time_sec = result.get("time_sec", 0.0)
        time_min = result.get("time_min", time_sec)
        time_max = result.get("time_max", time_sec)
        mem_mb = result.get("mem_mb", 0.0)
        mem_min = result.get("mem_min", mem_mb)
        mem_max = result.get("mem_max", mem_mb)
        mem_method = result.get("mem_method", "none")
        log(
            f"  {description}: {time_sec:.2f}s [{time_min:.2f}-{time_max:.2f}], {mem_mb:.1f}MB [{mem_min:.1f}-{mem_max:.1f}] ({mem_method})"
        )
    else:
        log(f"  {description}: {result.get('status', 'error')}")

    return result


def run_r_single_legacy(N, d, task="KDE", n_runs=1):
    """Legacy single-process R benchmark (kept for reference).

    NOTE: This has ~0.12s startup overhead per test. Use run_r_single() instead.
    """
    fastlpr_r_dir = str(REPO_ROOT / "fastLPR_R").replace("\\", "/")
    temp_dir = str(OUTPUT_DIR).replace("\\", "/")
    gt_dir = str(GT_DIR).replace("\\", "/")

    # Grid mode: load evaluation grid from GT file
    grid_code = ""
    eval_points = "x_zs"
    if EVAL_MODE == "grid":
        grid_code = get_grid_r_code(N, d, gt_dir)
        eval_points = "x_grid"

    # Get M_INTERNAL based on dimension (d=1/2: 16384, d=3: 32768)
    m_internal = get_m_internal(d)

    script = f'''
# NOTE: Do NOT disable OpenMP for R - Rcpp/RcppArmadillo relies on OpenMP for
# matrix operations. Disabling it causes 10x slowdown at large N. (2026-01-09 fix)
# Previous comment was wrong: "1 thread is 3x faster" only true for tiny N=1024,
# but for large N (65536+), OpenMP provides 10x speedup.

# Load fastLPR package via setup.R (ensures Rcpp bindings are available)
setwd("{fastlpr_r_dir}")
source("setup.R")

# Verify Rcpp is available (required for benchmark)
if (!rcpp_available()) {{
    stop("FATAL: Rcpp acceleration not available. Please install package: R CMD INSTALL .")
}}

N <- {N}
d <- {d}
SEED <- {SEED}
H0 <- {H0}
NOISE_STD <- {NOISE_STD}
N_RUNS <- {n_runs}

gt_dir <- "{gt_dir}"
gt_file <- file.path(gt_dir, sprintf("gt_d%d_N%d.mat", d, N))

# Load GT file if exists (N <= 65536), otherwise generate data (large N)
if (file.exists(gt_file)) {{
    if (!requireNamespace("R.matlab", quietly = TRUE)) {{
        cat("[R] Installing R.matlab package...\\n", file = stderr())
        install.packages("R.matlab", repos = "https://cloud.r-project.org", quiet = TRUE)
    }}
    mat <- R.matlab::readMat(gt_file)
    x_orig <- as.matrix(mat[['x.orig']])
    x_zs <- as.matrix(mat[['x.zs']])
    if ("{task}" != "KDE") {{
        y <- as.matrix(mat[['y']])
    }}
    cat(sprintf("[R] Using GT data: %s\\n", gt_file), file = stderr())
}} else {{
    # Generate data for large N (> 65536) - no accuracy comparison needed
    set.seed(SEED)
    x_orig <- matrix(runif(N * d), nrow = N, ncol = d)
    x_mean <- colMeans(x_orig)
    x_std <- apply(x_orig, 2, function(col) sd(col) * sqrt((length(col)-1)/length(col)))
    x_zs <- scale(x_orig, center = x_mean, scale = x_std)
    if ("{task}" != "KDE") {{
        y_true <- if (d == 1) sin(2 * pi * x_orig) else matrix(sin(2 * pi * rowMeans(x_orig)), ncol = 1)
        y <- y_true + NOISE_STD * rnorm(N)
    }}
    cat(sprintf("[R] Generated data for N=%d (no GT file)\\n", N), file = stderr())
}}

h_N <- H0 * N^(-1/(d+4))
h <- rep(h_N, d)
hlist <- matrix(h, nrow = 1)
# Fixed M for fair cross-method comparison (same as evaluation grid)
# 3D uses 100000 for better accuracy (22^3=10648 too coarse, 46^3=97336 better)
M_INTERNAL <- {m_internal}
grid_size <- ceiling(M_INTERNAL^(1/d))
opt <- list(order = 0, calc_dof = FALSE, N = grid_size, accuracy = {NUFFT_ACCURACY})

{grid_code}

# y is loaded from GT file for LPR tasks (no need to generate)

invisible(gc(reset = TRUE, full = TRUE))

# Warmup with SMALL data (N=32) - trigger JIT without allocating full buffers
warmup_n <- 32
set.seed({SEED})
warmup_x <- matrix(runif(warmup_n * {d}), nrow = warmup_n, ncol = {d})
warmup_x_mean <- colMeans(warmup_x)
warmup_x_std <- apply(warmup_x, 2, function(col) sd(col) * sqrt((length(col)-1)/length(col)))
warmup_x_zs <- scale(warmup_x, center = warmup_x_mean, scale = warmup_x_std)
# FIX: Create 1xd bandwidth matrix (not 1x1) to match actual benchmark
warmup_h_N <- {H0} * warmup_n^(-1/({d}+4))
warmup_h <- matrix(rep(warmup_h_N, {d}), nrow = 1)
warmup_opt <- list(order = 0, calc_dof = FALSE, N = 20, accuracy = {NUFFT_ACCURACY})
if ("{task}" == "KDE") {{
    invisible(cv_fastkde(warmup_x_zs, warmup_h, warmup_opt))
}} else {{
    warmup_y <- if ({d} == 1) sin(2 * pi * warmup_x) else sin(2 * pi * rowMeans(warmup_x))
    warmup_y <- warmup_y + {NOISE_STD} * rnorm(warmup_n)
    invisible(cv_fastlpr(warmup_x_zs, warmup_y, warmup_h, warmup_opt))
}}
rm(warmup_n, warmup_x, warmup_x_zs, warmup_h_N, warmup_h, warmup_opt)
if ("{task}" != "KDE") rm(warmup_y)

# Double gc + sleep to stabilize memory before baseline capture (fix Issue #7 spike)
invisible(gc(reset = TRUE, full = TRUE))
Sys.sleep(0.1)
invisible(gc(reset = TRUE, full = TRUE))

cat("READY\\n")
flush(stdout())
Sys.sleep(0.05)  # Allow parent to capture baseline

n_runs_test <- if (N_RUNS > 1) N_RUNS else if (N <= 64) 5 else 1

times <- numeric(n_runs_test)
result <- NULL
for (run in 1:n_runs_test) {{
    cat("RUN_START\\n"); flush(stdout())
    t0 <- Sys.time()
    if ("{task}" == "KDE") {{
        result <- cv_fastkde(x_zs, hlist, opt)
    }} else {{
        result <- cv_fastlpr(x_zs, y, hlist, opt)
    }}
    times[run] <- as.numeric(Sys.time() - t0, units = "secs")
    cat("RUN_END\\n"); flush(stdout())
}}

cat(sprintf("TIME:%f\\n", median(times)))
cat(sprintf("TIME_MIN:%f\\n", min(times)))
cat(sprintf("TIME_MAX:%f\\n", max(times)))
cat(sprintf("TIME_STD:%f\\n", if(length(times)>1) sd(times) else 0))

# Save result for accuracy computation (only for N <= 65536)
if (N <= 65536 && !is.null(result)) {{
    # Evaluate interpolator at eval points (x_zs or x_grid depending on mode)
    # KDE: result$fpp is a list with $evaluate method
    # LPR: result$fpp_yhat can be:
    #   - function (1D case from stats::approxfun)
    #   - list with $evaluate method (multi-D case from fastlpr_gridinterp)
    if ("{task}" == "KDE") {{
        result_vec <- as.vector(result$fpp$evaluate({eval_points}))
    }} else {{
        # Handle both function and list cases for fpp_yhat
        if (is.function(result$fpp_yhat)) {{
            result_vec <- as.vector(result$fpp_yhat({eval_points}))
        }} else if (is.list(result$fpp_yhat) && !is.null(result$fpp_yhat$evaluate)) {{
            result_vec <- as.vector(result$fpp_yhat$evaluate({eval_points}))
        }} else {{
            result_vec <- NULL
            cat("WARNING: Unknown fpp_yhat type\\n")
        }}
    }}
    if (!is.null(result_vec)) {{
        result_file <- file.path("{temp_dir}", paste0("bench_r_{task}_{d}_{N}_", format(Sys.time(), "%H%M%S"), ".csv"))
        write.table(result_vec, result_file, row.names = FALSE, col.names = FALSE)
        cat(sprintf("RESULT_FILE:%s\\n", result_file))
    }}
}}
'''
    temp_script = OUTPUT_DIR / f"temp_r_bench_{task}_{d}_{N}.R"
    temp_script.parent.mkdir(parents=True, exist_ok=True)
    temp_script.write_text(script)
    result = run_with_memory_monitor(
        [R_PATH, str(temp_script)], f"R {task} d={d} N={N}", timeout=600
    )
    safe_unlink(temp_script)
    return _parse_result(result, f"R {task} d={d} N={N}")


# ==============================================================================
# MATLAB BENCHMARKS (Batched Execution to Avoid Startup Overhead)
# ==============================================================================

# Cache for batched MATLAB results
_MATLAB_BATCH_RESULTS = {}


def run_matlab_batch(test_list, n_runs=1):
    """Run ALL MATLAB benchmarks in a single MATLAB process.

    Args:
        test_list: List of (N, d, task) tuples to run
        n_runs: Number of runs per test

    Returns:
        Dictionary keyed by (N, d, task) with results
    """
    global _MATLAB_BATCH_RESULTS

    if not test_list:
        return {}

    util_path = str(REPO_ROOT / "fastLPR" / "utility").replace("\\", "/")
    core_path = str(REPO_ROOT / "fastLPR" / "utility" / "core").replace("\\", "/")
    temp_dir = str(OUTPUT_DIR).replace("\\", "/")
    gt_dir = str(GT_DIR).replace("\\", "/")

    # Build test array for MATLAB
    test_array_str = "tests = {"
    for N, d, task in test_list:
        test_array_str += f"{N}, {d}, '{task}';\n"
    test_array_str += "};"

    # Grid mode: generate MATLAB code for evaluation grid
    grid_mode = EVAL_MODE == "grid"
    eval_points_var = "x_grid" if grid_mode else "x_zs"

    matlab_code = f"""
addpath('{util_path}');
addpath('{core_path}');

% Test parameters
SEED = {SEED};
H0 = {H0};
NOISE_STD = {NOISE_STD};
N_RUNS = {n_runs};
temp_dir = '{temp_dir}';
GRID_MODE = {"true" if grid_mode else "false"};

{test_array_str}
num_tests = size(tests, 1);

% Warmup with small dataset (first test's parameters)
rng(SEED);
d = tests{{1, 2}};
x_small = rand(100, d);
[x_small_zs, ~, ~] = zscore(x_small);
y_small = sin(2*pi*x_small) + NOISE_STD * randn(100, 1);
if d > 1
    y_small = sin(2*pi*mean(x_small, 2)) + NOISE_STD * randn(100, 1);
end
h_small = 0.3;
opt_small.order = 0;
opt_small.accuracy = {NUFFT_ACCURACY};
opt_small.calc_dof = false;
opt_small.N = 20;
try
    cv_fastKDE(x_small_zs, h_small, opt_small);
    cv_fastLPR(x_small_zs, y_small, h_small, opt_small);
catch
end
clear x_small x_small_zs y_small h_small opt_small;

% Stabilize memory before baseline capture
pause(0.1);

fprintf(2, 'READY\\n');

% Wait for external memory monitor to capture baseline
% (5 samples at 50ms intervals = 250ms + margin)
pause(0.5);

% Run all tests
for t = 1:num_tests
    N = tests{{t, 1}};
    d = tests{{t, 2}};
    task = tests{{t, 3}};

    % Try to load ground truth data for fair cross-language comparison
    gt_file = fullfile('{gt_dir}', sprintf('gt_d%d_N%d.mat', d, N));
    if exist(gt_file, 'file')
        gt = load(gt_file);
        x_zs = gt.x_zs;
        h_N = gt.h_N;
        if strcmp(task, 'LPR')
            y = gt.y;
        end
        fprintf(2, '[MATLAB] Using GT data: %s\\n', gt_file);
    else
        % Fallback: generate data (for large N where no ground truth exists)
        rng(SEED);
        x_orig = rand(N, d);
        [x_zs, ~, ~] = zscore(x_orig);
        h_N = H0 * N^(-1/(d+4));
        if strcmp(task, 'LPR')
            if d == 1
                y_true = sin(2*pi*x_orig);
            else
                y_true = sin(2*pi*mean(x_orig, 2));
            end
            y = y_true + NOISE_STD * randn(N, 1);
        end
        fprintf(2, '[MATLAB] Generated data for N=%d (no GT file)\\n', N);
    end

    % Grid mode: load x_grid from GT file for exact match with ground truth
    if GRID_MODE
        gt_grid_file = fullfile('{gt_dir}', sprintf('gt_d%d_N%d_grid.mat', d, N));
        if exist(gt_grid_file, 'file')
            gt_grid = load(gt_grid_file);
            x_grid = gt_grid.x_grid;
            fprintf(2, '[MATLAB] Using x_grid from GT file: %d x %d\\n', size(x_grid, 1), size(x_grid, 2));
        else
            % Fallback for large N: generate grid (power-of-two per dimension, FFT-friendly)
            x_min = min(x_zs, [], 1);
            x_max = max(x_zs, [], 1);
            % NO margin - stay within interpolatable range
            if d == 1
                M_per_dim = 16384;
            elseif d == 2
                M_per_dim = 128;
            else
                M_per_dim = 32;
            end
            if d == 1
                x_grid = linspace(x_min, x_max, M_per_dim)';
            else
                grid_axes = cell(1, d);
                for i = 1:d
                    grid_axes{{i}} = linspace(x_min(i), x_max(i), M_per_dim);
                end
                [mesh{{1:d}}] = ndgrid(grid_axes{{:}});
                x_grid = zeros(M_per_dim^d, d);
                for i = 1:d
                    x_grid(:, i) = mesh{{i}}(:);
                end
            end
            fprintf(2, '[MATLAB] Generated grid: %d x %d (no GT file)\\n', size(x_grid, 1), size(x_grid, 2));
        end
        eval_pts = x_grid;
    else
        eval_pts = x_zs;
    end

    h = h_N * ones(1, d);

    opt.order = 0;
    opt.accuracy = {NUFFT_ACCURACY};
    opt.calc_dof = false;
    % Fixed internal grid size for fair cross-method comparison
    % d=1/2: 16384 (FFT-friendly), d=3: 32768 (=32^3)
    if d == 3
        M_INTERNAL = {M_INTERNAL_FULL_3D};
    else
        M_INTERNAL = {M_INTERNAL_FULL};
    end
    % In quick mode, use smaller M_INTERNAL
    if {M_INTERNAL} == {M_INTERNAL_QUICK}
        M_INTERNAL = {M_INTERNAL_QUICK};
    end
    opt.N = ceil(M_INTERNAL^(1/d)) * ones(1, d);
    opt.y_grid_opt.Method = 'linear';  % O(N) for fair cross-language comparison
    opt.y_grid_opt.ExtrapolationMethod = 'linear';  % Required by griddedInterpolant

    % Warmup run for this specific test (excluded from timing)
    try
        if strcmp(task, 'KDE')
            cv_fastKDE(x_zs, h, opt);
        else
            cv_fastLPR(x_zs, y, h, opt);
        end
    catch
    end

    % Timed runs
    fprintf('TEST_START:%d_%d_%s\\n', N, d, task);
    times = zeros(N_RUNS, 1);
    mem_deltas = zeros(N_RUNS, 1);  % Memory delta per run (MATLAB internal)
    result = [];
    err_msg = '';

    for run = 1:N_RUNS
        % Measure memory before (MATLAB internal)
        mem_before = memory;
        mem_used_before = mem_before.MemUsedMATLAB;

        fprintf(2, 'RUN_START\\n');
        try
            tic;
            if strcmp(task, 'KDE')
                result = cv_fastKDE(x_zs, h, opt);
            else
                result = cv_fastLPR(x_zs, y, h, opt);
            end
            times(run) = toc;

            % Measure memory after (MATLAB internal)
            mem_after = memory;
            mem_used_after = mem_after.MemUsedMATLAB;
            mem_deltas(run) = (mem_used_after - mem_used_before) / 1024 / 1024;  % Convert to MB
        catch ME
            times(run) = nan;
            mem_deltas(run) = nan;
            err_msg = ME.message;
        end
        fprintf(2, 'RUN_END\\n');
    end

    % Report results
    if any(~isnan(times))
        valid_times = times(~isnan(times));
        fprintf('TIME:%f\\n', median(valid_times));
        fprintf('TIME_MIN:%f\\n', min(valid_times));
        fprintf('TIME_MAX:%f\\n', max(valid_times));
        fprintf('TIME_STD:%f\\n', std(valid_times));

        % Report MATLAB internal memory measurement
        valid_mems = mem_deltas(~isnan(mem_deltas));
        if ~isempty(valid_mems)
            fprintf('MEM_MATLAB:%f\\n', max(valid_mems));  % Use max for peak
            fprintf('MEM_MATLAB_MIN:%f\\n', min(valid_mems));
            fprintf('MEM_MATLAB_MAX:%f\\n', max(valid_mems));
        end

        % Save result for accuracy computation (only for N <= 65536)
        if N <= 65536 && ~isempty(result)
            try
                if strcmp(task, 'KDE')
                    result_vec = result.fpp(eval_pts);
                else
                    result_vec = result.fpp_yhat(eval_pts);
                end
                result_vec = result_vec(:);
                result_file = fullfile(temp_dir, sprintf('bench_m_%s_%d_%d.mat', task, d, N));
                save(result_file, 'result_vec', '-v7');
                fprintf('RESULT_FILE:%s\\n', result_file);
            catch
            end
        end
    else
        fprintf('TIME:nan\\n');
        fprintf('ERROR:%s\\n', err_msg);
    end
    fprintf('TEST_END:%d_%d_%s\\n', N, d, task);

    % Aggressive memory cleanup for accurate per-test measurement
    clear result x_zs y x_grid eval_pts result_vec x_orig h opt times err_msg;
    clear ans;  % Clear any leftover answer variable

    % Force Java garbage collection (helps with some MATLAB internals)
    java.lang.System.gc();

    pause(0.1);  % Allow MATLAB to release memory
end

exit(0);
"""

    log(f"Running MATLAB batch: {len(test_list)} tests in single process...")
    result = run_with_memory_monitor(
        [MATLAB_CMD, "-batch", matlab_code],
        f"MATLAB batch ({len(test_list)} tests)",
        timeout=max(600, len(test_list) * 120),  # Allow more time for large batches
    )

    # Parse batch results from stdout
    results = {}
    stdout = result.get("stdout", "")
    current_test = None
    current_times = {}
    test_index = 0  # Track which test we're on for memory distribution

    # Get per-run peaks for distribution (each test has n_runs peaks)
    run_peaks = result.get("run_peaks", [])

    for line in stdout.split("\n"):
        line = line.strip()
        if line.startswith("TEST_START:"):
            # Parse N_d_task format
            parts = line.split(":")[1].split("_")
            if len(parts) >= 3:
                N = int(parts[0])
                d = int(parts[1])
                task = parts[2]
                current_test = (N, d, task)
                current_times = {
                    "time_sec": None,
                    "time_min": None,
                    "time_max": None,
                    "time_std": None,
                    "result_file": None,
                    "mem_matlab": None,
                    "mem_matlab_min": None,
                    "mem_matlab_max": None,
                }
        elif line.startswith("TEST_END:") and current_test:
            # Distribute per-run peaks to this test
            # Each test has n_runs peaks at indices [test_index*n_runs : (test_index+1)*n_runs]
            start_idx = test_index * n_runs
            end_idx = start_idx + n_runs
            test_peaks = (
                run_peaks[start_idx:end_idx] if start_idx < len(run_peaks) else []
            )

            if test_peaks:
                import statistics

                mem_median = statistics.median(test_peaks)
                mem_min = min(test_peaks)
                mem_max = max(test_peaks)
                mem_std = statistics.stdev(test_peaks) if len(test_peaks) > 1 else 0.0
            else:
                mem_median = mem_min = mem_max = mem_std = 0.0

            # Use MATLAB internal memory measurement consistently (don't mix with JobObject)
            # MATLAB_internal measures MemUsedMATLAB delta which is more accurate
            mem_matlab = current_times.get("mem_matlab")
            mem_matlab_min = current_times.get("mem_matlab_min")
            mem_matlab_max = current_times.get("mem_matlab_max")
            if mem_matlab is not None:
                # Always use MATLAB internal measurement for consistency
                # Even if 0, it's more accurate than JobObject for MATLAB
                final_mem = mem_matlab
                final_mem_min = (
                    mem_matlab_min if mem_matlab_min is not None else mem_matlab
                )
                final_mem_max = (
                    mem_matlab_max if mem_matlab_max is not None else mem_matlab
                )
                final_mem_median = mem_matlab
                final_mem_std = 0.0
                mem_method = "MATLAB_internal"
            else:
                # Fallback to JobObject only if MATLAB internal not available at all
                final_mem = mem_max
                final_mem_min = mem_min
                final_mem_max = mem_max
                final_mem_median = mem_median
                final_mem_std = mem_std
                mem_method = result.get("mem_method", "none")

            results[current_test] = {
                "time_sec": current_times.get("time_sec"),
                "time_min": current_times.get("time_min"),
                "time_max": current_times.get("time_max"),
                "time_std": current_times.get("time_std"),
                "result_file": current_times.get("result_file"),
                "mem_mb": final_mem,
                "mem_median": final_mem_median,
                "mem_min": final_mem_min,
                "mem_max": final_mem_max,
                "mem_std": final_mem_std,
                "mem_method": mem_method,
                "baseline_mb": result.get("baseline_mb", 0),
                "status": "success" if current_times.get("time_sec") else "error",
            }
            current_test = None
            test_index += 1
        elif line.startswith("TIME:") and current_test:
            try:
                current_times["time_sec"] = float(line.split(":")[1])
            except ValueError:
                pass
        elif line.startswith("TIME_MIN:") and current_test:
            try:
                current_times["time_min"] = float(line.split(":")[1])
            except ValueError:
                pass
        elif line.startswith("TIME_MAX:") and current_test:
            try:
                current_times["time_max"] = float(line.split(":")[1])
            except ValueError:
                pass
        elif line.startswith("TIME_STD:") and current_test:
            try:
                current_times["time_std"] = float(line.split(":")[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_FILE:") and current_test:
            current_times["result_file"] = line.split(":", 1)[1].strip()
        elif line.startswith("MEM_MATLAB:") and current_test:
            try:
                current_times["mem_matlab"] = float(line.split(":")[1])
            except ValueError:
                pass
        elif line.startswith("MEM_MATLAB_MIN:") and current_test:
            try:
                current_times["mem_matlab_min"] = float(line.split(":")[1])
            except ValueError:
                pass
        elif line.startswith("MEM_MATLAB_MAX:") and current_test:
            try:
                current_times["mem_matlab_max"] = float(line.split(":")[1])
            except ValueError:
                pass
        elif line.startswith("ERROR:") and current_test:
            pass  # Error already handled by time_sec being None

    _MATLAB_BATCH_RESULTS.update(results)
    log(f"MATLAB batch complete: {len(results)}/{len(test_list)} tests successful")
    return results


def run_matlab_single(N, d, task="KDE", n_runs=1):
    """Run single MATLAB benchmark - uses cached batch results if available."""
    global _MATLAB_BATCH_RESULTS

    # Check if result is cached from batch run
    key = (N, d, task)
    if key in _MATLAB_BATCH_RESULTS:
        result = _MATLAB_BATCH_RESULTS[key]
        # Unified log format: {description}: {time}s [{time_min}-{time_max}], {mem}MB [{mem_min}-{mem_max}] ({mem_method})
        time_sec = result.get("time_sec", 0)
        time_min = result.get("time_min", time_sec)
        time_max = result.get("time_max", time_sec)
        mem_mb = result.get("mem_mb", 0)
        mem_min = result.get("mem_min", mem_mb)
        mem_max = result.get("mem_max", mem_mb)
        mem_method = result.get("mem_method", "none")
        log(
            f"  MATLAB {task} d={d} N={N}: {time_sec:.2f}s [{time_min:.2f}-{time_max:.2f}], {mem_mb:.1f}MB [{mem_min:.1f}-{mem_max:.1f}] ({mem_method})"
        )
        return result

    # Fallback: run single test (should not happen in normal usage)
    util_path = str(REPO_ROOT / "fastLPR" / "utility").replace("\\", "/")
    core_path = str(REPO_ROOT / "fastLPR" / "utility" / "core").replace("\\", "/")
    temp_dir = str(OUTPUT_DIR).replace("\\", "/")
    gt_dir = str(GT_DIR).replace("\\", "/")
    grid_mode = EVAL_MODE == "grid"
    matlab_code = f"""
addpath('{util_path}');
addpath('{core_path}');

N = {N};
d = {d};
SEED = {SEED};
H0 = {H0};
NOISE_STD = {NOISE_STD};
N_RUNS = {n_runs};
GRID_MODE = {"true" if grid_mode else "false"};

gt_dir = '{gt_dir}';
gt_file = fullfile(gt_dir, sprintf('gt_d%d_N%d.mat', d, N));
if exist(gt_file, 'file')
    gt = load(gt_file);
    x_zs = gt.x_zs;
    h_N = gt.h_N;
    if strcmp('{task}', 'LPR')
        y = gt.y;
    end
    fprintf(2, '[MATLAB] Using GT data: %s\\n', gt_file);
else
    rng(SEED);
    x_orig = rand(N, d);
    [x_zs, ~, ~] = zscore(x_orig);
    h_N = H0 * N^(-1/(d+4));
    if strcmp('{task}', 'LPR')
        if d == 1
            y_true = sin(2*pi*x_orig);
        else
            y_true = sin(2*pi*mean(x_orig, 2));
        end
        y = y_true + NOISE_STD * randn(N, 1);
    end
    fprintf(2, '[MATLAB] Generated data for N=%d (no GT file)\\n', N);
end

% Grid mode: load x_grid from GT file for exact match with ground truth
if GRID_MODE
    gt_grid_file = fullfile('{gt_dir}', sprintf('gt_d%d_N%d_grid.mat', d, N));
    if exist(gt_grid_file, 'file')
        gt_grid = load(gt_grid_file);
        x_grid = gt_grid.x_grid;
        fprintf(2, '[MATLAB] Using x_grid from GT file: %d x %d\\n', size(x_grid, 1), size(x_grid, 2));
    else
        % Fallback for large N: generate grid (power-of-two per dimension, FFT-friendly)
        x_min = min(x_zs, [], 1);
        x_max = max(x_zs, [], 1);
        % NO margin - stay within interpolatable range
        if d == 1
            M_per_dim = 16384;
        elseif d == 2
            M_per_dim = 128;
        else
            M_per_dim = 32;
        end
        if d == 1
            x_grid = linspace(x_min, x_max, M_per_dim)';
        else
            grid_axes = cell(1, d);
            for i = 1:d
                grid_axes{{i}} = linspace(x_min(i), x_max(i), M_per_dim);
            end
            [mesh{{1:d}}] = ndgrid(grid_axes{{:}});
            x_grid = zeros(M_per_dim^d, d);
            for i = 1:d
                x_grid(:, i) = mesh{{i}}(:);
            end
        end
        fprintf(2, '[MATLAB] Generated grid: %d x %d (no GT file)\\n', size(x_grid, 1), size(x_grid, 2));
    end
    eval_pts = x_grid;
else
    eval_pts = x_zs;
end

h = h_N * ones(1, d);
opt.order = 0;
opt.accuracy = {NUFFT_ACCURACY};
opt.calc_dof = false;
% Fixed internal grid size for fair cross-method comparison
% d=1/2: 16384 (FFT-friendly), d=3: 32768 (=32^3)
if d == 3
    M_INTERNAL = {M_INTERNAL_FULL_3D};
else
    M_INTERNAL = {M_INTERNAL_FULL};
end
% In quick mode, use smaller M_INTERNAL
if {M_INTERNAL} == {M_INTERNAL_QUICK}
    M_INTERNAL = {M_INTERNAL_QUICK};
end
opt.N = ceil(M_INTERNAL^(1/d)) * ones(1, d);
opt.y_grid_opt.Method = 'linear';  % O(N) for fair cross-language comparison
opt.y_grid_opt.ExtrapolationMethod = 'linear';  % Required by griddedInterpolant

% Warmup with SMALL data (N=32) - trigger JIT without allocating full buffers
warmup_n = 32;
rng(SEED);
warmup_x = rand(warmup_n, d);
[warmup_x_zs, ~, ~] = zscore(warmup_x);
warmup_h = H0 * warmup_n^(-1/(d+4)) * ones(1, d);
warmup_opt.order = 0;
warmup_opt.accuracy = {NUFFT_ACCURACY};
warmup_opt.calc_dof = false;
warmup_opt.N = 20;
if strcmp('{task}', 'KDE')
    cv_fastKDE(warmup_x_zs, warmup_h, warmup_opt);
else
    warmup_y = sin(2*pi*warmup_x) + NOISE_STD * randn(warmup_n, 1);
    if d > 1
        warmup_y = sin(2*pi*mean(warmup_x, 2)) + NOISE_STD * randn(warmup_n, 1);
    end
    cv_fastLPR(warmup_x_zs, warmup_y, warmup_h, warmup_opt);
end
clear warmup_n warmup_x warmup_x_zs warmup_h warmup_opt warmup_y;

% Stabilize memory before baseline capture
pause(0.1);

fprintf(2, 'READY\\n');

% Wait for external memory monitor to capture baseline
% (5 samples at 50ms intervals = 250ms + margin)
pause(0.5);

times = zeros(N_RUNS, 1);
result = [];
for run = 1:N_RUNS
    fprintf(2, 'RUN_START\\n');
    tic;
    if strcmp('{task}', 'KDE')
        result = cv_fastKDE(x_zs, h, opt);
    else
        result = cv_fastLPR(x_zs, y, h, opt);
    end
    times(run) = toc;
    fprintf(2, 'RUN_END\\n');
end

fprintf('TIME:%f\\n', median(times));
fprintf('TIME_MIN:%f\\n', min(times));
fprintf('TIME_MAX:%f\\n', max(times));
fprintf('TIME_STD:%f\\n', std(times));

% Save result for accuracy computation (only for N <= 65536)
if N <= 65536 && ~isempty(result)
    % Evaluate interpolator at eval points (x_zs or x_grid depending on mode)
    if strcmp('{task}', 'KDE')
        result_vec = result.fpp(eval_pts);  % cv_fastKDE uses .fpp
    else
        result_vec = result.fpp_yhat(eval_pts);  % cv_fastLPR uses .fpp_yhat
    end
    result_vec = result_vec(:);
    result_file = fullfile('{temp_dir}', sprintf('bench_m_{task}_{d}_{N}.mat'));
    save(result_file, 'result_vec', '-v7');
    fprintf('RESULT_FILE:%s\\n', result_file);
end
exit(0);
"""
    result = run_with_memory_monitor(
        [MATLAB_CMD, "-batch", matlab_code], f"MATLAB {task} d={d} N={N}", timeout=600
    )
    return _parse_result(result, f"MATLAB {task} d={d} N={N}")


# ==============================================================================
# COMPETITOR METHODS
# ==============================================================================


def run_ks_single(N, d, n_runs=1):
    """Run R ks package benchmark."""
    temp_dir = str(OUTPUT_DIR).replace("\\", "/")
    gt_dir = str(GT_DIR).replace("\\", "/")

    # Grid mode: load evaluation grid from GT file
    grid_code = ""
    eval_points = "x"
    if EVAL_MODE == "grid":
        grid_code = (
            get_grid_r_code(N, d, gt_dir)
            + f"\n    eval_pts <- if ({d} == 1) as.vector(x_grid) else x_grid"
        )
        eval_points = "eval_pts"

    # Get M_INTERNAL based on dimension (d=1/2: 16384, d=3: 32768)
    m_internal = get_m_internal(d)

    script = f'''
library(ks)

gt_dir <- "{gt_dir}"
gt_file <- file.path(gt_dir, sprintf("gt_d%d_N%d.mat", {d}, {N}))

if (file.exists(gt_file)) {{
    if (!requireNamespace("R.matlab", quietly = TRUE)) {{
        install.packages("R.matlab", repos = "https://cloud.r-project.org", quiet = TRUE)
    }}
    mat <- R.matlab::readMat(gt_file)
    x_orig <- as.matrix(mat[['x.orig']])
    x_zs <- as.matrix(mat[['x.zs']])
    cat(sprintf("[R] Using GT data: %s\\n", gt_file), file = stderr())
}} else {{
    set.seed({SEED})
    x_orig <- matrix(runif({N} * {d}), nrow = {N}, ncol = {d})
    x_mean <- colMeans(x_orig)
    x_std <- apply(x_orig, 2, function(col) sd(col) * sqrt((length(col)-1)/length(col)))
    x_zs <- scale(x_orig, center = x_mean, scale = x_std)
    cat(sprintf("[R] Generated data for N=%d (no GT file)\\n", {N}), file = stderr())
}}

if ({d} == 1) {{
    x <- as.vector(x_zs)
}} else {{
    x <- x_zs
}}

h_N <- {H0} * {N}^(-1/({d}+4))
if ({d} == 1) {{
    H <- h_N^2
}} else {{
    H <- diag(rep(h_N^2, {d}))
}}

{grid_code}

# Double gc + sleep to stabilize memory before baseline capture (fix Issue #7 spike)
invisible(gc(reset = TRUE, full = TRUE))
Sys.sleep(0.1)
invisible(gc(reset = TRUE, full = TRUE))
cat("READY\\n")
flush(stdout())
Sys.sleep(0.05)  # Allow parent to capture baseline

times <- numeric({n_runs})
result <- NULL
# Fixed M for fair cross-method comparison
# 3D uses 100000 for better accuracy (22^3=10648 too coarse, 46^3=97336 better)
M_INTERNAL <- {m_internal}
grid_size <- ceiling(M_INTERNAL^(1/{d}))
for (run in 1:{n_runs}) {{
    cat("RUN_START\\n"); flush(stdout())
    t0 <- Sys.time()
    # Force binned=TRUE for consistent accuracy comparison across all N
    result <- kde(x, H = H, eval.points = {eval_points}, gridsize = rep(grid_size, {d}), binned = TRUE)
    times[run] <- as.numeric(Sys.time() - t0, units = "secs")
    cat("RUN_END\\n"); flush(stdout())
}}

cat(sprintf("TIME:%f\\n", median(times)))

# Save result for accuracy computation (only for N <= 65536)
if ({N} <= 65536 && !is.null(result)) {{
    result_vec <- as.vector(result$estimate)
    result_file <- file.path("{temp_dir}", paste0("bench_ks_{d}_{N}_", format(Sys.time(), "%H%M%S"), ".csv"))
    write.table(result_vec, result_file, row.names = FALSE, col.names = FALSE)
    cat(sprintf("RESULT_FILE:%s\\n", result_file))
}}
'''
    temp_script = OUTPUT_DIR / f"temp_ks_{d}_{N}.R"
    temp_script.write_text(script)
    result = run_with_memory_monitor([R_PATH, str(temp_script)], f"ks d={d} N={N}")
    safe_unlink(temp_script)
    return _parse_result(result, f"ks KDE d={d} N={N}")


# Cache for batched ks results
_KS_BATCH_RESULTS = {}


def run_ks_batch(test_list, n_runs=1):
    """Run ALL ks benchmarks in a single R process.

    Args:
        test_list: List of (N, d) tuples to run
        n_runs: Number of runs per test

    Returns:
        Dictionary keyed by (N, d) with results
    """
    global _KS_BATCH_RESULTS

    if not test_list:
        return {}

    temp_dir = str(OUTPUT_DIR).replace("\\", "/")
    gt_dir = str(GT_DIR).replace("\\", "/")

    # Build test list for R
    test_items = [f"  list(N={N}, d={d})" for N, d in test_list]
    test_list_str = "tests <- list(\n" + ",\n".join(test_items) + "\n)"

    # Grid mode
    grid_mode = EVAL_MODE == "grid"

    r_code = f'''
library(ks)

# Test parameters
SEED <- {SEED}
H0 <- {H0}
N_RUNS <- {n_runs}
temp_dir <- "{temp_dir}"
gt_dir <- "{gt_dir}"
GRID_MODE <- {"TRUE" if grid_mode else "FALSE"}
M_INTERNAL_FULL <- {M_INTERNAL_FULL}
M_INTERNAL_FULL_3D <- {M_INTERNAL_FULL_3D}
M_INTERNAL_QUICK <- {M_INTERNAL_QUICK}
M_INTERNAL_MODE <- {M_INTERNAL}

{test_list_str}
num_tests <- length(tests)

# Warmup for ALL dimensions to trigger JIT
set.seed(SEED)
for (d_warmup in 1:3) {{
    x_small <- matrix(rnorm(100 * d_warmup), nrow = 100, ncol = d_warmup)
    if (d_warmup == 1) {{
        tryCatch(kde(x_small, h = 0.3), error = function(e) NULL)
    }} else {{
        H_small <- diag(rep(0.3, d_warmup))
        tryCatch(kde(x_small, H = H_small), error = function(e) NULL)
    }}
}}
rm(x_small, d_warmup)
if (exists("H_small")) rm(H_small)

invisible(gc(reset = TRUE, full = TRUE))
Sys.sleep(0.1)
cat("READY\\n")
flush(stdout())
Sys.sleep(0.5)

# Run all tests
for (t in 1:num_tests) {{
    N <- tests[[t]]$N
    d <- tests[[t]]$d

    # Load or generate data
    gt_file <- file.path(gt_dir, sprintf("gt_d%d_N%d.mat", d, N))
    if (file.exists(gt_file)) {{
        if (!requireNamespace("R.matlab", quietly = TRUE)) {{
            install.packages("R.matlab", repos = "https://cloud.r-project.org", quiet = TRUE)
        }}
        mat <- R.matlab::readMat(gt_file)
        x_orig <- as.matrix(mat[['x.orig']])
        x_zs <- as.matrix(mat[['x.zs']])
        cat(sprintf("[R] Using GT: %s\\n", gt_file), file = stderr())
    }} else {{
        set.seed(SEED)
        x_orig <- matrix(runif(N * d), nrow = N, ncol = d)
        x_mean <- colMeans(x_orig)
        x_std <- apply(x_orig, 2, function(col) sd(col) * sqrt((length(col)-1)/length(col)))
        x_zs <- scale(x_orig, center = x_mean, scale = x_std)
        cat(sprintf("[R] Generated data N=%d\\n", N), file = stderr())
    }}

    if (d == 1) {{
        x <- as.vector(x_zs)
    }} else {{
        x <- x_zs
    }}

    h_N <- H0 * N^(-1/(d+4))
    if (d == 1) {{
        H <- h_N^2
    }} else {{
        H <- diag(rep(h_N^2, d))
    }}

    # M_INTERNAL based on dimension
    if (M_INTERNAL_MODE == M_INTERNAL_QUICK) {{
        M_INTERNAL <- M_INTERNAL_QUICK
    }} else {{
        M_INTERNAL <- if (d == 3) M_INTERNAL_FULL_3D else M_INTERNAL_FULL
    }}
    grid_size <- ceiling(M_INTERNAL^(1/d))

    # Grid mode: load x_grid
    eval_pts <- x
    if (GRID_MODE) {{
        gt_grid_file <- file.path(gt_dir, sprintf("gt_d%d_N%d_grid.mat", d, N))
        if (file.exists(gt_grid_file)) {{
            gt_grid <- R.matlab::readMat(gt_grid_file)
            x_grid <- as.matrix(gt_grid[['x.grid']])
            eval_pts <- if (d == 1) as.vector(x_grid) else x_grid
            cat(sprintf("[R] Using x_grid: %d x %d\\n", nrow(x_grid), ncol(x_grid)), file = stderr())
        }} else {{
            # Fallback: generate grid
            grid_1d <- seq(-3, 3, length.out = grid_size)
            if (d == 1) {{
                x_grid <- grid_1d
                eval_pts <- x_grid
            }} else if (d == 2) {{
                grid_expand <- expand.grid(grid_1d, grid_1d)
                x_grid <- as.matrix(grid_expand)
                eval_pts <- x_grid
            }} else {{
                grid_expand <- expand.grid(grid_1d, grid_1d, grid_1d)
                x_grid <- as.matrix(grid_expand)
                eval_pts <- x_grid
            }}
            cat(sprintf("[R] Using fallback grid\\n"), file = stderr())
        }}
    }}

    invisible(gc(reset = TRUE, full = TRUE))

    # Helper function to get R memory usage in MB using gc()
    get_r_mem_mb <- function() {{
        gc_info <- gc(verbose=FALSE)
        return(sum(gc_info[, 2]))
    }}

    # Timed runs
    n_runs_test <- if (N_RUNS > 1) N_RUNS else if (N <= 64) 5 else 1

    times <- numeric(n_runs_test)
    mem_deltas <- numeric(n_runs_test)
    result <- NULL
    for (run in 1:n_runs_test) {{
        invisible(gc(reset = TRUE, full = TRUE))
        mem_before <- get_r_mem_mb()
        cat("RUN_START\\n"); flush(stdout())
        t0 <- Sys.time()
        result <- kde(x, H = H, eval.points = eval_pts, gridsize = rep(grid_size, d), binned = TRUE)
        times[run] <- as.numeric(Sys.time() - t0, units = "secs")
        mem_after <- get_r_mem_mb()
        mem_deltas[run] <- max(0, mem_after - mem_before)
        cat("RUN_END\\n"); flush(stdout())
    }}

    # Output timing
    cat(sprintf("TIME:%f\\n", median(times)))
    cat(sprintf("TIME_MIN:%f\\n", min(times)))
    cat(sprintf("TIME_MAX:%f\\n", max(times)))
    cat(sprintf("TIME_STD:%f\\n", if(length(times) > 1) sd(times) else 0))

    # Output memory (gc-based)
    cat(sprintf("MEM_R:%f\\n", median(mem_deltas)))
    cat(sprintf("MEM_R_MIN:%f\\n", min(mem_deltas)))
    cat(sprintf("MEM_R_MAX:%f\\n", max(mem_deltas)))

    # Save result
    if (N <= 65536 && !is.null(result)) {{
        result_vec <- as.vector(result$estimate)
        result_file <- file.path(temp_dir, sprintf("ks_result_d%d_N%d.rds", d, N))
        saveRDS(result_vec, result_file)
        cat(sprintf("RESULT_FILE:%s\\n", result_file))
    }}

    cat(sprintf("TEST_END:%d_%d\\n", N, d)); flush(stdout())

    # Cleanup
    rm(x_orig, x_zs, x, H, times, result)
    if (exists("x_grid")) rm(x_grid)
    invisible(gc(reset = TRUE, full = TRUE))
}}
'''

    # Write and execute
    temp_script = OUTPUT_DIR / "temp_ks_batch.R"
    temp_script.parent.mkdir(parents=True, exist_ok=True)
    temp_script.write_text(r_code)

    log(f"[ks Batch] Running {len(test_list)} tests in single R process...")
    result = run_with_memory_monitor(
        [R_PATH, str(temp_script)],
        f"ks Batch ({len(test_list)} tests)",
        timeout=600 * len(test_list),
    )
    safe_unlink(temp_script)

    # Parse results
    if result["status"] != "success":
        log(f"[ks Batch] FAILED: {result.get('error', 'Unknown error')}")
        return {}

    results_dict = {}
    lines = result.get("stdout", "").split("\n")
    run_peaks = result.get("run_peaks", [])
    test_index = 0
    current_result = {}

    for line in lines:
        if line.startswith("TIME:"):
            current_result["time_sec"] = float(line.split(":")[1])
        elif line.startswith("TIME_MIN:"):
            current_result["time_min"] = float(line.split(":")[1])
        elif line.startswith("TIME_MAX:"):
            current_result["time_max"] = float(line.split(":")[1])
        elif line.startswith("TIME_STD:"):
            val = line.split(":")[1]
            current_result["time_std"] = (
                0.0 if val.upper() in ("NA", "NAN", "INF") else float(val)
            )
        elif line.startswith("MEM_R:"):
            current_result["mem_r"] = float(line.split(":")[1])
        elif line.startswith("MEM_R_MIN:"):
            current_result["mem_r_min"] = float(line.split(":")[1])
        elif line.startswith("MEM_R_MAX:"):
            current_result["mem_r_max"] = float(line.split(":")[1])
        elif line.startswith("RESULT_FILE:"):
            current_result["result_file"] = line.split(":", 1)[1].strip()
        elif line.startswith("TEST_END:"):
            if test_index < len(test_list):
                N, d = test_list[test_index]
                current_result["status"] = "success"

                # Use gc() memory measurement (more accurate than JobObject for R)
                mem_r = current_result.get("mem_r")
                mem_r_min = current_result.get("mem_r_min")
                mem_r_max = current_result.get("mem_r_max")
                if mem_r is not None:
                    current_result["mem_mb"] = mem_r
                    current_result["mem_median"] = mem_r
                    current_result["mem_min"] = (
                        mem_r_min if mem_r_min is not None else mem_r
                    )
                    current_result["mem_max"] = (
                        mem_r_max if mem_r_max is not None else mem_r
                    )
                    current_result["mem_std"] = 0.0
                    current_result["mem_method"] = "gc()"
                else:
                    # Fallback to JobObject
                    start_idx = test_index * n_runs
                    end_idx = start_idx + n_runs
                    test_peaks = (
                        run_peaks[start_idx:end_idx]
                        if start_idx < len(run_peaks)
                        else []
                    )

                    if test_peaks:
                        import statistics

                        mem_median = statistics.median(test_peaks)
                        mem_min = min(test_peaks)
                        mem_max = max(test_peaks)
                        mem_std = (
                            statistics.stdev(test_peaks) if len(test_peaks) > 1 else 0.0
                        )
                    else:
                        mem_median = mem_min = mem_max = mem_std = 0.0

                    current_result["mem_mb"] = mem_median
                    current_result["mem_median"] = mem_median
                    current_result["mem_min"] = mem_min
                    current_result["mem_max"] = mem_max
                    current_result["mem_std"] = mem_std
                    current_result["mem_method"] = result.get("mem_method", "none")

                current_result["baseline_mb"] = result.get("baseline_mb", 0)

                results_dict[(N, d)] = current_result.copy()
                current_result = {}
                test_index += 1

    log(f"[ks Batch] Completed {len(results_dict)}/{len(test_list)} tests")
    return results_dict


def run_ks_single_cached(N, d, n_runs=1):
    """Run single ks benchmark using batch mode."""
    global _KS_BATCH_RESULTS

    key = (N, d)
    if key in _KS_BATCH_RESULTS:
        result = _KS_BATCH_RESULTS[key]
    else:
        results = run_ks_batch([key], n_runs)
        _KS_BATCH_RESULTS.update(results)
        result = results.get(
            key,
            {
                "status": "error",
                "error": "Batch execution failed",
                "time_sec": 0.0,
                "mem_mb": 0.0,
            },
        )

    description = f"ks KDE d={d} N={N}"
    if result.get("status") == "success":
        time_sec = result.get("time_sec", 0.0)
        time_min = result.get("time_min", time_sec)
        time_max = result.get("time_max", time_sec)
        mem_mb = result.get("mem_mb", 0.0)
        mem_min = result.get("mem_min", mem_mb)
        mem_max = result.get("mem_max", mem_mb)
        mem_method = result.get("mem_method", "none")
        log(
            f"  {description}: {time_sec:.2f}s [{time_min:.2f}-{time_max:.2f}], {mem_mb:.1f}MB [{mem_min:.1f}-{mem_max:.1f}] ({mem_method})"
        )
    else:
        log(f"  {description}: {result.get('status', 'error')}")

    return result


def run_fksum_single(N, d, n_runs=1):
    """Run R FKSUM package benchmark (1D only)."""
    if d > 1:
        return {"time_sec": None, "mem_mb": 0, "status": "skip: 1D only"}

    temp_dir = str(OUTPUT_DIR).replace("\\", "/")
    gt_dir = str(GT_DIR).replace("\\", "/")

    # Grid mode: load evaluation grid from GT file (1D only for FKSUM)
    grid_code = ""
    eval_points = "x"
    if EVAL_MODE == "grid":
        # Use get_grid_r_code for consistent GT loading
        grid_code = get_grid_r_code(N, d, gt_dir)
        eval_points = "as.vector(x_grid)"

    # Get M_INTERNAL based on dimension (d=1/2: 16384, d=3: 32768)
    # Note: FKSUM is 1D only, so this will always be 10000
    m_internal = get_m_internal(d)

    script = f'''
library(FKSUM)

gt_dir <- "{gt_dir}"
gt_file <- file.path(gt_dir, sprintf("gt_d%d_N%d.mat", {d}, {N}))

if (file.exists(gt_file)) {{
    if (!requireNamespace("R.matlab", quietly = TRUE)) {{
        install.packages("R.matlab", repos = "https://cloud.r-project.org", quiet = TRUE)
    }}
    mat <- R.matlab::readMat(gt_file)
    x_orig <- as.matrix(mat[['x.orig']])
    x_zs <- as.matrix(mat[['x.zs']])
    cat(sprintf("[R] Using GT data: %s\\n", gt_file), file = stderr())
}} else {{
    set.seed({SEED})
    x_orig <- matrix(runif({N} * {d}), nrow = {N}, ncol = {d})
    x_mean <- colMeans(x_orig)
    x_std <- apply(x_orig, 2, function(col) sd(col) * sqrt((length(col)-1)/length(col)))
    x_zs <- scale(x_orig, center = x_mean, scale = x_std)
    cat(sprintf("[R] Generated data for N=%d (no GT file)\\n", {N}), file = stderr())
}}

x <- as.vector(x_zs)

h_N <- {H0} * {N}^(-1/({d}+4))

{grid_code}

# Double gc + sleep to stabilize memory before baseline capture (fix Issue #7 spike)
invisible(gc(reset = TRUE, full = TRUE))
Sys.sleep(0.1)
invisible(gc(reset = TRUE, full = TRUE))
cat("READY\\n")
flush(stdout())
Sys.sleep(0.05)  # Allow parent to capture baseline

times <- numeric({n_runs})
result <- NULL
# Fixed M for fair cross-method comparison
# Note: FKSUM is 1D only, so M_INTERNAL will always be 10000
M_INTERNAL <- {m_internal}
for (run in 1:{n_runs}) {{
    cat("RUN_START\\n"); flush(stdout())
    t0 <- Sys.time()
    result <- fk_density(x, h = h_N, ngrid = M_INTERNAL, x_eval = {eval_points})
    times[run] <- as.numeric(Sys.time() - t0, units = "secs")
    cat("RUN_END\\n"); flush(stdout())
}}

cat(sprintf("TIME:%f\\n", median(times)))

# Save result for accuracy computation (only for N <= 65536)
if ({N} <= 65536 && !is.null(result)) {{
    result_vec <- as.vector(result$y)
    result_file <- file.path("{temp_dir}", paste0("bench_fksum_{d}_{N}_", format(Sys.time(), "%H%M%S"), ".csv"))
    write.table(result_vec, result_file, row.names = FALSE, col.names = FALSE)
    cat(sprintf("RESULT_FILE:%s\\n", result_file))
}}
'''
    temp_script = OUTPUT_DIR / f"temp_fksum_{d}_{N}.R"
    temp_script.write_text(script)
    result = run_with_memory_monitor([R_PATH, str(temp_script)], f"FKSUM d={d} N={N}")
    safe_unlink(temp_script)
    return _parse_result(result, f"FKSUM KDE d={d} N={N}")


# Cache for batched FKSUM results
_FKSUM_BATCH_RESULTS = {}


def run_fksum_batch(test_list, n_runs=1):
    """Run ALL FKSUM benchmarks in a single R process (1D only).

    Args:
        test_list: List of (N, d) tuples to run (d must be 1)
        n_runs: Number of runs per test

    Returns:
        Dictionary keyed by (N, d) with results
    """
    global _FKSUM_BATCH_RESULTS

    # Filter to 1D only
    test_list = [(N, d) for N, d in test_list if d == 1]
    if not test_list:
        return {}

    temp_dir = str(OUTPUT_DIR).replace("\\", "/")
    gt_dir = str(GT_DIR).replace("\\", "/")

    test_items = [f"  list(N={N}, d={d})" for N, d in test_list]
    test_list_str = "tests <- list(\n" + ",\n".join(test_items) + "\n)"

    grid_mode = EVAL_MODE == "grid"

    r_code = f'''
library(FKSUM)

SEED <- {SEED}
H0 <- {H0}
N_RUNS <- {n_runs}
temp_dir <- "{temp_dir}"
gt_dir <- "{gt_dir}"
GRID_MODE <- {"TRUE" if grid_mode else "FALSE"}
M_INTERNAL <- {M_INTERNAL}

{test_list_str}
num_tests <- length(tests)

# Warmup
set.seed(SEED)
x_small <- rnorm(100)
tryCatch(fk_density(x_small, h = 0.3), error = function(e) NULL)
rm(x_small)

invisible(gc(reset = TRUE, full = TRUE))
Sys.sleep(0.1)
cat("READY\\n")
flush(stdout())
Sys.sleep(0.5)

for (t in 1:num_tests) {{
    N <- tests[[t]]$N
    d <- tests[[t]]$d

    gt_file <- file.path(gt_dir, sprintf("gt_d%d_N%d.mat", d, N))
    if (file.exists(gt_file)) {{
        if (!requireNamespace("R.matlab", quietly = TRUE)) {{
            install.packages("R.matlab", repos = "https://cloud.r-project.org", quiet = TRUE)
        }}
        mat <- R.matlab::readMat(gt_file)
        x_zs <- as.matrix(mat[['x.zs']])
        cat(sprintf("[R] Using GT: %s\\n", gt_file), file = stderr())
    }} else {{
        set.seed(SEED)
        x_orig <- matrix(runif(N * d), nrow = N, ncol = d)
        x_mean <- colMeans(x_orig)
        x_std <- apply(x_orig, 2, function(col) sd(col) * sqrt((length(col)-1)/length(col)))
        x_zs <- scale(x_orig, center = x_mean, scale = x_std)
        cat(sprintf("[R] Generated data N=%d\\n", N), file = stderr())
    }}

    x <- as.vector(x_zs)
    h_N <- H0 * N^(-1/(d+4))

    # Grid mode: load x_grid
    eval_pts <- x
    if (GRID_MODE) {{
        gt_grid_file <- file.path(gt_dir, sprintf("gt_d%d_N%d_grid.mat", d, N))
        if (file.exists(gt_grid_file)) {{
            gt_grid <- R.matlab::readMat(gt_grid_file)
            x_grid <- as.matrix(gt_grid[['x.grid']])
            eval_pts <- as.vector(x_grid)
            cat(sprintf("[R] Using x_grid\\n"), file = stderr())
        }}
    }}

    invisible(gc(reset = TRUE, full = TRUE))

    # Helper function to get R memory usage in MB using gc()
    get_r_mem_mb <- function() {{
        gc_info <- gc(verbose=FALSE)
        return(sum(gc_info[, 2]))
    }}

    n_runs_test <- if (N_RUNS > 1) N_RUNS else if (N <= 64) 5 else 1

    times <- numeric(n_runs_test)
    mem_deltas <- numeric(n_runs_test)
    result <- NULL
    for (run in 1:n_runs_test) {{
        invisible(gc(reset = TRUE, full = TRUE))
        mem_before <- get_r_mem_mb()
        cat("RUN_START\\n"); flush(stdout())
        t0 <- Sys.time()
        result <- fk_density(x, h = h_N, ngrid = M_INTERNAL, x_eval = eval_pts)
        times[run] <- as.numeric(Sys.time() - t0, units = "secs")
        mem_after <- get_r_mem_mb()
        mem_deltas[run] <- max(0, mem_after - mem_before)
        cat("RUN_END\\n"); flush(stdout())
    }}

    cat(sprintf("TIME:%f\\n", median(times)))
    cat(sprintf("TIME_MIN:%f\\n", min(times)))
    cat(sprintf("TIME_MAX:%f\\n", max(times)))
    cat(sprintf("TIME_STD:%f\\n", if(length(times) > 1) sd(times) else 0))

    # Output memory (gc-based)
    cat(sprintf("MEM_R:%f\\n", median(mem_deltas)))
    cat(sprintf("MEM_R_MIN:%f\\n", min(mem_deltas)))
    cat(sprintf("MEM_R_MAX:%f\\n", max(mem_deltas)))

    if (N <= 65536 && !is.null(result)) {{
        result_vec <- as.vector(result$y)
        result_file <- file.path(temp_dir, sprintf("fksum_result_d%d_N%d.rds", d, N))
        saveRDS(result_vec, result_file)
        cat(sprintf("RESULT_FILE:%s\\n", result_file))
    }}

    cat(sprintf("TEST_END:%d_%d\\n", N, d)); flush(stdout())

    rm(x_zs, x, times, result)
    if (exists("x_grid")) rm(x_grid)
    invisible(gc(reset = TRUE, full = TRUE))
}}
'''

    temp_script = OUTPUT_DIR / "temp_fksum_batch.R"
    temp_script.parent.mkdir(parents=True, exist_ok=True)
    temp_script.write_text(r_code)

    log(f"[FKSUM Batch] Running {len(test_list)} tests in single R process...")
    result = run_with_memory_monitor(
        [R_PATH, str(temp_script)],
        f"FKSUM Batch ({len(test_list)} tests)",
        timeout=600 * len(test_list),
    )
    safe_unlink(temp_script)

    if result["status"] != "success":
        log(f"[FKSUM Batch] FAILED: {result.get('error', 'Unknown error')}")
        return {}

    results_dict = {}
    lines = result.get("stdout", "").split("\n")
    run_peaks = result.get("run_peaks", [])
    test_index = 0
    current_result = {}

    for line in lines:
        if line.startswith("TIME:"):
            current_result["time_sec"] = float(line.split(":")[1])
        elif line.startswith("TIME_MIN:"):
            current_result["time_min"] = float(line.split(":")[1])
        elif line.startswith("TIME_MAX:"):
            current_result["time_max"] = float(line.split(":")[1])
        elif line.startswith("TIME_STD:"):
            val = line.split(":")[1]
            current_result["time_std"] = (
                0.0 if val.upper() in ("NA", "NAN", "INF") else float(val)
            )
        elif line.startswith("MEM_R:"):
            current_result["mem_r"] = float(line.split(":")[1])
        elif line.startswith("MEM_R_MIN:"):
            current_result["mem_r_min"] = float(line.split(":")[1])
        elif line.startswith("MEM_R_MAX:"):
            current_result["mem_r_max"] = float(line.split(":")[1])
        elif line.startswith("RESULT_FILE:"):
            current_result["result_file"] = line.split(":", 1)[1].strip()
        elif line.startswith("TEST_END:"):
            if test_index < len(test_list):
                N, d = test_list[test_index]
                current_result["status"] = "success"

                # Use gc() memory measurement (more accurate than JobObject for R)
                mem_r = current_result.get("mem_r")
                mem_r_min = current_result.get("mem_r_min")
                mem_r_max = current_result.get("mem_r_max")
                if mem_r is not None:
                    current_result["mem_mb"] = mem_r
                    current_result["mem_median"] = mem_r
                    current_result["mem_min"] = (
                        mem_r_min if mem_r_min is not None else mem_r
                    )
                    current_result["mem_max"] = (
                        mem_r_max if mem_r_max is not None else mem_r
                    )
                    current_result["mem_std"] = 0.0
                    current_result["mem_method"] = "gc()"
                else:
                    # Fallback to JobObject
                    start_idx = test_index * n_runs
                    end_idx = start_idx + n_runs
                    test_peaks = (
                        run_peaks[start_idx:end_idx]
                        if start_idx < len(run_peaks)
                        else []
                    )

                    if test_peaks:
                        import statistics

                        mem_median = statistics.median(test_peaks)
                        mem_min = min(test_peaks)
                        mem_max = max(test_peaks)
                        mem_std = (
                            statistics.stdev(test_peaks) if len(test_peaks) > 1 else 0.0
                        )
                    else:
                        mem_median = mem_min = mem_max = mem_std = 0.0

                    current_result["mem_mb"] = mem_median
                    current_result["mem_median"] = mem_median
                    current_result["mem_min"] = mem_min
                    current_result["mem_max"] = mem_max
                    current_result["mem_std"] = mem_std
                    current_result["mem_method"] = result.get("mem_method", "none")

                current_result["baseline_mb"] = result.get("baseline_mb", 0)

                results_dict[(N, d)] = current_result.copy()
                current_result = {}
                test_index += 1

    log(f"[FKSUM Batch] Completed {len(results_dict)}/{len(test_list)} tests")
    return results_dict


def run_fksum_single_cached(N, d, n_runs=1):
    """Run single FKSUM benchmark using batch mode."""
    if d > 1:
        return {"time_sec": None, "mem_mb": 0, "status": "skip: 1D only"}

    global _FKSUM_BATCH_RESULTS

    key = (N, d)
    if key in _FKSUM_BATCH_RESULTS:
        result = _FKSUM_BATCH_RESULTS[key]
    else:
        results = run_fksum_batch([key], n_runs)
        _FKSUM_BATCH_RESULTS.update(results)
        result = results.get(
            key,
            {
                "status": "error",
                "error": "Batch execution failed",
                "time_sec": 0.0,
                "mem_mb": 0.0,
            },
        )

    description = f"FKSUM KDE d={d} N={N}"
    if result.get("status") == "success":
        time_sec = result.get("time_sec", 0.0)
        time_min = result.get("time_min", time_sec)
        time_max = result.get("time_max", time_sec)
        mem_mb = result.get("mem_mb", 0.0)
        mem_min = result.get("mem_min", mem_mb)
        mem_max = result.get("mem_max", mem_mb)
        mem_method = result.get("mem_method", "none")
        log(
            f"  {description}: {time_sec:.2f}s [{time_min:.2f}-{time_max:.2f}], {mem_mb:.1f}MB [{mem_min:.1f}-{mem_max:.1f}] ({mem_method})"
        )
    else:
        log(f"  {description}: {result.get('status', 'error')}")

    return result


def run_locfit_single(N, d, n_runs=1):
    """Run R locfit package benchmark."""
    temp_dir = str(OUTPUT_DIR).replace("\\", "/")
    gt_dir = str(GT_DIR).replace("\\", "/")
    if d == 1:
        formula = "y ~ lp(x, h = h_N, deg = 0)"
        df_create = "df <- data.frame(x = as.vector(x_zs), y = y)"
    elif d == 2:
        formula = "y ~ lp(x1, x2, h = h_N, deg = 0)"
        df_create = "df <- data.frame(x1 = x_zs[, 1], x2 = x_zs[, 2], y = y)"
    else:
        formula = "y ~ lp(x1, x2, x3, h = h_N, deg = 0)"
        df_create = (
            "df <- data.frame(x1 = x_zs[, 1], x2 = x_zs[, 2], x3 = x_zs[, 3], y = y)"
        )

    # Grid mode: load x_grid from GT file and create grid dataframe for prediction
    grid_code = ""
    predict_data = "df"
    if EVAL_MODE == "grid":
        # Load x_grid from GT file using get_grid_r_code
        base_grid_code = get_grid_r_code(N, d, gt_dir)
        if d == 1:
            grid_code = (
                base_grid_code
                + """
grid_df <- data.frame(x = as.vector(x_grid))
"""
            )
        elif d == 2:
            grid_code = (
                base_grid_code
                + """
grid_df <- data.frame(x1 = x_grid[, 1], x2 = x_grid[, 2])
"""
            )
        else:
            grid_code = (
                base_grid_code
                + """
grid_df <- data.frame(x1 = x_grid[, 1], x2 = x_grid[, 2], x3 = x_grid[, 3])
"""
            )
        predict_data = "grid_df"

    script = f'''
library(locfit)

gt_dir <- "{gt_dir}"
gt_file <- file.path(gt_dir, sprintf("gt_d%d_N%d.mat", {d}, {N}))

if (file.exists(gt_file)) {{
    if (!requireNamespace("R.matlab", quietly = TRUE)) {{
        install.packages("R.matlab", repos = "https://cloud.r-project.org", quiet = TRUE)
    }}
    mat <- R.matlab::readMat(gt_file)
    x_orig <- as.matrix(mat[['x.orig']])
    x_zs <- as.matrix(mat[['x.zs']])
    y <- as.matrix(mat[['y']])
    cat(sprintf("[R] Using GT data: %s\\n", gt_file), file = stderr())
}} else {{
    set.seed({SEED})
    x_orig <- matrix(runif({N} * {d}), nrow = {N}, ncol = {d})
    x_mean <- colMeans(x_orig)
    x_std <- apply(x_orig, 2, function(col) sd(col) * sqrt((length(col)-1)/length(col)))
    x_zs <- scale(x_orig, center = x_mean, scale = x_std)
    y_true <- if ({d} == 1) sin(2 * pi * x_orig) else matrix(sin(2 * pi * rowMeans(x_orig)), ncol = 1)
    y <- y_true + {NOISE_STD} * rnorm({N})
    cat(sprintf("[R] Generated data for N=%d (no GT file)\\n", {N}), file = stderr())
}}

{df_create}

h_N <- {H0} * {N}^(-1/({d}+4))

{grid_code}

# Double gc + sleep to stabilize memory before baseline capture (fix Issue #7 spike)
invisible(gc(reset = TRUE, full = TRUE))
Sys.sleep(0.1)
invisible(gc(reset = TRUE, full = TRUE))
cat("READY\\n")
flush(stdout())
Sys.sleep(0.05)  # Allow parent to capture baseline

times <- numeric({n_runs})
result <- NULL
for (run in 1:{n_runs}) {{
    cat("RUN_START\\n"); flush(stdout())
    t0 <- Sys.time()
    result <- tryCatch({{
        fit <- locfit({formula}, data = df)
        predict(fit, {predict_data})
    }}, error = function(e) {{
        cat(sprintf("ERROR:%s\\n", e$message))
        NULL
    }})
    times[run] <- as.numeric(Sys.time() - t0, units = "secs")
    cat("RUN_END\\n"); flush(stdout())
}}

cat(sprintf("TIME:%f\\n", median(times, na.rm = TRUE)))

# Save result for accuracy computation (only for N <= 65536)
if ({N} <= 65536 && !is.null(result)) {{
    result_vec <- as.vector(result)
    result_file <- file.path("{temp_dir}", paste0("bench_locfit_{d}_{N}_", format(Sys.time(), "%H%M%S"), ".csv"))
    write.table(result_vec, result_file, row.names = FALSE, col.names = FALSE)
    cat(sprintf("RESULT_FILE:%s\\n", result_file))
}}
'''
    temp_script = OUTPUT_DIR / f"temp_locfit_{d}_{N}.R"
    temp_script.write_text(script)
    result = run_with_memory_monitor([R_PATH, str(temp_script)], f"locfit d={d} N={N}")
    safe_unlink(temp_script)
    return _parse_result(result, f"locfit LPR d={d} N={N}")


def run_npregfast_single(N, d, n_runs=1):
    """Run R npregfast package benchmark (1D only)."""
    if d > 1:
        return {"time_sec": None, "mem_mb": 0, "status": "skip: 1D only"}

    temp_dir = str(OUTPUT_DIR).replace("\\", "/")
    gt_dir = str(GT_DIR).replace("\\", "/")
    grid_mode = EVAL_MODE == "grid"

    # Grid mode: load evaluation points from GT file
    grid_code = ""
    eval_code = ""
    if grid_mode:
        # Load x_grid from GT file using get_grid_r_code
        grid_code = (
            get_grid_r_code(N, d, gt_dir)
            + """
eval_df <- data.frame(x = as.vector(x_grid))
"""
        )
        eval_code = """
    # Grid mode: evaluate at grid points
    result_vec <- as.vector(predict(result, newdata = eval_df)$Estimation[, 1])
"""
    else:
        eval_code = """
    # npregfast predict() crashes with segfault when newdata > ~47000 points
    # Workaround: use smaller grid for large N, interpolate back to original points
    if (N > 40000) {
        # Use 40000 point grid to avoid npregfast bug
        x_grid <- seq(min(x_zs), max(x_zs), length.out = 40000)
        pred_df <- data.frame(x = x_grid)
        pred_vals <- predict(result, newdata = pred_df)$Estimation[, 1]
        # Interpolate to original points
        result_vec <- approx(x_grid, pred_vals, xout = x_zs, rule = 2)$y
    } else {
        result_vec <- as.vector(predict(result, newdata = df)$Estimation[, 1])
    }
"""

    script = f'''
library(npregfast)

# Bandwidth formula (same as all other methods)
H0 <- {H0}
h_N <- H0 * {N}^(-1/({d}+4))

gt_dir <- "{gt_dir}"
gt_file <- file.path(gt_dir, sprintf("gt_d%d_N%d.mat", {d}, {N}))

if (file.exists(gt_file)) {{
    if (!requireNamespace("R.matlab", quietly = TRUE)) {{
        install.packages("R.matlab", repos = "https://cloud.r-project.org", quiet = TRUE)
    }}
    mat <- R.matlab::readMat(gt_file)
    x_orig <- as.matrix(mat[['x.orig']])
    x_zs <- as.matrix(mat[['x.zs']])
    y <- as.matrix(mat[['y']])
    cat(sprintf("[R] Using GT data: %s\\n", gt_file), file = stderr())
}} else {{
    set.seed({SEED})
    x_orig <- matrix(runif({N} * {d}), nrow = {N}, ncol = {d})
    x_mean <- colMeans(x_orig)
    x_std <- apply(x_orig, 2, function(col) sd(col) * sqrt((length(col)-1)/length(col)))
    x_zs <- scale(x_orig, center = x_mean, scale = x_std)
    y_true <- sin(2 * pi * x_orig)
    y <- y_true + {NOISE_STD} * rnorm({N})
    cat(sprintf("[R] Generated data for N=%d (no GT file)\\n", {N}), file = stderr())
}}

df <- data.frame(x = x_zs, y = y)
N <- {N}  # For use in eval code
{grid_code}

# Double gc + sleep to stabilize memory before baseline capture (fix Issue #7 spike)
invisible(gc(reset = TRUE, full = TRUE))
Sys.sleep(0.1)
invisible(gc(reset = TRUE, full = TRUE))
cat("READY\\n")
flush(stdout())
Sys.sleep(0.05)  # Allow parent to capture baseline

times <- numeric({n_runs})
result <- NULL
for (run in 1:{n_runs}) {{
    cat("RUN_START\\n"); flush(stdout())
    t0 <- Sys.time()
    # h=h_N: our standard bandwidth, kbin=1000: increased binning nodes (kbin=10000 causes segfault)
    # NOTE: npregfast default is p=3 (local cubic); we set p=2 (local quadratic) since p=1 segfaults
    result <- frfast(y ~ x, data = df, nboot = 0, h = h_N, kbin = 1000, p = 2)
    times[run] <- as.numeric(Sys.time() - t0, units = "secs")
    cat("RUN_END\\n"); flush(stdout())
}}

cat(sprintf("TIME:%f\\n", median(times)))

# Save result for accuracy computation (only for N <= 65536)
if ({N} <= 65536 && !is.null(result)) {{
{eval_code}
    result_file <- file.path("{temp_dir}", paste0("bench_npregfast_{d}_{N}_", format(Sys.time(), "%H%M%S"), ".csv"))
    write.table(result_vec, result_file, row.names = FALSE, col.names = FALSE)
    cat(sprintf("RESULT_FILE:%s\\n", result_file))
}}
'''
    temp_script = OUTPUT_DIR / f"temp_npregfast_{d}_{N}.R"
    temp_script.write_text(script)
    result = run_with_memory_monitor(
        [R_PATH, str(temp_script)], f"npregfast d={d} N={N}"
    )
    safe_unlink(temp_script)
    return _parse_result(result, f"npregfast LPR d={d} N={N}")


# Cache for batched locfit results
_LOCFIT_BATCH_RESULTS = {}


def run_locfit_batch(test_list, n_runs=1):
    """Run ALL locfit benchmarks in a single R process.

    Args:
        test_list: List of (N, d) tuples to run
        n_runs: Number of runs per test

    Returns:
        Dictionary keyed by (N, d) with results
    """
    global _LOCFIT_BATCH_RESULTS

    if not test_list:
        return {}

    temp_dir = str(OUTPUT_DIR).replace("\\", "/")
    gt_dir = str(GT_DIR).replace("\\", "/")

    test_items = [f"  list(N={N}, d={d})" for N, d in test_list]
    test_list_str = "tests <- list(\n" + ",\n".join(test_items) + "\n)"

    grid_mode = EVAL_MODE == "grid"

    r_code = f'''
library(locfit)

SEED <- {SEED}
H0 <- {H0}
NOISE_STD <- {NOISE_STD}
N_RUNS <- {n_runs}
temp_dir <- "{temp_dir}"
gt_dir <- "{gt_dir}"
GRID_MODE <- {"TRUE" if grid_mode else "FALSE"}

{test_list_str}
num_tests <- length(tests)

# Warmup for ALL dimensions to trigger JIT
set.seed(SEED)
for (d_warmup in 1:3) {{
    x_small <- matrix(rnorm(100 * d_warmup), nrow = 100, ncol = d_warmup)
    y_small <- sin(2*pi*rowMeans(x_small)) + 0.1*rnorm(100)
    if (d_warmup == 1) {{
        df_small <- data.frame(x = x_small, y = y_small)
        tryCatch(locfit(y ~ lp(x, h = 0.3, deg = 0), data = df_small), error = function(e) NULL)
    }} else if (d_warmup == 2) {{
        df_small <- data.frame(x1 = x_small[,1], x2 = x_small[,2], y = y_small)
        tryCatch(locfit(y ~ lp(x1, x2, h = 0.3, deg = 0), data = df_small), error = function(e) NULL)
    }} else {{
        df_small <- data.frame(x1 = x_small[,1], x2 = x_small[,2], x3 = x_small[,3], y = y_small)
        tryCatch(locfit(y ~ lp(x1, x2, x3, h = 0.3, deg = 0), data = df_small), error = function(e) NULL)
    }}
}}
rm(x_small, y_small, df_small, d_warmup)

invisible(gc(reset = TRUE, full = TRUE))
Sys.sleep(0.1)
cat("READY\\n")
flush(stdout())
Sys.sleep(0.5)

for (t in 1:num_tests) {{
    N <- tests[[t]]$N
    d <- tests[[t]]$d

    # Load or generate data
    gt_file <- file.path(gt_dir, sprintf("gt_d%d_N%d.mat", d, N))
    if (file.exists(gt_file)) {{
        if (!requireNamespace("R.matlab", quietly = TRUE)) {{
            install.packages("R.matlab", repos = "https://cloud.r-project.org", quiet = TRUE)
        }}
        mat <- R.matlab::readMat(gt_file)
        x_orig <- as.matrix(mat[['x.orig']])
        x_zs <- as.matrix(mat[['x.zs']])
        y <- as.matrix(mat[['y']])
        cat(sprintf("[R] Using GT: %s\\n", gt_file), file = stderr())
    }} else {{
        set.seed(SEED)
        x_orig <- matrix(runif(N * d), nrow = N, ncol = d)
        x_mean <- colMeans(x_orig)
        x_std <- apply(x_orig, 2, function(col) sd(col) * sqrt((length(col)-1)/length(col)))
        x_zs <- scale(x_orig, center = x_mean, scale = x_std)
        y_true <- if (d == 1) sin(2 * pi * x_orig) else matrix(sin(2 * pi * rowMeans(x_orig)), ncol = 1)
        y <- y_true + NOISE_STD * rnorm(N)
        cat(sprintf("[R] Generated data N=%d\\n", N), file = stderr())
    }}

    h_N <- H0 * N^(-1/(d+4))

    # Create dataframe based on dimension
    if (d == 1) {{
        df <- data.frame(x = as.vector(x_zs), y = y)
        formula_str <- "y ~ lp(x, h = h_N, deg = 0)"
    }} else if (d == 2) {{
        df <- data.frame(x1 = x_zs[, 1], x2 = x_zs[, 2], y = y)
        formula_str <- "y ~ lp(x1, x2, h = h_N, deg = 0)"
    }} else {{
        df <- data.frame(x1 = x_zs[, 1], x2 = x_zs[, 2], x3 = x_zs[, 3], y = y)
        formula_str <- "y ~ lp(x1, x2, x3, h = h_N, deg = 0)"
    }}

    # Grid mode: load x_grid
    predict_data <- df
    if (GRID_MODE) {{
        gt_grid_file <- file.path(gt_dir, sprintf("gt_d%d_N%d_grid.mat", d, N))
        if (file.exists(gt_grid_file)) {{
            gt_grid <- R.matlab::readMat(gt_grid_file)
            x_grid <- as.matrix(gt_grid[['x.grid']])
            if (d == 1) {{
                predict_data <- data.frame(x = as.vector(x_grid))
            }} else if (d == 2) {{
                predict_data <- data.frame(x1 = x_grid[, 1], x2 = x_grid[, 2])
            }} else {{
                predict_data <- data.frame(x1 = x_grid[, 1], x2 = x_grid[, 2], x3 = x_grid[, 3])
            }}
            cat(sprintf("[R] Using x_grid: %d x %d\\n", nrow(x_grid), ncol(x_grid)), file = stderr())
        }}
    }}

    invisible(gc(reset = TRUE, full = TRUE))

    # Helper function to get R memory usage in MB using gc()
    get_r_mem_mb <- function() {{
        gc_info <- gc(verbose=FALSE)
        return(sum(gc_info[, 2]))
    }}

    # Timed runs
    n_runs_test <- if (N_RUNS > 1) N_RUNS else if (N <= 64) 5 else 1

    times <- numeric(n_runs_test)
    mem_deltas <- numeric(n_runs_test)
    result <- NULL
    for (run in 1:n_runs_test) {{
        invisible(gc(reset = TRUE, full = TRUE))
        mem_before <- get_r_mem_mb()
        cat("RUN_START\\n"); flush(stdout())
        t0 <- Sys.time()
        result <- tryCatch({{
            fit <- locfit(as.formula(formula_str), data = df)
            predict(fit, predict_data)
        }}, error = function(e) {{
            cat(sprintf("[R] ERROR: %s\\n", e$message), file = stderr())
            NULL
        }})
        times[run] <- as.numeric(Sys.time() - t0, units = "secs")
        mem_after <- get_r_mem_mb()
        mem_deltas[run] <- max(0, mem_after - mem_before)
        cat("RUN_END\\n"); flush(stdout())
    }}

    # Output timing
    cat(sprintf("TIME:%f\\n", median(times, na.rm = TRUE)))
    cat(sprintf("TIME_MIN:%f\\n", min(times, na.rm = TRUE)))
    cat(sprintf("TIME_MAX:%f\\n", max(times, na.rm = TRUE)))
    cat(sprintf("TIME_STD:%f\\n", if(length(times) > 1) sd(times, na.rm = TRUE) else 0))

    # Output memory (gc-based)
    cat(sprintf("MEM_R:%f\\n", median(mem_deltas)))
    cat(sprintf("MEM_R_MIN:%f\\n", min(mem_deltas)))
    cat(sprintf("MEM_R_MAX:%f\\n", max(mem_deltas)))

    # Save result
    if (N <= 65536 && !is.null(result)) {{
        result_vec <- as.vector(result)
        result_file <- file.path(temp_dir, sprintf("locfit_result_d%d_N%d.rds", d, N))
        saveRDS(result_vec, result_file)
        cat(sprintf("RESULT_FILE:%s\\n", result_file))
    }}

    cat(sprintf("TEST_END:%d_%d\\n", N, d)); flush(stdout())

    # Cleanup
    rm(x_orig, x_zs, y, df, times, result)
    if (exists("x_grid")) rm(x_grid)
    if (exists("predict_data")) rm(predict_data)
    invisible(gc(reset = TRUE, full = TRUE))
}}
'''

    temp_script = OUTPUT_DIR / "temp_locfit_batch.R"
    temp_script.parent.mkdir(parents=True, exist_ok=True)
    temp_script.write_text(r_code)

    log(f"[locfit Batch] Running {len(test_list)} tests in single R process...")
    result = run_with_memory_monitor(
        [R_PATH, str(temp_script)],
        f"locfit Batch ({len(test_list)} tests)",
        timeout=600 * len(test_list),
    )
    safe_unlink(temp_script)

    if result["status"] != "success":
        log(f"[locfit Batch] FAILED: {result.get('error', 'Unknown error')}")
        return {}

    results_dict = {}
    lines = result.get("stdout", "").split("\n")
    run_peaks = result.get("run_peaks", [])
    test_index = 0
    current_result = {}

    for line in lines:
        if line.startswith("TIME:"):
            current_result["time_sec"] = float(line.split(":")[1])
        elif line.startswith("TIME_MIN:"):
            current_result["time_min"] = float(line.split(":")[1])
        elif line.startswith("TIME_MAX:"):
            current_result["time_max"] = float(line.split(":")[1])
        elif line.startswith("TIME_STD:"):
            val = line.split(":")[1]
            current_result["time_std"] = (
                0.0 if val.upper() in ("NA", "NAN", "INF") else float(val)
            )
        elif line.startswith("MEM_R:"):
            current_result["mem_r"] = float(line.split(":")[1])
        elif line.startswith("MEM_R_MIN:"):
            current_result["mem_r_min"] = float(line.split(":")[1])
        elif line.startswith("MEM_R_MAX:"):
            current_result["mem_r_max"] = float(line.split(":")[1])
        elif line.startswith("RESULT_FILE:"):
            current_result["result_file"] = line.split(":", 1)[1].strip()
        elif line.startswith("TEST_END:"):
            if test_index < len(test_list):
                N, d = test_list[test_index]
                current_result["status"] = "success"

                # Use gc() memory measurement (more accurate than JobObject for R)
                mem_r = current_result.get("mem_r")
                mem_r_min = current_result.get("mem_r_min")
                mem_r_max = current_result.get("mem_r_max")
                if mem_r is not None:
                    current_result["mem_mb"] = mem_r
                    current_result["mem_median"] = mem_r
                    current_result["mem_min"] = (
                        mem_r_min if mem_r_min is not None else mem_r
                    )
                    current_result["mem_max"] = (
                        mem_r_max if mem_r_max is not None else mem_r
                    )
                    current_result["mem_std"] = 0.0
                    current_result["mem_method"] = "gc()"
                else:
                    # Fallback to JobObject
                    start_idx = test_index * n_runs
                    end_idx = start_idx + n_runs
                    test_peaks = (
                        run_peaks[start_idx:end_idx]
                        if start_idx < len(run_peaks)
                        else []
                    )

                    if test_peaks:
                        import statistics

                        mem_median = statistics.median(test_peaks)
                        mem_min = min(test_peaks)
                        mem_max = max(test_peaks)
                        mem_std = (
                            statistics.stdev(test_peaks) if len(test_peaks) > 1 else 0.0
                        )
                    else:
                        mem_median = mem_min = mem_max = mem_std = 0.0

                    current_result["mem_mb"] = mem_median
                    current_result["mem_median"] = mem_median
                    current_result["mem_min"] = mem_min
                    current_result["mem_max"] = mem_max
                    current_result["mem_std"] = mem_std
                    current_result["mem_method"] = result.get("mem_method", "none")

                current_result["baseline_mb"] = result.get("baseline_mb", 0)

                results_dict[(N, d)] = current_result.copy()
                current_result = {}
                test_index += 1

    log(f"[locfit Batch] Completed {len(results_dict)}/{len(test_list)} tests")
    return results_dict


def run_locfit_single_cached(N, d, n_runs=1):
    """Run single locfit benchmark using batch mode."""
    global _LOCFIT_BATCH_RESULTS

    key = (N, d)
    if key in _LOCFIT_BATCH_RESULTS:
        result = _LOCFIT_BATCH_RESULTS[key]
    else:
        results = run_locfit_batch([key], n_runs)
        _LOCFIT_BATCH_RESULTS.update(results)
        result = results.get(
            key,
            {
                "status": "error",
                "error": "Batch execution failed",
                "time_sec": 0.0,
                "mem_mb": 0.0,
            },
        )

    description = f"locfit LPR d={d} N={N}"
    if result.get("status") == "success":
        time_sec = result.get("time_sec", 0.0)
        time_min = result.get("time_min", time_sec)
        time_max = result.get("time_max", time_sec)
        mem_mb = result.get("mem_mb", 0.0)
        mem_min = result.get("mem_min", mem_mb)
        mem_max = result.get("mem_max", mem_mb)
        mem_method = result.get("mem_method", "none")
        log(
            f"  {description}: {time_sec:.2f}s [{time_min:.2f}-{time_max:.2f}], {mem_mb:.1f}MB [{mem_min:.1f}-{mem_max:.1f}] ({mem_method})"
        )
    else:
        log(f"  {description}: {result.get('status', 'error')}")

    return result


# Cache for batched npregfast results
_NPREGFAST_BATCH_RESULTS = {}


def run_npregfast_batch(test_list, n_runs=1):
    """Run ALL npregfast benchmarks in a single R process (1D only).

    Args:
        test_list: List of (N, d) tuples to run (d must be 1)
        n_runs: Number of runs per test

    Returns:
        Dictionary keyed by (N, d) with results
    """
    global _NPREGFAST_BATCH_RESULTS

    # Filter to 1D only
    test_list = [(N, d) for N, d in test_list if d == 1]
    if not test_list:
        return {}

    temp_dir = str(OUTPUT_DIR).replace("\\", "/")
    gt_dir = str(GT_DIR).replace("\\", "/")

    test_items = [f"  list(N={N}, d={d})" for N, d in test_list]
    test_list_str = "tests <- list(\n" + ",\n".join(test_items) + "\n)"

    grid_mode = EVAL_MODE == "grid"

    r_code = f'''
library(npregfast)

SEED <- {SEED}
H0 <- {H0}
NOISE_STD <- {NOISE_STD}
N_RUNS <- {n_runs}
temp_dir <- "{temp_dir}"
gt_dir <- "{gt_dir}"
GRID_MODE <- {"TRUE" if grid_mode else "FALSE"}

{test_list_str}
num_tests <- length(tests)

# Warmup
set.seed(SEED)
x_small <- rnorm(100)
y_small <- sin(2*pi*x_small) + 0.1*rnorm(100)
df_small <- data.frame(x = x_small, y = y_small)
tryCatch(frfast(y ~ x, data = df_small, nboot = 0, h = 0.3, p = 2), error = function(e) NULL)
rm(x_small, y_small, df_small)

invisible(gc(reset = TRUE, full = TRUE))
Sys.sleep(0.1)
cat("READY\\n")
flush(stdout())
Sys.sleep(0.5)

for (t in 1:num_tests) {{
    N <- tests[[t]]$N
    d <- tests[[t]]$d

    h_N <- H0 * N^(-1/(d+4))

    gt_file <- file.path(gt_dir, sprintf("gt_d%d_N%d.mat", d, N))
    if (file.exists(gt_file)) {{
        if (!requireNamespace("R.matlab", quietly = TRUE)) {{
            install.packages("R.matlab", repos = "https://cloud.r-project.org", quiet = TRUE)
        }}
        mat <- R.matlab::readMat(gt_file)
        x_zs <- as.matrix(mat[['x.zs']])
        y <- as.matrix(mat[['y']])
        cat(sprintf("[R] Using GT: %s\\n", gt_file), file = stderr())
    }} else {{
        set.seed(SEED)
        x_orig <- matrix(runif(N * d), nrow = N, ncol = d)
        x_mean <- colMeans(x_orig)
        x_std <- apply(x_orig, 2, function(col) sd(col) * sqrt((length(col)-1)/length(col)))
        x_zs <- scale(x_orig, center = x_mean, scale = x_std)
        y_true <- sin(2 * pi * x_orig)
        y <- y_true + NOISE_STD * rnorm(N)
        cat(sprintf("[R] Generated data N=%d\\n", N), file = stderr())
    }}

    df <- data.frame(x = x_zs, y = y)

    # Grid mode: load x_grid
    eval_df <- NULL
    if (GRID_MODE) {{
        gt_grid_file <- file.path(gt_dir, sprintf("gt_d%d_N%d_grid.mat", d, N))
        if (file.exists(gt_grid_file)) {{
            gt_grid <- R.matlab::readMat(gt_grid_file)
            x_grid <- as.matrix(gt_grid[['x.grid']])
            eval_df <- data.frame(x = as.vector(x_grid))
            cat(sprintf("[R] Using x_grid\\n"), file = stderr())
        }}
    }}

    invisible(gc(reset = TRUE, full = TRUE))

    # Helper function to get R memory usage in MB using gc()
    get_r_mem_mb <- function() {{
        gc_info <- gc(verbose=FALSE)
        return(sum(gc_info[, 2]))
    }}

    n_runs_test <- if (N_RUNS > 1) N_RUNS else if (N <= 64) 5 else 1

    times <- numeric(n_runs_test)
    mem_deltas <- numeric(n_runs_test)
    result <- NULL
    for (run in 1:n_runs_test) {{
        invisible(gc(reset = TRUE, full = TRUE))
        mem_before <- get_r_mem_mb()
        cat("RUN_START\\n"); flush(stdout())
        t0 <- Sys.time()
        result <- tryCatch({{
            frfast(y ~ x, data = df, nboot = 0, h = h_N, kbin = 1000, p = 2)
        }}, error = function(e) {{
            cat(sprintf("[R] ERROR: %s\\n", e$message), file = stderr())
            NULL
        }})
        times[run] <- as.numeric(Sys.time() - t0, units = "secs")
        mem_after <- get_r_mem_mb()
        mem_deltas[run] <- max(0, mem_after - mem_before)
        cat("RUN_END\\n"); flush(stdout())
    }}

    cat(sprintf("TIME:%f\\n", median(times)))
    cat(sprintf("TIME_MIN:%f\\n", min(times)))
    cat(sprintf("TIME_MAX:%f\\n", max(times)))
    cat(sprintf("TIME_STD:%f\\n", if(length(times) > 1) sd(times) else 0))

    # Output memory (gc-based)
    cat(sprintf("MEM_R:%f\\n", median(mem_deltas)))
    cat(sprintf("MEM_R_MIN:%f\\n", min(mem_deltas)))
    cat(sprintf("MEM_R_MAX:%f\\n", max(mem_deltas)))

    if (N <= 65536 && !is.null(result)) {{
        if (GRID_MODE && !is.null(eval_df)) {{
            result_vec <- as.vector(predict(result, newdata = eval_df)$Estimation[, 1])
        }} else {{
            # npregfast predict() crashes with segfault when newdata > ~47000 points
            if (N > 40000) {{
                x_grid_tmp <- seq(min(x_zs), max(x_zs), length.out = 40000)
                pred_df <- data.frame(x = x_grid_tmp)
                pred_vals <- predict(result, newdata = pred_df)$Estimation[, 1]
                result_vec <- approx(x_grid_tmp, pred_vals, xout = x_zs, rule = 2)$y
            }} else {{
                result_vec <- as.vector(predict(result, newdata = df)$Estimation[, 1])
            }}
        }}
        result_file <- file.path(temp_dir, sprintf("npregfast_result_d%d_N%d.rds", d, N))
        saveRDS(result_vec, result_file)
        cat(sprintf("RESULT_FILE:%s\\n", result_file))
    }}

    cat(sprintf("TEST_END:%d_%d\\n", N, d)); flush(stdout())

    rm(x_zs, y, df, times, result)
    if (exists("x_grid")) rm(x_grid)
    if (exists("eval_df")) rm(eval_df)
    invisible(gc(reset = TRUE, full = TRUE))
}}
'''

    temp_script = OUTPUT_DIR / "temp_npregfast_batch.R"
    temp_script.parent.mkdir(parents=True, exist_ok=True)
    temp_script.write_text(r_code)

    log(f"[npregfast Batch] Running {len(test_list)} tests in single R process...")
    result = run_with_memory_monitor(
        [R_PATH, str(temp_script)],
        f"npregfast Batch ({len(test_list)} tests)",
        timeout=600 * len(test_list),
    )
    safe_unlink(temp_script)

    if result["status"] != "success":
        log(f"[npregfast Batch] FAILED: {result.get('error', 'Unknown error')}")
        return {}

    results_dict = {}
    lines = result.get("stdout", "").split("\n")
    run_peaks = result.get("run_peaks", [])
    test_index = 0
    current_result = {}

    for line in lines:
        if line.startswith("TIME:"):
            current_result["time_sec"] = float(line.split(":")[1])
        elif line.startswith("TIME_MIN:"):
            current_result["time_min"] = float(line.split(":")[1])
        elif line.startswith("TIME_MAX:"):
            current_result["time_max"] = float(line.split(":")[1])
        elif line.startswith("TIME_STD:"):
            val = line.split(":")[1]
            current_result["time_std"] = (
                0.0 if val.upper() in ("NA", "NAN", "INF") else float(val)
            )
        elif line.startswith("MEM_R:"):
            current_result["mem_r"] = float(line.split(":")[1])
        elif line.startswith("MEM_R_MIN:"):
            current_result["mem_r_min"] = float(line.split(":")[1])
        elif line.startswith("MEM_R_MAX:"):
            current_result["mem_r_max"] = float(line.split(":")[1])
        elif line.startswith("RESULT_FILE:"):
            current_result["result_file"] = line.split(":", 1)[1].strip()
        elif line.startswith("TEST_END:"):
            if test_index < len(test_list):
                N, d = test_list[test_index]
                current_result["status"] = "success"

                # Use gc() memory measurement (more accurate than JobObject for R)
                mem_r = current_result.get("mem_r")
                mem_r_min = current_result.get("mem_r_min")
                mem_r_max = current_result.get("mem_r_max")
                if mem_r is not None:
                    current_result["mem_mb"] = mem_r
                    current_result["mem_median"] = mem_r
                    current_result["mem_min"] = (
                        mem_r_min if mem_r_min is not None else mem_r
                    )
                    current_result["mem_max"] = (
                        mem_r_max if mem_r_max is not None else mem_r
                    )
                    current_result["mem_std"] = 0.0
                    current_result["mem_method"] = "gc()"
                else:
                    # Fallback to JobObject
                    start_idx = test_index * n_runs
                    end_idx = start_idx + n_runs
                    test_peaks = (
                        run_peaks[start_idx:end_idx]
                        if start_idx < len(run_peaks)
                        else []
                    )

                    if test_peaks:
                        import statistics

                        mem_median = statistics.median(test_peaks)
                        mem_min = min(test_peaks)
                        mem_max = max(test_peaks)
                        mem_std = (
                            statistics.stdev(test_peaks) if len(test_peaks) > 1 else 0.0
                        )
                    else:
                        mem_median = mem_min = mem_max = mem_std = 0.0

                    current_result["mem_mb"] = mem_median
                    current_result["mem_median"] = mem_median
                    current_result["mem_min"] = mem_min
                    current_result["mem_max"] = mem_max
                    current_result["mem_std"] = mem_std
                    current_result["mem_method"] = result.get("mem_method", "none")

                current_result["baseline_mb"] = result.get("baseline_mb", 0)

                results_dict[(N, d)] = current_result.copy()
                current_result = {}
                test_index += 1

    log(f"[npregfast Batch] Completed {len(results_dict)}/{len(test_list)} tests")
    return results_dict


def run_npregfast_single_cached(N, d, n_runs=1):
    """Run single npregfast benchmark using batch mode."""
    if d > 1:
        return {"time_sec": None, "mem_mb": 0, "status": "skip: 1D only"}

    global _NPREGFAST_BATCH_RESULTS

    key = (N, d)
    if key in _NPREGFAST_BATCH_RESULTS:
        result = _NPREGFAST_BATCH_RESULTS[key]
    else:
        results = run_npregfast_batch([key], n_runs)
        _NPREGFAST_BATCH_RESULTS.update(results)
        result = results.get(
            key,
            {
                "status": "error",
                "error": "Batch execution failed",
                "time_sec": 0.0,
                "mem_mb": 0.0,
            },
        )

    description = f"npregfast LPR d={d} N={N}"
    if result.get("status") == "success":
        time_sec = result.get("time_sec", 0.0)
        time_min = result.get("time_min", time_sec)
        time_max = result.get("time_max", time_sec)
        mem_mb = result.get("mem_mb", 0.0)
        mem_min = result.get("mem_min", mem_mb)
        mem_max = result.get("mem_max", mem_mb)
        mem_method = result.get("mem_method", "none")
        log(
            f"  {description}: {time_sec:.2f}s [{time_min:.2f}-{time_max:.2f}], {mem_mb:.1f}MB [{mem_min:.1f}-{mem_max:.1f}] ({mem_method})"
        )
    else:
        log(f"  {description}: {result.get('status', 'error')}")

    return result


def run_stopt_single(N, d, n_runs=1):
    """Run StOpt-NW benchmark."""
    stopt_wrapper = REPO_ROOT / "benchmark" / "scripts" / "cpp" / "stopt_wrapper.py"

    # Grid mode: load evaluation grid from GT file
    gt_dir = str(GT_DIR).replace("\\", "/")
    grid_code = ""
    xq_param = "None"
    if EVAL_MODE == "grid":
        grid_code = f'''
# Grid mode: load x_grid from GT file for exact match with ground truth
import os
from scipy.io import loadmat
gt_grid_file = os.path.join("{gt_dir}", f'gt_d{{d}}_N{{N}}_grid.mat')
if os.path.exists(gt_grid_file):
    gt_grid = loadmat(gt_grid_file)
    x_grid = np.asarray(gt_grid['x_grid'], dtype=np.float64)
    print(f"[StOpt] Using x_grid from GT file: {{x_grid.shape}}", file=sys.stderr)
else:
    # Fallback for large N: generate grid (power-of-two per dimension, FFT-friendly)
    M_per_dim = {{1: 16384, 2: 128, 3: 32}}[d]
    x_min = x_zs.min(axis=0)
    x_max = x_zs.max(axis=0)
    # NO margin - stay within interpolatable range
    if d == 1:
        x_grid = np.linspace(x_min[0], x_max[0], M_per_dim).reshape(-1, 1)
    else:
        grid_axes = [np.linspace(x_min[i], x_max[i], M_per_dim) for i in range(d)]
        mesh = np.meshgrid(*grid_axes, indexing='ij')
        x_grid = np.stack(mesh, axis=-1).reshape(-1, d, order='F')  # Column-major to match MATLAB/R
    print(f"[StOpt] Generated x_grid (no GT file): {{x_grid.shape}}", file=sys.stderr)
'''
        xq_param = "x_grid"

    script = f'''
import sys
import time
import gc
import tracemalloc
import tempfile
import os
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, r"{stopt_wrapper.parent}")

try:
    from stopt_wrapper import stopt_nw, HAS_STOPT
    if not HAS_STOPT:
        print("TIME:nan")
        print("STATUS:error: StOpt not available")
        sys.exit(1)
except ImportError as e:
    print("TIME:nan")
    print(f"STATUS:error: {{e}}")
    sys.exit(1)

N = {N}
d = {d}
SEED = {SEED}
H0 = {H0}
NOISE_STD = {NOISE_STD}
N_RUNS = {n_runs}
GRID_MODE = {EVAL_MODE == "grid"}

gt_dir = r"{gt_dir}"
gt_file = os.path.join(gt_dir, f'gt_d{{d}}_N{{N}}.mat')
if os.path.exists(gt_file):
    gt = loadmat(gt_file)
    x_zs = np.asarray(gt['x_zs'], dtype=np.float64)
    h_N = float(gt['h_N'].flatten()[0])
    y = np.asarray(gt['y'], dtype=np.float64).flatten()
    print(f"[StOpt] Using GT data: {{gt_file}}", file=sys.stderr)
else:
    np.random.seed(SEED)
    x_orig = np.random.rand(N, d)
    x_mean = np.mean(x_orig, axis=0)
    x_std = np.std(x_orig, axis=0, ddof=0)
    x_zs = (x_orig - x_mean) / x_std

    y_true = np.sin(2 * np.pi * x_orig) if d == 1 else np.sin(2 * np.pi * np.mean(x_orig, axis=1, keepdims=True))
    y = y_true.flatten() + NOISE_STD * np.random.randn(N)

    h_N = H0 * N ** (-1 / (d + 4))
    print(f"[StOpt] Generated data for N={{N}} (no GT file)", file=sys.stderr)

{grid_code}

# Double gc + sleep to stabilize memory before baseline capture
gc.collect()
time.sleep(0.1)
gc.collect()

print("READY", flush=True)
time.sleep(0.05)  # Allow parent to capture baseline before algorithm starts

times = []
yhat = None
for run in range(N_RUNS):
    print("RUN_START", flush=True)
    t0 = time.perf_counter()
    yhat, _ = stopt_nw(x_zs, y, bandwidth=h_N, xq={xq_param})
    times.append(time.perf_counter() - t0)
    print("RUN_END", flush=True)

print(f"TIME:{{np.median(times)}}")

# Compute accuracy internally (MSE vs Direct) when possible
# This avoids relying on RESULT_FILE for StOpt.
accuracy = float('nan')
if N <= 65536 and yhat is not None:
    try:
        if GRID_MODE:
            gt_grid_file = os.path.join(gt_dir, f'gt_d{{d}}_N{{N}}_grid.mat')
            if os.path.exists(gt_grid_file):
                gt_grid = loadmat(gt_grid_file)
                gt_vec = np.asarray(gt_grid['nw_gt'], dtype=np.float64).flatten()
            else:
                gt_vec = None
        else:
            if os.path.exists(gt_file):
                gt_vec = np.asarray(gt['nw_gt'], dtype=np.float64).flatten()
            else:
                gt_vec = None

        yhat_vec = np.asarray(yhat, dtype=np.float64).flatten()
        if gt_vec is not None and len(gt_vec) == len(yhat_vec):
            accuracy = float(np.nanmean((yhat_vec - gt_vec) ** 2))
    except Exception:
        accuracy = float('nan')
print(f"ACCURACY:{{accuracy}}")

# Save result for accuracy computation (only for N <= 65536)
if N <= 65536 and yhat is not None:
    result_vec = yhat.flatten()
    result_file = tempfile.mktemp(suffix=".npy", prefix="bench_stopt_")
    np.save(result_file, result_vec)
    print(f"RESULT_FILE:{{result_file}}")
'''
    result = run_with_memory_monitor(
        [PYTHON_PATH, "-c", script], f"StOpt d={d} N={N}", timeout=600
    )
    return _parse_result(result, f"StOpt LPR d={d} N={N}")


# Cache for batched StOpt results
_STOPT_BATCH_RESULTS = {}


def run_stopt_batch(test_list, n_runs=1):
    """Run ALL StOpt-NW benchmarks in a single Python process.

    Args:
        test_list: List of (N, d) tuples to run
        n_runs: Number of runs per test

    Returns:
        Dictionary keyed by (N, d) with results
    """
    global _STOPT_BATCH_RESULTS

    if not test_list:
        return {}

    stopt_wrapper = REPO_ROOT / "benchmark" / "scripts" / "cpp" / "stopt_wrapper.py"
    temp_dir = str(OUTPUT_DIR).replace("\\", "/")
    gt_dir = str(GT_DIR).replace("\\", "/")

    # Build test list for Python
    test_items = [f"    ({N}, {d})" for N, d in test_list]
    test_list_str = "tests = [\n" + ",\n".join(test_items) + "\n]"

    # Grid mode settings
    grid_mode = EVAL_MODE == "grid"

    script = f'''
import sys
import time
import gc
import tracemalloc
import tempfile
import os
import numpy as np
from scipy.io import loadmat
import psutil  # For process-level memory measurement (tracks C++ allocations)

sys.path.insert(0, r"{stopt_wrapper.parent}")

try:
    from stopt_wrapper import stopt_nw, HAS_STOPT
    if not HAS_STOPT:
        print("STATUS:error: StOpt not available")
        sys.exit(1)
except ImportError as e:
    print(f"STATUS:error: {{e}}")
    sys.exit(1)

SEED = {SEED}
H0 = {H0}
NOISE_STD = {NOISE_STD}
N_RUNS = {n_runs}
GRID_MODE = {grid_mode}

gt_dir = r"{gt_dir}"
temp_dir = r"{temp_dir}"

{test_list_str}

# Warmup to avoid first-test cold-start overhead
np.random.seed(SEED)
for d_warmup in (1, 2, 3):
    try:
        n_warm = 128
        x_small = np.random.rand(n_warm, d_warmup)
        x_mean = np.mean(x_small, axis=0)
        x_std = np.std(x_small, axis=0, ddof=0)
        x_small_zs = (x_small - x_mean) / x_std
        y_small = (
            np.sin(2 * np.pi * x_small)
            if d_warmup == 1
            else np.sin(2 * np.pi * np.mean(x_small, axis=1, keepdims=True))
        ).flatten()
        h_small = H0 * n_warm ** (-1 / (d_warmup + 4))
        xq_small = np.random.rand(64, d_warmup)
        _ = stopt_nw(x_small_zs, y_small, bandwidth=h_small, xq=xq_small)
        del x_small, x_small_zs, y_small, xq_small
    except Exception:
        pass

# Double gc + sleep to stabilize memory before baseline capture
gc.collect()
time.sleep(0.1)
gc.collect()

print("READY", flush=True)
time.sleep(0.05)  # Allow parent to capture baseline before algorithm starts

for N, d in tests:
    # Load or generate data
    gt_file = os.path.join(gt_dir, f'gt_d{{d}}_N{{N}}.mat')
    if os.path.exists(gt_file):
        gt = loadmat(gt_file)
        x_zs = np.asarray(gt['x_zs'], dtype=np.float64)
        h_N = float(gt['h_N'].flatten()[0])
        y = np.asarray(gt['y'], dtype=np.float64).flatten()
    else:
        np.random.seed(SEED)
        x_orig = np.random.rand(N, d)
        x_mean = np.mean(x_orig, axis=0)
        x_std = np.std(x_orig, axis=0, ddof=0)
        x_zs = (x_orig - x_mean) / x_std
        y_true = np.sin(2 * np.pi * x_orig) if d == 1 else np.sin(2 * np.pi * np.mean(x_orig, axis=1, keepdims=True))
        y = y_true.flatten() + NOISE_STD * np.random.randn(N)
        h_N = H0 * N ** (-1 / (d + 4))

    # Grid mode: load or generate evaluation grid
    xq = None
    if GRID_MODE:
        gt_grid_file = os.path.join(gt_dir, f'gt_d{{d}}_N{{N}}_grid.mat')
        if os.path.exists(gt_grid_file):
            gt_grid = loadmat(gt_grid_file)
            xq = np.asarray(gt_grid['x_grid'], dtype=np.float64)
        else:
            # Fallback: generate grid (power-of-two per dimension, FFT-friendly)
            x_min = x_zs.min(axis=0)
            x_max = x_zs.max(axis=0)
            # NO margin - stay within interpolatable range
            M_per_dim = {{1: 16384, 2: 128, 3: 32}}[d]
            if d == 1:
                xq = np.linspace(x_min[0], x_max[0], M_per_dim).reshape(-1, 1)
            else:
                grid_axes = [np.linspace(x_min[i], x_max[i], M_per_dim) for i in range(d)]
                mesh = np.meshgrid(*grid_axes, indexing='ij')
                xq = np.stack(mesh, axis=-1).reshape(-1, d, order='F')

# Run benchmark with internal process RSS peak delta (includes C++ allocations)
    import psutil
    import threading

    process = psutil.Process(os.getpid())

    def rss_mb() -> float:
        return float(process.memory_info().rss) / (1024 * 1024)

    def baseline_mb(samples: int = 5, interval: float = 0.01) -> float:
        vals = []
        for _ in range(samples):
            vals.append(rss_mb())
            time.sleep(interval)
        return max(vals) if vals else rss_mb()

    times = []
    mem_deltas = []  # Per-run RSS peak deltas in MB
    yhat = None
    n_runs_test = N_RUNS if N_RUNS > 1 else (5 if N <= 64 else 1)
    for run in range(n_runs_test):
        gc.collect()
        base = baseline_mb()
        peak = [base]
        stop_evt = threading.Event()

        def poll():
            while not stop_evt.is_set():
                v = rss_mb()
                if v > peak[0]:
                    peak[0] = v
                time.sleep(0.001)

        t_poll = threading.Thread(target=poll, daemon=True)
        t_poll.start()

        print("RUN_START", flush=True)
        t0 = time.perf_counter()
        yhat, _ = stopt_nw(x_zs, y, bandwidth=h_N, xq=xq)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        stop_evt.set()
        t_poll.join(timeout=1)

        mem_deltas.append(max(0.0, peak[0] - base))
        print("RUN_END", flush=True)

    # Output timing statistics
    print(f"TIME:{{np.median(times)}}")
    print(f"TIME_MIN:{{np.min(times)}}")
    print(f"TIME_MAX:{{np.max(times)}}")
    print(f"TIME_STD:{{np.std(times)}}")

    # Output RSS peak delta statistics
    print(f"MEM_STOPT:{{float(np.median(mem_deltas))}}")
    print(f"MEM_STOPT_MIN:{{float(np.min(mem_deltas))}}")
    print(f"MEM_STOPT_MAX:{{float(np.max(mem_deltas))}}")
    print(f"MEM_STOPT_STD:{{float(np.std(mem_deltas))}}")


    # Save result for accuracy computation (only for N <= 65536)
    if N <= 65536 and yhat is not None:
        result_vec = yhat.flatten()
        result_file = os.path.join(temp_dir, f"bench_stopt_d{{d}}_N{{N}}.npy")
        np.save(result_file, result_vec)
        print(f"RESULT_FILE:{{result_file}}")
    else:
        print("RESULT_FILE:none")

    # Accuracy vs Direct (MSE) when GT exists
    accuracy = float('nan')
    if N <= 65536 and yhat is not None:
        try:
            gt_vec = None
            if GRID_MODE:
                if os.path.exists(gt_grid_file):
                    gt_grid_acc = loadmat(gt_grid_file)
                    gt_vec = np.asarray(gt_grid_acc['nw_gt'], dtype=np.float64).flatten()
            else:
                if os.path.exists(gt_file):
                    gt_acc = loadmat(gt_file)
                    gt_vec = np.asarray(gt_acc['nw_gt'], dtype=np.float64).flatten()

            yhat_vec = np.asarray(yhat, dtype=np.float64).flatten()
            if gt_vec is not None and len(gt_vec) == len(yhat_vec):
                accuracy = float(np.nanmean((yhat_vec - gt_vec) ** 2))
        except Exception:
            accuracy = float('nan')

    print(f"ACCURACY:{{accuracy}}")

    print("TEST_END", flush=True)
'''

    log(f"[StOpt Batch] Running {len(test_list)} tests in single Python process...")

    result = run_subprocess_capture(
        [PYTHON_PATH, "-c", script],
        f"StOpt Batch ({len(test_list)} tests)",
        timeout=3600,  # 1 hour for large batches
    )

    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    if result.get("status") != "success":
        log(f"[StOpt Batch] status={result.get('status')}")
        if stderr:
            log(f"[StOpt Batch] stderr (tail):\n{stderr[-2000:]}")
        if stdout:
            log(f"[StOpt Batch] stdout (tail):\n{stdout[-2000:]}")
        return {}

    # Parse batched results
    results = {}
    lines = stdout.split("\n")

    current_test_idx = 0
    current_time = None
    current_time_min = None
    current_time_max = None
    current_time_std = None
    current_result_file = None
    current_accuracy = None
    current_mem_stopt = None
    current_mem_stopt_min = None
    current_mem_stopt_max = None
    current_mem_stopt_std = None

    # Get run_peaks for memory distribution
    run_peaks = result.get("run_peaks", [])

    for line in lines:
        line = line.strip()
        if (
            line.startswith("TIME:")
            and not line.startswith("TIME_MIN")
            and not line.startswith("TIME_MAX")
            and not line.startswith("TIME_STD")
        ):
            try:
                current_time = float(line.split(":")[1])
            except (ValueError, IndexError):
                current_time = None
        elif line.startswith("TIME_MIN:"):
            try:
                current_time_min = float(line.split(":")[1])
            except (ValueError, IndexError):
                current_time_min = None
        elif line.startswith("TIME_MAX:"):
            try:
                current_time_max = float(line.split(":")[1])
            except (ValueError, IndexError):
                current_time_max = None
        elif line.startswith("TIME_STD:"):
            try:
                current_time_std = float(line.split(":")[1])
            except (ValueError, IndexError):
                current_time_std = None
        elif (
            line.startswith("MEM_STOPT:")
            and not line.startswith("MEM_STOPT_MIN")
            and not line.startswith("MEM_STOPT_MAX")
            and not line.startswith("MEM_STOPT_STD")
        ):
            try:
                current_mem_stopt = float(line.split(":")[1])
            except (ValueError, IndexError):
                current_mem_stopt = None
        elif line.startswith("MEM_STOPT_MIN:"):
            try:
                current_mem_stopt_min = float(line.split(":")[1])
            except (ValueError, IndexError):
                current_mem_stopt_min = None
        elif line.startswith("MEM_STOPT_MAX:"):
            try:
                current_mem_stopt_max = float(line.split(":")[1])
            except (ValueError, IndexError):
                current_mem_stopt_max = None
        elif line.startswith("MEM_STOPT_STD:"):
            try:
                current_mem_stopt_std = float(line.split(":")[1])
            except (ValueError, IndexError):
                current_mem_stopt_std = None
        elif line.startswith("RESULT_FILE:"):
            rf = line.split(":", 1)[1].strip()
            current_result_file = rf if rf != "none" else None
        elif line.startswith("ACCURACY:"):
            try:
                current_accuracy = float(line.split(":", 1)[1])
            except (ValueError, IndexError):
                current_accuracy = None
        elif line == "TEST_END":
            if current_test_idx < len(test_list):
                N, d = test_list[current_test_idx]

                # Calculate memory from run_peaks (fallback)
                mem_mb = 0.0
                if run_peaks and current_test_idx < len(run_peaks):
                    if current_test_idx == 0:
                        mem_mb = run_peaks[0]
                    else:
                        mem_mb = max(
                            0,
                            run_peaks[current_test_idx]
                            - run_peaks[current_test_idx - 1],
                        )

                # Prefer internal RSS peak-delta measurement if available
                if current_mem_stopt is not None and current_mem_stopt > 0:
                    final_mem = current_mem_stopt
                    final_mem_min = (
                        current_mem_stopt_min
                        if current_mem_stopt_min is not None
                        else current_mem_stopt
                    )
                    final_mem_max = (
                        current_mem_stopt_max
                        if current_mem_stopt_max is not None
                        else current_mem_stopt
                    )
                    final_mem_std = (
                        current_mem_stopt_std
                        if current_mem_stopt_std is not None
                        else 0.0
                    )
                    mem_method = "psutil_rss_internal"
                else:
                    final_mem = mem_mb
                    final_mem_min = mem_mb
                    final_mem_max = mem_mb
                    final_mem_std = 0.0
                    mem_method = result.get("mem_method", "none")

                results[(N, d)] = {
                    "status": "success" if current_time is not None else "error",
                    "time_sec": current_time if current_time is not None else 0.0,
                    "time_min": current_time_min
                    if current_time_min is not None
                    else (current_time or 0.0),
                    "time_max": current_time_max
                    if current_time_max is not None
                    else (current_time or 0.0),
                    "time_std": current_time_std
                    if current_time_std is not None
                    else 0.0,
                    "mem_mb": final_mem,
                    "mem_median": final_mem,
                    "mem_min": final_mem_min,
                    "mem_max": final_mem_max,
                    "mem_std": final_mem_std,
                    "mem_method": mem_method,
                    "baseline_mb": 0.0,
                    "result_file": current_result_file,
                    "accuracy_internal": current_accuracy,
                }

                # Reset for next test
                current_time = None
                current_time_min = None
                current_time_max = None
                current_time_std = None
                current_result_file = None
                current_accuracy = None
                current_mem_stopt = None
                current_mem_stopt_min = None
                current_mem_stopt_max = None
                current_mem_stopt_std = None
                current_test_idx += 1

    log(f"[StOpt Batch] Completed {current_test_idx}/{len(test_list)} tests")
    return results


def run_stopt_single_cached(N, d, n_runs=1):
    """Run StOpt-NW benchmark using cached batch results."""
    global _STOPT_BATCH_RESULTS

    key = (N, d)
    if key in _STOPT_BATCH_RESULTS:
        result = _STOPT_BATCH_RESULTS[key]
    else:
        # Batch not run yet - run single test in batch mode
        results = run_stopt_batch([key], n_runs)
        _STOPT_BATCH_RESULTS.update(results)
        result = results.get(
            key,
            {
                "status": "error",
                "error": "Batch execution failed",
                "time_sec": 0.0,
                "mem_mb": 0.0,
            },
        )

    # Log result
    description = f"StOpt LPR d={d} N={N}"
    if result.get("status") == "success":
        time_sec = result.get("time_sec", 0.0)
        time_min = result.get("time_min", time_sec)
        time_max = result.get("time_max", time_sec)
        mem_mb = result.get("mem_mb", 0.0)
        mem_min = result.get("mem_min", mem_mb)
        mem_max = result.get("mem_max", mem_mb)
        mem_method = result.get("mem_method", "none")
        log(
            f"  {description}: {time_sec:.2f}s [{time_min:.2f}-{time_max:.2f}], {mem_mb:.1f}MB [{mem_min:.1f}-{mem_max:.1f}] ({mem_method})"
        )
    else:
        log(f"  {description}: {result.get('status', 'error')}")

    return result


# Cache for batched Direct MATLAB results
_DIRECT_BATCH_RESULTS = {}


def run_direct_batch(test_list, n_runs=1):
    """Run ALL Direct MATLAB benchmarks in a single MATLAB process.

    Args:
        test_list: List of (N, d, task) tuples to run
        n_runs: Number of runs per test

    Returns:
        Dictionary keyed by (N, d, task) with results
    """
    global _DIRECT_BATCH_RESULTS

    if not test_list:
        return {}

    util_path = str(REPO_ROOT / "fastLPR" / "utility").replace("\\", "/")
    core_path = str(REPO_ROOT / "fastLPR" / "utility" / "core").replace("\\", "/")

    # Build test array for MATLAB
    test_array_str = "tests = {"
    for N, d, task in test_list:
        test_array_str += f"{N}, {d}, '{task}';\n"
    test_array_str += "};"

    matlab_code = f"""
addpath('{util_path}');
addpath('{core_path}');

% Test parameters
SEED = {SEED};
H0 = {H0};
NOISE_STD = {NOISE_STD};
N_RUNS = {n_runs};

{test_array_str}
num_tests = size(tests, 1);

% Warmup: run DirectKDE and DirectNW for all dimensions AND multiple N values
% MATLAB JIT has different optimization paths for different array sizes
% Use N=32, 64, 128 to cover small array JIT paths
for warmup_n = [32, 64, 128]
    for warmup_d = 1:3
        rng(SEED);
        warmup_x = rand(warmup_n, warmup_d);
        [warmup_x_zs, ~, ~] = zscore(warmup_x);
        warmup_h = H0 * warmup_n^(-1/(warmup_d+4));
        if warmup_d == 1
            warmup_y = sin(2*pi*warmup_x) + NOISE_STD * randn(warmup_n, 1);
        else
            warmup_y = sin(2*pi*mean(warmup_x, 2)) + NOISE_STD * randn(warmup_n, 1);
        end
        [~, ~] = DirectKDE(warmup_x_zs, warmup_h, warmup_x_zs);
        [~, ~, ~] = DirectNW(warmup_x_zs, warmup_y, warmup_h, warmup_x_zs);
    end
end
clear warmup_n warmup_d warmup_x warmup_x_zs warmup_h warmup_y;

fprintf(2, 'READY\\n');

% Wait for external memory monitor to capture baseline
% (5 samples at 50ms intervals = 250ms + margin)
pause(0.5);

% Run all tests
for t = 1:num_tests
    N = tests{{t, 1}};
    d = tests{{t, 2}};
    task = tests{{t, 3}};

    % Generate data with consistent seed
    rng(SEED);
    x_orig = rand(N, d);
    [x_zs, ~, ~] = zscore(x_orig);

    h_N = H0 * N^(-1/(d+4));

    if strcmp(task, 'LPR')
        if d == 1
            y_true = sin(2*pi*x_orig);
        else
            y_true = sin(2*pi*mean(x_orig, 2));
        end
        y = y_true + NOISE_STD * randn(N, 1);
    end

    % Timed runs
    fprintf('TEST_START:%d_%d_%s\\n', N, d, task);

    % Per-test warmup to reduce MATLAB JIT cold-start inflation
    warmup_n = min(N, 128);
    try
        if strcmp(task, 'KDE')
            [~, ~] = DirectKDE(x_zs(1:warmup_n, :), h_N, x_zs(1:warmup_n, :));
        else
            [~, ~, ~] = DirectNW(x_zs(1:warmup_n, :), y(1:warmup_n), h_N, x_zs(1:warmup_n, :));
        end
    catch ME
        % Ignore warmup errors
    end
    clear warmup_n ME;

    times = zeros(N_RUNS, 1);
    theoretical_mem = 0;  % Theoretical memory from dbg structure
    err_msg = '';
 
    for run = 1:N_RUNS
        fprintf(2, 'RUN_START\\n');
        try
            tic;
            if strcmp(task, 'KDE')
                [result, dbg] = DirectKDE(x_zs, h_N, x_zs);
            else
                [result, ~, dbg] = DirectNW(x_zs, y, h_N, x_zs);
            end
            times(run) = toc;

            % Get theoretical memory from dbg structure (only need once)
            if run == 1
                theoretical_mem = dbg.theoretical_mem_mb;
            end
        catch ME
            times(run) = nan;
            err_msg = ME.message;
        end
        fprintf(2, 'RUN_END\\n');
    end

    % Report results
    if any(~isnan(times))
        valid_times = times(~isnan(times));
        fprintf('TIME:%f\\n', median(valid_times));
        fprintf('TIME_MIN:%f\\n', min(valid_times));
        fprintf('TIME_MAX:%f\\n', max(valid_times));
        fprintf('TIME_STD:%f\\n', std(valid_times));

        % Report theoretical memory (O(N*M) baseline)
        fprintf('MEM_DIRECT:%f\\n', theoretical_mem);
        fprintf('MEM_DIRECT_MIN:%f\\n', theoretical_mem);
        fprintf('MEM_DIRECT_MAX:%f\\n', theoretical_mem);
    else
        fprintf('TIME:nan\\n');
        fprintf('ERROR:%s\\n', err_msg);
    end
    fprintf('TEST_END:%d_%d_%s\\n', N, d, task);
end

exit(0);
"""

    log(f"Running Direct batch: {len(test_list)} tests in single process...")

    # Set EVAL_MODE in environment - subprocess inherits from os.environ
    os.environ["EVAL_MODE"] = EVAL_MODE

    result = run_with_memory_monitor(
        [MATLAB_CMD, "-batch", matlab_code],
        f"Direct batch ({len(test_list)} tests)",
        timeout=max(600, len(test_list) * 300),  # Direct is O(N^2), needs more time
    )

    # Parse batch results from stdout/stderr
    # MATLAB batch mode can route output to stderr depending on platform.
    results = {}
    stdout = result.get("stdout", "") + "\n" + result.get("stderr", "")
    current_test = None
    current_times = {}
    test_index = 0  # Track which test we're on for memory distribution

    # Get per-run peaks for distribution (each test has n_runs peaks)
    run_peaks = result.get("run_peaks", [])

    for line in stdout.split("\n"):
        line = line.strip()
        if line.startswith("TEST_START:"):
            parts = line.split(":")[1].split("_")
            if len(parts) >= 3:
                N = int(parts[0])
                d = int(parts[1])
                task = parts[2]
                current_test = (N, d, task)
                current_times = {
                    "time_sec": None,
                    "time_min": None,
                    "time_max": None,
                    "time_std": None,
                }
        elif line.startswith("TEST_END:") and current_test:
            # Distribute per-run peaks to this test (same logic as MATLAB batch)
            start_idx = test_index * n_runs
            end_idx = start_idx + n_runs
            test_peaks = (
                run_peaks[start_idx:end_idx] if start_idx < len(run_peaks) else []
            )

            if test_peaks:
                import statistics

                mem_median = statistics.median(test_peaks)
                mem_min = min(test_peaks)
                mem_max = max(test_peaks)
                mem_std = statistics.stdev(test_peaks) if len(test_peaks) > 1 else 0.0
            else:
                mem_median = mem_min = mem_max = mem_std = 0.0

            # Use MATLAB internal memory measurement consistently (don't mix with JobObject)
            # Direct_internal measures MemUsedMATLAB delta which is more accurate
            mem_direct = current_times.get("mem_direct")
            mem_direct_min = current_times.get("mem_direct_min")
            mem_direct_max = current_times.get("mem_direct_max")
            if mem_direct is not None:
                # Always use MATLAB internal measurement for consistency
                # Even if 0, it's more accurate than JobObject for MATLAB
                final_mem = mem_direct
                final_mem_min = (
                    mem_direct_min if mem_direct_min is not None else mem_direct
                )
                final_mem_max = (
                    mem_direct_max if mem_direct_max is not None else mem_direct
                )
                final_mem_median = mem_direct
                final_mem_std = 0.0
                mem_method = "MATLAB_internal"  # Same as fastLPR MATLAB batch
            else:
                # Fallback to JobObject only if MATLAB internal not available at all
                final_mem = mem_max
                final_mem_min = mem_min
                final_mem_max = mem_max
                final_mem_median = mem_median
                final_mem_std = mem_std
                mem_method = result.get("mem_method", "none")

            results[current_test] = {
                "time_sec": current_times.get("time_sec"),
                "time_min": current_times.get("time_min"),
                "time_max": current_times.get("time_max"),
                "time_std": current_times.get("time_std"),
                "result_file": current_times.get("result_file"),
                "mem_mb": final_mem,
                "mem_median": final_mem_median,
                "mem_min": final_mem_min,
                "mem_max": final_mem_max,
                "mem_std": final_mem_std,
                "mem_method": mem_method,
                "baseline_mb": result.get("baseline_mb", 0),
                "status": "success" if current_times.get("time_sec") else "error",
            }
            current_test = None
            test_index += 1
        elif line.startswith("TIME:") and current_test:
            try:
                current_times["time_sec"] = float(line.split(":")[1])
            except ValueError:
                pass
        elif line.startswith("TIME_MIN:") and current_test:
            try:
                current_times["time_min"] = float(line.split(":")[1])
            except ValueError:
                pass
        elif line.startswith("TIME_MAX:") and current_test:
            try:
                current_times["time_max"] = float(line.split(":")[1])
            except ValueError:
                pass
        elif line.startswith("TIME_STD:") and current_test:
            try:
                current_times["time_std"] = float(line.split(":")[1])
            except ValueError:
                pass
        elif line.startswith("MEM_DIRECT:") and current_test:
            try:
                current_times["mem_direct"] = float(line.split(":")[1])
            except ValueError:
                pass
        elif line.startswith("MEM_DIRECT_MIN:") and current_test:
            try:
                current_times["mem_direct_min"] = float(line.split(":")[1])
            except ValueError:
                pass
        elif line.startswith("MEM_DIRECT_MAX:") and current_test:
            try:
                current_times["mem_direct_max"] = float(line.split(":")[1])
            except ValueError:
                pass

    _DIRECT_BATCH_RESULTS.update(results)
    log(f"Direct batch complete: {len(results)}/{len(test_list)} tests successful")
    return results


def run_direct_single(N, d, task="KDE", n_runs=1):
    """Run MATLAB Direct O(N^2) benchmark - uses cached batch results if available."""
    global _DIRECT_BATCH_RESULTS

    if N > 65536:
        return {"time_sec": None, "mem_mb": 0, "status": "skip: N > 65536"}

    # Check if result is cached from batch run
    key = (N, d, task)
    if key in _DIRECT_BATCH_RESULTS:
        result = _DIRECT_BATCH_RESULTS[key]
        method_name = "DirectKDE" if task == "KDE" else "DirectNW"
        # Unified log format: {description}: {time}s [{time_min}-{time_max}], {mem}MB [{mem_min}-{mem_max}] ({mem_method})
        time_sec = result.get("time_sec", 0)
        time_min = result.get("time_min", time_sec)
        time_max = result.get("time_max", time_sec)
        mem_mb = result.get("mem_mb", 0)
        mem_min = result.get("mem_min", mem_mb)
        mem_max = result.get("mem_max", mem_mb)
        mem_method = result.get("mem_method", "none")
        if isinstance(time_sec, (int, float)) and time_sec is not None:
            log(
                f"  {method_name} d={d} N={N}: {time_sec:.2f}s [{time_min:.2f}-{time_max:.2f}], {mem_mb:.1f}MB [{mem_min:.1f}-{mem_max:.1f}] ({mem_method})"
            )
        else:
            log(f"  {method_name} d={d} N={N}: N/A")
        return result

    # Fallback: run single test (should not happen in normal usage)
    util_path = str(REPO_ROOT / "fastLPR" / "utility").replace("\\", "/")
    core_path = str(REPO_ROOT / "fastLPR" / "utility" / "core").replace("\\", "/")

    if task == "KDE":
        method_call = "[result, ~] = DirectKDE(x_zs, h_N, x_zs, opt);"
        method_name = "DirectKDE"
        data_gen = ""
    else:
        method_call = "[result, ~, ~] = DirectNW(x_zs, y, h_N, x_zs, opt);"
        method_name = "DirectNW"
        data_gen = f"""
if d == 1
    y_true = sin(2*pi*x_orig);
else
    y_true = sin(2*pi*mean(x_orig, 2));
end
y = y_true + {NOISE_STD} * randn(N, 1);
"""

    matlab_code = f"""
addpath('{util_path}');
addpath('{core_path}');

N = {N};
d = {d};
SEED = {SEED};
H0 = {H0};
NOISE_STD = {NOISE_STD};
N_RUNS = {n_runs};

rng(SEED);
x_orig = rand(N, d);
[x_zs, ~, ~] = zscore(x_orig);

h_N = H0 * N^(-1/(d+4));
opt.block_size = 1e10;

{data_gen}

fprintf(2, 'READY\\n');
pause(0.05);  % Allow parent to capture baseline

times = zeros(N_RUNS, 1);
for run = 1:N_RUNS
    fprintf(2, 'RUN_START\\n');
    tic;
    {method_call}
    times(run) = toc;
    fprintf(2, 'RUN_END\\n');
end

fprintf('TIME:%f\\n', median(times));
exit(0);
"""
    result = run_with_memory_monitor(
        [MATLAB_CMD, "-batch", matlab_code], f"{method_name} d={d} N={N}", timeout=600
    )
    return _parse_result(result, f"{method_name} d={d} N={N}")


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def _parse_result(result, description):
    """Parse result from run_with_memory_monitor."""
    parsed = {
        "time_sec": result.get("time_sec"),
        "mem_mb": result.get("mem_mb", 0),
        "mem_median": result.get("mem_median", result.get("mem_mb", 0)),
        "mem_min": result.get("mem_min", result.get("mem_mb", 0)),
        "mem_max": result.get("mem_max", result.get("mem_mb", 0)),
        "mem_std": result.get("mem_std", 0),
        "mem_method": result.get(
            "mem_method", "none"
        ),  # Track memory measurement method
        "baseline_mb": result.get("baseline_mb", 0),
        "status": result.get("status", "unknown"),
        "time_min": None,
        "time_max": None,
        "time_std": None,
        "result_file": None,  # Path to result file for accuracy computation
        "accuracy_internal": None,  # Optional internal accuracy (e.g., StOpt)
    }

    # Global convention: N_RUNS=3 and record peak memory.
    # Keep mem_median/min/max for diagnostics, but use mem_max as primary mem_mb.
    parsed["mem_mb"] = parsed.get("mem_max", parsed.get("mem_mb", 0))

    # Parse additional timing info and result file from stdout
    stdout = result.get("stdout", "")
    for line in stdout.split("\n"):
        line = line.strip()
        if line.startswith("TIME_MIN:"):
            try:
                parsed["time_min"] = float(line.split(":")[1])
            except ValueError:
                pass
        elif line.startswith("TIME_MAX:"):
            try:
                parsed["time_max"] = float(line.split(":")[1])
            except ValueError:
                pass
        elif line.startswith("TIME_STD:"):
            try:
                parsed["time_std"] = float(line.split(":")[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_FILE:"):
            parsed["result_file"] = line.split(":", 1)[1].strip()
        elif line.startswith("ACCURACY:"):
            try:
                parsed["accuracy_internal"] = float(line.split(":", 1)[1])
            except ValueError:
                pass

    # Log result (2026-01-09: show both time and mem statistics)
    if parsed["status"] == "success":
        time_sec = parsed["time_sec"] or 0.0
        time_min = parsed.get("time_min") or time_sec
        time_max = parsed.get("time_max") or time_sec
        mem_mb = parsed["mem_mb"] or 0.0
        mem_min = parsed.get("mem_min") or mem_mb
        mem_max = parsed.get("mem_max") or mem_mb
        mem_method = parsed.get("mem_method", "none")
        # Format: time (median) [min-max], mem (median) [min-max] (method)
        log(
            f"  {description}: {time_sec:.2f}s [{time_min:.2f}-{time_max:.2f}], {mem_mb:.1f}MB [{mem_min:.1f}-{mem_max:.1f}] ({mem_method})"
        )
    else:
        log(f"  {description}: {parsed['status']}")

    return parsed


# ==============================================================================
# ALL METHOD DEFINITIONS
# ==============================================================================

ALL_METHODS = {
    "fastlpr": {
        "kde": {
            "python": run_python_single_cached,
            "r": run_r_single,
            "matlab": run_matlab_single,
        },
        "lpr": {
            "python": run_python_single_cached,
            "r": run_r_single,
            "matlab": run_matlab_single,
        },
    },
    "ks": {"kde": {"r": run_ks_single_cached}},
    "fksum": {"kde": {"r": run_fksum_single_cached}},
    "locfit": {"lpr": {"r": run_locfit_single_cached}},
    "npregfast": {"lpr": {"r": run_npregfast_single_cached}},
    "stopt": {"lpr": {"c++": run_stopt_single_cached}},
    "direct": {
        "kde": {
            "matlab": lambda N, d, n_runs=1: run_direct_single(N, d, "KDE", n_runs)
        },
        "lpr": {
            "matlab": lambda N, d, n_runs=1: run_direct_single(N, d, "LPR", n_runs)
        },
    },
}


def get_jss_method_name(internal_name: str, task: str) -> str:
    """Map internal method names to JSS-standard names for plot compatibility.

    Args:
        internal_name: Internal method name (e.g., 'fksum', 'stopt', 'direct')
        task: Task type ('KDE' or 'LPR')

    Returns:
        JSS-standard method name for publication figures
    """
    # fastlpr is already handled separately as 'fastKDE'/'fastLPR'
    mapping = {
        "ks": "ks",  # Unchanged
        "fksum": "FKSUM",  # Uppercase
        "locfit": "locfit",  # Unchanged
        "npregfast": "npregfast",  # Unchanged
        "stopt": "StOpt-NW",  # Full name with suffix
        "direct": f"Direct{task}",  # DirectKDE or DirectNW
    }
    return mapping.get(internal_name, internal_name)


def run_all_benchmarks(n_values, dimensions, methods, n_runs):
    """Run all benchmarks and collect results."""
    results = []

    # Calculate total tests
    total_tests = 0
    for method_name in methods:
        method_def = ALL_METHODS.get(method_name, {})
        for task_name, task_def in method_def.items():
            for lang in task_def.keys():
                for d in dimensions:
                    for N in n_values:
                        if N > MAX_N_PER_DIM.get(d, 2**25):
                            continue
                        if method_name in ["fksum", "npregfast"] and d > 1:
                            continue
                        if method_name == "direct" and N > 65536:
                            continue
                        total_tests += 1

    completed = 0

    log(f"\n{'=' * 60}")
    log(f"Starting {total_tests} benchmark tests")
    log(f"N values: {len(n_values)} ({min(n_values):,} to {max(n_values):,})")
    log(f"Dimensions: {dimensions}")
    log(f"Methods: {methods}")
    log(f"N_RUNS: {n_runs}")
    log(f"{'=' * 60}\n")

    # Pre-batch all Python tests to avoid startup overhead
    # Sort by task (KDE first, then LPR) and N descending within each task
    # This prevents memory reuse between KDE and LPR at the same N
    if "fastlpr" in methods:
        python_tests = []
        for d in dimensions:
            for N in n_values:
                if N > MAX_N_PER_DIM.get(d, 2**25):
                    continue
                python_tests.append((N, d, "KDE"))
                python_tests.append((N, d, "LPR"))
        # Sort by task (KDE=0, LPR=1) then by N descending
        python_tests.sort(key=lambda x: (0 if x[2] == "KDE" else 1, -x[0]))
        if python_tests:
            log(
                f"\n=== PRE-BATCHING Python fastKDE/fastLPR ({len(python_tests)} tests) ==="
            )
            batch_results = run_python_batch(python_tests, n_runs)
            _PYTHON_BATCH_RESULTS.update(batch_results)

    # Pre-batch all MATLAB tests to avoid startup overhead (~20s per test saved)
    # Sort by N descending to get accurate memory measurements for large N
    # (MATLAB reuses memory, so running large N first captures actual allocation)
    if "fastlpr" in methods:
        matlab_tests = []
        for d in dimensions:
            for N in n_values:
                if N > MAX_N_PER_DIM.get(d, 2**25):
                    continue
                matlab_tests.append((N, d, "KDE"))
                matlab_tests.append((N, d, "LPR"))
        # Sort by task (KDE first, then LPR) then by N descending
        # This ensures KDE tests run first (large N to small N), then LPR tests
        # Prevents memory reuse between KDE and LPR at same N
        matlab_tests.sort(key=lambda x: (0 if x[2] == "KDE" else 1, -x[0]))
        if matlab_tests:
            log(
                f"\n=== PRE-BATCHING MATLAB fastKDE/fastLPR ({len(matlab_tests)} tests) ==="
            )
            run_matlab_batch(matlab_tests, n_runs)

    # Pre-batch all R fastLPR tests to avoid startup overhead (~0.12s per test saved)
    if "fastlpr" in methods:
        r_tests = []
        for d in dimensions:
            for N in n_values:
                if N > MAX_N_PER_DIM.get(d, 2**25):
                    continue
                r_tests.append((N, d, "KDE"))
                r_tests.append((N, d, "LPR"))
        if r_tests:
            log(f"\n=== PRE-BATCHING R fastKDE/fastLPR ({len(r_tests)} tests) ===")
            batch_results = run_r_batch(r_tests, get_n_runs_r(n_runs))
            _R_BATCH_RESULTS.update(batch_results)

    # Pre-batch Direct MATLAB tests too
    if "direct" in methods:
        direct_tests = []
        for d in dimensions:
            for N in n_values:
                if N > 65536:
                    continue
                if N > MAX_N_PER_DIM.get(d, 2**25):
                    continue
                direct_tests.append((N, d, "KDE"))
                direct_tests.append((N, d, "LPR"))
        if direct_tests:
            log(f"\n=== PRE-BATCHING MATLAB Direct ({len(direct_tests)} tests) ===")
            run_direct_batch(direct_tests, n_runs)

    # Pre-batch ks tests (R KDE competitor)
    if "ks" in methods:
        ks_tests = []
        for d in dimensions:
            for N in n_values:
                if N > MAX_N_PER_DIM.get(d, 2**25):
                    continue
                ks_tests.append((N, d))
        if ks_tests:
            log(f"\n=== PRE-BATCHING R ks ({len(ks_tests)} tests) ===")
            batch_results = run_ks_batch(ks_tests, get_n_runs_r(n_runs))
            _KS_BATCH_RESULTS.update(batch_results)

    # Pre-batch FKSUM tests (R KDE competitor, 1D only)
    if "fksum" in methods:
        fksum_tests = []
        for d in dimensions:
            if d > 1:  # FKSUM is 1D only
                continue
            for N in n_values:
                if N > MAX_N_PER_DIM.get(d, 2**25):
                    continue
                fksum_tests.append((N, d))
        if fksum_tests:
            log(f"\n=== PRE-BATCHING R FKSUM ({len(fksum_tests)} tests) ===")
            batch_results = run_fksum_batch(fksum_tests, get_n_runs_r(n_runs))
            _FKSUM_BATCH_RESULTS.update(batch_results)

    # Pre-batch locfit tests (R LPR competitor)
    if "locfit" in methods:
        locfit_tests = []
        for d in dimensions:
            for N in n_values:
                if N > MAX_N_PER_DIM.get(d, 2**25):
                    continue
                locfit_tests.append((N, d))
        if locfit_tests:
            log(f"\n=== PRE-BATCHING R locfit ({len(locfit_tests)} tests) ===")
            batch_results = run_locfit_batch(locfit_tests, get_n_runs_r(n_runs))
            _LOCFIT_BATCH_RESULTS.update(batch_results)

    # Pre-batch npregfast tests (R LPR competitor, 1D only)
    if "npregfast" in methods:
        npregfast_tests = []
        for d in dimensions:
            if d > 1:  # npregfast is 1D only
                continue
            for N in n_values:
                if N > MAX_N_PER_DIM.get(d, 2**25):
                    continue
                npregfast_tests.append((N, d))
        if npregfast_tests:
            log(f"\n=== PRE-BATCHING R npregfast ({len(npregfast_tests)} tests) ===")
            batch_results = run_npregfast_batch(npregfast_tests, get_n_runs_r(n_runs))
            _NPREGFAST_BATCH_RESULTS.update(batch_results)

    # Pre-batch stopt tests (C++ LPR competitor)
    if "stopt" in methods:
        stopt_tests = []
        for d in dimensions:
            for N in n_values:
                if N > MAX_N_PER_DIM.get(d, 2**25):
                    continue
                stopt_tests.append((N, d))
        if stopt_tests:
            log(f"\n=== PRE-BATCHING C++ StOpt ({len(stopt_tests)} tests) ===")
            batch_results = run_stopt_batch(stopt_tests, n_runs)
            _STOPT_BATCH_RESULTS.update(batch_results)

    for method_name in methods:
        method_def = ALL_METHODS.get(method_name)
        if method_def is None:
            log(f"WARNING: Unknown method '{method_name}', skipping")
            continue

        log(f"\n=== {method_name.upper()} Benchmarks ===")

        for task_name, task_def in method_def.items():
            task_upper = task_name.upper()
            log(f"\n--- {task_upper} ---")

            for lang, func in task_def.items():
                for d in dimensions:
                    for N in n_values:
                        # Skip if N exceeds dimension-specific limit
                        if N > MAX_N_PER_DIM.get(d, 2**25):
                            continue

                        # Method-specific skip conditions (must match total_tests calculation)
                        if method_name in ["fksum", "npregfast"] and d > 1:
                            continue
                        if method_name == "direct" and N > 65536:
                            continue

                        completed += 1

                        log(
                            f"\n[{completed}/{total_tests}] {method_name} {task_upper} d={d} N={N:,} ({lang})"
                        )

                        try:
                            if method_name == "fastlpr":
                                result = func(N, d, task_upper, n_runs)
                                result_method = f"fast{task_upper}"
                            else:
                                result = func(N, d, n_runs)
                                result_method = get_jss_method_name(
                                    method_name, task_upper
                                )

                            # Compute accuracy_vs_direct (mode-aware loading)
                            result_file = result.get("result_file")
                            acc_stats = {
                                "mse": np.nan,
                                "acc_mask_applied": False,
                                "acc_mask_ratio": np.nan,
                                "acc_mask_n_total": 0,
                                "acc_mask_n_used": 0,
                            }

                            if method_name == "direct":
                                # Direct methods ARE the baseline, so MSE = 0
                                accuracy = 0.0
                            elif N <= 65536:
                                # Collect mask diagnostics whenever we have an output vector.
                                if result_file is not None:
                                    acc_stats = compute_accuracy_vs_direct_stats(
                                        result_file, task_upper, d, N, mode=EVAL_MODE
                                    )

                                # Prefer method-provided accuracy if available (e.g., StOpt)
                                accuracy_internal = result.get("accuracy_internal")
                                if method_name == "stopt" and isinstance(
                                    accuracy_internal, (int, float)
                                ):
                                    accuracy = float(accuracy_internal)
                                else:
                                    accuracy = float(acc_stats.get("mse", np.nan))
                            else:
                                # No ground truth for N > 65536
                                accuracy = np.nan

                            # Record failed tests with status (fix survivorship bias)
                            # Previously: failed tests were skipped, causing missing data points
                            if (
                                N <= 65536
                                and result_file is None
                                and method_name != "direct"
                                and not (
                                    isinstance(accuracy, (int, float))
                                    and np.isfinite(accuracy)
                                )
                            ):
                                log(
                                    f"  Recording failed result: no output (method likely failed)"
                                )
                                results.append(
                                    {
                                        "method": get_jss_method_name(
                                            method_name, task_upper
                                        ),
                                        "task": task_upper,
                                        "lang": lang.upper()
                                        if lang != "c++"
                                        else "C++",
                                        "d": d,
                                        "N": N,
                                        "time_sec": result.get(
                                            "time_sec"
                                        ),  # May have timing even if no output
                                        "mem_mb": result.get(
                                            "mem_max", result.get("mem_mb", 0)
                                        ),
                                        "mem_median": result.get("mem_median", 0),
                                        "mem_min": result.get("mem_min", 0),
                                        "mem_max": result.get("mem_max", 0),
                                        "mem_std": result.get("mem_std", 0),
                                        "mem_method": result.get("mem_method", "none"),
                                        "accuracy_vs_direct": np.nan,  # Cannot compute without output
                                        "acc_mask_applied": acc_stats.get(
                                            "acc_mask_applied", False
                                        ),
                                        "acc_mask_ratio": acc_stats.get(
                                            "acc_mask_ratio", np.nan
                                        ),
                                        "acc_mask_n_total": acc_stats.get(
                                            "acc_mask_n_total", 0
                                        ),
                                        "acc_mask_n_used": acc_stats.get(
                                            "acc_mask_n_used", 0
                                        ),
                                        "time_min": result.get("time_min"),
                                        "time_max": result.get("time_max"),
                                        "time_std": result.get("time_std"),
                                        "status": "failed: no output",
                                    }
                                )
                                continue

                            results.append(
                                {
                                    "method": result_method,
                                    "task": task_upper,
                                    "lang": lang.upper() if lang != "c++" else "C++",
                                    "d": d,
                                    "N": N,
                                    "time_sec": result["time_sec"],
                                    "mem_mb": result.get(
                                        "mem_max", result.get("mem_mb", 0)
                                    ),
                                    "mem_median": result.get(
                                        "mem_median", result.get("mem_mb", 0)
                                    ),
                                    "mem_min": result.get(
                                        "mem_min", result.get("mem_mb", 0)
                                    ),
                                    "mem_max": result.get(
                                        "mem_max", result.get("mem_mb", 0)
                                    ),
                                    "mem_std": result.get("mem_std", 0),
                                    "mem_method": result.get("mem_method", "none"),
                                    "accuracy_vs_direct": accuracy,
                                    "acc_mask_applied": acc_stats.get(
                                        "acc_mask_applied", False
                                    ),
                                    "acc_mask_ratio": acc_stats.get(
                                        "acc_mask_ratio", np.nan
                                    ),
                                    "acc_mask_n_total": acc_stats.get(
                                        "acc_mask_n_total", 0
                                    ),
                                    "acc_mask_n_used": acc_stats.get(
                                        "acc_mask_n_used", 0
                                    ),
                                    "time_min": result.get("time_min"),
                                    "time_max": result.get("time_max"),
                                    "time_std": result.get("time_std"),
                                    "status": result["status"],
                                }
                            )

                        except Exception as e:
                            log(f"  EXCEPTION: {e}")
                            results.append(
                                {
                                    "method": get_jss_method_name(
                                        method_name, task_upper
                                    ),
                                    "task": task_upper,
                                    "lang": lang.upper() if lang != "c++" else "C++",
                                    "d": d,
                                    "N": N,
                                    "time_sec": None,
                                    "mem_mb": 0,
                                    "mem_median": 0,
                                    "mem_min": 0,
                                    "mem_max": 0,
                                    "mem_std": 0,
                                    "mem_method": "none",
                                    "accuracy_vs_direct": np.nan,
                                    "acc_mask_applied": False,
                                    "acc_mask_ratio": np.nan,
                                    "acc_mask_n_total": 0,
                                    "acc_mask_n_used": 0,
                                    "time_min": None,
                                    "time_max": None,
                                    "time_std": None,
                                    "status": f"exception: {str(e)[:100]}",
                                }
                            )

    return results


def check_environment(methods: list) -> bool:
    """Comprehensive environment assertion before benchmark execution.

    Validates that all required components are properly installed and accessible.
    Prints status for each check and exits if critical components are missing.

    Args:
        methods: List of methods to benchmark (determines which checks to run)

    Returns:
        bool: True if all required checks pass
    """
    import subprocess
    import shutil

    log("=" * 60)
    log("Environment Validation")
    log("=" * 60)

    errors = []
    warnings = []

    # Determine which language checks are needed
    need_python = any(
        m in methods for m in ["fastlpr", "fastKDE", "fastLPR", "DirectKDE", "DirectNW"]
    )
    need_r = any(
        m in methods for m in ["fastlpr", "ks", "FKSUM", "locfit", "npregfast"]
    )
    need_matlab = any(
        m in methods for m in ["fastlpr", "direct", "DirectKDE", "DirectNW"]
    )
    need_cpp = "stopt" in methods or "StOpt-NW" in methods

    # ==========================================================================
    # Python Checks
    # ==========================================================================
    if need_python:
        log("[Python] Checking components...")

        # Check 1: fastlpr import
        try:
            from fastlpr import cv_fastkde

            log("  [OK] fastlpr package importable")
        except ImportError as e:
            errors.append(f"Python: fastlpr package not installed - {e}")
            log(f"  [ERROR] fastlpr package MISSING: {e}")

        # Check 2: NUFFT backend
        try:
            from fastlpr.nufft import get_backend

            backend_obj, config = get_backend()
            backend_name = config.name
            log(f"  [OK] NUFFT backend: {backend_name}")
        except ImportError as e:
            errors.append(f"Python: NUFFT backend not available - {e}")
            log(f"  [ERROR] NUFFT backend MISSING: {e}")

        # Check 3: psutil for memory monitoring
        try:
            import psutil

            log("  [OK] psutil available for memory monitoring")
        except ImportError:
            errors.append(
                "Python: psutil not installed (required for memory monitoring)"
            )
            log("  [ERROR] psutil MISSING")

    # ==========================================================================
    # R Checks
    # ==========================================================================
    if need_r:
        log("[R] Checking components...")

        # Check R executable
        r_path = Path(R_PATH)
        if r_path.exists():
            log(f"  [OK] Rscript found: {R_PATH}")
        else:
            errors.append(f"R: Rscript not found at {R_PATH}")
            log(f"  [ERROR] Rscript NOT FOUND: {R_PATH}")

        # Check Rcpp DLL for fastLPR_R
        rcpp_dll = REPO_ROOT / "fastLPR_R" / "src" / "fastlpr.dll"
        if rcpp_dll.exists():
            log(f"  [OK] Rcpp DLL exists: {rcpp_dll.name}")
        else:
            # Also check for .so on Linux
            rcpp_so = REPO_ROOT / "fastLPR_R" / "src" / "fastlpr.so"
            if rcpp_so.exists():
                log(f"  [OK] Rcpp SO exists: {rcpp_so.name}")
            else:
                warnings.append(
                    "R: Rcpp DLL/SO not found - fastLPR_R will use pure R (10-35x slower)"
                )
                log("  [WARN] Rcpp DLL/SO not found (will use pure R, 10-35x slower)")

        # Check R packages
        if r_path.exists():
            r_packages = ["ks", "FKSUM", "locfit", "npregfast"]
            for pkg in r_packages:
                try:
                    result = subprocess.run(
                        [str(r_path), "-e", f'library({pkg}); cat("OK")'],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if "OK" in result.stdout:
                        log(f"  [OK] R package '{pkg}' available")
                    else:
                        warnings.append(f"R: Package '{pkg}' not installed")
                        log(f"  [WARN] R package '{pkg}' not installed")
                except subprocess.TimeoutExpired:
                    log(f"  - R package '{pkg}' check timed out")
                except Exception as e:
                    log(f"  - R package '{pkg}' check failed: {e}")

    # ==========================================================================
    # MATLAB Checks
    # ==========================================================================
    if need_matlab:
        log("[MATLAB] Checking components...")

        # Check MATLAB responsive
        matlab_cmd = shutil.which(MATLAB_CMD)
        if matlab_cmd:
            log(f"  [OK] MATLAB found: {matlab_cmd}")
            try:
                result = subprocess.run(
                    [MATLAB_CMD, "-batch", "disp('OK')"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if "OK" in result.stdout:
                    log("  [OK] MATLAB responsive")
                else:
                    warnings.append("MATLAB: Not responding correctly")
                    log("  [WARN] MATLAB not responding as expected")
            except subprocess.TimeoutExpired:
                warnings.append("MATLAB: Startup timed out (>60s)")
                log("  [WARN] MATLAB startup timed out")
            except Exception as e:
                warnings.append(f"MATLAB: Check failed - {e}")
                log(f"  [WARN] MATLAB check failed: {e}")
        else:
            errors.append(f"MATLAB: Command '{MATLAB_CMD}' not found in PATH")
            log(f"  [ERROR] MATLAB not found in PATH")

        # Check toolbox path (cv_fastKDE function)
        if matlab_cmd:
            try:
                result = subprocess.run(
                    [
                        MATLAB_CMD,
                        "-batch",
                        f"addpath(genpath('{REPO_ROOT}')); disp(exist('cv_fastKDE','file'))",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if "2" in result.stdout:
                    log("  [OK] cv_fastKDE function accessible")
                else:
                    warnings.append("MATLAB: cv_fastKDE function not found on path")
                    log("  [WARN] cv_fastKDE not on MATLAB path")
            except Exception as e:
                log(f"  - Toolbox path check failed: {e}")

    # ==========================================================================
    # C++/StOpt Checks
    # ==========================================================================
    if need_cpp:
        log("[C++/StOpt] Checking components...")

        # Check StOpt DLLs/PYD
        stopt_bin = REPO_ROOT / "external" / "StOpt" / "BUILD_MINGW" / "bin"
        if stopt_bin.exists():
            # Check for both .dll and .pyd files (Python extensions)
            dlls = list(stopt_bin.glob("*.dll")) + list(stopt_bin.glob("*.pyd"))
            if len(dlls) > 0:
                log(
                    f"  [OK] StOpt modules found: {len(dlls)} files in {stopt_bin.name}/"
                )
            else:
                errors.append("C++: StOpt modules not found in BUILD_MINGW/bin/")
                log("  [ERROR] StOpt modules not found")
        else:
            errors.append(f"C++: StOpt BUILD_MINGW directory not found")
            log(f"  [ERROR] StOpt BUILD_MINGW not found")

        # Check StOpt wrapper
        try:
            sys.path.insert(0, str(SCRIPT_DIR / "cpp"))
            from stopt_wrapper import HAS_STOPT, STOPT_IMPORT_ERROR

            if HAS_STOPT:
                log("  [OK] StOpt wrapper importable and StOptReg available")
            else:
                detail = STOPT_IMPORT_ERROR or "unknown import error"
                errors.append(
                    "C++: StOptReg module not available (HAS_STOPT=False). "
                    f"Python={sys.version.split()[0]} ({sys.executable}). "
                    f"Error={detail}"
                )
                log("  [ERROR] StOptReg module not available")
                log(f"          Python={sys.version.split()[0]} ({sys.executable})")
                log(f"          Error={detail}")
        except ImportError as e:
            errors.append(f"C++: StOpt wrapper import failed - {e}")
            log(f"  [ERROR] StOpt wrapper MISSING: {e}")

    # ==========================================================================
    # Data Checks
    # ==========================================================================
    log("[Data] Checking ground truth...")
    # Mode-aware pattern: grid mode needs _grid.mat files
    if EVAL_MODE == "grid":
        gt_pattern = "gt_*_grid.mat"
    else:
        gt_pattern = "gt_d[123]_N*.mat"
    gt_files = (
        [
            f
            for f in GT_DIR.glob(gt_pattern)
            if not (EVAL_MODE != "grid" and "_grid.mat" in f.name)
        ]
        if GT_DIR.exists()
        else []
    )
    expected_count = len(N_VALUES_DIRECT) * len(DIMENSIONS)
    if len(gt_files) >= expected_count:
        log(f"  [OK] Ground truth: {len(gt_files)}/{expected_count} files")
    elif len(gt_files) > 0:
        warnings.append(
            f"Ground truth incomplete: {len(gt_files)}/{expected_count} (mode={EVAL_MODE})"
        )
        log(
            f"  [WARN] Ground truth incomplete: {len(gt_files)}/{expected_count} (mode={EVAL_MODE})"
        )
    else:
        warnings.append(
            f"No ground truth files for mode={EVAL_MODE} (will be auto-generated)"
        )
        log(
            f"  [WARN] No ground truth files for mode={EVAL_MODE} (will be auto-generated)"
        )

    # ==========================================================================
    # Summary
    # ==========================================================================
    log("=" * 60)
    if errors:
        log(f"[FAIL] FAILED: {len(errors)} critical error(s)")
        for err in errors:
            log(f"   ERROR: {err}")
        log("")
        log("Cannot proceed with benchmark. Please fix the above errors.")
        return False
    elif warnings:
        log(f"[WARN] PASSED with {len(warnings)} warning(s)")
        for warn in warnings:
            log(f"   WARNING: {warn}")
        log("")
        log(
            "Proceeding with benchmark (some methods may be slower or skip accuracy)..."
        )
        return True
    else:
        log("[PASS] ALL CHECKS PASSED")
        return True


def check_ground_truth(skip_generate=False):
    """Check if ground truth files exist and auto-generate if missing.

    Args:
        skip_generate: If True, skip auto-generation even if files are missing

    Returns:
        bool: True if ground truth is available
    """
    if not GT_DIR.exists():
        GT_DIR.mkdir(parents=True, exist_ok=True)

    # Mode-aware ground truth file pattern (Req 2.4)
    if EVAL_MODE == "grid":
        gt_pattern = "gt_*_grid.mat"  # Grid mode: gt_d{d}_N{N}_grid.mat
    else:
        gt_pattern = "gt_d[123]_N*.mat"  # Data-point mode: gt_d{d}_N{N}.mat

    gt_files = [
        f
        for f in GT_DIR.glob(gt_pattern)
        if not (EVAL_MODE != "grid" and "_grid.mat" in f.name)
    ]

    # Check expected files for N <= 65536 (Direct method limit)
    expected_count = len(N_VALUES_DIRECT) * len(
        DIMENSIONS
    )  # 12 N values * 3 dims = 36 files

    if len(gt_files) >= expected_count:
        log(f"Ground truth: {len(gt_files)}/{expected_count} files (mode={EVAL_MODE})")
        return True

    if len(gt_files) > 0:
        log(
            f"Ground truth: {len(gt_files)}/{expected_count} files (mode={EVAL_MODE}, incomplete)"
        )
    else:
        log(
            f"[WARNING] No ground truth files for mode={EVAL_MODE} in data/ground_truth/"
        )

    if skip_generate:
        log("   Skipping generation (--skip-gt). Accuracy will return NaN.")
        return False
    else:
        log("Auto-generating ground truth using MATLAB...")
        return generate_ground_truth()


def generate_ground_truth():
    """Generate ground truth files using MATLAB benchmark_direct.m.

    Returns:
        bool: True if generation succeeded
    """
    matlab_script = (
        REPO_ROOT / "benchmark" / "scripts" / "matlab" / "benchmark_direct.m"
    )
    if not matlab_script.exists():
        log(f"[ERROR] MATLAB script not found: {matlab_script}")
        return False

    log("Running MATLAB benchmark_direct.m...")
    log("(This generates O(N^2) ground truth for N up to 65536)")

    import subprocess

    try:
        # Set EVAL_MODE in environment - subprocess inherits from os.environ
        os.environ["EVAL_MODE"] = EVAL_MODE

        # Run MATLAB in batch mode
        cmd = [
            MATLAB_CMD,
            "-batch",
            f"run('{str(matlab_script).replace(chr(92), '/')}')",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            cwd=str(REPO_ROOT),
        )

        if result.returncode == 0:
            gt_files = list(GT_DIR.glob("gt_*.mat"))
            log(f"Ground truth generation complete: {len(gt_files)} files created")
            return True
        else:
            log(f"[ERROR] MATLAB failed with return code {result.returncode}")
            if result.stderr:
                log(f"  stderr: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        log("[ERROR] MATLAB timeout (1 hour)")
        return False
    except FileNotFoundError:
        log(f"[ERROR] MATLAB not found: {MATLAB_CMD}")
        log("   Set MATLAB_CMD environment variable if MATLAB is not in PATH")
        return False
    except Exception as e:
        log(f"[ERROR] Failed to run MATLAB: {e}")
        return False


def main():
    """Main entry point."""
    if sys.version_info < (3, 12) or sys.version_info >= (3, 14):
        print(
            "ERROR: Unsupported Python version for this repo. "
            "Expected >=3.12,<3.14 (StOptReg is built for Python 3.12).\n"
            f"Current: {sys.version.split()[0]} ({sys.executable})\n\n"
            "Fix: use the pinned environment from the repo root:\n"
            "  uv run python -V\n"
            "  uv run python benchmark/scripts/run_all_benchmarks.py --methods all\n"
        )
        return 1

    available_methods = list(ALL_METHODS.keys())

    parser = argparse.ArgumentParser(
        description="Unified Benchmark Runner with psutil RSS Delta Monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
    # Run all methods (default N_RUNS=3)
    uv run python benchmark/scripts/run_all_benchmarks.py --methods all

    # Quick run (N_RUNS=1)
    uv run python benchmark/scripts/run_all_benchmarks.py --quick --methods fastlpr

    # Run specific methods
    uv run python benchmark/scripts/run_all_benchmarks.py --methods fastlpr ks stopt

    # Skip ground truth auto-generation
    uv run python benchmark/scripts/run_all_benchmarks.py --skip-gt --methods fastlpr

Available Methods: {", ".join(available_methods)}
""",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Use N_RUNS=1 instead of default {N_RUNS_DEFAULT}",
    )
    parser.add_argument(
        "--mode",
        choices=["data_point", "grid"],
        default="grid",
        help="Evaluation mode: grid (eval on GT x_grid; fallback to power-of-two grid) or data_point (eval at x)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Methods to benchmark (default: all)",
    )
    parser.add_argument(
        "--dims", nargs="+", type=int, default=[1, 2, 3], help="Dimensions to test"
    )
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_FILE), help="Output CSV file"
    )
    parser.add_argument(
        "--skip-gt",
        action="store_true",
        help="Skip ground truth auto-generation (accuracy will be NaN)",
    )

    args = parser.parse_args()

    # Handle 'all' methods
    if "all" in args.methods:
        methods = available_methods
    else:
        methods = args.methods
        for m in methods:
            if m not in available_methods:
                print(f"ERROR: Unknown method '{m}'. Available: {available_methods}")
                return 1

    n_runs = N_RUNS_QUICK if args.quick else N_RUNS_DEFAULT
    n_values = N_VALUES_QUICK if args.quick else N_VALUES_FULL

    # Set global M_INTERNAL based on quick mode
    global M_INTERNAL
    M_INTERNAL = M_INTERNAL_QUICK if args.quick else M_INTERNAL_FULL

    # Set global evaluation mode
    global EVAL_MODE
    EVAL_MODE = args.mode

    # Update output filename for grid mode
    output_file = args.output
    if args.mode == "grid" and args.output == str(OUTPUT_FILE):
        output_file = str(OUTPUT_DIR / "benchmark_results_grid.csv")

    # Auto-archive existing results and figures before new run
    archive_dir = REPO_ROOT / "dev" / "archive" / "benchmark"
    output_path = Path(output_file)
    figures_dir = REPO_ROOT / "benchmark" / "figures"

    if output_path.exists() or any(figures_dir.glob("fig7_*.png")):
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_archive = archive_dir / timestamp
        run_archive.mkdir(parents=True, exist_ok=True)

        archived = []
        # Archive existing CSV
        if output_path.exists():
            import shutil

            shutil.move(str(output_path), str(run_archive / output_path.name))
            archived.append(output_path.name)

        # Archive existing figures
        for fig in figures_dir.glob("fig7_*.*"):
            import shutil

            shutil.move(str(fig), str(run_archive / fig.name))
            archived.append(fig.name)

        if archived:
            log(
                f"[Archive] Moved {len(archived)} old files to: {run_archive.relative_to(REPO_ROOT)}"
            )
            for f in archived:
                log(f"  - {f}")

    # Environment validation - exit early if critical components missing
    if not check_environment(methods):
        return 1

    log(f"{'=' * 60}")
    log("Unified Benchmark Runner - psutil Memory Monitoring")
    log(f"{'=' * 60}")
    log(f"psutil available: {check_psutil()}")
    mem_info = get_memory_metric_info()
    log(f"Memory metric: {mem_info['metric']} ({mem_info['description']})")
    log(f"Platform: {mem_info['platform']}")
    log(f"Evaluation mode: {args.mode}")
    log(f"Quick mode: {args.quick}")
    log(f"Methods: {methods}")
    log(f"Dimensions: {args.dims}")
    log(f"N range: {min(n_values):,} to {max(n_values):,} ({len(n_values)} values)")
    log(f"M_INTERNAL: {M_INTERNAL}")
    log(f"N_RUNS: {n_runs}")
    log(f"Output: {output_file}")
    check_ground_truth(skip_generate=args.skip_gt)
    log(f"{'=' * 60}")

    results = run_all_benchmarks(n_values, args.dims, methods, n_runs)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    log(f"\nSaved {len(df)} results to: {args.output}")

    log("\nDONE!")
    return 0


if __name__ == "__main__":
    main()
