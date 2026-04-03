"""
StOpt Kernel Regression Benchmark Wrapper
"""

import os
import sys
import time
import numpy as np

# Add StOptReg module path
# Path: benchmark/scripts/cpp/ -> benchmark/scripts/ -> benchmark/ -> repo_root/
WRAPPER_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.abspath(os.path.join(WRAPPER_DIR, ".."))
BENCHMARK_DIR = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(BENCHMARK_DIR, ".."))
STOPT_BIN_DIR = os.path.join(REPO_ROOT, "external", "StOpt", "BUILD_MINGW", "bin")

HAS_STOPT = False
STOPT_IMPORT_ERROR = None

if os.path.isdir(STOPT_BIN_DIR):
    if STOPT_BIN_DIR not in sys.path:
        sys.path.insert(0, STOPT_BIN_DIR)

    # On Windows, extension modules may require dependent DLLs to be discoverable.
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        try:
            # Keep the handle alive; otherwise CPython may close it immediately.
            _STOPT_DLL_DIR_HANDLE = os.add_dll_directory(STOPT_BIN_DIR)
        except OSError:
            _STOPT_DLL_DIR_HANDLE = None

try:
    import StOptReg

    HAS_STOPT = True
except Exception as e:
    HAS_STOPT = False
    STOPT_IMPORT_ERROR = str(e)


KERNEL_SIGMA = {
    "gaussian": 1.0,
    "laplacian": np.sqrt(2),
    "epanechnikov": 1.0 / np.sqrt(5),
}


def convert_bandwidth(h_gaussian, target_kernel):
    if target_kernel not in KERNEL_SIGMA:
        raise ValueError(f"Unknown kernel: {target_kernel}")
    sigma_target = KERNEL_SIGMA[target_kernel]
    return h_gaussian * sigma_target


def stopt_local_regression(x, y, bandwidth=0.1, factor=1.0, linear=True, xq=None):
    if not HAS_STOPT:
        raise ImportError("StOptReg module not available.")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.shape[0] > x.shape[1]:
        x = x.T
    y = np.asarray(y, dtype=np.float64).ravel()
    bandwidth = np.asarray(bandwidth, dtype=np.float64)
    if bandwidth.ndim == 0:
        bandwidth = np.array([float(bandwidth)])
    factor = np.asarray(factor, dtype=np.float64)
    if factor.ndim == 0:
        factor = np.array([float(factor)])
    if xq is not None:
        xq = np.asarray(xq, dtype=np.float64)
        if xq.ndim == 1:
            xq = xq.reshape(1, -1)
        elif xq.shape[0] > xq.shape[1]:
            xq = xq.T
        predict_at_xq = True
    else:
        predict_at_xq = False
    t0 = time.time()
    regressor = StOptReg.LocalGridKernelRegression(False, x, bandwidth, factor, linear)
    if predict_at_xq:
        coef = regressor.getCoordBasisFunction(y)
        yhat = regressor.getValues(xq, coef)
    else:
        yhat = regressor.getAllSimulations(y)
    elapsed = time.time() - t0
    return np.asarray(yhat), elapsed


def stopt_nw(x, y, bandwidth=0.1, factor=1.0, xq=None):
    return stopt_local_regression(x, y, bandwidth, factor, linear=False, xq=xq)


def stopt_ll(x, y, bandwidth=0.1, factor=1.0, xq=None):
    return stopt_local_regression(x, y, bandwidth, factor, linear=True, xq=xq)


def stopt_laplacian_nw(x, y, bandwidth_gaussian, factor=1.0, xq=None):
    """StOpt LaplacianGridKernelRegression (Nadaraya-Watson)."""
    if not HAS_STOPT:
        raise ImportError("StOptReg module not available.")

    h_laplacian = convert_bandwidth(bandwidth_gaussian, "laplacian")

    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.shape[0] > x.shape[1]:
        x = x.T

    d = x.shape[0]  # dimensionality
    y = np.asarray(y, dtype=np.float64).ravel()

    # Bandwidth as column vector (d, 1) - required by LaplacianGridKernelRegression
    h_laplacian = np.asarray(h_laplacian, dtype=np.float64)
    if h_laplacian.ndim == 0:
        h_arr = np.full((d, 1), float(h_laplacian), dtype=np.float64)
    elif h_laplacian.ndim == 1:
        h_arr = h_laplacian.reshape(-1, 1)
    else:
        h_arr = h_laplacian

    # Factor as scalar float
    factor = float(factor)

    # Query points
    if xq is not None:
        xq = np.asarray(xq, dtype=np.float64)
        if xq.ndim == 1:
            xq = xq.reshape(1, -1)
        elif xq.shape[0] > xq.shape[1]:
            xq = xq.T
        predict_at_xq = True
    else:
        predict_at_xq = False

    t0 = time.time()

    # LaplacianGridKernelRegression: (bZeroDate, particles[d,N], bandwidth[d,1], factor, bLin)
    # bLin=False for NW (constant), bLin=True for local linear
    regressor = StOptReg.LaplacianGridKernelRegression(
        False,  # bZeroDate
        x,  # particles (d, N)
        h_arr,  # bandwidth (d, 1) column vector
        factor,  # factPoint (scalar)
        False,  # bLin: False=NW, True=local linear
    )

    if predict_at_xq:
        coef = regressor.getCoordBasisFunction(y)
        yhat = regressor.getValues(xq, coef)
    else:
        yhat = regressor.getAllSimulations(y)

    elapsed = time.time() - t0

    return np.asarray(yhat), elapsed


def silverman_bandwidth(N, d):
    return 1.0 * (N ** (-1.0 / (4 + d)))


def run_stopt_benchmark(x, y, d, N, linear=True, bandwidth=None):
    if not HAS_STOPT:
        err = STOPT_IMPORT_ERROR or "StOptReg not available"
        return {"status": "skip", "error": err}
    if bandwidth is None:
        bandwidth = silverman_bandwidth(N, d) * 0.5
    try:
        yhat, time_sec = stopt_local_regression(
            x, y, bandwidth=bandwidth, factor=1.0, linear=linear
        )
        return {
            "status": "success",
            "time_sec": time_sec,
            "yhat": yhat,
            "bandwidth": bandwidth,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
