# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Benchmarking and profiling utilities for fastLPR.

This module provides tools for:
- Timing execution
- Memory profiling
- Complexity analysis
- Result saving/loading
- Figure comparison
"""

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class Timer:
    """Context manager for timing code execution."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        print(f"{self.name}: {self.elapsed:.4f} seconds")

    def get_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return self.elapsed if self.elapsed is not None else 0.0


class MemoryProfiler:
    """Memory profiling utility."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        if HAS_PSUTIL:
            self.process = psutil.Process()
        else:
            self.process = None
        self.start_memory = None
        self.peak_memory = None
        self.end_memory = None

    def __enter__(self):
        if self.process:
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = self.start_memory
        else:
            self.start_memory = 0.0
            self.peak_memory = 0.0
        return self

    def __exit__(self, *args):
        if self.process:
            self.end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = max(self.peak_memory, self.end_memory)
            delta = self.end_memory - self.start_memory
            print(
                f"{self.name} - Memory: Start={self.start_memory:.1f}MB, "
                f"End={self.end_memory:.1f}MB, Delta={delta:.1f}MB"
            )
        else:
            self.end_memory = 0.0
            print(
                f"{self.name} - Memory profiling not available (psutil not installed)"
            )

    def update_peak(self):
        """Update peak memory usage."""
        if self.process:
            current = self.process.memory_info().rss / 1024 / 1024
            self.peak_memory = max(self.peak_memory, current)

    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        return {
            "start_mb": self.start_memory or 0.0,
            "end_mb": self.end_memory or 0.0,
            "peak_mb": self.peak_memory or 0.0,
            "delta_mb": (self.end_memory or 0.0) - (self.start_memory or 0.0),
        }


class ComplexityAnalyzer:
    """Analyze computational complexity."""

    @staticmethod
    def analyze_timing(sizes: List[int], times: List[float]) -> Dict[str, Any]:
        """
        Analyze timing complexity.

        Fits timing data to various complexity models:
        - O(N): linear
        - O(N log N): linearithmic
        - O(N^2): quadratic

        Returns best fit and R^2 values.
        """
        sizes_arr = np.array(sizes, dtype=float)
        times_arr = np.array(times, dtype=float)

        # Normalize
        sizes_norm = sizes_arr / sizes_arr[0]
        times_norm = times_arr / times_arr[0]

        results = {}

        # O(N) - linear
        predicted_linear = sizes_norm
        r2_linear = 1 - np.sum((times_norm - predicted_linear) ** 2) / np.sum(
            (times_norm - np.mean(times_norm)) ** 2
        )
        results["O(N)"] = {"r2": r2_linear, "predicted": predicted_linear}

        # O(N log N) - linearithmic
        predicted_nlogn = sizes_norm * np.log(sizes_norm + 1e-10)  # Avoid log(0)
        if predicted_nlogn[0] != 0:
            predicted_nlogn = predicted_nlogn / predicted_nlogn[0]
        r2_nlogn = 1 - np.sum((times_norm - predicted_nlogn) ** 2) / np.sum(
            (times_norm - np.mean(times_norm)) ** 2
        )
        results["O(N log N)"] = {"r2": r2_nlogn, "predicted": predicted_nlogn}

        # O(N^2) - quadratic
        predicted_n2 = sizes_norm**2
        r2_n2 = 1 - np.sum((times_norm - predicted_n2) ** 2) / np.sum(
            (times_norm - np.mean(times_norm)) ** 2
        )
        results["O(N^2)"] = {"r2": r2_n2, "predicted": predicted_n2}

        # Find best fit
        best_complexity = max(results.keys(), key=lambda k: results[k]["r2"])

        return {
            "best_fit": best_complexity,
            "r2_scores": {k: v["r2"] for k, v in results.items()},
            "sizes": sizes,
            "times": times,
            "analysis": f"Best fit: {best_complexity} (R2 = {results[best_complexity]['r2']:.4f})",
        }


def save_results_csv(
    filepath: Path, data: Dict[str, np.ndarray], metadata: Optional[Dict] = None
):
    """
    Save results to CSV file.

    Parameters
    ----------
    filepath : Path
        Output CSV file path
    data : dict
        Dictionary of arrays to save (keys become column names)
    metadata : dict, optional
        Metadata to save as comments at top of file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="") as f:
        # Write metadata as comments
        if metadata:
            f.write(f"# Metadata: {json.dumps(metadata)}\n")

        # Write data
        writer = csv.writer(f)

        # Header
        headers = list(data.keys())
        writer.writerow(headers)

        # Determine number of rows
        n_rows = len(next(iter(data.values())))

        # Write rows
        for i in range(n_rows):
            row = []
            for key in headers:
                val = data[key][i] if i < len(data[key]) else ""
                # Handle complex numbers
                if isinstance(val, complex):
                    row.append(f"{val.real}+{val.imag}j")
                else:
                    row.append(val)
            writer.writerow(row)

    print(f"Saved results to {filepath}")


def load_results_csv(filepath: Path) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Load results from CSV file.

    Returns
    -------
    data : dict
        Dictionary of arrays
    metadata : dict
        Metadata from file
    """
    filepath = Path(filepath)

    metadata = {}
    data = {}

    with open(filepath, "r") as f:
        # Read metadata
        first_line = f.readline()
        if first_line.startswith("# Metadata:"):
            metadata = json.loads(first_line[11:].strip())
            reader = csv.DictReader(f)
        else:
            f.seek(0)
            reader = csv.DictReader(f)

        # Read data
        rows = list(reader)
        if not rows:
            return {}, metadata

        # Initialize arrays
        for key in rows[0].keys():
            data[key] = []

        # Fill arrays
        for row in rows:
            for key, val in row.items():
                # Try to parse as number
                try:
                    # Check for complex number
                    if "+" in val and "j" in val:
                        data[key].append(complex(val))
                    else:
                        data[key].append(float(val))
                except (ValueError, AttributeError):
                    data[key].append(val)

        # Convert to numpy arrays
        for key in data:
            data[key] = np.array(data[key])

    return data, metadata


def save_timing_results(
    filepath: Path, timings: Dict[str, float], metadata: Optional[Dict] = None
):
    """Save timing results to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "timings": timings,
        "metadata": metadata or {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved timing results to {filepath}")


def save_profiling_results(
    filepath: Path,
    memory_stats: Dict[str, float],
    timing_stats: Dict[str, float],
    metadata: Optional[Dict] = None,
):
    """Save profiling results to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "memory": memory_stats,
        "timing": timing_stats,
        "metadata": metadata or {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved profiling results to {filepath}")


def compare_results(
    python_data: Dict[str, np.ndarray],
    matlab_data: Dict[str, np.ndarray],
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Compare Python and MATLAB results.

    Returns
    -------
    comparison : dict
        Comparison statistics including max error, mean error, etc.
    """
    comparison = {}

    for key in python_data.keys():
        if key not in matlab_data:
            comparison[key] = {"status": "missing_in_matlab"}
            continue

        py_val = python_data[key]
        ml_val = matlab_data[key]

        # Check shapes
        if py_val.shape != ml_val.shape:
            comparison[key] = {
                "status": "shape_mismatch",
                "python_shape": py_val.shape,
                "matlab_shape": ml_val.shape,
            }
            continue

        # Compute errors
        abs_error = np.abs(py_val - ml_val)
        rel_error = abs_error / (np.abs(ml_val) + 1e-10)

        comparison[key] = {
            "status": "match" if np.max(abs_error) < tolerance else "mismatch",
            "max_abs_error": float(np.max(abs_error)),
            "mean_abs_error": float(np.mean(abs_error)),
            "max_rel_error": float(np.max(rel_error)),
            "mean_rel_error": float(np.mean(rel_error)),
            "tolerance": tolerance,
        }

    return comparison
