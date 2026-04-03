# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Timing utilities for performance profiling.

Provides tic/toc style timing and logging for identifying bottlenecks.
"""

import time
from contextlib import contextmanager
from typing import Optional

# Global timing state
_timing_enabled = False
_timing_stack = []


def enable_timing():
    """Enable timing output."""
    global _timing_enabled
    _timing_enabled = True


def disable_timing():
    """Disable timing output."""
    global _timing_enabled
    _timing_enabled = False


def is_timing_enabled():
    """Check if timing is enabled."""
    return _timing_enabled


@contextmanager
def timer(name: str, verbose: Optional[bool] = None):
    """
    Context manager for timing code blocks.

    Parameters
    ----------
    name : str
        Name of the code block being timed
    verbose : bool, optional
        If True, print timing. If None, use global timing state.

    Examples
    --------
    >>> with timer("My operation"):
    ...     # code to time
    ...     pass
    """
    if verbose is None:
        verbose = _timing_enabled

    if not verbose:
        yield
        return

    indent = "  " * len(_timing_stack)
    _timing_stack.append(name)

    try:
        print(f"{indent}[TIMER] {name}...", flush=True)
    except UnicodeEncodeError:
        print(
            f"{indent}[TIMER] {name}...".encode("utf-8", errors="ignore").decode(
                "utf-8"
            ),
            flush=True,
        )

    t0 = time.time()

    try:
        yield
    finally:
        elapsed = time.time() - t0
        _timing_stack.pop()
        try:
            print(f"{indent}[OK] {name}: {elapsed:.4f}s", flush=True)
        except UnicodeEncodeError:
            print(
                f"{indent}[OK] {name}: {elapsed:.4f}s".encode(
                    "utf-8", errors="ignore"
                ).decode("utf-8"),
                flush=True,
            )


def tic():
    """Start a timer (MATLAB-style)."""
    return time.time()


def toc(t0: float, name: str = "Elapsed", verbose: Optional[bool] = None):
    """
    Print elapsed time since tic (MATLAB-style).

    Parameters
    ----------
    t0 : float
        Start time from tic()
    name : str
        Name to print
    verbose : bool, optional
        If True, print timing. If None, use global timing state.

    Returns
    -------
    float
        Elapsed time in seconds
    """
    elapsed = time.time() - t0

    if verbose is None:
        verbose = _timing_enabled

    if verbose:
        print(f"{name}: {elapsed:.4f}s", flush=True)

    return elapsed
