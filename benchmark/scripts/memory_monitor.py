# Copyright (c) 2024 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later
"""
External Memory Monitor using psutil USS/RSS Delta

Provides consistent cross-language memory measurement by monitoring
subprocess memory from the parent process.

Key Features:
- USS (Unique Set Size) by default, with RSS fallback for accuracy
- Multi-sample baseline capture (5 samples, max) for stability
- Signal-based baseline capture (subprocess prints "READY")
- Tracks peak memory during algorithm execution
- Reports: peak - baseline = algorithm memory only
- Works for Python, R, MATLAB, C++ subprocesses

Usage:
    from memory_monitor import MemoryMonitor, run_with_memory_monitor

    # Low-level API
    proc = subprocess.Popen(cmd, ...)
    monitor = MemoryMonitor(proc.pid)
    monitor.start()
    # ... wait for READY signal ...
    monitor.capture_baseline()
    # ... wait for process to finish ...
    mem_mb = monitor.stop()

    # High-level API
    result = run_with_memory_monitor(cmd, description, timeout=300)
    # result = {'time_sec': ..., 'mem_mb': ..., 'status': ...}

Author: Ying Wang, Min Li
Date: 2025-12-26
"""

import subprocess
import sys
import threading
import time
from typing import Optional, Dict, Any, List

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Windows Job Objects API for precise peak memory tracking
HAS_JOB_OBJECTS = False
if sys.platform == "win32":
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32

        # Set proper function prototypes for 64-bit compatibility
        # Without these, HANDLE (64-bit) gets truncated to c_int (32-bit)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]

        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]

        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID,
            wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)
        ]

        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

        # Job Object constants
        JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
        PROCESS_SET_QUOTA = 0x0100
        PROCESS_TERMINATE = 0x0001
        PROCESS_QUERY_INFORMATION = 0x0400

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ('ReadOperationCount', ctypes.c_ulonglong),
                ('WriteOperationCount', ctypes.c_ulonglong),
                ('OtherOperationCount', ctypes.c_ulonglong),
                ('ReadTransferCount', ctypes.c_ulonglong),
                ('WriteTransferCount', ctypes.c_ulonglong),
                ('OtherTransferCount', ctypes.c_ulonglong),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ('PerProcessUserTimeLimit', ctypes.c_ulonglong),
                ('PerJobUserTimeLimit', ctypes.c_ulonglong),
                ('LimitFlags', wintypes.DWORD),
                ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t),
                ('ActiveProcessLimit', wintypes.DWORD),
                ('Affinity', ctypes.c_size_t),
                ('PriorityClass', wintypes.DWORD),
                ('SchedulingClass', wintypes.DWORD),
                ('IoInfo', IO_COUNTERS),
                ('ProcessMemoryLimit', ctypes.c_size_t),
                ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t),
                ('PeakJobMemoryUsed', ctypes.c_size_t),
            ]

        HAS_JOB_OBJECTS = True
    except Exception:
        HAS_JOB_OBJECTS = False

# Windows GetProcessMemoryInfo API for accurate per-process peak tracking
HAS_PROCESS_MEMORY_INFO = False
if sys.platform == "win32":
    try:
        import ctypes
        from ctypes import wintypes

        # Load psapi.dll for GetProcessMemoryInfo
        psapi = ctypes.windll.psapi

        # PROCESS_MEMORY_COUNTERS_EX structure
        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ('cb', wintypes.DWORD),
                ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t),  # Current private bytes
                ('PeakPagefileUsage', ctypes.c_size_t),  # Peak private bytes
                ('PrivateUsage', ctypes.c_size_t),
            ]

        # Set function prototype
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD
        ]

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

        def get_process_peak_memory(pid: int) -> int:
            """Get peak private bytes for a process using GetProcessMemoryInfo.

            This is more accurate than Job Objects for individual processes because
            it tracks the peak for the specific process, not the job.

            Returns peak private bytes, or 0 if failed.
            """
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return 0
            try:
                pmc = PROCESS_MEMORY_COUNTERS_EX()
                pmc.cb = ctypes.sizeof(pmc)
                if psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                    return pmc.PeakPagefileUsage
                return 0
            finally:
                kernel32.CloseHandle(handle)

        HAS_PROCESS_MEMORY_INFO = True
    except Exception:
        HAS_PROCESS_MEMORY_INFO = False


class JobObjectMonitor:
    """Windows Job Object based memory monitor for precise peak tracking.

    Uses both Job Objects and GetProcessMemoryInfo for maximum accuracy:
    - Job Objects: OS-level peak tracking without polling
    - GetProcessMemoryInfo: Per-process peak private bytes (PeakPagefileUsage)

    Returns the maximum of both methods for best accuracy.
    """

    def __init__(self):
        self.job_handle = None
        self.process_handle = None
        self._pid = None
        self._baseline_peak = 0
        self._baseline_process_peak = 0  # From GetProcessMemoryInfo
        self._cached_peak_mb = 0.0  # Cache last known peak (survives process exit)
        self._lock = threading.Lock()  # Thread safety for baselines

    def create_job(self) -> bool:
        """Create a new Job Object."""
        if not HAS_JOB_OBJECTS:
            return False
        self.job_handle = kernel32.CreateJobObjectW(None, None)
        # HANDLE failure returns 0 (NULL), not Python None
        return self.job_handle is not None and self.job_handle != 0

    def assign_process(self, pid: int) -> bool:
        """Assign a process to the Job Object."""
        if not self.job_handle:
            return False
        self._pid = pid
        self.process_handle = kernel32.OpenProcess(
            PROCESS_SET_QUOTA | PROCESS_TERMINATE | PROCESS_QUERY_INFORMATION,
            False, pid
        )
        if not self.process_handle:
            return False
        return kernel32.AssignProcessToJobObject(self.job_handle, self.process_handle)

    def capture_baseline(self):
        """Capture current peak as baseline (called after READY signal)."""
        info = self._query_job_info()
        with self._lock:
            if info:
                self._baseline_peak = info.PeakJobMemoryUsed
            # Also capture from GetProcessMemoryInfo
            if HAS_PROCESS_MEMORY_INFO and self._pid:
                self._baseline_process_peak = get_process_peak_memory(self._pid)

    def update_peak_cache(self):
        """Update cached peak memory (call while process is still running)."""
        peak = self._query_peak_mb()
        if peak > 0:
            with self._lock:
                self._cached_peak_mb = max(self._cached_peak_mb, peak)

    def _query_job_info(self) -> Optional['JOBOBJECT_EXTENDED_LIMIT_INFORMATION']:
        """Query Job Object for extended limit information."""
        if not self.job_handle:
            return None
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        ret_len = wintypes.DWORD()
        if kernel32.QueryInformationJobObject(
            self.job_handle,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
            ctypes.byref(ret_len)
        ):
            return info
        return None

    def _query_peak_mb(self) -> float:
        """Query current peak memory delta from both methods (internal).

        Returns the maximum of Job Object and GetProcessMemoryInfo peaks.
        """
        job_delta_mb = 0.0
        process_delta_mb = 0.0

        # Method 1: Job Object peak
        info = self._query_job_info()
        if info:
            with self._lock:
                job_delta = info.PeakJobMemoryUsed - self._baseline_peak
            job_delta_mb = max(0, job_delta) / (1024 * 1024)

        # Method 2: GetProcessMemoryInfo peak (often more accurate for short runs)
        if HAS_PROCESS_MEMORY_INFO and self._pid:
            current_peak = get_process_peak_memory(self._pid)
            if current_peak > 0:  # Only if process still exists
                with self._lock:
                    process_delta = current_peak - self._baseline_process_peak
                process_delta_mb = max(0, process_delta) / (1024 * 1024)

        return max(job_delta_mb, process_delta_mb)

    def get_peak_memory_mb(self) -> float:
        """Get peak memory delta (peak - baseline) in MB.

        Returns the maximum of:
        1. Current query result (if process still exists)
        2. Cached peak (from earlier queries while process was running)
        """
        # Try to query current peak
        current_peak = self._query_peak_mb()

        # Update cache if we got a valid reading
        if current_peak > 0:
            with self._lock:
                self._cached_peak_mb = max(self._cached_peak_mb, current_peak)

        # Return cached peak (survives process exit)
        with self._lock:
            return self._cached_peak_mb

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics compatible with MemoryMonitor interface."""
        peak_mb = self.get_peak_memory_mb()
        return {
            'mem_mb': peak_mb,
            'mem_median': peak_mb,
            'mem_min': peak_mb,
            'mem_max': peak_mb,
            'mem_std': 0.0,
            'run_peaks': [peak_mb],
            'method': 'JobObject'
        }

    def close(self):
        """Clean up handles."""
        if self.process_handle:
            kernel32.CloseHandle(self.process_handle)
            self.process_handle = None
        if self.job_handle:
            kernel32.CloseHandle(self.job_handle)
            self.job_handle = None


class MemoryMonitor:
    """Monitor RSS delta of a subprocess using psutil with signal-based baseline.

    Waits for subprocess to signal "READY" before capturing baseline RSS.
    This isolates algorithm memory from runtime/interpreter overhead.

    Flow:
    1. Subprocess loads libraries, generates data
    2. Subprocess prints "READY" (flush)
    3. Parent sees "READY", calls capture_baseline()
    4. For each iteration:
       - Subprocess prints "RUN_START" -> parent calls start_run()
       - Subprocess runs algorithm
       - Subprocess prints "RUN_END" -> parent calls end_run()
    5. Result: list of per-iteration (peak - baseline) values
    """

    def __init__(self, pid: int, interval: float = 0.001, high_precision: bool = True,
                 use_uss: bool = True, baseline_samples: int = 5, baseline_interval: float = 0.05):
        """
        Args:
            pid: Process ID to monitor
            interval: Sampling interval in seconds (default 1ms for high-frequency capture)
            high_precision: If True, use busy-wait during runs for accurate 10KHz sampling.
                           Windows time.sleep() has ~1ms minimum resolution, but busy-wait
                           achieves true 0.1ms intervals. Only active during RUN_START/RUN_END.
            use_uss: If True, try to use USS (Unique Set Size) instead of RSS for more
                    accurate memory measurement. Falls back to RSS if USS unavailable.
            baseline_samples: Number of samples to take for baseline (default 5, take max)
            baseline_interval: Interval between baseline samples in seconds (default 50ms)
        """
        self.pid = pid
        self.interval = interval
        self.high_precision = high_precision
        self.use_uss = use_uss
        self.baseline_samples = baseline_samples
        self.baseline_interval = baseline_interval
        self.baseline_rss = 0
        self.peak_rss = 0
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._baseline_captured = False
        self._lock = threading.Lock()
        # Per-iteration tracking
        self._run_peaks: List[float] = []  # Peak RSS deltas per iteration (in MB)
        self._in_run = False
        self._run_baseline = 0
        self._run_peak = 0

        # Async baseline state (non-blocking request from reader threads)
        self._baseline_requested = threading.Event()
        self._baseline_samples: List[int] = []
        self._baseline_next_sample_ts: float = 0.0
        self._baseline_finalized = False

        # Wake-up event for immediate sampling
        self._wake_up = threading.Event()

    def _get_process_memory(self, process) -> int:
        """Get memory in bytes using RSS (most reliable for numpy allocations).

        Includes memory for the process and all child processes.

        Note: On Windows, Private Bytes can be misleading for numpy allocations
        because numpy may use memory-mapped files or other mechanisms that don't
        show up in Private Bytes. RSS is more reliable for cross-platform
        memory measurement.
        """
        try:
            total_mem = 0

            # Use RSS for all platforms (most reliable for numpy allocations)
            total_mem = int(process.memory_info().rss)

            # Add child processes
            for child in process.children(recursive=True):
                try:
                    total_mem += int(child.memory_info().rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                    pass
            return total_mem
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0

    def capture_baseline(self):
        """Capture baseline memory using multi-sampling (called when subprocess signals READY).

        Takes multiple samples and uses the maximum to reduce baseline variability
        caused by memory allocator behavior and GC timing.
        """
        if not HAS_PSUTIL:
            return
        try:
            process = psutil.Process(self.pid)
            samples = []

            # Take multiple samples
            for _ in range(self.baseline_samples):
                mem = self._get_process_memory(process)
                if mem > 0:
                    samples.append(mem)
                time.sleep(self.baseline_interval)

            baseline_value = max(samples) if samples else self._get_process_memory(process)
            with self._lock:
                self.baseline_rss = baseline_value
                self.peak_rss = self.baseline_rss  # Reset peak to baseline
                self._baseline_captured = True
                self._baseline_samples = []
                self._baseline_finalized = True
                self._baseline_requested.clear()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def request_baseline_capture(self):
        """Request baseline capture asynchronously (non-blocking).

        Called from reader threads when the subprocess signals READY.
        """
        if not HAS_PSUTIL:
            return
        with self._lock:
            if self._baseline_finalized:
                return
            if self._baseline_requested.is_set():
                return
            self._baseline_samples = []
            self._baseline_next_sample_ts = time.perf_counter()
            self._baseline_requested.set()

    def _finalize_baseline(self):
        """Finalize baseline from collected samples (internal, idempotent)."""
        if not HAS_PSUTIL:
            return

        with self._lock:
            if self._baseline_finalized:
                return
            samples = list(self._baseline_samples)

        baseline_value = max(samples) if samples else 0
        if baseline_value <= 0:
            try:
                process = psutil.Process(self.pid)
                baseline_value = self._get_process_memory(process)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                baseline_value = 0

        with self._lock:
            if self._baseline_finalized:
                return
            self._baseline_samples = []
            self.baseline_rss = baseline_value
            self.peak_rss = self.baseline_rss  # Reset peak to baseline
            if baseline_value > 0:
                self._baseline_captured = True
            self._baseline_finalized = True
            self._baseline_requested.clear()

    def start_run(self):
        """Called when subprocess signals RUN_START - capture per-run baseline.

        This method captures the baseline immediately (single sample for speed)
        and wakes up the monitor thread for high-precision sampling.

        IMPORTANT: The subprocess should wait ~50ms after RUN_START before
        allocating memory to ensure the baseline is captured correctly.
        """
        if not HAS_PSUTIL:
            return
        try:
            process = psutil.Process(self.pid)
            needs_finalize = False
            with self._lock:
                if self._in_run:
                    return
                needs_finalize = (
                    self._baseline_requested.is_set() and not self._baseline_finalized
                )

            if needs_finalize:
                self._finalize_baseline()

            # Capture baseline immediately (single sample for speed)
            # Subprocess should wait ~50ms after RUN_START before allocating
            run_baseline = self._get_process_memory(process)

            with self._lock:
                if self._in_run:
                    return
                self._run_baseline = run_baseline
                self._run_peak = run_baseline
                self._in_run = True

            # Wake up the monitor thread immediately for high-precision sampling
            self._wake_up.set()

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def end_run(self):
        """Called when subprocess signals RUN_END - record peak delta for this run.

        This method captures the final peak memory at RUN_END time to ensure
        we don't miss any allocations that happened between monitor samples.
        """
        if not HAS_PSUTIL:
            with self._lock:
                if self._in_run:
                    self._run_peaks.append(0.0)
                    self._in_run = False
            return

        try:
            # Capture final memory reading at RUN_END time
            process = psutil.Process(self.pid)
            final_mem = self._get_process_memory(process)

            with self._lock:
                if self._in_run:
                    # Update peak with final reading (in case monitor missed it)
                    if final_mem > 0:
                        self._run_peak = max(self._run_peak, final_mem)

                    delta_mb = (self._run_peak - self._run_baseline) / (1024 * 1024)
                    # Ensure non-negative (memory can fluctuate)
                    self._run_peaks.append(max(0.0, delta_mb))
                    self._in_run = False
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            with self._lock:
                if self._in_run:
                    delta_mb = (self._run_peak - self._run_baseline) / (1024 * 1024)
                    self._run_peaks.append(max(0.0, delta_mb))
                    self._in_run = False

    def _monitor_loop(self):
        """Background thread to monitor peak memory.

        Uses adaptive timing:
        - Normal sleep (~1ms resolution) when idle
        - Busy-wait (true 0.1ms resolution) during RUN_START/RUN_END if high_precision=True
        """
        if not HAS_PSUTIL:
            return
        try:
            process = psutil.Process(self.pid)

            while self.running:
                current_rss = self._get_process_memory(process)
                if current_rss > 0:
                    finalize_baseline = False
                    with self._lock:
                        if (
                            self._baseline_requested.is_set()
                            and not self._baseline_finalized
                        ):
                            now = time.perf_counter()
                            if now >= self._baseline_next_sample_ts:
                                self._baseline_samples.append(current_rss)
                                self._baseline_next_sample_ts = (
                                    now + self.baseline_interval
                                )
                                if len(self._baseline_samples) >= self.baseline_samples:
                                    finalize_baseline = True
                        self.peak_rss = max(self.peak_rss, current_rss)
                        # Also track per-run peak if in a run
                        if self._in_run:
                            self._run_peak = max(self._run_peak, current_rss)
                    if finalize_baseline:
                        self._finalize_baseline()

                # Adaptive timing: busy-wait during runs for accuracy, sleep otherwise
                with self._lock:
                    in_run = self._in_run

                if in_run and self.high_precision:
                    # Busy-wait for true 10KHz sampling during measurement
                    target = time.perf_counter() + self.interval
                    while time.perf_counter() < target and self.running:
                        pass
                else:
                    # Normal sleep when not measuring (saves CPU)
                    # Use Event.wait() so we can be woken up immediately when a run starts
                    self._wake_up.wait(timeout=max(0.001, self.interval))
                    self._wake_up.clear()

        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            pass  # Process ended, that's fine

    def start(self):
        """Start monitoring in background thread."""
        if not HAS_PSUTIL:
            return
        self.running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        """Stop monitoring and return RSS delta in MB."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)

        needs_finalize = False
        with self._lock:
            needs_finalize = (
                self._baseline_requested.is_set() and not self._baseline_finalized
            )
        if needs_finalize:
            self._finalize_baseline()

        # Return delta (peak - baseline) to isolate algorithm cost
        with self._lock:
            if self._baseline_captured and self.peak_rss > self.baseline_rss:
                delta = self.peak_rss - self.baseline_rss
            else:
                delta = self.peak_rss  # Fallback to peak if baseline not captured

        return delta / (1024 * 1024)  # Convert bytes to MB

    @property
    def baseline_mb(self) -> float:
        """Get baseline RSS in MB."""
        with self._lock:
            return self.baseline_rss / (1024 * 1024)

    @property
    def run_peaks_mb(self) -> List[float]:
        """Get list of per-run peak memory deltas in MB."""
        with self._lock:
            return list(self._run_peaks)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics from per-run peaks (median/min/max/std).

        Returns dict with keys: mem_median, mem_min, mem_max, mem_std, mem_mb (same as median),
        and run_peaks (list of per-run memory deltas for batch distribution).
        Falls back to single peak if no per-run data.
        """
        import statistics
        with self._lock:
            if self._run_peaks:
                peaks = list(self._run_peaks)  # Copy for safety
                mem_median = statistics.median(peaks)
                return {
                    'mem_mb': mem_median,  # For backward compatibility
                    'mem_median': mem_median,
                    'mem_min': min(peaks),
                    'mem_max': max(peaks),
                    'mem_std': statistics.stdev(peaks) if len(peaks) > 1 else 0.0,
                    'run_peaks': peaks,  # Raw per-run peaks for batch distribution
                }
            else:
                # Fallback to old behavior (single peak)
                if self._baseline_captured and self.peak_rss > self.baseline_rss:
                    delta = (self.peak_rss - self.baseline_rss) / (1024 * 1024)
                else:
                    delta = self.peak_rss / (1024 * 1024)
                return {
                    'mem_mb': delta,
                    'mem_median': delta,
                    'mem_min': delta,
                    'mem_max': delta,
                    'mem_std': 0.0,
                    'run_peaks': [delta],  # Single peak as list
                }


def run_with_memory_monitor(
    cmd: List[str],
    description: str = "",
    timeout: int = 300,
    ready_signal: str = "READY"
) -> Dict[str, Any]:
    """Run subprocess with signal-based RSS delta monitoring.

    Memory is measured EXTERNALLY using psutil with signal-based baseline:
    1. Start subprocess and memory monitor
    2. Wait for subprocess to print ready_signal (after loading libs, before algorithm)
    3. Capture baseline RSS at that moment
    4. For each iteration, detect RUN_START/RUN_END signals and track per-run peaks
    5. Return: median of per-run (peak - baseline) values

    This isolates algorithm memory from runtime/interpreter overhead.

    Args:
        cmd: Command to run as list (e.g., ['python', 'script.py'])
        description: Description for logging
        timeout: Timeout in seconds (default 300)
        ready_signal: Signal string to watch for (default "READY")

    Returns:
        dict with keys: time_sec, mem_mb, mem_median, mem_min, mem_max, mem_std,
                       baseline_mb, status, stdout, stderr
    """
    mem_monitor = None
    job_monitor = None  # Job Object monitor (preferred on Windows)
    proc = None
    stdout_lines = []
    stderr_lines = []
    ready_seen = threading.Event()

    def read_stdout(pipe, lines_list, monitor):
        """Thread function to read stdout and detect signals."""
        nonlocal ready_seen, job_monitor
        try:
            for line in iter(pipe.readline, ''):
                if not line:
                    break
                lines_list.append(line)
                line_stripped = line.strip()
                # Check for READY signal
                if line_stripped == ready_signal and not ready_seen.is_set():
                    ready_seen.set()
                    if monitor:
                        monitor.request_baseline_capture()
                    if job_monitor:
                        job_monitor.capture_baseline()
                # Check for per-run signals
                elif line_stripped == "RUN_START":
                    if monitor:
                        monitor.start_run()
                elif line_stripped == "RUN_END":
                    if monitor:
                        monitor.end_run()
                    # Update Job Object peak cache while process still running
                    if job_monitor:
                        job_monitor.update_peak_cache()
        except Exception:
            pass  # Pipe closed or error

    def read_stderr(pipe, lines_list, monitor):
        """Thread function to read stderr and detect signals.

        Note: MATLAB uses stderr for signals because stderr is unbuffered,
        while stdout may be block-buffered in batch mode.
        """
        nonlocal ready_seen, job_monitor
        try:
            for line in iter(pipe.readline, ''):
                if not line:
                    break
                lines_list.append(line)
                line_stripped = line.strip()
                # Check for READY signal (also on stderr for MATLAB)
                if line_stripped == ready_signal and not ready_seen.is_set():
                    ready_seen.set()
                    if monitor:
                        monitor.request_baseline_capture()
                    if job_monitor:
                        job_monitor.capture_baseline()
                # Check for per-run signals
                elif line_stripped == "RUN_START":
                    if monitor:
                        monitor.start_run()
                elif line_stripped == "RUN_END":
                    if monitor:
                        monitor.end_run()
                    # Update Job Object peak cache while process still running
                    if job_monitor:
                        job_monitor.update_peak_cache()
        except Exception:
            pass  # Pipe closed or error

    try:
        # Start subprocess with unbuffered output
        # Use CREATE_BREAKAWAY_FROM_JOB on Windows to allow assigning to our Job Object
        # (child process may inherit parent's Job Object from shell/terminal)
        creationflags = 0
        if sys.platform == 'win32':
            creationflags = subprocess.CREATE_BREAKAWAY_FROM_JOB

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            creationflags=creationflags
        )

        # Create Job Object for precise peak memory tracking (Windows)
        if HAS_JOB_OBJECTS:
            job_monitor = JobObjectMonitor()
            if job_monitor.create_job():
                if not job_monitor.assign_process(proc.pid):
                    # Assignment failed, fall back to psutil
                    job_monitor.close()
                    job_monitor = None
            else:
                job_monitor = None

        # Start external memory monitor (fallback or complement)
        if HAS_PSUTIL:
            mem_monitor = MemoryMonitor(proc.pid, interval=0.0001)  # 0.1ms sampling
            mem_monitor.start()

        start_time = time.perf_counter()

        # Start reader threads (Windows-compatible approach)
        stdout_thread = threading.Thread(
            target=read_stdout,
            args=(proc.stdout, stdout_lines, mem_monitor),
            daemon=True
        )
        stderr_thread = threading.Thread(
            target=read_stderr,
            args=(proc.stderr, stderr_lines, mem_monitor),
            daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()

        # Wait for process with timeout
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            if mem_monitor:
                mem_monitor.stop()
            if job_monitor:
                job_monitor.close()
            return {
                'time_sec': None,
                'mem_mb': 0,
                'mem_median': 0,
                'mem_min': 0,
                'mem_max': 0,
                'mem_std': 0,
                'mem_method': 'none',
                'baseline_mb': 0,
                'status': 'timeout',
                'stdout': ''.join(stdout_lines),
                'stderr': ''.join(stderr_lines)
            }

        # Wait for reader threads to finish
        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)

        wall_time = time.perf_counter() - start_time
        stdout = ''.join(stdout_lines)
        stderr = ''.join(stderr_lines)

        # Stop memory monitor and get stats (Job Object preferred over psutil)
        mem_stats = {'mem_mb': 0.0, 'mem_median': 0.0, 'mem_min': 0.0, 'mem_max': 0.0, 'mem_std': 0.0}
        baseline_mb = 0.0
        mem_method = 'none'

        if mem_monitor:
            mem_monitor.stop()
            mem_stats = mem_monitor.get_memory_stats()
            baseline_mb = mem_monitor.baseline_mb
            mem_method = 'psutil'

        # Prefer Job Object stats if available (more accurate peak tracking)
        # BUT preserve psutil's run_peaks for batch mode (JobObject only tracks single peak)
        if job_monitor:
            job_stats = job_monitor.get_memory_stats()
            if job_stats['mem_mb'] > 0:
                # Preserve psutil's run_peaks - JobObject only has single peak
                psutil_run_peaks = mem_stats.get('run_peaks', [])
                mem_stats = job_stats
                # Restore psutil's per-run peaks if they exist and JobObject only has 1
                if len(psutil_run_peaks) > 1 and len(job_stats.get('run_peaks', [])) <= 1:
                    mem_stats['run_peaks'] = psutil_run_peaks
                mem_method = 'JobObject'
            job_monitor.close()

        # Parse TIME from output (if subprocess reports it)
        time_sec = None
        for line in stdout.split('\n'):
            line = line.strip()
            if line.startswith('TIME:'):
                try:
                    time_sec = float(line.split(':')[1])
                except ValueError:
                    pass

        if time_sec is None:
            time_sec = wall_time  # Fallback to wall time

        if proc.returncode != 0:
            return {
                'time_sec': time_sec,
                'mem_mb': mem_stats['mem_mb'],
                'mem_median': mem_stats['mem_median'],
                'mem_min': mem_stats['mem_min'],
                'mem_max': mem_stats['mem_max'],
                'mem_std': mem_stats['mem_std'],
                'mem_method': mem_method,
                'baseline_mb': baseline_mb,
                'status': f'error: returncode={proc.returncode}',
                'stdout': stdout,
                'stderr': stderr
            }

        return {
            'time_sec': time_sec,
            'mem_mb': mem_stats['mem_mb'],
            'mem_median': mem_stats['mem_median'],
            'mem_min': mem_stats['mem_min'],
            'mem_max': mem_stats['mem_max'],
            'mem_std': mem_stats['mem_std'],
            'mem_method': mem_method,
            'run_peaks': mem_stats.get('run_peaks', []),  # Per-run peaks for batch distribution
            'baseline_mb': baseline_mb,
            'status': 'success',
            'stdout': stdout,
            'stderr': stderr
        }

    except subprocess.TimeoutExpired:
        if proc:
            proc.kill()
        if mem_monitor:
            mem_monitor.stop()
        if job_monitor:
            job_monitor.close()
        return {
            'time_sec': None,
            'mem_mb': 0,
            'mem_median': 0,
            'mem_min': 0,
            'mem_max': 0,
            'mem_std': 0,
            'mem_method': 'none',
            'baseline_mb': 0,
            'status': 'timeout',
            'stdout': ''.join(stdout_lines),
            'stderr': ''.join(stderr_lines)
        }

    except Exception as e:
        if proc and proc.poll() is None:
            proc.kill()
        if mem_monitor:
            mem_monitor.stop()
        if job_monitor:
            job_monitor.close()
        return {
            'time_sec': None,
            'mem_mb': 0,
            'mem_median': 0,
            'mem_min': 0,
            'mem_max': 0,
            'mem_std': 0,
            'mem_method': 'none',
            'baseline_mb': 0,
            'status': f'error: {str(e)[:100]}',
            'stdout': ''.join(stdout_lines),
            'stderr': ''.join(stderr_lines)
        }


# Export check
def check_psutil() -> bool:
    """Check if psutil is available."""
    return HAS_PSUTIL


def get_memory_metric_info() -> Dict[str, Any]:
    """Detect which memory metric will be used on this platform.

    Returns:
        dict with keys:
            - metric: str ('RSS')
            - platform: str (sys.platform)
            - psutil_available: bool
            - description: str (human-readable explanation)
    """
    if not HAS_PSUTIL:
        return {
            'metric': 'N/A',
            'platform': sys.platform,
            'psutil_available': False,
            'description': 'psutil not available - memory monitoring disabled'
        }

    # Always use RSS for cross-platform consistency and numpy allocation accuracy
    return {
        'metric': 'RSS',
        'platform': sys.platform,
        'psutil_available': True,
        'description': 'Resident Set Size (most reliable for numpy allocations)'
    }
