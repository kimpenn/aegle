"""Performance measurement and baseline tracking utilities.

This module provides tools for benchmarking analysis functions and tracking
performance baselines to detect regressions over time.
"""

import json
import logging
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def benchmark_function(
    func: Callable,
    *args,
    warmup: bool = True,
    n_runs: int = 1,
    **kwargs,
) -> Tuple[Any, float]:
    """Accurately time a function execution.

    Parameters
    ----------
    func : Callable
        Function to benchmark
    *args
        Positional arguments to pass to func
    warmup : bool
        Whether to run function once before timing (default: True)
    n_runs : int
        Number of times to run and average (default: 1)
    **kwargs
        Keyword arguments to pass to func

    Returns
    -------
    result : Any
        Return value from the function
    elapsed_time : float
        Elapsed time in seconds (averaged over n_runs if n_runs > 1)

    Examples
    --------
    >>> result, time_sec = benchmark_function(my_func, arg1, arg2, kwarg1=value)
    >>> print(f"Function took {time_sec:.3f} seconds")
    """
    # Warmup run to avoid cold-start effects
    if warmup:
        _ = func(*args, **kwargs)

    # Benchmark run(s)
    start_time = time.perf_counter()
    for _ in range(n_runs):
        result = func(*args, **kwargs)
    end_time = time.perf_counter()

    elapsed_time = (end_time - start_time) / n_runs

    return result, elapsed_time


def record_baseline(
    stage_name: str,
    time_seconds: float,
    baseline_file: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Record a performance baseline to JSON file.

    Parameters
    ----------
    stage_name : str
        Name of the benchmark stage (e.g., "normalization_small")
    time_seconds : float
        Execution time in seconds
    baseline_file : Path
        Path to baseline JSON file
    metadata : Optional[Dict[str, Any]]
        Additional metadata to store (e.g., dataset size, config)

    Notes
    -----
    - Preserves previous baselines for history tracking
    - Includes timestamp and system info automatically
    - Creates parent directories if they don't exist
    """
    baseline_file = Path(baseline_file)
    baseline_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing baselines
    if baseline_file.exists():
        with open(baseline_file, "r") as f:
            baselines = json.load(f)
    else:
        baselines = {}

    # Get system info
    system_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }

    # Store baseline with metadata
    baseline_entry = {
        "time_seconds": time_seconds,
        "timestamp": datetime.now().isoformat(),
        "system_info": system_info,
    }

    if metadata:
        baseline_entry["metadata"] = metadata

    # Preserve history: move current baseline to history
    if stage_name in baselines:
        if "history" not in baselines[stage_name]:
            baselines[stage_name]["history"] = []
        baselines[stage_name]["history"].append(
            {
                "time_seconds": baselines[stage_name]["time_seconds"],
                "timestamp": baselines[stage_name]["timestamp"],
            }
        )
        # Keep only last 10 historical entries
        baselines[stage_name]["history"] = baselines[stage_name]["history"][-10:]

    baselines[stage_name] = baseline_entry

    # Save updated baselines
    with open(baseline_file, "w") as f:
        json.dump(baselines, f, indent=2)

    logger.info(f"Recorded baseline for '{stage_name}': {time_seconds:.4f}s")


def compare_to_baseline(
    stage_name: str,
    current_time: float,
    baseline_file: Path,
    tolerance: float = 0.20,
) -> Tuple[str, float, Optional[float]]:
    """Compare current performance to baseline.

    Parameters
    ----------
    stage_name : str
        Name of the benchmark stage
    current_time : float
        Current execution time in seconds
    baseline_file : Path
        Path to baseline JSON file
    tolerance : float
        Acceptable slowdown ratio (default: 0.20 = 20% slower)

    Returns
    -------
    status : str
        One of: "faster", "similar", "slower", "no_baseline"
    ratio : float
        Current time / baseline time (e.g., 1.5 = 50% slower)
    baseline_time : Optional[float]
        Baseline time in seconds, or None if no baseline exists

    Examples
    --------
    >>> status, ratio, baseline = compare_to_baseline("norm_small", 0.5, baseline_file)
    >>> if status == "slower":
    ...     print(f"WARNING: {ratio:.2f}x slower than baseline!")
    """
    baseline_file = Path(baseline_file)

    if not baseline_file.exists():
        logger.warning(f"No baseline file found at {baseline_file}")
        return "no_baseline", 1.0, None

    with open(baseline_file, "r") as f:
        baselines = json.load(f)

    if stage_name not in baselines:
        logger.warning(f"No baseline found for stage '{stage_name}'")
        return "no_baseline", 1.0, None

    baseline_time = baselines[stage_name]["time_seconds"]
    ratio = current_time / baseline_time

    if ratio < (1.0 - tolerance / 2):
        status = "faster"
    elif ratio > (1.0 + tolerance):
        status = "slower"
    else:
        status = "similar"

    logger.info(
        f"Stage '{stage_name}': {current_time:.4f}s vs baseline {baseline_time:.4f}s "
        f"(ratio: {ratio:.2f}x, status: {status})"
    )

    return status, ratio, baseline_time


def load_baselines(baseline_file: Path) -> Dict[str, Any]:
    """Load all baselines from JSON file.

    Parameters
    ----------
    baseline_file : Path
        Path to baseline JSON file

    Returns
    -------
    baselines : Dict[str, Any]
        Dictionary of all baselines, or empty dict if file doesn't exist
    """
    baseline_file = Path(baseline_file)

    if not baseline_file.exists():
        return {}

    with open(baseline_file, "r") as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string.

    Parameters
    ----------
    seconds : float
        Time in seconds

    Returns
    -------
    formatted : str
        Human-readable time string (e.g., "1.23s", "123ms", "12.3µs")

    Examples
    --------
    >>> format_time(1.234)
    '1.23s'
    >>> format_time(0.123)
    '123ms'
    >>> format_time(0.000123)
    '123µs'
    """
    if seconds >= 1.0:
        return f"{seconds:.2f}s"
    elif seconds >= 0.001:
        return f"{seconds * 1000:.0f}ms"
    else:
        return f"{seconds * 1_000_000:.1f}µs"
