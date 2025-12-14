"""GPU utilities for accelerated analysis operations.

This module provides utilities for GPU detection, memory management, and
backend selection for analysis operations like clustering and normalization.
Supports both CuPy and PyTorch backends for maximum flexibility.
"""

import logging
from typing import Optional, Dict

# Global flags for GPU backend availability
_GPU_AVAILABLE = None
_CUPY_AVAILABLE = None
_TORCH_AVAILABLE = None


def is_gpu_available() -> bool:
    """Check if CUDA GPU is available via CuPy or PyTorch.

    This function checks for GPU availability by testing both CuPy and PyTorch.
    Results are cached to avoid repeated checks.

    Returns:
        bool: True if at least one GPU backend (CuPy or PyTorch) is available

    Example:
        >>> from aegle_analysis.gpu_utils import is_gpu_available
        >>> if is_gpu_available():
        ...     # Use GPU-accelerated analysis
        ...     pass
        ... else:
        ...     # Fall back to CPU
        ...     pass
    """
    global _GPU_AVAILABLE, _CUPY_AVAILABLE, _TORCH_AVAILABLE

    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE

    # Check CuPy
    try:
        import cupy as cp
        _ = cp.array([1, 2, 3])
        device_count = cp.cuda.runtime.getDeviceCount()
        _CUPY_AVAILABLE = device_count > 0
    except ImportError:
        _CUPY_AVAILABLE = False
    except Exception:
        _CUPY_AVAILABLE = False

    # Check PyTorch
    try:
        import torch
        _TORCH_AVAILABLE = torch.cuda.is_available()
    except ImportError:
        _TORCH_AVAILABLE = False
    except Exception:
        _TORCH_AVAILABLE = False

    # GPU is available if either backend works
    _GPU_AVAILABLE = _CUPY_AVAILABLE or _TORCH_AVAILABLE

    return _GPU_AVAILABLE


def get_gpu_memory_info(device_id: int = 0) -> Optional[Dict[str, float]]:
    """Get GPU memory information.

    Queries GPU memory stats using CuPy or PyTorch, whichever is available.
    Prefers CuPy for consistency with aegle/gpu_utils.py.

    Args:
        device_id: GPU device ID to query (default: 0)

    Returns:
        dict: Memory info with keys:
            - device_id: GPU device number
            - total_mb: Total GPU memory in MB
            - free_mb: Free GPU memory in MB
            - used_mb: Used GPU memory in MB
        None: If GPU unavailable or query fails

    Example:
        >>> from aegle_analysis.gpu_utils import get_gpu_memory_info
        >>> mem_info = get_gpu_memory_info()
        >>> if mem_info:
        ...     print(f"GPU has {mem_info['free_mb']:.0f} MB free")
    """
    if not is_gpu_available():
        return None

    # Try CuPy first
    try:
        import cupy as cp
        device = cp.cuda.Device(device_id)
        free_memory, total_memory = device.mem_info  # Returns (free, total) in bytes
        used_memory = total_memory - free_memory

        return {
            'device_id': device_id,
            'total_mb': total_memory / 1e6,
            'free_mb': free_memory / 1e6,
            'used_mb': used_memory / 1e6,
        }
    except Exception:
        pass

    # Try PyTorch as fallback
    try:
        import torch
        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            # Get memory stats
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            reserved_memory = torch.cuda.memory_reserved(device_id)
            allocated_memory = torch.cuda.memory_allocated(device_id)
            free_memory = total_memory - reserved_memory

            return {
                'device_id': device_id,
                'total_mb': total_memory / 1e6,
                'free_mb': free_memory / 1e6,
                'used_mb': allocated_memory / 1e6,
            }
    except Exception:
        pass

    # Fallback: try nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.free,memory.used',
             '--format=csv,noheader,nounits', '-i', str(device_id)],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            total, free, used = map(float, result.stdout.strip().split(','))
            return {
                'device_id': device_id,
                'total_mb': total,
                'free_mb': free,
                'used_mb': used,
            }
    except Exception:
        pass

    return None


def log_gpu_info(logger: logging.Logger) -> None:
    """Log GPU availability, name, and memory information.

    Logs comprehensive GPU information to the provided logger. If GPU is
    unavailable, logs a message indicating CPU-only mode.

    Args:
        logger: Logger instance to use for output

    Example:
        >>> import logging
        >>> from aegle_analysis.gpu_utils import log_gpu_info
        >>> logger = logging.getLogger(__name__)
        >>> log_gpu_info(logger)
        [INFO] GPU detected: NVIDIA RTX A6000 (48576 MB total, 47234 MB free)
    """
    if not is_gpu_available():
        logger.info("GPU not available, using CPU")
        return

    try:
        # Try to get GPU name and memory
        gpu_name = None
        mem_info = get_gpu_memory_info()

        # Get GPU name from CuPy
        try:
            import cupy as cp
            device = cp.cuda.Device(0)
            gpu_name = device.attributes['Name'].decode('utf-8')
        except Exception:
            pass

        # Get GPU name from PyTorch if CuPy failed
        if gpu_name is None:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                pass

        # Get GPU name from nvidia-smi if both failed
        if gpu_name is None:
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader', '-i', '0'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    gpu_name = result.stdout.strip()
            except Exception:
                pass

        # Format log message
        if gpu_name and mem_info:
            logger.info(
                f"GPU detected: {gpu_name} "
                f"({mem_info['total_mb']:.0f} MB total, {mem_info['free_mb']:.0f} MB free)"
            )
        elif gpu_name:
            logger.info(f"GPU detected: {gpu_name}")
        elif mem_info:
            logger.info(
                f"GPU detected ({mem_info['total_mb']:.0f} MB total, "
                f"{mem_info['free_mb']:.0f} MB free)"
            )
        else:
            logger.info("GPU detected (details unavailable)")

    except Exception as e:
        logger.warning(f"GPU available but failed to query details: {e}")


def select_compute_backend(
    use_gpu: bool,
    fallback_to_cpu: bool,
    logger: logging.Logger
) -> str:
    """Select compute backend (GPU or CPU) based on availability and config.

    Determines whether to use GPU or CPU for analysis operations based on:
    1. User configuration (use_gpu flag)
    2. GPU availability
    3. Fallback preferences (fallback_to_cpu flag)

    Logs the decision and rationale to help with debugging.

    Args:
        use_gpu: Whether user requested GPU acceleration
        fallback_to_cpu: Whether to fall back to CPU if GPU unavailable
        logger: Logger instance for decision logging

    Returns:
        str: "gpu" or "cpu" indicating selected backend

    Raises:
        RuntimeError: If GPU requested but unavailable and fallback disabled

    Example:
        >>> import logging
        >>> from aegle_analysis.gpu_utils import select_compute_backend
        >>> logger = logging.getLogger(__name__)
        >>> backend = select_compute_backend(
        ...     use_gpu=True,
        ...     fallback_to_cpu=True,
        ...     logger=logger
        ... )
        >>> if backend == "gpu":
        ...     # Use GPU-accelerated analysis
        ...     pass
        ... else:
        ...     # Use CPU-based analysis
        ...     pass
    """
    if not use_gpu:
        logger.info("GPU acceleration disabled by configuration, using CPU")
        return "cpu"

    gpu_available = is_gpu_available()

    if gpu_available:
        logger.info("GPU acceleration enabled and available, using GPU")
        log_gpu_info(logger)
        return "gpu"

    # GPU requested but not available
    if fallback_to_cpu:
        logger.warning(
            "GPU acceleration requested but GPU not available, "
            "falling back to CPU"
        )
        return "cpu"
    else:
        error_msg = (
            "GPU acceleration requested but GPU not available. "
            "Set fallback_to_cpu=True to allow CPU fallback, or install "
            "CuPy (pip install cupy-cuda12x) or PyTorch with CUDA support."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
