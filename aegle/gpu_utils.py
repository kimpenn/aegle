"""GPU utilities for accelerated computing.

This module provides utilities for GPU detection, memory management, and
batch size estimation for CuPy-accelerated operations.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Global flag for CuPy availability
_CUPY_AVAILABLE = None
_GPU_MEMORY_INFO = {}


def is_cupy_available():
    """Check if CuPy is available and functional.

    Returns:
        bool: True if CuPy is installed and at least one GPU is detected
    """
    global _CUPY_AVAILABLE

    if _CUPY_AVAILABLE is not None:
        return _CUPY_AVAILABLE

    try:
        import cupy as cp
        # Test basic operation
        _ = cp.array([1, 2, 3])
        device_count = cp.cuda.runtime.getDeviceCount()
        _CUPY_AVAILABLE = device_count > 0
        if _CUPY_AVAILABLE:
            logger.info(f"CuPy available with {device_count} GPU(s)")
        else:
            logger.warning("CuPy installed but no GPUs detected")
        return _CUPY_AVAILABLE
    except ImportError:
        logger.warning("CuPy not installed - GPU acceleration unavailable")
        _CUPY_AVAILABLE = False
        return False
    except Exception as e:
        logger.warning(f"CuPy available but initialization failed: {e}")
        _CUPY_AVAILABLE = False
        return False


def get_gpu_memory_info(device_id=0):
    """Get GPU memory information.

    Args:
        device_id: GPU device ID to query (default: 0)

    Returns:
        dict: Memory info with keys:
            - device_id: GPU device number
            - total_gb: Total GPU memory in GB
            - free_gb: Free GPU memory in GB
            - used_gb: Used GPU memory in GB
            - pool_used_gb: Memory used by CuPy pool in GB
            - pool_total_gb: Total memory allocated by CuPy pool in GB
        None: If GPU unavailable or query fails
    """
    if not is_cupy_available():
        return None

    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()

        # Get device properties
        device = cp.cuda.Device(device_id)
        free_memory, total_memory = device.mem_info  # Returns (free, total) in bytes
        used_memory = total_memory - free_memory

        # Pool memory
        pool_used = mempool.used_bytes()
        pool_total = mempool.total_bytes()

        info = {
            'device_id': device_id,
            'total_gb': total_memory / 1e9,
            'free_gb': free_memory / 1e9,
            'used_gb': used_memory / 1e9,
            'pool_used_gb': pool_used / 1e9,
            'pool_total_gb': pool_total / 1e9,
        }

        _GPU_MEMORY_INFO[device_id] = info
        return info
    except Exception as e:
        logger.warning(f"Failed to get GPU memory info: {e}")
        return None


def estimate_gpu_batch_size(
    n_channels,
    image_shape,
    n_cells,
    dtype=np.float32,
    safety_factor=0.7
):
    """Estimate optimal GPU batch size based on available memory.

    Args:
        n_channels: Total number of channels to process
        image_shape: (height, width) of the image
        n_cells: Number of cells (max label)
        dtype: Data type for computations (default: np.float32)
        safety_factor: Use this fraction of available memory (default: 0.7)

    Returns:
        int: Recommended batch size (number of channels per batch)
    """
    mem_info = get_gpu_memory_info()
    if mem_info is None:
        logger.warning("Could not determine GPU memory, using conservative batch size of 5")
        return 5

    available_gb = mem_info['free_gb'] * safety_factor
    logger.info(
        f"GPU memory: {mem_info['free_gb']:.2f} GB free, "
        f"using {available_gb:.2f} GB for batching"
    )

    # Estimate memory per channel:
    # - Channel data: height * width * dtype_size
    # - Masks (2): height * width * 2 * 8 bytes (int64) - shared across batches
    # - Intermediate arrays: height * width * dtype_size * 2 (weights, squared)
    # - Output arrays: n_cells * dtype_size * 3 (sums for cell, nucleus, variance)

    dtype_size = np.dtype(dtype).itemsize
    h, w = image_shape
    pixels = h * w

    # Memory for masks (allocated once, shared across batches)
    masks_gb = pixels * 8 * 2 / 1e9  # 2 int64 masks

    # Memory per channel
    per_channel_gb = (
        pixels * dtype_size / 1e9 +              # 1 channel data
        pixels * dtype_size * 2 / 1e9 +          # intermediate arrays (weights, squared)
        n_cells * dtype_size * 3 / 1e9           # output arrays per channel
    )

    # Available memory for channels (after allocating masks)
    available_for_channels = available_gb - masks_gb

    if available_for_channels <= 0:
        logger.warning(
            f"Insufficient GPU memory for masks ({masks_gb:.2f} GB required, "
            f"{available_gb:.2f} GB available). Using batch size 1."
        )
        return 1

    # Estimate batch size
    estimated_batch = max(1, int(available_for_channels / per_channel_gb))

    # Cap at total channels
    batch_size = min(estimated_batch, n_channels)

    logger.info(
        f"Estimated GPU batch size: {batch_size} channels "
        f"({per_channel_gb:.2f} GB per channel, "
        f"{masks_gb:.2f} GB for masks, "
        f"{available_for_channels:.2f} GB available for channels)"
    )

    return batch_size


def clear_gpu_memory():
    """Clear GPU memory pool.

    Frees all unused blocks in CuPy's memory pool. Useful for managing
    memory between large operations.
    """
    if not is_cupy_available():
        return

    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        logger.debug("Cleared GPU memory pools")
    except Exception as e:
        logger.warning(f"Failed to clear GPU memory: {e}")


def log_gpu_memory(prefix="GPU memory"):
    """Log current GPU memory usage.

    Args:
        prefix: String to prepend to log message
    """
    mem_info = get_gpu_memory_info()
    if mem_info:
        logger.info(
            f"{prefix}: {mem_info['used_gb']:.2f}/{mem_info['total_gb']:.2f} GB used "
            f"({mem_info['free_gb']:.2f} GB free), "
            f"pool: {mem_info['pool_used_gb']:.2f}/{mem_info['pool_total_gb']:.2f} GB"
        )
    else:
        logger.debug(f"{prefix}: Unable to query GPU memory")
