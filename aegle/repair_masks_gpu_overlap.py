"""GPU-accelerated overlap computation for mask repair.

This module provides GPU-accelerated cell-nucleus overlap matrix computation
using CuPy, offering 5-10x speedup over CPU for large samples.

IMPORTANT: Returns sparse CSR matrices to avoid OOM on production-scale data.
Dense format would require 3.60 TiB for D18_0 (1.99M cells × 1.99M nuclei).
Sparse format requires ~20 MB (156,522x reduction).
"""

import logging
import numpy as np
from typing import Tuple, Optional
from scipy.sparse import coo_matrix, csr_matrix, issparse

logger = logging.getLogger(__name__)


def compute_overlap_matrix_gpu_auto(
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    batch_size: int = 0,
    use_bincount: bool = True,
    bincount_chunk_size: int = 0,
    overlap_num_gpus: int = 1,
    fallback_enabled: bool = True,
) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Smart wrapper with automatic fallback chain for overlap computation.

    Fallback chain:
        1. Phase 5c (chunked bincount) - if use_bincount=True
           - Multi-GPU if overlap_num_gpus > 1
        2. Phase 4 (sequential cp.unique) - if GPU available
           - Multi-GPU if overlap_num_gpus > 1
        3. CPU fallback - if GPU fails or unavailable

    This provides robust operation: tries fastest method first, falls back if needed.

    Args:
        cell_mask: 2D cell segmentation mask (H, W) with integer labels
        nucleus_mask: 2D nucleus segmentation mask (H, W) with integer labels
        batch_size: Number of cells to process per GPU batch (0=auto)
                   Only used for Phase 4 fallback
        use_bincount: If True, try Phase 5c bincount first (default: True)
        bincount_chunk_size: Cells per chunk for Phase 5c (default: 20000)
                            Set to 0 for auto-sizing, None to disable chunking
        overlap_num_gpus: Number of GPUs to use for overlap computation (default: 1)
        fallback_enabled: If True, fall back on errors (default: True)

    Returns:
        Tuple of:
            - overlap_sparse: scipy.sparse.csr_matrix (n_cells, n_nuclei)
            - cell_labels: (n_cells,) array of cell label IDs
            - nucleus_labels: (n_nuclei,) array of nucleus label IDs

    Raises:
        RuntimeError: If all methods fail and fallback_enabled=False
    """
    from aegle.gpu_utils import is_cupy_available

    # Proactive decision: if auto-sizing, check if Phase 5c is viable before attempting
    calculated_chunk_size = bincount_chunk_size
    if use_bincount and bincount_chunk_size == 0 and is_cupy_available():
        # Extract cell and nucleus counts for proactive decision
        cell_labels_temp = np.unique(cell_mask)
        cell_labels_temp = cell_labels_temp[cell_labels_temp != 0]
        n_cells_temp = len(cell_labels_temp)

        nucleus_labels_temp = np.unique(nucleus_mask)
        nucleus_labels_temp = nucleus_labels_temp[nucleus_labels_temp != 0]
        n_nuclei_temp = len(nucleus_labels_temp)

        if n_cells_temp > 0 and n_nuclei_temp > 0:
            # Make proactive decision about Phase 5c vs Phase 4
            calculated_chunk_size, _, use_phase5c = calculate_optimal_chunk_size(
                n_cells_temp, n_nuclei_temp, num_gpus=overlap_num_gpus
            )

            if not use_phase5c:
                # Skip Phase 5c entirely, go directly to Phase 4
                logger.info(
                    "Proactive decision: skipping Phase 5c (chunk too small), "
                    "using Phase 4 directly"
                )
                use_bincount = False  # Disable bincount, fall through to Phase 4

    # Try Phase 5c bincount (fastest)
    if use_bincount and is_cupy_available():
        try:
            logger.info("Trying Phase 5c (chunked bincount approach)...")

            # Use multi-GPU if requested
            if overlap_num_gpus > 1:
                # For multi-GPU, need to calculate chunk size if auto-sizing
                if calculated_chunk_size == 0:
                    # Should have been calculated in proactive decision
                    calculated_chunk_size = 20000  # Fallback

                return compute_overlap_matrix_gpu_bincount_multi_gpu(
                    cell_mask, nucleus_mask,
                    chunk_size=calculated_chunk_size,
                    num_gpus=overlap_num_gpus
                )
            else:
                return compute_overlap_matrix_gpu_bincount(
                    cell_mask, nucleus_mask, chunk_size=calculated_chunk_size
                )
        except Exception as e:
            if fallback_enabled:
                logger.warning(f"Phase 5c failed: {e}")
                logger.info("Falling back to Phase 4 (sequential cp.unique)...")
            else:
                raise RuntimeError(f"Phase 5c failed (fallback disabled): {e}") from e

    # Try Phase 4 sequential (baseline)
    if is_cupy_available():
        try:
            if use_bincount:  # Only log if we're falling back from bincount
                logger.info("Using Phase 4 (sequential cp.unique)...")

            # Use multi-GPU if requested
            if overlap_num_gpus > 1:
                return compute_overlap_matrix_gpu_multi_gpu(
                    cell_mask, nucleus_mask, batch_size, num_gpus=overlap_num_gpus
                )
            else:
                return compute_overlap_matrix_gpu(cell_mask, nucleus_mask, batch_size)
        except Exception as e:
            if fallback_enabled:
                logger.warning(f"Phase 4 GPU failed: {e}")
                logger.info("Falling back to CPU...")
            else:
                raise RuntimeError(f"Phase 4 failed (fallback disabled): {e}") from e

    # Final fallback: CPU
    logger.info("Using CPU fallback...")
    return _compute_overlap_matrix_cpu(cell_mask, nucleus_mask)


def compute_overlap_matrix_gpu(
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    batch_size: int = 0,
) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Compute cell-nucleus overlap matrix on GPU (SPARSE CSR FORMAT).

    For each cell-nucleus pair, determines if they share any pixels.
    This is the GPU-accelerated version of the implicit overlap computation
    in the CPU repair pipeline.

    IMPORTANT: Returns scipy.sparse.csr_matrix to avoid OOM on production data.
    For D18_0 (1.99M cells), dense would require 3.60 TiB; sparse requires ~20 MB.

    Args:
        cell_mask: 2D cell segmentation mask (H, W) with integer labels
        nucleus_mask: 2D nucleus segmentation mask (H, W) with integer labels
        batch_size: Number of cells to process per GPU batch (0=auto)
                   Used to avoid OOM on very large masks

    Returns:
        Tuple of:
            - overlap_sparse: scipy.sparse.csr_matrix (n_cells, n_nuclei) where
                             overlap_sparse[i, j] = 1 if cell i overlaps nucleus j
            - cell_labels: (n_cells,) array of cell label IDs
            - nucleus_labels: (n_nuclei,) array of nucleus label IDs

    Example:
        >>> overlap, cell_ids, nucleus_ids = compute_overlap_matrix_gpu(cell_mask, nucleus_mask)
        >>> # Find which nuclei overlap with cell i
        >>> overlap_row = overlap.getrow(i)  # Get row i (sparse CSR)
        >>> nuclei_indices = overlap_row.indices  # Column indices of nonzeros
        >>> nuclei_for_cell_i = nucleus_ids[nuclei_indices]
    """
    # Check GPU availability
    from aegle.gpu_utils import is_cupy_available, clear_gpu_memory, log_gpu_memory

    if not is_cupy_available():
        logger.warning("GPU not available, falling back to CPU version")
        return _compute_overlap_matrix_cpu(cell_mask, nucleus_mask)

    try:
        import cupy as cp

        logger.debug("Computing overlap matrix on GPU...")
        log_gpu_memory("GPU memory before overlap computation")

        # Ensure masks are 2D and squeeze if needed
        cell_mask = np.asarray(cell_mask).squeeze()
        nucleus_mask = np.asarray(nucleus_mask).squeeze()

        if cell_mask.shape != nucleus_mask.shape:
            raise ValueError(
                f"Mask shapes must match: cell_mask {cell_mask.shape} "
                f"vs nucleus_mask {nucleus_mask.shape}"
            )

        # Get unique labels (excluding background 0)
        cell_labels = np.unique(cell_mask)
        cell_labels = cell_labels[cell_labels != 0].astype(np.int32)

        nucleus_labels = np.unique(nucleus_mask)
        nucleus_labels = nucleus_labels[nucleus_labels != 0].astype(np.int32)

        n_cells = len(cell_labels)
        n_nuclei = len(nucleus_labels)

        logger.info(f"Computing overlap matrix: {n_cells} cells × {n_nuclei} nuclei")

        if n_cells == 0 or n_nuclei == 0:
            # Empty case - return empty sparse matrix
            overlap_sparse = csr_matrix((n_cells, n_nuclei), dtype=np.uint8)
            logger.info("Empty overlap matrix (no cells or nuclei)")
            return overlap_sparse, cell_labels, nucleus_labels

        # Transfer masks to GPU
        cell_mask_gpu = cp.asarray(cell_mask, dtype=cp.int32)
        nucleus_mask_gpu = cp.asarray(nucleus_mask, dtype=cp.int32)

        # Flatten for vectorized operations
        cell_flat = cell_mask_gpu.ravel()
        nucleus_flat = nucleus_mask_gpu.ravel()

        # Pre-allocate COO storage for sparse matrix construction
        # Avoids dense allocation which would require 3.60 TiB for D18_0
        row_indices = []  # Will store cell indices (i) for each overlap
        col_indices = []  # Will store nucleus indices (j) for each overlap

        # Determine batch size if auto
        if batch_size == 0:
            # Estimate based on memory: need to store nucleus IDs for each cell's pixels
            # Conservative estimate: process cells that collectively have <100M pixels
            avg_pixels_per_cell = cell_mask.size / n_cells if n_cells > 0 else 1000
            target_total_pixels = 100_000_000  # 100M pixels
            batch_size = max(1, int(target_total_pixels / avg_pixels_per_cell))
            batch_size = min(batch_size, n_cells)  # Cap at total cells
            logger.debug(
                f"Auto batch size: {batch_size} cells "
                f"(avg {avg_pixels_per_cell:.0f} pixels/cell)"
            )

        # Create label→index mapping for fast lookup
        # This is CRITICAL for handling non-contiguous labels (e.g., [1, 5, 7, 23])
        nucleus_label_to_idx = {label: idx for idx, label in enumerate(nucleus_labels)}
        nucleus_label_to_idx_keys = cp.asarray(
            list(nucleus_label_to_idx.keys()), dtype=cp.int32
        )
        nucleus_label_to_idx_values = cp.asarray(
            list(nucleus_label_to_idx.values()), dtype=cp.int32
        )

        # Process cells in batches to avoid OOM
        n_batches = (n_cells + batch_size - 1) // batch_size

        # Initialize progress tracking (Agent A's enhancement)
        import time
        from aegle.gpu_utils import get_gpu_memory_info

        total_cells = n_cells
        cells_processed = 0
        start_time = time.time()
        last_log_cells = 0
        log_interval = 50000  # Log every 50K cells

        for batch_idx in range(0, n_cells, batch_size):
            batch_end = min(batch_idx + batch_size, n_cells)
            batch_num = batch_idx // batch_size + 1

            if n_batches > 1:
                logger.debug(
                    f"Processing GPU batch {batch_num}/{n_batches} "
                    f"(cells {batch_idx+1}-{batch_end})"
                )

            # Process each cell in the batch
            for i in range(batch_idx, batch_end):
                cell_id = cell_labels[i]

                # Create boolean mask for current cell's pixels
                cell_pixels_mask = cell_flat == cell_id

                # Get nucleus IDs at cell pixel locations
                nucleus_ids_at_cell = nucleus_flat[cell_pixels_mask]

                # Find unique nucleus IDs (excluding 0=background)
                overlapping_nuclei = cp.unique(nucleus_ids_at_cell)
                overlapping_nuclei = overlapping_nuclei[overlapping_nuclei != 0]

                # Convert to CPU for dictionary lookup
                overlapping_nuclei_cpu = cp.asnumpy(overlapping_nuclei)

                # Accumulate sparse entries using label→index mapping
                for nucleus_id in overlapping_nuclei_cpu:
                    if nucleus_id in nucleus_label_to_idx:
                        j = nucleus_label_to_idx[nucleus_id]
                        row_indices.append(i)  # Cell index
                        col_indices.append(j)  # Nucleus index

            # Update progress (Agent A's enhancement)
            cells_processed = batch_end
            batch_time = time.time() - start_time

            # Progress logging every log_interval cells
            if cells_processed - last_log_cells >= log_interval or cells_processed == total_cells:
                elapsed = time.time() - start_time
                rate = cells_processed / elapsed if elapsed > 0 else 0
                eta_sec = (total_cells - cells_processed) / rate if rate > 0 else 0
                eta_min = eta_sec / 60

                progress_pct = 100 * cells_processed / total_cells
                avg_overlaps = len(row_indices) / cells_processed if cells_processed > 0 else 0

                # GPU memory
                mem_info = get_gpu_memory_info()
                gpu_used = mem_info['used_gb'] if mem_info else 0
                gpu_total = mem_info['total_gb'] if mem_info else 0

                logger.info(
                    f"Stage 3 progress: {cells_processed:,}/{total_cells:,} cells "
                    f"({progress_pct:.1f}%) - "
                    f"{rate:.0f} cells/sec - "
                    f"ETA: {eta_min:.1f} min"
                )
                logger.info(
                    f"  GPU memory: {gpu_used:.1f}/{gpu_total:.1f} GB, "
                    f"Avg overlaps/cell: {avg_overlaps:.2f}"
                )

                last_log_cells = cells_processed

            # Clear GPU memory after each batch
            if batch_num < n_batches:
                clear_gpu_memory()

        # Clean up GPU arrays
        del cell_mask_gpu, nucleus_mask_gpu, cell_flat, nucleus_flat
        clear_gpu_memory()

        log_gpu_memory("GPU memory after overlap computation")

        # Build sparse matrix (COO → CSR for fast row access)
        if len(row_indices) == 0:
            # No overlaps found (rare edge case)
            overlap_sparse = csr_matrix((n_cells, n_nuclei), dtype=np.uint8)
            logger.info("No overlaps found (all cells lack nuclei or vice versa)")
        else:
            # Build COO matrix from accumulated indices
            data = np.ones(len(row_indices), dtype=np.uint8)  # All values are 1 (True)
            overlap_coo = coo_matrix(
                (data, (row_indices, col_indices)),
                shape=(n_cells, n_nuclei),
                dtype=np.uint8,
            )

            # Convert to CSR for efficient row slicing (O(1) access per row)
            overlap_sparse = overlap_coo.tocsr()

        # Calculate statistics
        nnz = overlap_sparse.nnz
        density = nnz / (n_cells * n_nuclei) if n_cells * n_nuclei > 0 else 0
        size_mb = (
            overlap_sparse.data.nbytes
            + overlap_sparse.indices.nbytes
            + overlap_sparse.indptr.nbytes
        ) / 1e6

        logger.info(
            f"Overlap matrix computed (sparse CSR): "
            f"{nnz:,} overlaps out of {n_cells * n_nuclei:,} possible pairs "
            f"({density * 100:.3f}% density), "
            f"{size_mb:.2f} MB sparse storage"
        )

        return overlap_sparse, cell_labels, nucleus_labels

    except Exception as e:
        logger.error(f"GPU overlap computation failed: {e}")
        logger.warning("Falling back to CPU version")
        clear_gpu_memory()
        return _compute_overlap_matrix_cpu(cell_mask, nucleus_mask)


def compute_overlap_matrix_gpu_multi_gpu(
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    batch_size: int = 0,
    num_gpus: int = 2,
) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Multi-GPU Phase 4: Sequential cp.unique with cell-level round-robin distribution.

    Distributes cells across multiple GPUs using round-robin assignment (GPU 0: cells
    0,2,4..., GPU 1: cells 1,3,5...) for balanced workload. Each GPU processes its
    assigned cells independently using cp.unique, then results are combined.

    Expected speedup: ~2x for 2 GPUs on large samples.

    Args:
        cell_mask: 2D cell segmentation mask (H, W) with integer labels
        nucleus_mask: 2D nucleus segmentation mask (H, W) with integer labels
        batch_size: Number of cells per batch (unused, kept for API compatibility)
        num_gpus: Number of GPUs to use (default: 2)

    Returns:
        Tuple of:
            - overlap_sparse: scipy.sparse.csr_matrix (n_cells, n_nuclei)
            - cell_labels: (n_cells,) array of cell label IDs
            - nucleus_labels: (n_nuclei,) array of nucleus label IDs

    Raises:
        RuntimeError: If GPU processing fails
    """
    from aegle.gpu_utils import is_cupy_available, clear_gpu_memory, log_gpu_memory
    from concurrent.futures import ThreadPoolExecutor
    import cupy as cp
    import time

    if not is_cupy_available():
        raise RuntimeError("GPU required for multi-GPU overlap computation")

    try:
        start_time = time.time()
        logger.info(f"Phase 4 Multi-GPU: Computing overlap with {num_gpus} GPUs...")
        log_gpu_memory("GPU memory before multi-GPU Phase 4")

        # Validate GPU count
        actual_gpu_count = cp.cuda.runtime.getDeviceCount()
        if num_gpus > actual_gpu_count:
            logger.warning(
                f"Requested {num_gpus} GPUs but only {actual_gpu_count} available"
            )
            num_gpus = actual_gpu_count

        # Get labels
        cell_labels = np.unique(cell_mask)
        cell_labels = cell_labels[cell_labels != 0].astype(np.int32)
        nucleus_labels = np.unique(nucleus_mask)
        nucleus_labels = nucleus_labels[nucleus_labels != 0].astype(np.int32)

        n_cells = len(cell_labels)
        n_nuclei = len(nucleus_labels)

        logger.info(f"Computing overlap matrix: {n_cells:,} cells × {n_nuclei:,} nuclei")

        if n_cells == 0 or n_nuclei == 0:
            overlap_sparse = csr_matrix((n_cells, n_nuclei), dtype=np.uint8)
            return overlap_sparse, cell_labels, nucleus_labels

        # Round-robin cell distribution
        gpu_cell_indices = [[] for _ in range(num_gpus)]
        for cell_idx in range(n_cells):
            gpu_id = cell_idx % num_gpus
            gpu_cell_indices[gpu_id].append(cell_idx)

        for gpu_id in range(num_gpus):
            logger.debug(f"  GPU {gpu_id}: {len(gpu_cell_indices[gpu_id]):,} cells")

        # Worker function
        def process_gpu_cells(gpu_id, cell_indices_list):
            """Process assigned cells on a specific GPU."""
            cp.cuda.Device(gpu_id).use()

            logger.debug(f"  [GPU {gpu_id}] Starting with {len(cell_indices_list):,} cells")

            # Transfer masks
            cell_flat = cp.asarray(cell_mask.ravel(), dtype=cp.int32)
            nucleus_flat = cp.asarray(nucleus_mask.ravel(), dtype=cp.int32)

            overlap_dict = {}
            progress_interval = max(1, len(cell_indices_list) // 10)

            for local_idx, cell_idx in enumerate(cell_indices_list):
                cell_label = cell_labels[cell_idx]

                # Find pixels for this cell
                cell_pixels = (cell_flat == cell_label)
                overlapping_nuclei = cp.unique(nucleus_flat[cell_pixels])
                overlapping_nuclei = overlapping_nuclei[overlapping_nuclei != 0]

                if len(overlapping_nuclei) > 0:
                    nucleus_indices = np.searchsorted(
                        nucleus_labels, cp.asnumpy(overlapping_nuclei)
                    )
                    for nuc_idx in nucleus_indices:
                        overlap_dict[(cell_idx, nuc_idx)] = 1

                if (local_idx + 1) % progress_interval == 0:
                    pct = 100 * (local_idx + 1) / len(cell_indices_list)
                    logger.debug(
                        f"  [GPU {gpu_id}] Progress: {local_idx+1:,}/{len(cell_indices_list):,} "
                        f"({pct:.1f}%)"
                    )

            logger.debug(f"  [GPU {gpu_id}] Completed all {len(cell_indices_list):,} cells")

            del cell_flat, nucleus_flat
            cp.get_default_memory_pool().free_all_blocks()

            return overlap_dict

        # Execute in parallel
        parallel_start = time.time()

        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for gpu_id in range(num_gpus):
                if len(gpu_cell_indices[gpu_id]) > 0:
                    future = executor.submit(
                        process_gpu_cells, gpu_id, gpu_cell_indices[gpu_id]
                    )
                    futures.append(future)

            all_overlap_dicts = []
            for future in futures:
                all_overlap_dicts.append(future.result())

        parallel_time = time.time() - parallel_start
        logger.info(f"  Parallel processing completed in {parallel_time:.2f}s")

        # Combine dictionaries
        combined_dict = {}
        for overlap_dict in all_overlap_dicts:
            combined_dict.update(overlap_dict)

        # Build sparse matrix
        if combined_dict:
            rows, cols = zip(*combined_dict.keys())
            data = np.ones(len(rows), dtype=np.uint8)
            overlap_sparse = csr_matrix(
                (data, (rows, cols)), shape=(n_cells, n_nuclei), dtype=np.uint8
            )
        else:
            overlap_sparse = csr_matrix((n_cells, n_nuclei), dtype=np.uint8)

        total_time = time.time() - start_time
        nnz = overlap_sparse.nnz
        logger.info(
            f"  ✓ Multi-GPU Phase 4: {nnz:,} overlaps in {total_time:.2f}s "
            f"({n_cells / total_time:.0f} cells/sec)"
        )

        return overlap_sparse, cell_labels, nucleus_labels

    except Exception as e:
        logger.error(f"Multi-GPU Phase 4 failed: {e}")
        clear_gpu_memory()
        raise RuntimeError(f"Multi-GPU overlap failed: {e}") from e


def calculate_optimal_chunk_size(
    n_cells: int,
    n_nuclei: int,
    num_gpus: int = 1,
    safety_margin: float = 0.85,
    min_viable_chunk: int = 500,
) -> Tuple[int, float, bool]:
    """Calculate optimal chunk size from available VRAM with proactive Phase decision.

    This function determines the best chunk size for Phase 5c bincount processing
    based on available GPU memory, and decides whether Phase 5c is viable or if
    Phase 4 should be used instead.

    Decision logic:
        1. Calculate optimal chunk size from available VRAM
        2. If chunk_size < min_viable_chunk: Use Phase 4 (overhead not worth it)
        3. If chunk_size >= n_cells * 0.9: Use single-pass bincount (all cells fit)
        4. Otherwise: Use chunked Phase 5c with calculated chunk size

    Args:
        n_cells: Number of cells in the sample
        n_nuclei: Number of nuclei in the sample
        num_gpus: Number of GPUs available (default: 1)
        safety_margin: VRAM utilization factor (default: 0.85 = 85%)
        min_viable_chunk: Minimum chunk size for Phase 5c viability (default: 500)

    Returns:
        Tuple of:
            - chunk_size: Optimal chunk size (cells per chunk)
            - estimated_memory_gb: Estimated memory per chunk in GB
            - use_phase5c: True if Phase 5c should be used, False to use Phase 4

    Example:
        For D11_img0018_1 with 580k cells, 512k nuclei, 47 GB free VRAM:
            chunk_size, mem_gb, use_phase5c = calculate_optimal_chunk_size(580717, 512882)
            # Returns: (9756, 41.0, True) - use Phase 5c with 9756 cells/chunk
    """
    from aegle.gpu_utils import get_gpu_memory_info

    # Get available VRAM (use first GPU's memory, assume symmetric)
    mem_info = get_gpu_memory_info()
    if not mem_info:
        logger.warning(
            "  Could not detect GPU memory, using conservative default chunk_size=10000"
        )
        return 10000, (10000 * n_nuclei * 8) / 1e9, True

    free_vram_bytes = mem_info['free_gb'] * 1e9

    # Calculate optimal chunk size: (free_VRAM * safety_margin) / (n_nuclei * 8 bytes)
    # Each chunk creates a (chunk_size × n_nuclei) matrix, 8 bytes per element (int64)
    optimal_chunk_size = int((free_vram_bytes * safety_margin) / (n_nuclei * 8))

    # Cap at total cells (no point in chunks larger than total)
    optimal_chunk_size = min(optimal_chunk_size, n_cells)

    # Ensure at least 1 cell per chunk
    optimal_chunk_size = max(optimal_chunk_size, 1)

    # Estimate memory usage per chunk
    estimated_memory_gb = (optimal_chunk_size * n_nuclei * 8) / 1e9

    # Proactive decision: should we use Phase 5c or go directly to Phase 4?
    if optimal_chunk_size < min_viable_chunk:
        # Chunk too small - Phase 5c overhead not worth it
        logger.info(
            f"  Chunk size {optimal_chunk_size:,} < min_viable ({min_viable_chunk:,})"
        )
        logger.info(
            f"  Phase 5c overhead too high for small chunks, will use Phase 4 multi-GPU"
        )
        use_phase5c = False
    elif optimal_chunk_size >= n_cells * 0.9:
        # Can fit ~all cells in single pass
        logger.info(
            f"  Chunk size {optimal_chunk_size:,} >= 90% of cells ({n_cells:,})"
        )
        logger.info(
            f"  Using single-pass bincount (all cells fit in memory)"
        )
        use_phase5c = True
    else:
        # Sweet spot: chunked Phase 5c is optimal
        n_chunks_est = (n_cells + optimal_chunk_size - 1) // optimal_chunk_size
        logger.info(
            f"  Optimal chunk size: {optimal_chunk_size:,} cells/chunk "
            f"(~{n_chunks_est} chunks, {estimated_memory_gb:.2f} GB/chunk)"
        )
        logger.info(
            f"  Using chunked Phase 5c with multi-GPU ({num_gpus} GPUs)"
        )
        use_phase5c = True

    return optimal_chunk_size, estimated_memory_gb, use_phase5c


def compute_overlap_matrix_gpu_bincount(
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    chunk_size: int = 0,
) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Phase 5c: Chunked bincount-based overlap computation (SPARSE CSR FORMAT).

    This is a highly optimized GPU implementation that uses chunked processing
    to avoid OOM errors on large samples while maintaining high speedup.

    Expected speedup: 85-90x over Phase 4 baseline (9 min → 6-8 sec on D11_0).

    Algorithm:
        1. Extract valid pixels (non-zero in both masks)
        2. Remap labels to contiguous indices using np.searchsorted()
        3. Process cells in chunks to avoid OOM:
           - For each chunk of cells:
             * Compute linear indices for chunk
             * Run cp.bincount() on chunk (fits in memory)
             * Extract non-zero entries
           - Combine all chunks into sparse CSR matrix

    Memory usage: chunk_size × n_nuclei × 8 bytes per chunk (configurable).
    For chunk_size=20000 and n_nuclei=80000: ~12.7 GB per chunk.

    Args:
        cell_mask: 2D cell segmentation mask (H, W) with integer labels
        nucleus_mask: 2D nucleus segmentation mask (H, W) with integer labels
        chunk_size: Number of cells to process per chunk (default: 20000)
                   Set to 0 for auto-sizing based on available VRAM.
                   Set to None to disable chunking (use original algorithm).

    Returns:
        Tuple of:
            - overlap_sparse: scipy.sparse.csr_matrix (n_cells, n_nuclei)
            - cell_labels: (n_cells,) array of cell label IDs
            - nucleus_labels: (n_nuclei,) array of nucleus label IDs

    Raises:
        ValueError: If mask shapes don't match
        RuntimeError: If GPU processing fails (should fallback to Phase 4)
    """
    # Check GPU availability
    from aegle.gpu_utils import is_cupy_available, clear_gpu_memory, log_gpu_memory

    if not is_cupy_available():
        logger.warning("GPU not available for bincount approach, falling back")
        raise RuntimeError("GPU required for bincount overlap computation")

    try:
        import cupy as cp
        import time
        from aegle.gpu_utils import get_gpu_memory_info

        start_time = time.time()
        logger.info("Phase 5c: Computing overlap matrix using bincount approach...")
        log_gpu_memory("GPU memory before bincount overlap")

        # Ensure masks are 2D and squeeze if needed
        cell_mask = np.asarray(cell_mask).squeeze()
        nucleus_mask = np.asarray(nucleus_mask).squeeze()

        if cell_mask.shape != nucleus_mask.shape:
            raise ValueError(
                f"Mask shapes must match: cell_mask {cell_mask.shape} "
                f"vs nucleus_mask {nucleus_mask.shape}"
            )

        # Get unique labels (excluding background 0)
        logger.debug("Step 0: Extracting unique labels...")
        step_start = time.time()

        cell_labels = np.unique(cell_mask)
        cell_labels = cell_labels[cell_labels != 0].astype(np.int32)

        nucleus_labels = np.unique(nucleus_mask)
        nucleus_labels = nucleus_labels[nucleus_labels != 0].astype(np.int32)

        n_cells = len(cell_labels)
        n_nuclei = len(nucleus_labels)

        logger.info(
            f"  Found {n_cells:,} cells × {n_nuclei:,} nuclei "
            f"({time.time() - step_start:.2f}s)"
        )

        if n_cells == 0 or n_nuclei == 0:
            # Empty case - return empty sparse matrix
            overlap_sparse = csr_matrix((n_cells, n_nuclei), dtype=np.uint8)
            logger.info("Empty overlap matrix (no cells or nuclei)")
            return overlap_sparse, cell_labels, nucleus_labels

        # STEP 1: Extract valid pixels (non-zero in both masks)
        logger.debug("Step 1: Extracting valid pixels...")
        step_start = time.time()

        # Transfer masks to GPU
        cell_mask_gpu = cp.asarray(cell_mask, dtype=cp.int32)
        nucleus_mask_gpu = cp.asarray(nucleus_mask, dtype=cp.int32)

        # Flatten masks
        cell_flat = cell_mask_gpu.ravel()
        nucleus_flat = nucleus_mask_gpu.ravel()

        # Create valid pixel mask (non-zero in both)
        valid_mask = (cell_flat != 0) & (nucleus_flat != 0)
        n_valid_pixels = int(cp.sum(valid_mask))

        # Extract valid pixel IDs
        cell_ids_valid = cell_flat[valid_mask]
        nucleus_ids_valid = nucleus_flat[valid_mask]

        logger.info(
            f"  Valid pixels: {n_valid_pixels:,} / {cell_mask.size:,} "
            f"({100 * n_valid_pixels / cell_mask.size:.1f}%) "
            f"({time.time() - step_start:.2f}s)"
        )

        mem_info = get_gpu_memory_info()
        if mem_info:
            logger.debug(
                f"  GPU memory after extraction: "
                f"{mem_info['used_gb']:.1f}/{mem_info['total_gb']:.1f} GB"
            )

        # Free original masks
        del cell_mask_gpu, nucleus_mask_gpu, cell_flat, nucleus_flat, valid_mask
        clear_gpu_memory()

        # STEP 2: Remap labels to contiguous indices [0, n-1]
        logger.debug("Step 2: Remapping labels to contiguous indices...")
        step_start = time.time()

        # Sort labels for binary search
        cell_labels_sorted = np.sort(cell_labels)
        nucleus_labels_sorted = np.sort(nucleus_labels)

        # Transfer valid IDs to CPU for searchsorted (faster than GPU for this size)
        cell_ids_valid_cpu = cp.asnumpy(cell_ids_valid)
        nucleus_ids_valid_cpu = cp.asnumpy(nucleus_ids_valid)

        # Binary search: label → index
        cell_indices_cpu = np.searchsorted(cell_labels_sorted, cell_ids_valid_cpu)
        nucleus_indices_cpu = np.searchsorted(nucleus_labels_sorted, nucleus_ids_valid_cpu)

        logger.info(
            f"  Remapped {n_valid_pixels:,} pixels to indices "
            f"({time.time() - step_start:.2f}s)"
        )

        # Transfer indices back to GPU
        # Use int64 to prevent overflow when n_cells * n_nuclei > 2^31
        cell_indices = cp.asarray(cell_indices_cpu, dtype=cp.int64)
        nucleus_indices = cp.asarray(nucleus_indices_cpu, dtype=cp.int64)

        # Free CPU arrays
        del cell_ids_valid, nucleus_ids_valid, cell_ids_valid_cpu, nucleus_ids_valid_cpu
        del cell_indices_cpu, nucleus_indices_cpu
        clear_gpu_memory()

        mem_info = get_gpu_memory_info()
        if mem_info:
            logger.debug(
                f"  GPU memory after remapping: "
                f"{mem_info['used_gb']:.1f}/{mem_info['total_gb']:.1f} GB"
            )

        # STEP 3: Auto-size chunk if requested
        if chunk_size == 0:
            # Use proactive adaptive chunk sizing
            chunk_size, estimated_mem_gb, use_phase5c = calculate_optimal_chunk_size(
                n_cells, n_nuclei, num_gpus=1, safety_margin=0.85, min_viable_chunk=500
            )

            if not use_phase5c:
                # Chunk size too small for Phase 5c to be efficient
                logger.warning(
                    f"Adaptive sizing determined Phase 5c not optimal "
                    f"(chunk_size={chunk_size:,} < min_viable)"
                )
                logger.warning("Fallback to Phase 4 should have been triggered earlier")
                # Raise exception to trigger fallback in wrapper
                raise RuntimeError(
                    f"Chunk size {chunk_size:,} too small for efficient Phase 5c operation"
                )

        # Check if chunking is needed
        needs_chunking = (chunk_size is not None) and (chunk_size < n_cells)

        if not needs_chunking:
            # Use original non-chunked algorithm
            logger.info("Processing all cells in single pass (no chunking)")

            # STEP 3: Compute linear indices for 2D → 1D raveling
            logger.debug("Step 3: Computing linear indices (2D → 1D)...")
            step_start = time.time()

            linear_indices = cell_indices * n_nuclei + nucleus_indices

            logger.info(
                f"  Computed {n_valid_pixels:,} linear indices "
                f"({time.time() - step_start:.2f}s)"
            )

            # Free intermediate arrays
            del cell_indices, nucleus_indices
            clear_gpu_memory()

            # STEP 4: Single bincount call
            logger.debug("Step 4: Running cp.bincount()...")
            step_start = time.time()

            overlap_counts = cp.bincount(linear_indices, minlength=n_cells * n_nuclei)

            bincount_time = time.time() - step_start
            logger.info(
                f"  cp.bincount() completed in {bincount_time:.2f}s "
                f"(processing {n_valid_pixels:,} pixels)"
            )

            # Free linear indices
            del linear_indices
            clear_gpu_memory()

            # STEP 5: Reshape and convert to sparse CSR matrix
            logger.debug("Step 5: Converting to sparse CSR matrix...")
            step_start = time.time()

            overlap_matrix = overlap_counts.reshape(n_cells, n_nuclei)
            overlap_bool = overlap_matrix > 0
            overlap_bool_cpu = cp.asnumpy(overlap_bool)

            del overlap_counts, overlap_matrix, overlap_bool
            clear_gpu_memory()

            overlap_sparse = csr_matrix(overlap_bool_cpu, dtype=np.uint8)

        else:
            # Use chunked algorithm to avoid OOM
            n_chunks = (n_cells + chunk_size - 1) // chunk_size
            chunk_memory_gb = (chunk_size * n_nuclei * 8) / 1e9

            logger.info(
                f"Using chunked bincount: {chunk_size:,} cells/chunk, "
                f"{n_chunks} chunks, {chunk_memory_gb:.2f} GB per chunk"
            )

            # STEP 3-5: Chunked processing
            all_rows = []
            all_cols = []

            for chunk_idx in range(n_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, n_cells)
                chunk_n_cells = chunk_end - chunk_start

                logger.debug(f"  Chunk {chunk_idx + 1}/{n_chunks}: cells [{chunk_start:,}, {chunk_end:,})")

                # Filter pixels belonging to this chunk
                chunk_mask = (cell_indices >= chunk_start) & (cell_indices < chunk_end)
                n_chunk_pixels = int(cp.sum(chunk_mask))

                if n_chunk_pixels == 0:
                    logger.debug(f"    Skipping (no pixels)")
                    continue

                logger.debug(f"    Pixels in chunk: {n_chunk_pixels:,}")

                # Get indices for this chunk (renumber cells to [0, chunk_n_cells))
                chunk_cell_indices = cell_indices[chunk_mask] - chunk_start
                chunk_nucleus_indices = nucleus_indices[chunk_mask]

                # Compute linear indices for this chunk
                linear_indices = chunk_cell_indices * n_nuclei + chunk_nucleus_indices

                # Bincount for this chunk
                chunk_start_time = time.time()
                counts = cp.bincount(linear_indices, minlength=chunk_n_cells * n_nuclei)
                chunk_time = time.time() - chunk_start_time

                # Reshape to 2D and find non-zero entries
                chunk_dense = counts.reshape(chunk_n_cells, n_nuclei)
                chunk_rows, chunk_cols = cp.where(chunk_dense > 0)

                # Adjust row indices to global coordinates
                chunk_rows_global = chunk_rows + chunk_start

                # Store (transfer to CPU)
                all_rows.append(cp.asnumpy(chunk_rows_global))
                all_cols.append(cp.asnumpy(chunk_cols))

                n_overlaps = len(chunk_rows)
                logger.debug(
                    f"    Bincount: {chunk_time:.2f}s, found {n_overlaps:,} overlaps"
                )

                # Free GPU memory
                del chunk_cell_indices, chunk_nucleus_indices, linear_indices
                del counts, chunk_dense, chunk_rows, chunk_cols, chunk_rows_global
                clear_gpu_memory()

            # Free indices arrays
            del cell_indices, nucleus_indices
            clear_gpu_memory()

            # Combine all chunks into sparse matrix
            logger.debug("Combining chunks into sparse CSR matrix...")
            step_start = time.time()

            all_rows_combined = np.concatenate(all_rows)
            all_cols_combined = np.concatenate(all_cols)
            all_data = np.ones(len(all_rows_combined), dtype=np.uint8)

            overlap_sparse = csr_matrix(
                (all_data, (all_rows_combined, all_cols_combined)),
                shape=(n_cells, n_nuclei),
                dtype=np.uint8
            )

        logger.info(
            f"  Converted to sparse CSR in {time.time() - step_start:.2f}s"
        )

        log_gpu_memory("GPU memory after bincount overlap")

        # Calculate statistics
        nnz = overlap_sparse.nnz
        density = nnz / (n_cells * n_nuclei) if n_cells * n_nuclei > 0 else 0
        size_mb = (
            overlap_sparse.data.nbytes
            + overlap_sparse.indices.nbytes
            + overlap_sparse.indptr.nbytes
        ) / 1e6

        total_time = time.time() - start_time

        logger.info(
            f"Phase 5c overlap matrix computed (sparse CSR): "
            f"{nnz:,} overlaps out of {n_cells * n_nuclei:,} possible pairs "
            f"({density * 100:.3f}% density), "
            f"{size_mb:.2f} MB sparse storage"
        )
        logger.info(
            f"Phase 5c total time: {total_time:.2f}s "
            f"({n_cells / total_time:.0f} cells/sec)"
        )

        return overlap_sparse, cell_labels_sorted, nucleus_labels_sorted

    except Exception as e:
        logger.error(f"Phase 5c bincount overlap computation failed: {e}")
        logger.warning("Will need to fallback to Phase 4 implementation")
        clear_gpu_memory()
        raise RuntimeError(f"Bincount overlap failed: {e}") from e


def compute_overlap_matrix_gpu_bincount_multi_gpu(
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    chunk_size: int,
    num_gpus: int = 2,
) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Multi-GPU Phase 5c: Chunked bincount with parallel GPU processing.

    Distributes chunks across multiple GPUs using round-robin assignment and
    ThreadPoolExecutor for parallel processing. Provides near-linear speedup
    (~2x for 2 GPUs).

    Algorithm:
        1. Extract valid pixels and remap labels (same as single-GPU)
        2. Calculate chunks and distribute round-robin across GPUs
        3. Each GPU processes its assigned chunks independently
        4. Combine results on CPU after parallel execution

    Args:
        cell_mask: 2D cell segmentation mask (H, W) with integer labels
        nucleus_mask: 2D nucleus segmentation mask (H, W) with integer labels
        chunk_size: Number of cells to process per chunk
        num_gpus: Number of GPUs to use (default: 2)

    Returns:
        Tuple of:
            - overlap_sparse: scipy.sparse.csr_matrix (n_cells, n_nuclei)
            - cell_labels: (n_cells,) array of cell label IDs
            - nucleus_labels: (n_nuclei,) array of nucleus label IDs

    Raises:
        RuntimeError: If GPU processing fails
    """
    from aegle.gpu_utils import is_cupy_available, clear_gpu_memory, log_gpu_memory
    from concurrent.futures import ThreadPoolExecutor
    import cupy as cp
    import time

    if not is_cupy_available():
        raise RuntimeError("GPU required for multi-GPU bincount overlap")

    try:
        start_time = time.time()
        logger.info(f"Phase 5c Multi-GPU: Computing overlap with {num_gpus} GPUs...")
        log_gpu_memory("GPU memory before multi-GPU bincount overlap")

        # Validate GPU count
        actual_gpu_count = cp.cuda.runtime.getDeviceCount()
        if num_gpus > actual_gpu_count:
            logger.warning(
                f"Requested {num_gpus} GPUs but only {actual_gpu_count} available, "
                f"using {actual_gpu_count}"
            )
            num_gpus = actual_gpu_count

        # Ensure masks are 2D
        cell_mask = np.asarray(cell_mask).squeeze()
        nucleus_mask = np.asarray(nucleus_mask).squeeze()

        if cell_mask.shape != nucleus_mask.shape:
            raise ValueError(f"Mask shapes must match")

        # STEP 0: Get unique labels (on CPU, will be shared across GPUs)
        step_start = time.time()
        cell_labels = np.unique(cell_mask)
        cell_labels = cell_labels[cell_labels != 0].astype(np.int32)
        nucleus_labels = np.unique(nucleus_mask)
        nucleus_labels = nucleus_labels[nucleus_labels != 0].astype(np.int32)

        n_cells = len(cell_labels)
        n_nuclei = len(nucleus_labels)

        logger.info(
            f"  Found {n_cells:,} cells × {n_nuclei:,} nuclei ({time.time() - step_start:.2f}s)"
        )

        if n_cells == 0 or n_nuclei == 0:
            overlap_sparse = csr_matrix((n_cells, n_nuclei), dtype=np.uint8)
            return overlap_sparse, cell_labels, nucleus_labels

        # STEP 1: Extract valid pixels (on GPU 0, will transfer to other GPUs)
        step_start = time.time()
        cp.cuda.Device(0).use()

        cell_mask_gpu = cp.asarray(cell_mask, dtype=cp.int32)
        nucleus_mask_gpu = cp.asarray(nucleus_mask, dtype=cp.int32)

        cell_flat = cell_mask_gpu.ravel()
        nucleus_flat = nucleus_mask_gpu.ravel()

        valid_mask = (cell_flat != 0) & (nucleus_flat != 0)
        n_valid_pixels = int(cp.sum(valid_mask))

        cell_ids_valid = cell_flat[valid_mask]
        nucleus_ids_valid = nucleus_flat[valid_mask]

        logger.info(
            f"  Valid pixels: {n_valid_pixels:,} / {cell_mask.size:,} "
            f"({100 * n_valid_pixels / cell_mask.size:.1f}%) ({time.time() - step_start:.2f}s)"
        )

        del cell_mask_gpu, nucleus_mask_gpu, cell_flat, nucleus_flat, valid_mask
        clear_gpu_memory()

        # STEP 2: Remap labels to indices (on CPU, shared across GPUs)
        step_start = time.time()

        cell_labels_sorted = np.sort(cell_labels)
        nucleus_labels_sorted = np.sort(nucleus_labels)

        cell_ids_valid_cpu = cp.asnumpy(cell_ids_valid)
        nucleus_ids_valid_cpu = cp.asnumpy(nucleus_ids_valid)

        cell_indices_cpu = np.searchsorted(cell_labels_sorted, cell_ids_valid_cpu)
        nucleus_indices_cpu = np.searchsorted(nucleus_labels_sorted, nucleus_ids_valid_cpu)

        logger.info(
            f"  Remapped {n_valid_pixels:,} pixels to indices ({time.time() - step_start:.2f}s)"
        )

        # Clean up GPU 0
        del cell_ids_valid, nucleus_ids_valid
        cp.cuda.Device(0).use()
        clear_gpu_memory()

        # STEP 3: Distribute chunks across GPUs
        n_chunks = (n_cells + chunk_size - 1) // chunk_size
        chunk_memory_gb = (chunk_size * n_nuclei * 8) / 1e9

        logger.info(
            f"  Using {num_gpus} GPUs: {chunk_size:,} cells/chunk, "
            f"{n_chunks} chunks, {chunk_memory_gb:.2f} GB/chunk"
        )

        # Round-robin chunk distribution
        gpu_chunks = [[] for _ in range(num_gpus)]
        for chunk_idx in range(n_chunks):
            gpu_id = chunk_idx % num_gpus
            gpu_chunks[gpu_id].append(chunk_idx)

        for gpu_id in range(num_gpus):
            logger.debug(f"    GPU {gpu_id}: {len(gpu_chunks[gpu_id])} chunks")

        # Worker function for each GPU
        def process_gpu_chunks(gpu_id, chunk_indices):
            """Process assigned chunks on a specific GPU."""
            cp.cuda.Device(gpu_id).use()

            logger.debug(f"  [GPU {gpu_id}] Starting with {len(chunk_indices)} chunks")

            # Transfer indices to this GPU
            cell_indices = cp.asarray(cell_indices_cpu, dtype=cp.int64)
            nucleus_indices = cp.asarray(nucleus_indices_cpu, dtype=cp.int64)

            results = []

            # Milestone tracking for progress reporting
            logged_milestones = set()
            milestone_targets = [20, 40, 60, 80]
            total_chunks_for_gpu = len(chunk_indices)

            for local_idx, chunk_idx in enumerate(chunk_indices):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, n_cells)
                chunk_n_cells = chunk_end - chunk_start

                # Milestone-based progress logging at INFO level
                progress_pct = 100 * (local_idx + 1) / total_chunks_for_gpu
                for milestone in milestone_targets:
                    if progress_pct >= milestone and milestone not in logged_milestones:
                        logger.info(
                            f"  [GPU {gpu_id}] Overlap: {milestone}% complete "
                            f"({local_idx+1}/{total_chunks_for_gpu} chunks)"
                        )
                        logged_milestones.add(milestone)
                        break

                if local_idx % 2 == 0:
                    logger.debug(
                        f"  [GPU {gpu_id}] Chunk {chunk_idx+1}/{n_chunks} "
                        f"({local_idx+1}/{len(chunk_indices)} for this GPU): "
                        f"cells [{chunk_start:,}, {chunk_end:,})"
                    )

                # Filter pixels for this chunk
                chunk_mask = (cell_indices >= chunk_start) & (cell_indices < chunk_end)
                n_chunk_pixels = int(cp.sum(chunk_mask))

                if n_chunk_pixels == 0:
                    continue

                # Get indices for this chunk
                chunk_cell_indices = cell_indices[chunk_mask] - chunk_start
                chunk_nucleus_indices = nucleus_indices[chunk_mask]

                # Compute linear indices
                linear_indices = chunk_cell_indices * n_nuclei + chunk_nucleus_indices

                # Bincount for this chunk
                counts = cp.bincount(linear_indices, minlength=chunk_n_cells * n_nuclei)

                # Reshape and find non-zero entries
                chunk_dense = counts.reshape(chunk_n_cells, n_nuclei)
                chunk_rows, chunk_cols = cp.where(chunk_dense > 0)

                # Adjust row indices to global coordinates
                chunk_rows_global = chunk_rows + chunk_start

                # Store (transfer to CPU)
                results.append((
                    chunk_idx,
                    cp.asnumpy(chunk_rows_global),
                    cp.asnumpy(chunk_cols)
                ))

                # Free memory
                del chunk_cell_indices, chunk_nucleus_indices, linear_indices
                del counts, chunk_dense, chunk_rows, chunk_cols, chunk_rows_global

            logger.debug(f"  [GPU {gpu_id}] Completed all {len(chunk_indices)} chunks")

            # Clean up
            del cell_indices, nucleus_indices
            cp.get_default_memory_pool().free_all_blocks()

            return results

        # Execute in parallel
        parallel_start = time.time()

        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for gpu_id in range(num_gpus):
                if len(gpu_chunks[gpu_id]) > 0:
                    future = executor.submit(process_gpu_chunks, gpu_id, gpu_chunks[gpu_id])
                    futures.append(future)

            all_results = []
            for future in futures:
                all_results.extend(future.result())

        parallel_time = time.time() - parallel_start
        logger.info(f"  Parallel processing completed in {parallel_time:.2f}s")

        # STEP 4: Combine results
        step_start = time.time()

        # Sort by chunk index to maintain order
        all_results.sort(key=lambda x: x[0])

        all_rows = [rows for _, rows, _ in all_results]
        all_cols = [cols for _, _, cols in all_results]

        all_rows_combined = np.concatenate(all_rows)
        all_cols_combined = np.concatenate(all_cols)
        all_data = np.ones(len(all_rows_combined), dtype=np.uint8)

        overlap_sparse = csr_matrix(
            (all_data, (all_rows_combined, all_cols_combined)),
            shape=(n_cells, n_nuclei),
            dtype=np.uint8
        )

        logger.info(f"  Combined to sparse CSR in {time.time() - step_start:.2f}s")

        # Statistics
        total_time = time.time() - start_time
        nnz = overlap_sparse.nnz
        density = nnz / (n_cells * n_nuclei) if n_cells * n_nuclei > 0 else 0
        size_mb = (
            overlap_sparse.data.nbytes
            + overlap_sparse.indices.nbytes
            + overlap_sparse.indptr.nbytes
        ) / 1e6

        logger.info(
            f"  ✓ Multi-GPU Phase 5c: {nnz:,} overlaps "
            f"({density * 100:.3f}% density), {size_mb:.2f} MB"
        )
        logger.info(
            f"  Total time: {total_time:.2f}s ({n_cells / total_time:.0f} cells/sec)"
        )

        return overlap_sparse, cell_labels_sorted, nucleus_labels_sorted

    except Exception as e:
        logger.error(f"Multi-GPU Phase 5c failed: {e}")
        clear_gpu_memory()
        raise RuntimeError(f"Multi-GPU bincount overlap failed: {e}") from e


def _compute_overlap_matrix_cpu(
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
    """CPU fallback for overlap matrix computation (SPARSE CSR FORMAT).

    This implements the same logic as the GPU version but using NumPy.

    Args:
        cell_mask: 2D cell segmentation mask
        nucleus_mask: 2D nucleus segmentation mask

    Returns:
        Tuple of (overlap_sparse, cell_labels, nucleus_labels)
        where overlap_sparse is scipy.sparse.csr_matrix
    """
    logger.debug("Computing overlap matrix on CPU (sparse CSR)...")

    # Ensure masks are 2D
    cell_mask = np.asarray(cell_mask).squeeze()
    nucleus_mask = np.asarray(nucleus_mask).squeeze()

    if cell_mask.shape != nucleus_mask.shape:
        raise ValueError(
            f"Mask shapes must match: cell_mask {cell_mask.shape} "
            f"vs nucleus_mask {nucleus_mask.shape}"
        )

    # Get unique labels (excluding background 0)
    cell_labels = np.unique(cell_mask)
    cell_labels = cell_labels[cell_labels != 0].astype(np.int32)

    nucleus_labels = np.unique(nucleus_mask)
    nucleus_labels = nucleus_labels[nucleus_labels != 0].astype(np.int32)

    n_cells = len(cell_labels)
    n_nuclei = len(nucleus_labels)

    logger.info(f"Computing overlap matrix (CPU): {n_cells} cells × {n_nuclei} nuclei")

    if n_cells == 0 or n_nuclei == 0:
        # Empty case - return empty sparse matrix
        overlap_sparse = csr_matrix((n_cells, n_nuclei), dtype=np.uint8)
        logger.info("Empty overlap matrix (CPU)")
        return overlap_sparse, cell_labels, nucleus_labels

    # Pre-allocate COO storage for sparse matrix construction
    row_indices = []  # Cell indices
    col_indices = []  # Nucleus indices

    # Create label→index mapping
    nucleus_label_to_idx = {label: idx for idx, label in enumerate(nucleus_labels)}

    # Flatten masks for vectorized operations
    cell_flat = cell_mask.ravel()
    nucleus_flat = nucleus_mask.ravel()

    # Process each cell
    for i, cell_id in enumerate(cell_labels):
        # Boolean mask for current cell's pixels
        cell_pixels_mask = cell_flat == cell_id

        # Get nucleus IDs at cell pixel locations
        nucleus_ids_at_cell = nucleus_flat[cell_pixels_mask]

        # Find unique nucleus IDs (excluding 0=background)
        overlapping_nuclei = np.unique(nucleus_ids_at_cell)
        overlapping_nuclei = overlapping_nuclei[overlapping_nuclei != 0]

        # Accumulate sparse entries
        for nucleus_id in overlapping_nuclei:
            if nucleus_id in nucleus_label_to_idx:
                j = nucleus_label_to_idx[nucleus_id]
                row_indices.append(i)  # Cell index
                col_indices.append(j)  # Nucleus index

    # Build sparse matrix (COO → CSR)
    if len(row_indices) == 0:
        overlap_sparse = csr_matrix((n_cells, n_nuclei), dtype=np.uint8)
        logger.info("No overlaps found (CPU)")
    else:
        data = np.ones(len(row_indices), dtype=np.uint8)
        overlap_coo = coo_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_cells, n_nuclei),
            dtype=np.uint8,
        )
        overlap_sparse = overlap_coo.tocsr()

    # Calculate statistics
    nnz = overlap_sparse.nnz
    density = nnz / (n_cells * n_nuclei) if n_cells * n_nuclei > 0 else 0
    size_mb = (
        overlap_sparse.data.nbytes
        + overlap_sparse.indices.nbytes
        + overlap_sparse.indptr.nbytes
    ) / 1e6

    logger.info(
        f"Overlap matrix computed (CPU, sparse CSR): "
        f"{nnz:,} overlaps out of {n_cells * n_nuclei:,} possible pairs "
        f"({density * 100:.3f}% density), "
        f"{size_mb:.2f} MB sparse storage"
    )

    return overlap_sparse, cell_labels, nucleus_labels
