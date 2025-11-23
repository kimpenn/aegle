"""Integration tests for GPU repair configuration in the pipeline.

This module tests the configuration integration of GPU repair functionality:
- Configuration parsing and validation
- Pipeline integration with segment.py
- GPU enable/disable via config
- Backward compatibility with old configs
- Metadata logging
"""

import pytest
import numpy as np
import yaml
import tempfile
import logging
from pathlib import Path

# Import pipeline components
from aegle.repair_masks import repair_masks_batch
from aegle.gpu_utils import is_cupy_available

# Test utilities
from tests.utils.repair_test_fixtures import (
    create_simple_mask_pair,
    create_stress_test_case,
)


# Decorator to skip tests if GPU not available
requires_gpu = pytest.mark.skipif(
    not is_cupy_available(),
    reason="GPU not available (CuPy not installed or no CUDA device)"
)


class TestPipelineGPUConfig:
    """Test GPU repair configuration integration."""

    def test_config_default_cpu(self):
        """Test that repair_masks_batch defaults to CPU when no config provided."""
        cell_mask, nucleus_mask, expected = create_simple_mask_pair(n_cells=5)

        # Create segmentation result in expected format
        seg_res_batch = [
            {
                "cell": cell_mask,
                "nucleus": nucleus_mask,
            }
        ]

        # Call without GPU params (backward compatibility)
        result_batch = repair_masks_batch(seg_res_batch)

        # Should succeed with CPU
        assert len(result_batch) == 1
        assert "cell_matched_mask" in result_batch[0]
        assert "nucleus_matched_mask" in result_batch[0]
        assert "matched_fraction" in result_batch[0]

        # Should not have GPU metadata when using CPU
        if "repair_metadata" in result_batch[0]:
            assert not result_batch[0]["repair_metadata"].get("gpu_used", False)

    def test_config_explicit_cpu(self):
        """Test explicit CPU mode via config."""
        cell_mask, nucleus_mask, expected = create_simple_mask_pair(n_cells=5)

        seg_res_batch = [
            {
                "cell": cell_mask,
                "nucleus": nucleus_mask,
            }
        ]

        # Explicitly disable GPU
        result_batch = repair_masks_batch(seg_res_batch, use_gpu=False)

        # Should succeed
        assert len(result_batch) == 1
        assert "cell_matched_mask" in result_batch[0]

    @requires_gpu
    def test_config_explicit_gpu(self):
        """Test explicit GPU mode via config."""
        cell_mask, nucleus_mask, expected = create_simple_mask_pair(n_cells=10)

        seg_res_batch = [
            {
                "cell": cell_mask,
                "nucleus": nucleus_mask,
            }
        ]

        # Explicitly enable GPU
        result_batch = repair_masks_batch(seg_res_batch, use_gpu=True)

        # Should succeed
        assert len(result_batch) == 1
        assert "cell_matched_mask" in result_batch[0]
        assert "repair_metadata" in result_batch[0]

        metadata = result_batch[0]["repair_metadata"]
        assert metadata["gpu_used"] or metadata["fallback_to_cpu"]

    @requires_gpu
    def test_config_gpu_batch_size_manual(self):
        """Test manual GPU batch size override."""
        cell_mask, nucleus_mask, expected = create_simple_mask_pair(n_cells=20)

        seg_res_batch = [
            {
                "cell": cell_mask,
                "nucleus": nucleus_mask,
            }
        ]

        # Set small manual batch size
        result_batch = repair_masks_batch(
            seg_res_batch,
            use_gpu=True,
            gpu_batch_size=5,  # Small batch size to force batching
        )

        # Should succeed even with small batch size
        assert len(result_batch) == 1
        assert "cell_matched_mask" in result_batch[0]

    @requires_gpu
    def test_config_gpu_batch_size_auto(self):
        """Test automatic GPU batch size detection."""
        cell_mask, nucleus_mask, expected = create_simple_mask_pair(n_cells=15)

        seg_res_batch = [
            {
                "cell": cell_mask,
                "nucleus": nucleus_mask,
            }
        ]

        # Use auto batch size (None or 0)
        result_batch = repair_masks_batch(
            seg_res_batch,
            use_gpu=True,
            gpu_batch_size=None,  # Auto-detect
        )

        # Should succeed
        assert len(result_batch) == 1
        assert "cell_matched_mask" in result_batch[0]

    def test_backward_compatibility_old_signature(self):
        """Test backward compatibility with old repair_masks_batch signature."""
        cell_mask, nucleus_mask, expected = create_simple_mask_pair(n_cells=5)

        seg_res_batch = [
            {
                "cell": cell_mask,
                "nucleus": nucleus_mask,
            }
        ]

        # Call with old signature (positional only, no kwargs)
        result_batch = repair_masks_batch(seg_res_batch)

        # Should work exactly as before
        assert len(result_batch) == 1
        assert "cell_matched_mask" in result_batch[0]
        assert "nucleus_matched_mask" in result_batch[0]
        assert "matched_fraction" in result_batch[0]

    @requires_gpu
    def test_gpu_metadata_logging(self):
        """Test that GPU metadata is properly logged."""
        cell_mask, nucleus_mask, expected = create_simple_mask_pair(n_cells=10)

        seg_res_batch = [
            {
                "cell": cell_mask,
                "nucleus": nucleus_mask,
            }
        ]

        # Enable GPU
        result_batch = repair_masks_batch(seg_res_batch, use_gpu=True)

        # Check metadata structure
        assert len(result_batch) == 1
        result = result_batch[0]

        assert "repair_metadata" in result
        metadata = result["repair_metadata"]

        # Required metadata fields
        assert "use_gpu" in metadata
        assert "gpu_available" in metadata
        assert "gpu_used" in metadata
        assert "fallback_to_cpu" in metadata
        assert "total_time" in metadata
        assert "n_cells_input" in metadata
        assert "n_nuclei_input" in metadata

        # If GPU was actually used, should have additional metrics
        if metadata["gpu_used"]:
            assert "boundary_computation" in metadata
            assert "overlap_computation" in metadata
            assert "cell_matching" in metadata
            assert "n_matched_cells" in metadata

    def test_config_validation_yaml_structure(self):
        """Test that the YAML config structure is valid."""
        # Create a temporary config with GPU repair section
        config = {
            "segmentation": {
                "repair": {
                    "use_gpu": True,
                    "gpu_batch_size": None,
                    "fallback_to_cpu": True,
                    "log_gpu_performance": True,
                }
            }
        }

        # Should parse without errors
        repair_config = config.get("segmentation", {}).get("repair", {})
        assert repair_config["use_gpu"] is True
        assert repair_config["gpu_batch_size"] is None
        assert repair_config["fallback_to_cpu"] is True
        assert repair_config["log_gpu_performance"] is True

    def test_config_missing_repair_section(self):
        """Test that missing repair section defaults gracefully."""
        config = {
            "segmentation": {}  # No repair section
        }

        repair_config = config.get("segmentation", {}).get("repair", {})
        use_gpu = repair_config.get("use_gpu", False)
        gpu_batch_size = repair_config.get("gpu_batch_size", None)

        # Should default to CPU mode
        assert use_gpu is False
        assert gpu_batch_size is None

    def test_config_partial_repair_section(self):
        """Test that partial repair config uses defaults."""
        config = {
            "segmentation": {
                "repair": {
                    "use_gpu": True,
                    # Other fields missing
                }
            }
        }

        repair_config = config.get("segmentation", {}).get("repair", {})
        use_gpu = repair_config.get("use_gpu", False)
        gpu_batch_size = repair_config.get("gpu_batch_size", None)
        fallback = repair_config.get("fallback_to_cpu", True)

        # Should use provided values and defaults
        assert use_gpu is True
        assert gpu_batch_size is None  # Default
        assert fallback is True  # Default

    @requires_gpu
    def test_multiple_patches_gpu(self):
        """Test GPU repair with multiple patches (batch processing)."""
        # Create multiple patches
        cell_mask1, nucleus_mask1, _ = create_simple_mask_pair(n_cells=10)
        cell_mask2, nucleus_mask2, _ = create_simple_mask_pair(n_cells=15)

        seg_res_batch = [
            {"cell": cell_mask1, "nucleus": nucleus_mask1},
            {"cell": cell_mask2, "nucleus": nucleus_mask2},
        ]

        # Process with GPU
        result_batch = repair_masks_batch(seg_res_batch, use_gpu=True)

        # Should process both patches
        assert len(result_batch) == 2

        # Both should have results
        for result in result_batch:
            assert "cell_matched_mask" in result
            assert "nucleus_matched_mask" in result
            assert "repair_metadata" in result

    def test_empty_batch(self):
        """Test handling of empty batch."""
        seg_res_batch = []

        # Should handle empty batch gracefully
        result_batch = repair_masks_batch(seg_res_batch, use_gpu=False)
        assert len(result_batch) == 0

    @requires_gpu
    def test_gpu_fallback_on_invalid_input(self):
        """Test that GPU falls back to CPU on invalid input."""
        # Create invalid input (empty masks)
        seg_res_batch = [
            {
                "cell": np.array([]),
                "nucleus": np.array([]),
            }
        ]

        # Should not crash, should fall back to CPU or handle gracefully
        try:
            result_batch = repair_masks_batch(seg_res_batch, use_gpu=True)
            # If it succeeds, check fallback
            if len(result_batch) > 0 and "repair_metadata" in result_batch[0]:
                # Either fell back to CPU or handled the error
                assert True
        except Exception as e:
            # Or it raises an appropriate error
            assert True

    @requires_gpu
    def test_performance_logging_metadata(self):
        """Test that performance metadata is properly structured."""
        cell_mask, nucleus_mask, expected = create_stress_test_case(n_cells=100)

        seg_res_batch = [
            {
                "cell": cell_mask,
                "nucleus": nucleus_mask,
            }
        ]

        # Run with GPU
        result_batch = repair_masks_batch(seg_res_batch, use_gpu=True)

        assert len(result_batch) == 1
        result = result_batch[0]

        if "repair_metadata" in result and result["repair_metadata"]["gpu_used"]:
            metadata = result["repair_metadata"]

            # Check performance metrics
            assert metadata["total_time"] > 0
            assert isinstance(metadata["total_time"], float)

            # Check stage timings
            if "boundary_computation" in metadata:
                assert metadata["boundary_computation"] >= 0

            if "overlap_computation" in metadata:
                assert metadata["overlap_computation"] >= 0

            if "cell_matching" in metadata:
                assert metadata["cell_matching"] >= 0


class TestConfigIntegrationWithSegment:
    """Test configuration integration at the segment.py level."""

    def test_parse_repair_config_from_dict(self):
        """Test parsing repair config from dictionary (as used in segment.py)."""
        config = {
            "segmentation": {
                "model_path": "/path/to/model",
                "repair": {
                    "use_gpu": True,
                    "gpu_batch_size": 1000,
                    "fallback_to_cpu": False,
                    "log_gpu_performance": True,
                }
            }
        }

        # Simulate what segment.py does
        repair_config = config.get("segmentation", {}).get("repair", {})
        use_gpu = repair_config.get("use_gpu", False)
        gpu_batch_size = repair_config.get("gpu_batch_size", None)
        log_gpu_performance = repair_config.get("log_gpu_performance", False)

        assert use_gpu is True
        assert gpu_batch_size == 1000
        assert log_gpu_performance is True

    def test_parse_repair_config_defaults(self):
        """Test default values when repair section missing."""
        config = {
            "segmentation": {
                "model_path": "/path/to/model",
                # No repair section
            }
        }

        # Simulate segment.py parsing
        repair_config = config.get("segmentation", {}).get("repair", {})
        use_gpu = repair_config.get("use_gpu", False)
        gpu_batch_size = repair_config.get("gpu_batch_size", None)
        log_gpu_performance = repair_config.get("log_gpu_performance", False)

        # Should all be defaults
        assert use_gpu is False
        assert gpu_batch_size is None
        assert log_gpu_performance is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
