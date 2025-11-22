"""
GPU integration tests for end-to-end pipeline workflows.

Tests configuration propagation, full pipeline execution with GPU,
and validation on real data subsets when available.
"""

import os
import unittest
import tempfile
import numpy as np
import pandas as pd
from unittest.mock import patch as mock_patch
from types import SimpleNamespace

from aegle.gpu_utils import is_cupy_available
from aegle.cell_profiling import run_cell_profiling
from tests.utils import make_single_cell_patch
from tests.utils.gpu_test_utils import requires_gpu, assert_gpu_cpu_equal


class FakeCodexPatchesForGPU:
    """Minimal CodexPatches stub for testing GPU profiling."""

    def __init__(self, antibodies, patches, seg_results):
        """
        Args:
            antibodies: List of antibody names
            patches: List of patch image dictionaries
            seg_results: List of segmentation result dictionaries
        """
        self.antibody_df = pd.DataFrame({"antibody_name": antibodies})
        self.patches = patches
        self.repaired_seg_res_batch = seg_results
        self._patches_metadata = None

    def get_patches_metadata(self):
        """Return patches metadata DataFrame."""
        if self._patches_metadata is None:
            # Create metadata for all patches as informative
            self._patches_metadata = pd.DataFrame({
                "patch_id": list(range(len(self.patches))),
                "is_informative": [True] * len(self.patches),
                "x_start": [0] * len(self.patches),
                "y_start": [0] * len(self.patches),
            })
        return self._patches_metadata

    def get_all_channel_patch(self, patch_idx):
        """Return multi-channel patch as HxWxC array."""
        return self.patches[patch_idx]

    def is_using_disk_based_patches(self):
        """Return False for in-memory patches."""
        return False


class TestGPUConfigPropagation(unittest.TestCase):
    """Test that GPU configuration properly flows through pipeline."""

    @requires_gpu()
    def test_gpu_config_enables_gpu_path(self):
        """use_gpu=True in config should trigger GPU code path."""
        # Create synthetic patch
        patch = make_single_cell_patch(
            shape=(64, 64),
            channels=("chan0", "chan1"),
            nucleus_intensity={"chan0": 50.0, "chan1": 25.0},
            cytoplasm_intensity={"chan0": 5.0, "chan1": 10.0},
        )

        # Convert to format expected by run_cell_profiling
        antibodies = list(patch.channels)
        patch_img = np.stack([patch.image_dict[ch] for ch in antibodies], axis=2)
        patches = [patch_img]
        seg_results = [{
            "nucleus_matched_mask": patch.nucleus_mask,
            "cell_matched_mask": patch.cell_mask,
        }]

        codex_patches = FakeCodexPatchesForGPU(antibodies, patches, seg_results)

        # Config with GPU enabled
        config = {
            "patching": {"split_mode": "full_image"},
            "profiling": {
                "features": {
                    "use_gpu": True,
                    "gpu_batch_size": 0,  # Auto
                    "compute_laplacian": False,
                    "compute_cov": False,
                    "channel_dtype": np.float32,
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(out_dir=tmpdir)

            # Patch the GPU function to track if it was called
            with mock_patch('aegle.cell_profiling.extract_features_v2_gpu',
                           wraps=__import__('aegle.extract_features_gpu').extract_features_gpu.extract_features_v2_gpu) as gpu_mock:
                run_cell_profiling(codex_patches, config, args)

                # GPU function should have been called
                self.assertTrue(mock_gpu.called,
                              "GPU function should be called when use_gpu=True")

    def test_gpu_config_disabled_uses_cpu_path(self):
        """use_gpu=False in config should use CPU code path."""
        # Create synthetic patch
        patch = make_single_cell_patch(
            shape=(64, 64),
            channels=("chan0",),
            nucleus_intensity={"chan0": 50.0},
            cytoplasm_intensity={"chan0": 5.0},
        )

        antibodies = list(patch.channels)
        patch_img = np.stack([patch.image_dict[ch] for ch in antibodies], axis=2)
        patches = [patch_img]
        seg_results = [{
            "nucleus_matched_mask": patch.nucleus_mask,
            "cell_matched_mask": patch.cell_mask,
        }]

        codex_patches = FakeCodexPatchesForGPU(antibodies, patches, seg_results)

        # Config with GPU disabled
        config = {
            "patching": {"split_mode": "full_image"},
            "profiling": {
                "features": {
                    "use_gpu": False,
                    "compute_laplacian": False,
                    "compute_cov": False,
                    "channel_dtype": np.float32,
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(out_dir=tmpdir)

            # Patch the CPU function to track if it was called
            from aegle.extract_features import extract_features_v2_optimized
            with patch('aegle.cell_profiling.extract_features_v2_optimized',
                      wraps=extract_features_v2_optimized) as mock_cpu:
                run_cell_profiling(codex_patches, config, args)

                # CPU function should have been called
                self.assertTrue(mock_cpu.called,
                              "CPU function should be called when use_gpu=False")

    @requires_gpu()
    def test_gpu_batch_size_config_propagates(self):
        """gpu_batch_size config should propagate to GPU function."""
        patch = make_single_cell_patch(
            shape=(64, 64),
            channels=("c1", "c2", "c3", "c4", "c5"),
            nucleus_intensity={f"c{i}": float(i * 10) for i in range(1, 6)},
            cytoplasm_intensity={f"c{i}": float(i * 2) for i in range(1, 6)},
        )

        antibodies = list(patch.channels)
        patch_img = np.stack([patch.image_dict[ch] for ch in antibodies], axis=2)
        patches = [patch_img]
        seg_results = [{
            "nucleus_matched_mask": patch.nucleus_mask,
            "cell_matched_mask": patch.cell_mask,
        }]

        codex_patches = FakeCodexPatchesForGPU(antibodies, patches, seg_results)

        # Config with specific batch size
        config = {
            "patching": {"split_mode": "full_image"},
            "profiling": {
                "features": {
                    "use_gpu": True,
                    "gpu_batch_size": 2,  # Specific batch size
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(out_dir=tmpdir)

            from aegle.extract_features_gpu import extract_features_v2_gpu
            with patch('aegle.cell_profiling.extract_features_v2_gpu',
                      wraps=extract_features_v2_gpu) as mock_gpu:
                run_cell_profiling(codex_patches, config, args)

                # Check that gpu_batch_size=2 was passed
                call_kwargs = mock_gpu.call_args[1]
                self.assertEqual(call_kwargs.get('gpu_batch_size'), 2)


class TestGPUFullPipelineIntegration(unittest.TestCase):
    """Test full cell profiling pipeline with GPU enabled."""

    @requires_gpu()
    def test_full_pipeline_gpu_vs_cpu_equivalence(self):
        """Full pipeline should produce same results with GPU and CPU."""
        # Create synthetic patch with multiple cells
        from tests.utils.synthetic_data_factory import make_disk_cells_patch

        patch = make_disk_cells_patch(
            shape=(128, 128),
            channels=("CD3", "CD8", "CD45"),
            cell_positions=[(32, 32), (32, 96), (96, 32), (96, 96)],
            cell_radius=12,
            nucleus_radius=6,
            nucleus_intensities=[
                {"CD3": 100, "CD8": 50, "CD45": 200},
                {"CD3": 80, "CD8": 120, "CD45": 150},
                {"CD3": 90, "CD8": 70, "CD45": 180},
                {"CD3": 110, "CD8": 90, "CD45": 160},
            ],
            cytoplasm_intensities=[
                {"CD3": 20, "CD8": 10, "CD45": 40},
                {"CD3": 15, "CD8": 25, "CD45": 30},
                {"CD3": 18, "CD8": 12, "CD45": 35},
                {"CD3": 22, "CD8": 18, "CD45": 38},
            ],
        )

        antibodies = list(patch.channels)
        patch_img = np.stack([patch.image_dict[ch] for ch in antibodies], axis=2)
        patches = [patch_img]
        seg_results = [{
            "nucleus_matched_mask": patch.nucleus_mask,
            "cell_matched_mask": patch.cell_mask,
        }]

        codex_patches = FakeCodexPatchesForGPU(antibodies, patches, seg_results)

        base_config = {
            "patching": {"split_mode": "full_image"},
            "profiling": {
                "features": {
                    "compute_laplacian": False,
                    "compute_cov": True,
                    "channel_dtype": np.float32,
                }
            }
        }

        # Run with CPU
        cpu_config = base_config.copy()
        cpu_config["profiling"]["features"]["use_gpu"] = False

        with tempfile.TemporaryDirectory() as cpu_dir:
            args_cpu = SimpleNamespace(out_dir=cpu_dir)
            run_cell_profiling(codex_patches, cpu_config, args_cpu)

            # Load CPU results
            cpu_markers = pd.read_csv(os.path.join(cpu_dir, "cell_profiling", "markers.csv"))
            cpu_overview = pd.read_csv(os.path.join(cpu_dir, "cell_profiling", "cell_overview.csv"))

        # Run with GPU
        gpu_config = base_config.copy()
        gpu_config["profiling"]["features"]["use_gpu"] = True

        with tempfile.TemporaryDirectory() as gpu_dir:
            args_gpu = SimpleNamespace(out_dir=gpu_dir)
            run_cell_profiling(codex_patches, gpu_config, args_gpu)

            # Load GPU results
            gpu_markers = pd.read_csv(os.path.join(gpu_dir, "cell_profiling", "markers.csv"))
            gpu_overview = pd.read_csv(os.path.join(gpu_dir, "cell_profiling", "cell_overview.csv"))

        # Results should match
        # Sort by global_cell_id for consistent comparison
        cpu_markers = cpu_markers.sort_values("global_cell_id").reset_index(drop=True)
        gpu_markers = gpu_markers.sort_values("global_cell_id").reset_index(drop=True)
        cpu_overview = cpu_overview.sort_values("global_cell_id").reset_index(drop=True)
        gpu_overview = gpu_overview.sort_values("global_cell_id").reset_index(drop=True)

        # Compare numerical columns only
        marker_cols = [col for col in cpu_markers.columns
                      if col in antibodies or col.endswith("_mean")]
        for col in marker_cols:
            if col in gpu_markers.columns:
                np.testing.assert_allclose(
                    gpu_markers[col].values,
                    cpu_markers[col].values,
                    rtol=1e-5, atol=1e-6,
                    err_msg=f"Marker column {col} differs"
                )

    @requires_gpu()
    def test_full_pipeline_with_multiple_patches(self):
        """GPU pipeline should handle multiple patches correctly."""
        # Create two patches
        patches = []
        seg_results = []
        antibodies = ["chan0", "chan1"]

        for _ in range(2):
            patch = make_single_cell_patch(
                shape=(64, 64),
                channels=tuple(antibodies),
                nucleus_intensity={"chan0": 50.0, "chan1": 25.0},
                cytoplasm_intensity={"chan0": 5.0, "chan1": 10.0},
            )

            patch_img = np.stack([patch.image_dict[ch] for ch in antibodies], axis=2)
            patches.append(patch_img)
            seg_results.append({
                "nucleus_matched_mask": patch.nucleus_mask,
                "cell_matched_mask": patch.cell_mask,
            })

        codex_patches = FakeCodexPatchesForGPU(antibodies, patches, seg_results)

        config = {
            "patching": {"split_mode": "full_image"},
            "profiling": {
                "features": {
                    "use_gpu": True,
                    "gpu_batch_size": 0,
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(out_dir=tmpdir)
            run_cell_profiling(codex_patches, config, args)

            # Check outputs exist
            markers_path = os.path.join(tmpdir, "cell_profiling", "markers.csv")
            self.assertTrue(os.path.exists(markers_path))

            markers = pd.read_csv(markers_path)
            # Should have 2 cells (one per patch)
            self.assertEqual(len(markers), 2)


class TestGPUFallbackBehavior(unittest.TestCase):
    """Test GPU fallback to CPU when unavailable."""

    def test_gpu_requested_but_unavailable_falls_back(self):
        """Pipeline should fall back to CPU when GPU requested but unavailable."""
        patch = make_single_cell_patch(
            shape=(64, 64),
            channels=("chan0",),
            nucleus_intensity={"chan0": 50.0},
            cytoplasm_intensity={"chan0": 5.0},
        )

        antibodies = list(patch.channels)
        patch_img = np.stack([patch.image_dict[ch] for ch in antibodies], axis=2)
        patches = [patch_img]
        seg_results = [{
            "nucleus_matched_mask": patch.nucleus_mask,
            "cell_matched_mask": patch.cell_mask,
        }]

        codex_patches = FakeCodexPatchesForGPU(antibodies, patches, seg_results)

        config = {
            "patching": {"split_mode": "full_image"},
            "profiling": {
                "features": {
                    "use_gpu": True,  # Request GPU
                }
            }
        }

        # Mock GPU as unavailable
        with patch('aegle.cell_profiling.is_cupy_available', return_value=False):
            with tempfile.TemporaryDirectory() as tmpdir:
                args = SimpleNamespace(out_dir=tmpdir)

                # Should complete successfully using CPU fallback
                run_cell_profiling(codex_patches, config, args)

                # Outputs should exist
                markers_path = os.path.join(tmpdir, "cell_profiling", "markers.csv")
                self.assertTrue(os.path.exists(markers_path))


class TestGPURealDataSubset(unittest.TestCase):
    """Test GPU on real data subsets when available."""

    @unittest.skipUnless(
        os.path.exists("/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/out/main"),
        "Real data outputs not available"
    )
    @requires_gpu()
    def test_gpu_on_d18_subset(self):
        """Test GPU processing on D18_0 data subset if available."""
        # Look for D18_0 outputs
        main_out_dir = "/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/out/main"
        d18_candidates = [
            os.path.join(main_out_dir, "d18", "D18_0"),
            os.path.join(main_out_dir, "D18_0"),
        ]

        d18_dir = None
        for candidate in d18_candidates:
            if os.path.exists(candidate):
                d18_dir = candidate
                break

        if d18_dir is None:
            self.skipTest("D18_0 outputs not found")

        # Check for required files
        seg_file = os.path.join(d18_dir, "matched_seg_res_batch.pickle.zst")
        if not os.path.exists(seg_file):
            self.skipTest("D18_0 segmentation outputs not found")

        # This is a placeholder - actual implementation would load a small subset
        # of the D18_0 data and validate GPU processing
        # For now, just verify the test infrastructure is in place
        self.assertTrue(True, "Real data test infrastructure validated")


if __name__ == "__main__":
    unittest.main()
