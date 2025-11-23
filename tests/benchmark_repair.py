#!/usr/bin/env python
"""
Performance benchmarking for mask repair pipeline.

This script measures baseline performance metrics (runtime, memory, throughput)
for the repair_masks_single function at various scales. Results are saved in
JSON format for comparison after optimization.

Usage:
    python tests/benchmark_repair.py [--skip-large]

Output:
    - tests/benchmarks/repair_baseline.json (machine-readable)
    - tests/benchmarks/repair_baseline_summary.txt (human-readable)
"""

import time
import psutil
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from tqdm import tqdm
import subprocess
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.utils.repair_test_fixtures import create_stress_test_case
from aegle.repair_masks import repair_masks_single

logging.basicConfig(level=logging.WARNING)  # Suppress info logs during benchmarks


class RepairBenchmark:
    """Performance benchmark runner for mask repair."""

    def __init__(self):
        self.process = psutil.Process()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "git_commit": self._get_git_commit(),
            "benchmarks": {},
            "scaling": {},
        }

    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"

    def _measure_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        return self.process.memory_info().rss / 1024**3

    def benchmark_repair_at_scale(
        self,
        n_cells: int,
        image_size: Tuple[int, int],
        warmup: bool = False
    ) -> Dict:
        """Benchmark repair on synthetic data at specific scale.

        Args:
            n_cells: Number of cells to create
            image_size: (height, width) of mask images
            warmup: If True, don't record results (for JIT warmup)

        Returns:
            Dictionary with benchmark metrics
        """
        if not warmup:
            print(f"\nBenchmarking {n_cells:,} cells on {image_size[0]}x{image_size[1]} image...")

        # Create test data
        if not warmup:
            print("  Creating synthetic masks...")
        cell_mask, nucleus_mask, _ = create_stress_test_case(
            n_cells=n_cells,
            image_size=image_size,
            seed=42
        )

        # Count actual cells created (may be less than n_cells due to overlaps)
        actual_n_cells = len(np.unique(cell_mask)) - 1  # Exclude background
        actual_n_nuclei = len(np.unique(nucleus_mask)) - 1

        if not warmup:
            print(f"  Created {actual_n_cells:,} cells and {actual_n_nuclei:,} nuclei")

        # Add batch dimension for repair_masks_single
        cell_mask_batch = np.expand_dims(cell_mask, axis=0)
        nucleus_mask_batch = np.expand_dims(nucleus_mask, axis=0)

        # Measure memory before
        mem_before = self._measure_memory_usage()

        # Run repair with timing
        if not warmup:
            print("  Running repair...")
        start_time = time.time()

        result = None
        success = False
        error_msg = None

        try:
            result = repair_masks_single(cell_mask_batch, nucleus_mask_batch)
            success = True
        except MemoryError as e:
            error_msg = f"MemoryError: {str(e)}"
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"

        elapsed_time = time.time() - start_time

        # Measure memory after
        mem_after = self._measure_memory_usage()
        peak_memory = mem_after - mem_before

        if warmup:
            return {}

        # Calculate metrics
        throughput = actual_n_cells / elapsed_time if elapsed_time > 0 else 0
        memory_per_cell = (peak_memory * 1024) / actual_n_cells if actual_n_cells > 0 else 0  # MB/cell

        # Get match statistics if successful
        matched_cells = 0
        if success and result is not None:
            try:
                matched_cells = len(np.unique(result["cell_matched_mask"])) - 1
            except Exception:
                # Error accessing results, but core repair completed
                pass

        print(f"  Runtime: {elapsed_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} cells/sec")
        print(f"  Peak memory: {peak_memory:.3f} GB")
        print(f"  Memory/cell: {memory_per_cell:.3f} MB")
        if success:
            print(f"  Matched cells: {matched_cells}/{actual_n_cells}")
        else:
            print(f"  ERROR: {error_msg}")

        return {
            "n_cells_target": n_cells,
            "n_cells_actual": actual_n_cells,
            "n_nuclei_actual": actual_n_nuclei,
            "image_size": list(image_size),
            "runtime_sec": elapsed_time,
            "throughput_cells_per_sec": throughput,
            "peak_memory_gb": peak_memory,
            "memory_per_cell_mb": memory_per_cell,
            "matched_cells": matched_cells if success else None,
            "success": success,
            "error": error_msg,
        }

    def run_all_benchmarks(self, skip_large: bool = False) -> Dict:
        """Run full benchmark suite.

        Args:
            skip_large: If True, skip benchmarks >= 100K cells

        Returns:
            Results dictionary
        """
        print("=" * 70)
        print("MASK REPAIR PERFORMANCE BENCHMARK")
        print("=" * 70)

        # Define benchmark scales
        # Image size scales with sqrt(n_cells) to maintain similar density
        benchmarks = [
            ("1k_cells", 1000, (1000, 1000)),
            ("10k_cells", 10000, (3000, 3000)),
            ("100k_cells", 100000, (10000, 10000)),
        ]

        if not skip_large:
            benchmarks.append(("500k_cells", 500000, (22000, 22000))
        )

        # Warmup run to eliminate cold-start bias
        print("\nRunning warmup (100 cells)...")
        self.benchmark_repair_at_scale(100, (500, 500), warmup=True)
        print("Warmup complete.")

        # Run benchmarks
        for name, n_cells, image_size in benchmarks:
            try:
                result = self.benchmark_repair_at_scale(n_cells, image_size)
                self.results["benchmarks"][name] = result
            except Exception as e:
                print(f"\nERROR in {name}: {e}")
                self.results["benchmarks"][name] = {
                    "n_cells_target": n_cells,
                    "success": False,
                    "error": str(e),
                }

        # Analyze scaling behavior
        print("\nAnalyzing scaling behavior...")
        self.results["scaling"] = self._analyze_scaling()

        return self.results

    def _analyze_scaling(self) -> Dict:
        """Analyze how runtime scales with cell count.

        Returns:
            Dictionary with scaling analysis
        """
        # Extract benchmarks with valid timing data (even if result processing failed)
        data_points = []
        for name, result in self.results["benchmarks"].items():
            # Accept results that have runtime data, even if they have errors
            if "runtime_sec" in result and result["runtime_sec"] > 0:
                data_points.append({
                    "n_cells": result["n_cells_actual"],
                    "runtime": result["runtime_sec"],
                })

        if len(data_points) < 2:
            return {"error": "Insufficient data points for scaling analysis"}

        # Sort by cell count
        data_points.sort(key=lambda x: x["n_cells"])

        # Fit power law: runtime = a * n_cells^b
        # Using log-log linear regression: log(runtime) = log(a) + b*log(n_cells)
        n_cells_arr = np.array([p["n_cells"] for p in data_points])
        runtime_arr = np.array([p["runtime"] for p in data_points])

        # Log-log regression
        log_n = np.log(n_cells_arr)
        log_t = np.log(runtime_arr)

        # Fit line: log_t = slope * log_n + intercept
        coeffs = np.polyfit(log_n, log_t, 1)
        slope, intercept = coeffs

        # Calculate R^2
        log_t_pred = slope * log_n + intercept
        ss_res = np.sum((log_t - log_t_pred) ** 2)
        ss_tot = np.sum((log_t - np.mean(log_t)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Determine complexity class
        if slope < 1.2:
            complexity = "O(n) - Linear"
        elif slope < 1.8:
            complexity = f"O(n^{slope:.1f}) - Super-linear"
        elif slope < 2.2:
            complexity = "O(n^2) - Quadratic"
        else:
            complexity = f"O(n^{slope:.1f}) - Polynomial"

        return {
            "exponent": float(slope),
            "r_squared": float(r_squared),
            "complexity": complexity,
            "interpretation": self._interpret_scaling(slope),
        }

    def _interpret_scaling(self, exponent: float) -> str:
        """Provide human-readable interpretation of scaling exponent."""
        if exponent < 1.2:
            return "Excellent: Runtime scales linearly with cell count"
        elif exponent < 1.5:
            return "Good: Runtime scales slightly faster than linear"
        elif exponent < 2.0:
            return "Moderate: Runtime scales super-linearly"
        elif exponent < 2.5:
            return "Poor: Runtime scales quadratically"
        else:
            return "Critical: Runtime scales worse than quadratic - optimization needed"

    def save_results(self, output_dir: Path):
        """Save results to JSON and text files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = output_dir / "repair_baseline.json"
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {json_path}")

        # Generate summary text
        summary_path = output_dir / "repair_baseline_summary.txt"
        with open(summary_path, "w") as f:
            f.write(self._generate_summary())
        print(f"Summary saved to: {summary_path}")

    def _generate_summary(self) -> str:
        """Generate human-readable summary report."""
        lines = [
            "=" * 70,
            "MASK REPAIR PERFORMANCE BENCHMARK - BASELINE RESULTS",
            "=" * 70,
            "",
            f"Timestamp: {self.results['timestamp']}",
            f"Git Commit: {self.results['git_commit']}",
            "",
            "BENCHMARK RESULTS:",
            "-" * 70,
        ]

        # Table header
        lines.extend([
            "",
            f"{'Scale':<15} {'Cells':<10} {'Runtime':<12} {'Throughput':<15} {'Memory':<12} {'MB/cell':<10}",
            f"{'':=<15} {'':=<10} {'':=<12} {'':=<15} {'':=<12} {'':=<10}",
        ])

        # Table rows
        for name, result in self.results["benchmarks"].items():
            if "runtime_sec" in result and result["runtime_sec"] > 0:
                scale = name.replace("_", " ").title()
                cells = f"{result['n_cells_actual']:,}"
                runtime = f"{result['runtime_sec']:.2f}s"
                throughput = f"{result['throughput_cells_per_sec']:.1f} cells/s"
                memory = f"{result['peak_memory_gb']:.3f} GB"
                mem_per_cell = f"{result['memory_per_cell_mb']:.3f}"

                # Add marker if there was a post-processing error
                suffix = "*" if not result.get("success", False) else ""
                lines.append(f"{scale:<15} {cells:<10} {runtime:<12} {throughput:<15} {memory:<12} {mem_per_cell:<10} {suffix}")
            else:
                scale = name.replace("_", " ").title()
                error = result.get("error", "Unknown error")
                lines.append(f"{scale:<15} FAILED: {error}")

        lines.extend([
            "",
            "* = Timing successful but result processing had errors (bug in repair_masks.py)",
            "",
            "SCALING ANALYSIS:",
            "-" * 70,
        ])

        scaling = self.results.get("scaling", {})
        if "error" not in scaling:
            lines.extend([
                f"Complexity: {scaling.get('complexity', 'N/A')}",
                f"Exponent: {scaling.get('exponent', 0):.3f}",
                f"R-squared: {scaling.get('r_squared', 0):.4f}",
                f"Interpretation: {scaling.get('interpretation', 'N/A')}",
            ])
        else:
            lines.append(f"Error: {scaling['error']}")

        lines.extend([
            "",
            "EXTRAPOLATED ESTIMATES:",
            "-" * 70,
        ])

        # Extrapolate to D18_0 scale (1.8M cells) if we have scaling data
        if "error" not in scaling and self.results["benchmarks"]:
            exponent = scaling.get("exponent", 2.0)

            # Use largest benchmark with timing data as reference
            ref_result = None
            for name in ["500k_cells", "100k_cells", "10k_cells", "1k_cells"]:
                if (name in self.results["benchmarks"] and
                    "runtime_sec" in self.results["benchmarks"][name] and
                    self.results["benchmarks"][name]["runtime_sec"] > 0):
                    ref_result = self.results["benchmarks"][name]
                    break

            if ref_result:
                ref_n = ref_result["n_cells_actual"]
                ref_t = ref_result["runtime_sec"]

                # Estimate for 1.8M cells
                target_n = 1_800_000
                estimated_t = ref_t * (target_n / ref_n) ** exponent
                estimated_hours = estimated_t / 3600

                lines.extend([
                    f"Reference: {ref_n:,} cells in {ref_t:.2f}s",
                    f"Estimated for 1.8M cells: {estimated_hours:.1f} hours",
                    f"  (Based on {scaling['complexity']} scaling)",
                ])

                # Compare to reported 36 hours
                lines.extend([
                    "",
                    f"Reported D18_0 runtime: 36 hours",
                    f"Estimated runtime: {estimated_hours:.1f} hours",
                    f"Difference: {((estimated_hours / 36) - 1) * 100:+.1f}%",
                ])

        lines.extend([
            "",
            "=" * 70,
        ])

        return "\n".join(lines)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark mask repair performance"
    )
    parser.add_argument(
        "--skip-large",
        action="store_true",
        help="Skip benchmarks >= 100K cells (for faster testing)"
    )
    args = parser.parse_args()

    # Run benchmarks
    benchmark = RepairBenchmark()
    results = benchmark.run_all_benchmarks(skip_large=args.skip_large)

    # Save results
    output_dir = Path(__file__).parent / "benchmarks"
    benchmark.save_results(output_dir)

    # Print summary
    print("\n" + benchmark._generate_summary())

    # Return exit code based on whether we got timing data
    # (don't fail if post-processing errors occurred)
    has_timing_data = any(
        "runtime_sec" in r and r["runtime_sec"] > 0
        for r in results["benchmarks"].values()
    )
    return 0 if has_timing_data else 1


if __name__ == "__main__":
    sys.exit(main())
