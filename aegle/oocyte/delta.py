"""Incremental secondary-rescue artifacts built from a completed v6 run."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import tifffile
import zarr

from .config import OOCYTE_IMPLEMENTATION_VERSION, get_profile
from .detection import _atomic_write_csv, _atomic_write_json, candidate_score
from .io import extract_cyx_channel_patch, load_candidate_mask, save_candidate_mask
from .models import ScoredCandidateMask, SegmentationMetrics
from .rescue import run_secondary_rescue


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RescueDeltaBatchResult:
    out_dir: Path
    summary: pd.DataFrame
    artifact_paths: Dict[str, Path]


def _load_baseline_masks(
    sample_dir: Path,
    candidates: pd.DataFrame,
) -> Dict[str, ScoredCandidateMask]:
    masks = {}
    for row in candidates.to_dict("records"):
        candidate_id = str(row["detector_component_id"])
        persisted = load_candidate_mask(sample_dir / str(row["mask_path"]))
        masks[candidate_id] = ScoredCandidateMask(
            mask=persisted.mask,
            bbox=persisted.bbox,
            image_shape_yx=persisted.image_shape_yx,
            metrics=SegmentationMetrics(**persisted.metadata["metrics"]),
        )
    return masks


def generate_rescue_delta_batch(
    baseline_batch_dir: Path,
    *,
    out_dir: Path,
    profile_name: str = "donor13_v6_rescue_v1",
    sample_ids: Iterable[str] | None = None,
) -> RescueDeltaBatchResult:
    """Run only the secondary pass and persist a rescue-only review batch."""

    baseline_root = Path(baseline_batch_dir).resolve()
    destination = Path(out_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    profile = get_profile(profile_name)
    if profile.secondary_rescue is None:
        raise ValueError(f"profile {profile_name!r} has no secondary rescue pass")
    baseline_summary_path = baseline_root / "batch_summary.csv"
    if not baseline_summary_path.is_file():
        raise FileNotFoundError(
            f"baseline batch summary not found: {baseline_summary_path}"
        )
    baseline_summary = pd.read_csv(baseline_summary_path)
    requested = None if sample_ids is None else {str(value) for value in sample_ids}
    if requested is not None:
        known = set(baseline_summary["sample_id"].astype(str))
        missing = sorted(requested - known)
        if missing:
            raise ValueError(f"sample IDs not present in baseline batch: {missing}")

    summary_rows: List[Dict[str, object]] = []
    for summary_row in baseline_summary.to_dict("records"):
        sample_id = str(summary_row["sample_id"])
        if requested is not None and sample_id not in requested:
            continue
        if str(summary_row["status"]) != "complete":
            summary_rows.append(
                {
                    "sample_id": sample_id,
                    "status": "skipped_incomplete_baseline",
                    "rescue_candidate_count": 0,
                }
            )
            continue
        LOGGER.info("Secondary rescue starting sample %s", sample_id)
        started = time.perf_counter()
        baseline_sample_dir = baseline_root / sample_id
        sample_dir = destination / sample_id
        masks_dir = sample_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        manifest = json.loads(
            (baseline_sample_dir / "run_manifest.json").read_text()
        )
        image_path = Path(manifest["source_image"])
        channel_index = int(manifest["resolved_channel_index"])
        coarse = pd.read_csv(baseline_sample_dir / "coarse_candidates.csv")
        baseline_candidates = pd.read_csv(
            baseline_sample_dir / "candidates.csv"
        )
        baseline_masks = _load_baseline_masks(
            baseline_sample_dir,
            baseline_candidates,
        )
        with tifffile.TiffFile(image_path) as tif:
            series = tif.series[0]
            if series.axes != "CYX":
                raise ValueError(
                    f"secondary rescue requires CYX axes, got {series.axes!r}"
                )
            array = zarr.open(series.aszarr(), mode="r")
            image_shape = (int(array.shape[1]), int(array.shape[2]))

            def patch_reader(center_x: int, center_y: int, radius: int):
                return extract_cyx_channel_patch(
                    array,
                    channel_index,
                    (center_x, center_y),
                    radius,
                )

            result = run_secondary_rescue(
                patch_reader=patch_reader,
                image_shape_yx=image_shape,
                coarse_candidates=coarse,
                baseline_candidates=baseline_candidates,
                baseline_masks=baseline_masks,
                config=profile,
                score_candidate=candidate_score,
            )

        rescue_candidates = result.candidates[
            result.candidates.get("segmentation_pass", pd.Series(dtype=str))
            == "secondary_rescue"
        ].copy()
        mask_paths = []
        for candidate_id in rescue_candidates[
            "detector_component_id"
        ].astype(str):
            relative_path = Path("masks") / f"{candidate_id}.npz"
            candidate_mask = result.candidate_masks[candidate_id]
            save_candidate_mask(
                sample_dir / relative_path,
                mask=candidate_mask.mask,
                bbox=candidate_mask.bbox,
                image_shape_yx=candidate_mask.image_shape_yx,
                sample_id=sample_id,
                candidate_id=candidate_id,
                profile_name=profile.profile_name,
                profile_fingerprint=profile.fingerprint(),
                metrics=candidate_mask.metrics,
                implementation_version=OOCYTE_IMPLEMENTATION_VERSION,
            )
            mask_paths.append(str(relative_path))
        rescue_candidates["mask_path"] = mask_paths
        _atomic_write_csv(rescue_candidates, sample_dir / "candidates.csv")
        _atomic_write_csv(result.diagnostics, sample_dir / "rescue_diagnostics.csv")
        run_seconds = float(time.perf_counter() - started)
        sample_summary = {
            "schema_version": 1,
            "sample_id": sample_id,
            "status": "complete",
            "baseline_batch_dir": str(baseline_root),
            "baseline_profile_name": str(manifest.get("resolved_config", {}).get("profile_name", "")),
            "profile_name": profile.profile_name,
            "profile_fingerprint": profile.fingerprint(),
            "baseline_accepted_candidate_count": int(
                baseline_candidates["accepted"].astype(bool).sum()
            ),
            "rescue_candidate_count": int(len(rescue_candidates)),
            "rescue_diagnostic_count": int(len(result.diagnostics)),
            "runtime_seconds": run_seconds,
        }
        _atomic_write_json(sample_summary, sample_dir / "summary.json")
        _atomic_write_json(
            {
                "schema_version": 1,
                "sample_id": sample_id,
                "source_image": str(image_path),
                "resolved_channel_index": channel_index,
                "baseline_sample_dir": str(baseline_sample_dir),
                "resolved_config": profile.to_dict(),
                "profile_fingerprint": profile.fingerprint(),
                "implementation_version": OOCYTE_IMPLEMENTATION_VERSION,
            },
            sample_dir / "run_manifest.json",
        )
        summary_rows.append(
            {
                "sample_id": sample_id,
                "status": "complete",
                "baseline_accepted_candidate_count": int(
                    baseline_candidates["accepted"].astype(bool).sum()
                ),
                "rescue_candidate_count": int(len(rescue_candidates)),
                "rescue_diagnostic_count": int(len(result.diagnostics)),
                "runtime_seconds": run_seconds,
                "sample_dir": str(sample_dir),
            }
        )
        LOGGER.info(
            "Secondary rescue completed sample %s: %s candidates in %.1fs",
            sample_id,
            len(rescue_candidates),
            run_seconds,
        )

    summary = pd.DataFrame(summary_rows)
    summary_csv = destination / "batch_summary.csv"
    summary_json = destination / "batch_summary.json"
    _atomic_write_csv(summary, summary_csv)
    _atomic_write_json(
        {
            "schema_version": 1,
            "baseline_batch_dir": str(baseline_root),
            "profile_name": profile.profile_name,
            "profile_fingerprint": profile.fingerprint(),
            "samples": summary_rows,
        },
        summary_json,
    )
    return RescueDeltaBatchResult(
        out_dir=destination,
        summary=summary,
        artifact_paths={
            "batch_summary_csv": summary_csv,
            "batch_summary_json": summary_json,
        },
    )


__all__ = ["RescueDeltaBatchResult", "generate_rescue_delta_batch"]
