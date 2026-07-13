"""Manifest-driven standalone oocyte batch execution."""

from __future__ import annotations

import json
import os
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .config import OOCYTE_IMPLEMENTATION_VERSION, get_profile
from .detection import detect_oocytes
from .io import find_channel_index
from .qc import compile_batch_spatial_qc


@dataclass(frozen=True)
class OocyteSampleInput:
    sample_id: str
    image_path: Path
    pixel_size_um: float
    enabled: bool
    profile: str
    channel_index: int | None
    channel_name: str
    antibodies_path: Path | None


@dataclass(frozen=True)
class OocyteBatchResult:
    manifest_path: Path
    out_dir: Path
    summary: pd.DataFrame
    artifact_paths: Dict[str, Path]

    @property
    def failed_count(self) -> int:
        return int((self.summary["status"] == "failed").sum())


def _parse_bool(value: Any, *, default: bool) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _optional_text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"expected an integer, got {value!r}")
    return int(number)


def _resolve_path(value: Any, manifest_dir: Path) -> Path | None:
    text = _optional_text(value)
    if text is None:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def load_sample_manifest(
    manifest_path: Path,
    *,
    default_profile: str = "donor13_v6",
    default_channel_name: str = "UCHL1",
) -> List[OocyteSampleInput]:
    """Parse and structurally validate an oocyte sample CSV manifest."""

    path = Path(manifest_path).resolve()
    table = pd.read_csv(path, keep_default_na=True)
    required = {"sample_id", "image_path", "pixel_size_um"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"sample manifest missing columns: {sorted(missing)}")
    sample_ids = table["sample_id"].astype(str).str.strip()
    if (sample_ids == "").any():
        raise ValueError("sample manifest contains an empty sample_id")
    duplicates = sorted(sample_ids[sample_ids.duplicated()].unique())
    if duplicates:
        raise ValueError(f"sample manifest contains duplicate sample IDs: {duplicates}")

    rows = []
    for row_number, record in enumerate(table.to_dict("records"), start=2):
        try:
            sample_id = str(record["sample_id"]).strip()
            image_path = _resolve_path(record["image_path"], path.parent)
            if image_path is None:
                raise ValueError("image_path is empty")
            pixel_size_um = float(record["pixel_size_um"])
            if pixel_size_um <= 0:
                raise ValueError("pixel_size_um must be positive")
            enabled = _parse_bool(record.get("enabled"), default=True)
            profile = _optional_text(record.get("profile")) or default_profile
            get_profile(profile)
            channel_index = _optional_int(record.get("channel_index"))
            if channel_index is not None and channel_index < 0:
                raise ValueError("channel_index must not be negative")
            channel_name = (
                _optional_text(record.get("channel_name")) or default_channel_name
            )
            antibodies_path = _resolve_path(
                record.get("antibodies_path"),
                path.parent,
            )
            if enabled and channel_index is None and antibodies_path is None:
                raise ValueError(
                    "enabled rows require channel_index or antibodies_path"
                )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid sample manifest row {row_number}: {exc}") from exc
        rows.append(
            OocyteSampleInput(
                sample_id=sample_id,
                image_path=image_path,
                pixel_size_um=pixel_size_um,
                enabled=enabled,
                profile=profile,
                channel_index=channel_index,
                channel_name=channel_name,
                antibodies_path=antibodies_path,
            )
        )
    return rows


def _complete_record(
    sample: OocyteSampleInput,
    sample_out_dir: Path,
) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        result = detect_oocytes(
            sample.image_path,
            sample_id=sample.sample_id,
            out_dir=sample_out_dir,
            config=get_profile(sample.profile),
            channel_name=sample.channel_name,
            channel_index=sample.channel_index,
            antibodies_path=sample.antibodies_path,
            pixel_size_um=sample.pixel_size_um,
        )
        return {
            "sample_id": sample.sample_id,
            "status": "complete",
            "profile": result.profile_name,
            "profile_fingerprint": result.profile_fingerprint,
            "implementation_version": result.implementation_version,
            "image_path": str(result.image_path),
            "image_height": int(result.image_shape_yx[0]),
            "image_width": int(result.image_shape_yx[1]),
            "channel_index": int(result.channel_index),
            "coarse_candidate_count": int(len(result.coarse_candidates)),
            "refined_candidate_count": int(len(result.candidates)),
            "accepted_candidate_count": int(result.candidates["accepted"].sum())
            if not result.candidates.empty
            else 0,
            "runtime_seconds": float(result.runtime_seconds["total"]),
            "output_dir": str(sample_out_dir),
            "candidates_path": str(result.artifact_paths["candidates"]),
            "labels_path": str(result.artifact_paths["labels"]),
            "error_type": "",
            "error_message": "",
            "traceback": "",
            "resumed": False,
        }
    except Exception as exc:
        return {
            "sample_id": sample.sample_id,
            "status": "failed",
            "profile": sample.profile,
            "profile_fingerprint": "",
            "implementation_version": OOCYTE_IMPLEMENTATION_VERSION,
            "image_path": str(sample.image_path),
            "image_height": None,
            "image_width": None,
            "channel_index": sample.channel_index,
            "coarse_candidate_count": None,
            "refined_candidate_count": None,
            "accepted_candidate_count": None,
            "runtime_seconds": float(time.perf_counter() - started),
            "output_dir": str(sample_out_dir),
            "candidates_path": "",
            "labels_path": "",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "resumed": False,
        }


def _skipped_record(sample: OocyteSampleInput, sample_out_dir: Path) -> Dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "status": "skipped",
        "profile": sample.profile,
        "profile_fingerprint": "",
        "implementation_version": OOCYTE_IMPLEMENTATION_VERSION,
        "image_path": str(sample.image_path),
        "image_height": None,
        "image_width": None,
        "channel_index": sample.channel_index,
        "coarse_candidate_count": None,
        "refined_candidate_count": None,
        "accepted_candidate_count": None,
        "runtime_seconds": 0.0,
        "output_dir": str(sample_out_dir),
        "candidates_path": "",
        "labels_path": "",
        "error_type": "",
        "error_message": "disabled in manifest",
        "traceback": "",
        "resumed": False,
    }


def _resume_record(
    sample: OocyteSampleInput,
    sample_out_dir: Path,
) -> Dict[str, Any] | None:
    summary_path = sample_out_dir / "summary.json"
    manifest_path = sample_out_dir / "run_manifest.json"
    candidates_path = sample_out_dir / "candidates.csv"
    labels_path = sample_out_dir / "oocyte_labels.ome.tiff"
    if not all(
        path.is_file()
        for path in (summary_path, manifest_path, candidates_path, labels_path)
    ):
        return None
    try:
        summary = json.loads(summary_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        config = get_profile(sample.profile)
        if sample.pixel_size_um != config.pixel_size_um:
            config = replace(config, pixel_size_um=sample.pixel_size_um)
        expected_channel_index = sample.channel_index
        if expected_channel_index is None:
            if sample.antibodies_path is None:
                return None
            expected_channel_index = find_channel_index(
                sample.antibodies_path,
                sample.channel_name,
            )
        if str(Path(manifest["source_image"]).resolve()) != str(
            sample.image_path.resolve()
        ):
            return None
        if manifest.get("profile_fingerprint") != config.fingerprint():
            return None
        if manifest.get("implementation_version") != OOCYTE_IMPLEMENTATION_VERSION:
            return None
        if int(manifest["resolved_channel_index"]) != int(expected_channel_index):
            return None
        if summary.get("status") != "complete":
            return None
        return {
            "sample_id": sample.sample_id,
            "status": "complete",
            "profile": str(summary["profile_name"]),
            "profile_fingerprint": str(summary["profile_fingerprint"]),
            "implementation_version": str(summary["implementation_version"]),
            "image_path": str(sample.image_path),
            "image_height": int(summary["image_shape_yx"][0]),
            "image_width": int(summary["image_shape_yx"][1]),
            "channel_index": int(summary["channel_index"]),
            "coarse_candidate_count": int(summary["coarse_candidate_count"]),
            "refined_candidate_count": int(summary["refined_candidate_count"]),
            "accepted_candidate_count": int(summary["accepted_candidate_count"]),
            "runtime_seconds": float(summary["runtime_seconds"]["total"]),
            "output_dir": str(sample_out_dir),
            "candidates_path": str(candidates_path),
            "labels_path": str(labels_path),
            "error_type": "",
            "error_message": "",
            "traceback": "",
            "resumed": True,
        }
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _atomic_write_json(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_csv(table: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            table.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _resolved_manifest_row(sample: OocyteSampleInput) -> Dict[str, Any]:
    row = asdict(sample)
    row["image_path"] = str(sample.image_path)
    row["antibodies_path"] = (
        None if sample.antibodies_path is None else str(sample.antibodies_path)
    )
    return row


def detect_oocyte_batch(
    manifest_path: Path,
    *,
    out_dir: Path,
    jobs: int = 1,
    continue_on_error: bool = True,
    resume_completed: bool = True,
    default_profile: str = "donor13_v6",
    default_channel_name: str = "UCHL1",
) -> OocyteBatchResult:
    """Run all enabled manifest rows and preserve one status per input sample."""

    samples = load_sample_manifest(
        manifest_path,
        default_profile=default_profile,
        default_channel_name=default_channel_name,
    )
    destination = Path(out_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    enabled = [sample for sample in samples if sample.enabled]
    records_by_sample = {
        sample.sample_id: _skipped_record(sample, destination / sample.sample_id)
        for sample in samples
        if not sample.enabled
    }
    pending = []
    for sample in enabled:
        resumed = (
            _resume_record(sample, destination / sample.sample_id)
            if resume_completed
            else None
        )
        if resumed is None:
            pending.append(sample)
        else:
            records_by_sample[sample.sample_id] = resumed
    worker_count = max(1, int(jobs))
    worker_count = min(worker_count, len(pending)) if pending else 1

    if worker_count == 1:
        for sample in pending:
            record = _complete_record(sample, destination / sample.sample_id)
            records_by_sample[sample.sample_id] = record
            if record["status"] == "failed" and not continue_on_error:
                break
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _complete_record,
                    sample,
                    destination / sample.sample_id,
                ): sample
                for sample in pending
            }
            stop_requested = False
            for future in as_completed(futures):
                sample = futures[future]
                record = future.result()
                records_by_sample[sample.sample_id] = record
                if record["status"] == "failed" and not continue_on_error:
                    stop_requested = True
                    for pending_future in futures:
                        pending_future.cancel()
                    break
            if stop_requested:
                for sample in pending:
                    if sample.sample_id not in records_by_sample:
                        records_by_sample[sample.sample_id] = {
                            **_skipped_record(sample, destination / sample.sample_id),
                            "error_message": "not run after fail-fast error",
                        }

    for sample in pending:
        if sample.sample_id not in records_by_sample:
            records_by_sample[sample.sample_id] = {
                **_skipped_record(sample, destination / sample.sample_id),
                "error_message": "not run after fail-fast error",
            }
    records = [records_by_sample[sample.sample_id] for sample in samples]
    summary = pd.DataFrame(records)
    summary_csv = destination / "batch_summary.csv"
    summary_json = destination / "batch_summary.json"
    resolved_manifest = destination / "batch_manifest.json"
    _atomic_write_csv(summary, summary_csv)
    _atomic_write_json(records, summary_json)
    _atomic_write_json(
        {
            "schema_version": 1,
            "source_manifest": str(Path(manifest_path).resolve()),
            "jobs": worker_count,
            "continue_on_error": bool(continue_on_error),
            "resume_completed": bool(resume_completed),
            "samples": [_resolved_manifest_row(sample) for sample in samples],
        },
        resolved_manifest,
    )
    spatial_qc = compile_batch_spatial_qc(destination, summary)
    artifact_paths = {
        "batch_summary_csv": summary_csv,
        "batch_summary_json": summary_json,
        "batch_manifest": resolved_manifest,
        "spatial_qc_dir": spatial_qc.qc_dir,
        "spatial_qc_index": spatial_qc.overview_index_path,
        "duplicate_suspects": spatial_qc.duplicate_suspects_path,
    }
    if spatial_qc.overview_atlas_path is not None:
        artifact_paths["spatial_qc_atlas"] = spatial_qc.overview_atlas_path
    return OocyteBatchResult(
        manifest_path=Path(manifest_path).resolve(),
        out_dir=destination,
        summary=summary,
        artifact_paths=artifact_paths,
    )
