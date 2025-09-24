#!/usr/bin/env python3
"""Run preprocess experiments for tests with cleanup and assertions."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import numbers
import shutil

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency check
    raise SystemExit("PyYAML is required to run preprocess tests") from exc


@dataclass
class Experiment:
    name: str
    config_path: Path
    data_path: Path
    manual_annotation: Optional[Path]
    expected_labels: List[str]
    output_dir: Path


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PIPELINE_ROOT.parent
ROOT_DIR = WORKSPACE_ROOT

sys.path.insert(0, str(PIPELINE_ROOT))
from src.extract_tissue_regions import parse_napari_json_annotations
DATA_DIR = ROOT_DIR / "data"
CONFIG_ROOT = PIPELINE_ROOT / "exps" / "configs" / "preprocess" / "test"
LOG_ROOT = PIPELINE_ROOT / "logs" / "preprocess" / "test"

RUN_TISSUE_SCRIPT = PIPELINE_ROOT / "scripts" / "run_extract_tissue.sh"
RUN_ANTIBODY_SCRIPT = PIPELINE_ROOT / "scripts" / "run_extract_antibody.sh"
RUN_OVERVIEW_SCRIPT = PIPELINE_ROOT / "scripts" / "run_generate_overview.sh"

EXP_SET_NAME = "preprocess/test"


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def discover_experiments() -> Dict[str, Experiment]:
    experiments: Dict[str, Experiment] = {}
    if not CONFIG_ROOT.exists():
        return experiments

    for cfg_dir in sorted(CONFIG_ROOT.iterdir()):
        cfg_path = cfg_dir / "config.yaml"
        if not cfg_path.is_file():
            continue

        cfg = load_yaml(cfg_path)
        try:
            data_path = Path(cfg["data"]["file_name"])
        except KeyError as exc:
            raise KeyError(f"Missing data.file_name in {cfg_path}") from exc

        tissue_cfg = cfg.get("tissue_extraction", {})
        output_dir_cfg = tissue_cfg.get("output_dir")
        if not output_dir_cfg:
            raise KeyError(f"Missing tissue_extraction.output_dir in {cfg_path}")
        output_dir = Path(output_dir_cfg)

        manual_json = tissue_cfg.get("manual_mask_json")
        manual_annotation = Path(manual_json) if manual_json else None
        expected_labels = derive_expected_labels(manual_annotation)

        experiments[cfg_dir.name] = Experiment(
            name=cfg_dir.name,
            config_path=cfg_path,
            data_path=data_path,
            manual_annotation=manual_annotation,
            expected_labels=expected_labels,
            output_dir=output_dir,
        )

    return experiments


def derive_expected_labels(annotation_path: Optional[Path]) -> List[str]:
    if not annotation_path or not annotation_path.exists():
        return []

    if annotation_path.is_dir():
        labels: List[str] = []
        candidate_files = sorted(
            p for p in annotation_path.iterdir()
            if p.is_file() and p.suffix.lower() in {'.json', '.geojson'}
        )
        if not candidate_files:
            raise FileNotFoundError(f"No annotation files found in directory: {annotation_path}")
        for json_path in candidate_files:
            polygons, _ = parse_napari_json_annotations(str(json_path))
            if not polygons:
                logging.warning(f"No polygons detected in annotation file: {json_path}")
                continue
            base_label = json_path.name.split('.')[0]
            if len(polygons) == 1:
                labels.append(base_label)
            else:
                labels.extend(f"{base_label}_{idx + 1}" for idx in range(len(polygons)))
        if not labels:
            raise ValueError(f"No valid polygons parsed from directory: {annotation_path}")
        return labels

    return _labels_from_single_file(annotation_path)



def _labels_from_single_file(path: Path) -> List[str]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    entries: List[dict] = []

    if isinstance(data, dict) and data.get("type") == "FeatureCollection":
        for feature in data.get("features", []):
            if isinstance(feature, dict):
                entries.append(feature.get("properties", {}) or {})
        return [_safe_label(props, idx) for idx, props in enumerate(entries)]

    shapes_data: Iterable = ()
    if isinstance(data, list):
        shapes_data = data
    elif isinstance(data, dict):
        if "shapes" in data:
            shapes_data = data["shapes"]
        elif "data" in data:
            shapes_data = data["data"]
        else:
            shapes_data = data.values()

    labels: List[str] = []
    for idx, shape in enumerate(shapes_data):
        props: dict = {}
        if isinstance(shape, dict):
            props = shape.get("properties", {}) or {}
        labels.append(_safe_label(props, idx))
    return labels


def _safe_label(props: dict, fallback_index: int) -> str:
    label_val = props.get("idx")
    if label_val is None:
        label_val = props.get("name")
    if label_val is None and isinstance(props.get("classification"), dict):
        label_val = props["classification"].get("name")
    if label_val is None:
        label_val = fallback_index

    if isinstance(label_val, numbers.Integral):
        label_str = str(int(label_val))
    elif isinstance(label_val, numbers.Real):
        label_str = str(int(label_val)) if float(label_val).is_integer() else str(label_val)
    else:
        label_str = str(label_val).strip()

    if not label_str:
        label_str = str(fallback_index)

    safe = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in label_str)
    return safe or str(fallback_index)


def _format_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def clean_outputs(exp: Experiment, base_name: str) -> None:
    removed: List[str] = []
    log_path = LOG_ROOT / f"{exp.name}.log"
    if log_path.exists():
        log_path.unlink()
        removed.append(_format_path(log_path))

    patterns = [
        f"{base_name}_manual_*.ome.tiff",
        f"{base_name}_manual_*_ch*.jpg",
    ]
    for pattern in patterns:
        for path in exp.output_dir.glob(pattern):
            path.unlink()
            removed.append(_format_path(path))

    overlay_path = exp.output_dir / "manual_polygon_overlay.png"
    if overlay_path.exists():
        overlay_path.unlink()
        removed.append(_format_path(overlay_path))

    extras_dir = exp.data_path.parent / "extras"
    if extras_dir.exists():
        shutil.rmtree(extras_dir)
        removed.append(_format_path(extras_dir))

    if removed:
        print(f"[cleanup] Removed previous outputs for {exp.name}:")
        for item in removed:
            print(f"  - {item}")


def run_pipeline(exp: Experiment, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    common_args = [
        EXP_SET_NAME,
        exp.name,
        str(ROOT_DIR),
        str(DATA_DIR),
        str(CONFIG_ROOT),
        str(DATA_DIR),
    ]

    steps = [
        (RUN_TISSUE_SCRIPT, "tissue extraction"),
        (RUN_ANTIBODY_SCRIPT, "antibody extraction"),
        (RUN_OVERVIEW_SCRIPT, "overview generation"),
    ]

    with log_path.open("w", encoding="utf-8") as log_handle:
        for script_path, description in steps:
            if not script_path.exists():
                raise FileNotFoundError(f"Missing script: {script_path}")

            log_handle.write(f"\n=== Running {description}: {script_path.name} ===\n")
            log_handle.flush()

            cmd = [str(script_path), *common_args]
            subprocess.run(cmd, stdout=log_handle, stderr=subprocess.STDOUT, check=True)


def verify_outputs(exp: Experiment, base_name: str, started_at: float) -> None:
    if exp.expected_labels:
        ome_files = sorted(exp.output_dir.glob(f"{base_name}_manual_*.ome.tiff"))
        if len(ome_files) != len(exp.expected_labels):
            raise AssertionError(
                f"Expected {len(exp.expected_labels)} ROI OME files for {exp.name}, "
                f"found {len(ome_files)}"
            )

        for label in exp.expected_labels:
            ome_path = exp.output_dir / f"{base_name}_manual_{label}.ome.tiff"
            if not ome_path.exists():
                raise AssertionError(f"Missing ROI file: {ome_path}")
            if ome_path.stat().st_mtime < started_at:
                raise AssertionError(f"ROI file not refreshed: {ome_path}")

            jpg_candidates = sorted(
                exp.output_dir.glob(f"{base_name}_manual_{label}.ome_ch*.jpg")
            )
            if not jpg_candidates:
                raise AssertionError(f"Missing overview JPEG for label '{label}'")
            if all(path.stat().st_mtime < started_at for path in jpg_candidates):
                raise AssertionError(f"Overview JPEG not refreshed for label '{label}'")

    overlay_path = exp.output_dir / "manual_polygon_overlay.png"
    if not overlay_path.exists():
        raise AssertionError("manual_polygon_overlay.png was not created")
    if overlay_path.stat().st_mtime < started_at:
        raise AssertionError("manual_polygon_overlay.png was not updated")

    extras_dir = exp.data_path.parent / "extras"
    antibodies_file = extras_dir / "antibodies.tsv"
    if not antibodies_file.exists():
        raise AssertionError(f"Missing antibody TSV: {antibodies_file}")
    if antibodies_file.stat().st_mtime < started_at:
        raise AssertionError("antibodies.tsv was not regenerated")

    log_path = LOG_ROOT / f"{exp.name}.log"
    if not log_path.exists():
        raise AssertionError(f"Missing log file: {log_path}")
    log_text = log_path.read_text(encoding="utf-8")
    if "Overview JPEG generation completed." not in log_text:
        raise AssertionError("Overview step did not complete successfully (log check)")


def run_single_experiment(exp: Experiment, skip_cleanup: bool) -> None:
    base_name = Path(exp.data_path).stem
    if not skip_cleanup:
        clean_outputs(exp, base_name)

    started_at = time.time()
    log_path = LOG_ROOT / f"{exp.name}.log"
    run_pipeline(exp, log_path)
    verify_outputs(exp, base_name, started_at)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run preprocess test experiments")
    parser.add_argument(
        "--experiments",
        nargs="*",
        help="Optional subset of experiments to run (defaults to all configs)",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip removing existing outputs before running",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = discover_experiments()

    if not experiments:
        raise SystemExit(f"No experiments found under {CONFIG_ROOT}")

    selected: Dict[str, Experiment]
    if args.experiments:
        missing = [name for name in args.experiments if name not in experiments]
        if missing:
            raise SystemExit(f"Unknown experiments requested: {', '.join(missing)}")
        selected = {name: experiments[name] for name in args.experiments}
    else:
        selected = experiments

    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    failures: List[str] = []
    for exp in selected.values():
        print(f"\n▶ Running preprocess experiment: {exp.name}")
        try:
            run_single_experiment(exp, args.skip_cleanup)
        except Exception as exc:
            failures.append(f"{exp.name}: {exc}")
            print(f"❌ {exp.name} failed: {exc}")
        else:
            print(f"✅ {exp.name} passed")

    if failures:
        print("\nPreprocess tests completed with failures:")
        for failure in failures:
            print(f" - {failure}")
        raise SystemExit(1)

    print("\nAll preprocess experiments passed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)
